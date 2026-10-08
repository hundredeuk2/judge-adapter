# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Pre-submission CLIENT-SIDE cost gate and stop rules around one backend (RunGate), plus its estimate helpers.

NOT an account billing cap: the real charge is visible only in the Anthropic / RunPod consoles. No worker control.
Not a general budget framework: one backend, one sequential run, one cap.

Estimate rules:
- choice_endpoint: estimate seconds = max(settled seconds, wall clock since the first submission) + idle allowance
  + full reservation of every unsettled job; reservation = ttl 600 s + idle 60 s + 1 s. A job settles ONLY on an
  observed terminal state, at delay + execution + 1 s (full reservation without timing). USD = seconds x the
  configured per-second rate (no built-in rate).
- Haiku: committed = settled cost from usage + full reservation of every request whose usage is unknown (held,
  never zeroed). A request starts only if committed + its reservation <= cap. The reservation assumes input tokens
  <= UTF-8 body bytes + 200, charged at 2x the input price, plus max_tokens of output.

A local configuration failure (outcome local_config: endpoint / rate / key provider / missing key / client init,
submissions 0) sent nothing: its job reservation and first-submission clock are released, no Haiku reservation is
held, and the run stops. Reservations of jobs that were actually transmitted are never released this way.
"""
from __future__ import annotations

import json
import time

from . import contract as C
from . import prompt as P

IDLE_ALLOWANCE_SECONDS = 60.0            # assumption (12x the documented 5 s default); endpoint setting not visible
ROUNDING_ALLOWANCE_SECONDS = 1.0
JOB_RESERVATION_SECONDS = P.CHOICE_POLICY['ttl'] / 1000 + IDLE_ALLOWANCE_SECONDS + ROUNDING_ALLOWANCE_SECONDS  # 661
HAIKU_TOKEN_OVERHEAD = 200               # assumption: input tokens <= UTF-8 body bytes + 200 (message framing)
HAIKU_CACHE_MULTIPLIER = 2.0             # an unexpected cache path is charged conservatively


def haiku_reservation_usd(body):
    nbytes = len(json.dumps(body, ensure_ascii=False).encode('utf-8'))
    return round((nbytes + HAIKU_TOKEN_OVERHEAD) * P.HAIKU_PRICE_IN_PER_MTOK * HAIKU_CACHE_MULTIPLIER / 1e6
                 + P.HAIKU_MAX_TOKENS * P.HAIKU_PRICE_OUT_PER_MTOK / 1e6, 8)


def haiku_usd(usage):
    return usage['input_tokens'] * P.HAIKU_PRICE_IN_PER_MTOK / 1e6 + usage['output_tokens'] * P.HAIKU_PRICE_OUT_PER_MTOK / 1e6


class JobTime:
    """Time-based estimate for the choice endpoint, one worker."""

    def __init__(self, rate, clock):
        self.rate, self.clock = float(rate), clock
        self.first_submit = None
        self.settled_seconds = 0.0
        self.unsettled = {}

    def seconds(self, extra=0.0):
        sess = 0.0
        if self.first_submit is not None:
            sess = max(self.settled_seconds, self.clock() - self.first_submit) + IDLE_ALLOWANCE_SECONDS
        return sess + sum(self.unsettled.values()) + extra

    def usd(self, extra=0.0):
        return self.seconds(extra) * self.rate

    def reserve(self, label):
        if self.first_submit is None:
            self.first_submit = self.clock()
        self.unsettled[label] = JOB_RESERVATION_SECONDS

    def settle(self, label, timing):
        self.unsettled.pop(label)
        d, e = (timing or {}).get('delay_ms'), (timing or {}).get('execution_ms')
        sec = JOB_RESERVATION_SECONDS if d is None or e is None else (d + e) / 1000.0 + ROUNDING_ALLOWANCE_SECONDS
        self.settled_seconds += sec
        return sec


class RunGate:
    """Run-wide limits around one backend: max_submissions and a CLIENT-SIDE cost cap for the WHOLE run.

    NOT an account billing cap. Wraps a backend with the same call shape; refused calls return outcome
    not_submitted with 0 submissions. Call ``begin(request)`` before each ``JudgeAdapter.judge(request)``.
    - Calls are keyed by request id + direction, never by semantic hash. Two requests with identical content both
      run; the same request id + direction is never submitted twice.
    - two_direction: the first direction is admitted only if BOTH directions fit (2 submissions, 2 reservations).
    Stop rules: cap would be exceeded, max_submissions reached, first contract failure, an unsettled choice-endpoint job, Haiku
    usage unknown (e.g. HTTP error) or above its reservation, a local configuration failure (local_config: nothing
    transmitted; its reservation is released, nothing held). After a stop every later call is refused without a
    request (outcome not_submitted). Retries stay 0; status polls and cancels are counted apart from submissions.
    """

    def __init__(self, backend, mode, cap_usd, max_submissions, clock=time.perf_counter):
        if max_submissions is None or max_submissions <= 0:
            raise ValueError('max_submissions must be positive')
        if cap_usd is not None and cap_usd <= 0:
            raise ValueError('budget cap must be positive')
        self.inner = backend
        self.name, self.model = backend.name, backend.model
        self.mode = mode
        self.cap = float('inf') if cap_usd is None else float(cap_usd)
        self.max_submissions = int(max_submissions)
        self.clock = clock
        self.current = None
        self.submitted_keys = []
        self.submissions = self.status_polls = self.cancels = 0
        self.stopped = None
        self.job_time = JobTime(getattr(backend, 'usd_per_second', None) or 0.0, clock)   # haiku_direct: unused
        self.h_spent = self.h_held = 0.0
        self.unsettled_jobs, self.log = [], []

    def begin(self, req):
        self.current = req

    def rendered_input(self, sem):
        return self.inner.rendered_input(sem)

    def cost(self, calls):
        return self.inner.cost(calls)

    @property
    def cost_basis(self):
        return self.inner.cost_basis

    def close(self):
        self.inner.close()

    def committed_usd(self):
        return self.job_time.usd() if self.name == 'choice_endpoint' else self.h_spent + self.h_held

    def _refuse(self, key, why, stop=True):
        if stop and self.stopped is None:
            self.stopped = why
        rec = {'decision': 'refused', 'call_key': key, 'why': why, 'committed_usd': self.committed_usd()}
        self.log.append(rec)
        return {'outcome': C.NOT_SUBMITTED, 'detail': f'not submitted: {why}', 'submissions': 0, 'exchanges': [],
                'gate': rec}

    def call(self, sem, to_label):
        req = self.current
        if req is None:
            raise RuntimeError('RunGate.begin(request) must be called before judging')
        direction = 'AB' if to_label['opt_1'] == req.response_a.response_id else 'BA'
        key = f'{req.request_id}:{direction}'
        if self.stopped:
            return self._refuse(key, f'stopped earlier ({self.stopped})', stop=False)
        if key in self.submitted_keys:
            return self._refuse(key, f'{key} was already submitted (never submitted twice)')
        slots = 2 if (self.mode == 'two_direction' and direction == 'AB') else 1
        if self.submissions + slots > self.max_submissions:
            return self._refuse(key, f'max_submissions {self.max_submissions} would be exceeded '
                                     f'({self.submissions} used, this call needs {slots})')
        if self.name == 'choice_endpoint':
            idle = 0.0 if self.job_time.first_submit is not None else IDLE_ALLOWANCE_SECONDS
            need = self.job_time.usd(slots * JOB_RESERVATION_SECONDS + idle)
            reservation = JOB_RESERVATION_SECONDS * self.job_time.rate
        else:
            reservation = haiku_reservation_usd(P.haiku_body(sem))   # same size for both directions
            need = self.h_spent + self.h_held + slots * reservation
        if need > self.cap + 1e-12:
            return self._refuse(key, f'estimate with reservation ${need:.6f} > cap ${self.cap:.4f}')
        label = f'{len(self.submitted_keys) + 1}:{key}'
        first_before = self.job_time.first_submit
        if self.name == 'choice_endpoint':
            self.job_time.reserve(label)
        self.submitted_keys.append(key)
        r = self.inner.call(sem, to_label)
        if r.get('outcome') == C.LOCAL_CONFIG and not r.get('submissions'):
            return self._local_config(key, label, first_before, r)
        self.submissions += r.get('submissions', 0)
        self.status_polls += r.get('status_polls', 0)
        self.cancels += r.get('cancels', 0)
        stop = None
        if self.name == 'choice_endpoint':
            if r.get('job_final'):
                settle = {'settled_seconds': self.job_time.settle(label, r.get('server_timing'))}
            else:
                settle = {'unsettled_reservation_seconds': JOB_RESERVATION_SECONDS,
                          'cancel_sent': bool(r.get('cancels')), 'job_id': r.get('job_id')}
                self.unsettled_jobs.append({'label': label, **settle})
                stop = f'unsettled choice-endpoint job ({r.get("detail")}); its full reservation stays held'
        else:
            u = r.get('usage') or {}
            known = r.get('http_status') == 200 and all(isinstance(u.get(k), int) for k in (
                'input_tokens', 'output_tokens')) and not (u.get('cache_creation_input_tokens') or
                                                           u.get('cache_read_input_tokens'))
            if known:
                charged = haiku_usd(u)
                self.h_spent += charged
                settle = {'charged_usd': charged, 'reserved_usd': reservation}
                if charged > reservation + 1e-12:
                    stop = 'Haiku charge above its reservation'
            else:
                self.h_held += reservation
                settle = {'held_usd': reservation}
                stop = 'Haiku usage unknown or cache usage present; full reservation held'
        if r.get('outcome') == C.CONTRACT:
            stop = stop or f'contract failure ({r.get("detail")})'
        if stop and self.stopped is None:
            self.stopped = stop
        rec = {'decision': 'submitted', 'call_key': key, 'label': label, 'reservation_usd': reservation,
               'need_usd': need, **settle, 'outcome': r.get('outcome'), 'committed_after_usd': self.committed_usd(),
               'stop_after': stop}
        self.log.append(rec)
        r['gate'] = rec
        return r

    def _local_config(self, key, label, first_before, r):
        """Nothing was transmitted: undo this call's reservation (job + first-submission clock), hold nothing,
        stop the run. Reservations of transmitted (unsettled) jobs are never released here."""
        if self.name == 'choice_endpoint':
            self.job_time.unsettled.pop(label, None)
            self.job_time.first_submit = first_before
        self.submitted_keys.remove(key)
        stop = f'local configuration failure before transmission ({r.get("detail")})'
        if self.stopped is None:
            self.stopped = stop
        rec = {'decision': 'not_transmitted', 'call_key': key, 'outcome': C.LOCAL_CONFIG, 'released_reservation': True,
               'committed_after_usd': self.committed_usd(), 'stop_after': stop}
        self.log.append(rec)
        r['gate'] = rec
        return r

    def snapshot(self):
        return {'kind': 'CLIENT-SIDE ESTIMATE for this run; NOT the bill and NOT an account billing cap',
                'backend': self.name, 'mode': self.mode,
                'cap_usd': None if self.cap == float('inf') else self.cap, 'max_submissions': self.max_submissions,
                'submissions': self.submissions, 'status_polls': self.status_polls, 'cancels': self.cancels,
                'retries': 0, 'stopped': self.stopped, 'committed_usd': self.committed_usd(),
                'submitted_call_keys': list(self.submitted_keys),
                'choice_endpoint': {'usd_per_second_configured': self.job_time.rate,
                                    'rate_note': getattr(self.inner, 'rate_note', None),
                                    'idle_allowance_seconds': IDLE_ALLOWANCE_SECONDS,
                         'job_reservation_seconds': JOB_RESERVATION_SECONDS,
                         'settled_seconds': self.job_time.settled_seconds,
                         'unsettled_jobs': self.unsettled_jobs} if self.name == 'choice_endpoint' else None,
                'haiku': {'spent_usd': self.h_spent, 'held_usd': self.h_held} if self.name != 'choice_endpoint' else None,
                'log': self.log}

