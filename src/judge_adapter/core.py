# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""JudgeAdapter: request -> backend call(s) -> JudgeResult, with a trace on disk.

Modes (config ``mode``):
- single: one call, slot 1 = host position A.
- two_direction: AB then BA; labels are restored to response IDs before comparing. Same label -> that verdict;
  different labels -> tie with tie_kind 'inconsistent' (recorded separately from an explicit tie). If the first
  direction fails, the second is NOT sent (no extra spend) and the request is an error.

Never: retry, fallback to the other backend, turn an error into a tie, or read a key at import.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from . import contract as C
from . import prompt as P
from . import user_prompt as U
from .backends import ChoiceEndpointBackend, HaikuDirectBackend
from .keys import default_provider


def _sha_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest() if s is not None else None


class TraceWriter:
    """judge_trace.jsonl: one JudgeResult per line (no raw text). judge_raw.jsonl: raw HTTP bodies per call."""

    def __init__(self, trace_dir):
        self.dir = Path(trace_dir).resolve()     # absolute: hosts may chdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.trace = self.dir / 'judge_trace.jsonl'
        self.raw = self.dir / 'judge_raw.jsonl'
        self._lock = threading.Lock()      # host may judge in threads (FastChat --parallel)

    def _append(self, path, rec):
        line = json.dumps(rec, ensure_ascii=False, default=str) + '\n'
        with self._lock, open(path, 'a', encoding='utf-8') as f:
            f.write(line)

    def write_raw(self, call_id, exchanges):
        self._append(self.raw, {'call_id': call_id,
                                'exchanges': [{k: v for k, v in x.items() if k != 'json'} for x in exchanges]})
        return {'file': self.raw.name, 'call_id': call_id}

    def write_result(self, result):
        self._append(self.trace, result.to_dict())


FAKE_CHOICE_ENDPOINT_ID = 'fake-endpoint'     # fake transport only: an in-process stand-in, never contacted
FAKE_CHOICE_USD_PER_SECOND = 0.001             # fake transport only: PLACEHOLDER so the gate logic runs; not a price


def choice_endpoint_id(cfg):
    """Real: explicit config first, then the environment variable; None if neither is set (no built-in default).
    Fake: the configured ID or a placeholder (the fake transport never leaves the process; env is not consulted)."""
    if cfg.transport == 'fake':
        return cfg.choice_endpoint_id or FAKE_CHOICE_ENDPOINT_ID
    return cfg.choice_endpoint_id or os.environ.get(P.CHOICE_ENDPOINT_ENV, '').strip() or None


def build_backend(cfg, transport=None, key_provider=None, clock=time.perf_counter, sleep=time.sleep):
    """transport None -> real network (retries 0). key_provider None -> lazy reader of the explicit env file, else of
    the environment variable (read at the first real request only)."""
    if cfg.transport == 'real' and transport is not None:
        raise ValueError('transport "real" must not be given a custom transport')
    if cfg.transport == 'fake' and transport is None:
        raise ValueError('transport "fake" requires an injected fake transport')
    if cfg.backend == 'haiku_direct':
        kp = key_provider or default_provider(cfg.env_file, 'ANTHROPIC_API_KEY')
        return HaikuDirectBackend(kp, transport=transport, timeout=cfg.timeout_seconds, clock=clock)
    kp = key_provider or default_provider(cfg.env_file, 'RUNPOD_API_KEY')
    fake = cfg.transport == 'fake'
    rate = cfg.choice_usd_per_second
    if fake and rate is None:
        rate, note = FAKE_CHOICE_USD_PER_SECOND, 'FAKE PLACEHOLDER rate for model-free runs; not a price'
    else:
        note = 'user-configured rate, not verified'
    return ChoiceEndpointBackend(kp, transport=transport, timeout=cfg.timeout_seconds, clock=clock, sleep=sleep,
                                 endpoint_id=choice_endpoint_id(cfg), poll_interval=cfg.choice_poll_interval_seconds,
                                 max_status_polls=cfg.choice_max_status_polls, usd_per_second=rate,
                                 expected_model=cfg.choice_expected_model, rate_note=note)


class JudgeAdapter:
    def __init__(self, cfg, backend, clock=time.perf_counter):
        self.cfg = cfg
        self.backend = backend
        self.clock = clock
        self.trace = TraceWriter(cfg.trace_dir)
        self._local = threading.local()     # per-thread: the text view of the last request (host prompt field)

    @property
    def last_rendered_input(self):
        return getattr(self._local, 'rendered', None)

    @last_rendered_input.setter
    def last_rendered_input(self, value):
        self._local.rendered = value

    # --- input checks (explicit refusal; nothing is silently cut or dropped) -----------------------------------
    @staticmethod
    def _check(req):
        if req.reference is not None:
            return 'reference answers are not supported by the fixed judge input (not dropped silently)'
        if req.turn not in (1, 2):
            return f'turn {req.turn} not supported (1 or 2)'
        if len(req.question_turns) < req.turn:
            return 'fewer question turns than the judged turn'
        for r in (req.response_a, req.response_b):
            if len(r.turns) < req.turn:
                return f'response {r.response_id!r} has fewer turns than the judged turn'
            if not all(isinstance(t, str) for t in r.turns[:req.turn]):
                return f'response {r.response_id!r} has a non-text turn'
        ids = {req.response_a.response_id, req.response_b.response_id}
        if len(ids) != 2 or C.TIE in ids or not all(isinstance(i, str) and i for i in ids):
            return 'response ids must be two distinct non-empty strings other than "tie"'
        if req.criteria is not None and tuple(req.criteria) != P.CRITERIA_IDS:
            return f'criteria keys must be exactly {P.CRITERIA_IDS}'
        if U.uses_user_path(req.rubric, req.reference_turns):
            if req.instructions is not None or req.criteria is not None:
                return 'rubric / reference_turns cannot be combined with instructions or criteria (ambiguous standard)'
            return U.check(req.question_turns, req.rubric, req.reference_turns)
        return None

    def _one(self, req, direction):
        a, b = req.response_a, req.response_b
        s1, s2 = (a, b) if direction == 'AB' else (b, a)
        if U.uses_user_path(req.rubric, req.reference_turns):
            sem = U.semantic(req.question_turns, s1.turns, s2.turns, req.turn, req.rubric, req.reference_turns)
        else:
            sem = P.semantic(req.question_turns, s1.turns, s2.turns, req.turn, req.instructions, req.criteria)
        to_label = {'opt_1': s1.response_id, 'opt_2': s2.response_id, 'opt_tie': C.TIE}
        call_id = f'{req.request_id}:{direction}'
        if direction == 'AB':
            self.last_rendered_input = self.backend.rendered_input(sem)
        r = self.backend.call(sem, to_label)
        pointer = self.trace.write_raw(call_id, r.get('exchanges', []))
        rec = C.CallRecord(call_id=call_id, direction=direction, slot_ids={'1': s1.response_id, '2': s2.response_id},
                           outcome=r['outcome'], label=r.get('label'), detail=r.get('detail'),
                           submissions=r.get('submissions', 0), status_polls=r.get('status_polls', 0),
                           cancels=r.get('cancels', 0), http_status=r.get('http_status'),
                           client_seconds=r.get('client_seconds'), server_timing=r.get('server_timing'),
                           usage=r.get('usage'), raw_output_sha256=_sha_text(r.get('raw_text')),
                           raw_output_pointer=pointer, response_model=r.get('response_model'), job_id=r.get('job_id'),
                           job_status=r.get('job_status'), job_final=r.get('job_final'), gate=r.get('gate'))
        return rec, P.sha(sem)

    def judge(self, req):
        t0 = self.clock()
        self.last_rendered_input = None
        calls, sem_sha, failure = [], None, None
        why = self._check(req)
        if why:
            failure = {'reason': C.UNSUPPORTED_INPUT, 'detail': why, 'direction': None}
        else:
            for d in (('AB',) if self.cfg.mode == 'single' else ('AB', 'BA')):
                rec, s = self._one(req, d)
                calls.append(rec)
                sem_sha = sem_sha or s
                if rec.outcome != 'ok':
                    failure = {'reason': rec.outcome, 'detail': rec.detail, 'direction': d}
                    break                                   # no second direction after a failure
        status, verdict, winner, tie_kind = C.STATUS_ERROR, None, None, None
        if failure is None:
            labels = [c.label for c in calls]
            if len(set(labels)) == 1:
                lab = labels[0]
                if lab == C.TIE:
                    status, verdict, tie_kind = C.STATUS_TIE, 'tie', 'explicit'
                else:
                    status, winner = C.STATUS_OK, lab
                    verdict = 'A' if lab == req.response_a.response_id else 'B'
            else:
                status, verdict, tie_kind = C.STATUS_TIE, 'tie', 'inconsistent'
        usage = None
        if any(c.usage for c in calls):
            usage = {k: sum((c.usage or {}).get(k) or 0 for c in calls) for k in ('input_tokens', 'output_tokens')}
        usd = self.backend.cost(calls) if calls else 0.0
        result = C.JudgeResult(
            request_id=req.request_id, status=status, verdict=verdict, winner_id=winner, tie_kind=tie_kind,
            failure=failure, backend=self.backend.name, model=self.backend.model, mode=self.cfg.mode,
            transport=self.cfg.transport, calls=calls, submissions=sum(c.submissions for c in calls), retries=0,
            usage=usage,
            cost_estimate={'usd': usd, 'is_estimate': True, 'mock': self.cfg.transport == 'fake',
                           'basis': ('MOCK: fake transport; not a cost. ' if self.cfg.transport == 'fake' else '')
                           + self.backend.cost_basis},
            total_seconds=self.clock() - t0, semantic_sha256=sem_sha, host=dict(req.host),
            criteria_info=None if why else self._criteria_info(req))
        self.trace.write_result(result)
        return result

    @staticmethod
    def _criteria_info(req):
        if req.instructions is not None or req.criteria is not None:     # pre-existing adapter override (not the CLI)
            return {'prompt_version': 'adapter-instructions-override', 'rubric': {
                'source': 'adapter_instructions', 'sha256': U.sha_text(req.instructions or '')}, 'reference': None}
        if req.turn not in (1, 2):
            return None
        return U.criteria_info(req.turn, req.rubric, req.reference_turns)

    def close(self):
        self.backend.close()
