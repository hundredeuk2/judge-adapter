# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Two backends behind one call shape. One model submission per call; HTTP retries 0; no fallback.

- haiku_direct (default): one raw POST /v1/messages, the fixed 8-token direct prompt, strict parse.
- choice_endpoint: one POST /runsync to a user-served choice-judge worker on RunPod serverless whose endpoint ID is
  configured explicitly; if runsync returns IN_QUEUE / IN_PROGRESS, bounded GET /status of the SAME job (not a
  resubmission), then one cancel if still unresolved. No endpoint ID or no per-second rate configured -> every call
  is local_config with 0 transmissions (no key is read). The expected model name is optional: if set, a different
  output.model is a contract failure; if unset, the observed model is only recorded (calls[].response_model).

The transport is injectable (httpx.MockTransport in tests). With no transport, a real httpx transport with
retries=0 is used. The key provider is called lazily at the first request only.

Provider responses are parsed from the raw body. Scrubbing (the key, bearer tokens, key-like substrings) is applied
only on output paths: ``detail`` / error strings, and everything a trace writer stores. In-memory provider data
returned to the caller (parsed fields, ``raw_text``, ``exchanges[].text``) is unmodified, so a key-like substring
that is part of a legitimate payload is never rewritten before parsing or reuse.

A failure BEFORE any HTTP transmission (key provider error, missing key, client construction) is outcome
local_config with submissions 0, and the backend then refuses every later call of the run without retrying the key
read. Failures after transmission started (timeout, communication error) keep submissions 1 and unknown cost.
"""
from __future__ import annotations

import json
import threading
import time

import httpx

from . import contract as C
from . import prompt as P
from .keys import Scrubber

FORBIDDEN_HAIKU_KEYS = ('system', 'thinking', 'top_p', 'top_k', 'tools', 'tool_choice', 'stop_sequences', 'stream',
                        'metadata', 'output_config', 'cache_control', 'fallbacks')
HAIKU_STATUS_TO_REASON = {'truncated_max_tokens': C.TRUNCATED, 'refused': C.REFUSED,
                          'unexpected_stop_reason': C.FORMAT_ERROR, 'format_error': C.FORMAT_ERROR}
CHOICE_STATUS_TO_REASON = {'invalid': C.FORMAT_ERROR, 'choice_argmax_mismatch': C.FORMAT_ERROR,
                         'prob_tie': C.NO_DECISION, 'truncated': C.TRUNCATED, 'handler_error': C.BACKEND_ERROR}


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def check_haiku_body(body):
    bad = []
    if set(body) != set(P.HAIKU_BODY_KEYS):
        bad.append(f'body keys {sorted(body)} != {sorted(P.HAIKU_BODY_KEYS)}')
    bad += [f'forbidden key {k}' for k in FORBIDDEN_HAIKU_KEYS if k in body]
    if body.get('model') != P.HAIKU_MODEL or body.get('max_tokens') != P.HAIKU_MAX_TOKENS or \
            body.get('temperature') != P.HAIKU_TEMPERATURE:
        bad.append('model / max_tokens / temperature differ from the fixed settings')
    return bad


class _Base:
    name = model = None

    def __init__(self, key_provider, transport=None, timeout=120.0, clock=time.perf_counter):
        self._key_provider = key_provider
        self._transport = transport
        self._timeout = timeout
        self._http = None
        self._init_lock = threading.Lock()
        self.clock = clock
        self.scrub = Scrubber()
        self.local_failure = None           # set once by a pre-transmission failure; sticky for the run

    def _headers(self, key):
        raise NotImplementedError

    def _client(self):
        with self._init_lock:
            return self._client_locked()

    def _client_locked(self):
        if self._http is None:
            key = self._key_provider()          # first request only (never at import / construction)
            self.scrub.set_secret(key)
            tr = self._transport if self._transport is not None else httpx.HTTPTransport(retries=0)
            self._http = httpx.Client(transport=tr, timeout=httpx.Timeout(self._timeout), follow_redirects=False,
                                      headers=self._headers(key))
            del key
        return self._http

    def close(self):
        if self._http is not None:
            self._http.close()

    def _exchange(self, method, url, body=None, timeout=None):
        """One HTTP request. -> dict(outcome http|timeout|comm_error|local_config, http_status, text, json, seconds,
        error). local_config: nothing was transmitted (no key / client)."""
        t = self.clock()
        rec = {'method': method, 'url': url.split('?')[0]}
        if self.local_failure is None:
            try:
                client = self._client()        # key read + client construction; no byte has been sent yet
            except Exception as e:  # noqa: BLE001 - any local setup failure
                self.local_failure = self.scrub.text(f'{type(e).__name__}: {str(e)[:200]}')
        if self.local_failure is not None:
            rec.update(outcome='local_config', error=self.local_failure, seconds=self.clock() - t)
            return rec
        try:
            kw = {} if timeout is None else {'timeout': httpx.Timeout(timeout)}
            if method == 'POST':
                r = client.post(url, content=json.dumps(body, ensure_ascii=False).encode('utf-8'), **kw)
            else:
                r = client.get(url, **kw)
        except httpx.TimeoutException as e:
            rec.update(outcome='timeout', error=self.scrub.text(type(e).__name__), seconds=self.clock() - t)
            return rec
        except Exception as e:  # noqa: BLE001 - any transport failure: result unknown
            rec.update(outcome='comm_error', error=self.scrub.text(f'{type(e).__name__}: {str(e)[:200]}'),
                       seconds=self.clock() - t)
            return rec
        text = r.content.decode('utf-8', 'replace')      # the provider body as received: parsed, never rewritten
        try:
            js = json.loads(text)
        except ValueError:
            js = None
        rec.update(outcome='http', http_status=r.status_code, text=text, json=js if isinstance(js, dict) else None,
                   seconds=self.clock() - t)
        return rec

    def call(self, sem, to_label):
        """One judged call. Provider data in the result (parsed fields, ``raw_text``, ``exchanges[].text`` /
        ``json``) is the response as received. ``detail`` is an error message and is scrubbed here; trace writers
        scrub everything they store."""
        r = self._call(sem, to_label)
        if isinstance(r.get('detail'), str):
            r['detail'] = self.scrub.text(r['detail'])
        return r

    def _call(self, sem, to_label):
        raise NotImplementedError


class HaikuDirectBackend(_Base):
    name = 'haiku_direct'
    model = P.HAIKU_MODEL

    def _headers(self, key):
        return {'x-api-key': key, 'anthropic-version': P.ANTHROPIC_VERSION, 'content-type': 'application/json'}

    def rendered_input(self, sem):
        return P.haiku_text(sem)

    def _call(self, sem, to_label):
        body = P.haiku_body(sem)
        bad = check_haiku_body(body)
        if bad:
            return {'outcome': C.CONTRACT, 'detail': '; '.join(bad), 'submissions': 0, 'exchanges': []}
        x = self._exchange('POST', P.HAIKU_URL, body)
        if x['outcome'] == 'local_config':
            return {'outcome': C.LOCAL_CONFIG, 'submissions': 0, 'exchanges': [], 'client_seconds': x['seconds'],
                    'detail': f'local configuration failure before transmission ({x["error"]}); nothing sent'}
        out = {'submissions': 1, 'exchanges': [x], 'client_seconds': x['seconds'], 'http_status': x.get('http_status'),
               'raw_text': x.get('text')}
        if x['outcome'] == 'timeout':
            return {**out, 'outcome': C.TIMEOUT, 'detail': 'client timeout; result and usage unknown'}
        if x['outcome'] != 'http':
            return {**out, 'outcome': C.HTTP, 'detail': f'communication error ({x.get("error")}); usage unknown'}
        js = x['json'] or {}
        u = js.get('usage') if isinstance(js.get('usage'), dict) else None
        out['usage'] = {k: u.get(k) for k in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens',
                                              'cache_read_input_tokens')} if u is not None else None
        out['response_model'] = js.get('model')
        if x['http_status'] != 200:
            return {**out, 'outcome': C.HTTP, 'detail': f'HTTP {x["http_status"]} (retries 0)'}
        if js.get('model') != P.HAIKU_MODEL:
            return {**out, 'outcome': C.CONTRACT, 'detail': f'response model {js.get("model")!r} != {P.HAIKU_MODEL}'}
        ot = (out['usage'] or {}).get('output_tokens')
        if isinstance(ot, int) and ot > P.HAIKU_MAX_TOKENS:
            return {**out, 'outcome': C.CONTRACT, 'detail': f'output tokens {ot} > {P.HAIKU_MAX_TOKENS}'}
        blocks = js.get('content') if isinstance(js.get('content'), list) else []
        text = ''.join(b.get('text', '') for b in blocks if isinstance(b, dict) and b.get('type') == 'text')
        parsed = P.parse_haiku(js.get('stop_reason'), text, to_label)
        if parsed['status'] == 'ok':
            return {**out, 'outcome': 'ok', 'label': parsed['label'], 'detail': None}
        return {**out, 'outcome': HAIKU_STATUS_TO_REASON[parsed['status']],
                'detail': f'{parsed["status"]} (stop_reason {js.get("stop_reason")!r})'}

    @staticmethod
    def cost(call_records):
        it = ot = 0
        for c in call_records:
            u = c.usage or {}
            if c.submissions and not (isinstance(u.get('input_tokens'), int) and isinstance(u.get('output_tokens'), int)):
                return None                       # a submission with unknown usage: cost unknown, not zero
            it += u.get('input_tokens') or 0
            ot += u.get('output_tokens') or 0
        return round(it * P.HAIKU_PRICE_IN_PER_MTOK / 1e6 + ot * P.HAIKU_PRICE_OUT_PER_MTOK / 1e6, 8)

    cost_basis = ('ESTIMATE: response usage x list price claude-haiku-4-5 $1.00 in / $5.00 out per MTok (as checked '
                  '2026-09). Not the bill; the Anthropic console is the source of truth.')


class ChoiceEndpointBackend(_Base):
    name = 'choice_endpoint'
    UNSPECIFIED_MODEL = 'unspecified'

    def __init__(self, key_provider, transport=None, timeout=120.0, clock=time.perf_counter, sleep=time.sleep,
                 endpoint_id=None, poll_interval=5.0, max_status_polls=120, usd_per_second=None,
                 expected_model=None, rate_note='user-configured rate'):
        super().__init__(key_provider, transport, timeout, clock)
        self.endpoint_id = endpoint_id or None
        self.urls = P.choice_urls(self.endpoint_id) if self.endpoint_id else None    # invalid ID -> ValueError now
        self.sleep = sleep
        self.poll_interval = poll_interval
        self.max_status_polls = max_status_polls
        if usd_per_second is not None and not (isinstance(usd_per_second, (int, float)) and usd_per_second > 0):
            raise ValueError('choice_usd_per_second must be a positive number')
        self.usd_per_second = usd_per_second            # None -> a real call is refused as local_config
        self.rate_note = rate_note
        self.expected_model = expected_model or None
        self.model = self.expected_model or self.UNSPECIFIED_MODEL

    def _headers(self, key):
        return {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}

    def rendered_input(self, sem):
        return P.haiku_text(sem)    # text view of the same semantic input, for the host's prompt field only

    def _call(self, sem, to_label):
        body = P.choice_body(sem)
        t_sub = self.clock()
        if self.urls is None or self.usd_per_second is None:   # checked before any key read; nothing is sent
            if self.local_failure is None:
                self.local_failure = (
                    f'no choice endpoint ID configured (set AdapterConfig.choice_endpoint_id or the environment '
                    f'variable {P.CHOICE_ENDPOINT_ENV})' if self.urls is None else
                    'no per-second rate configured for the choice endpoint (set AdapterConfig.choice_usd_per_second); '
                    'cost could not be estimated or capped')
            x = {'outcome': 'local_config', 'error': self.local_failure}
        else:
            x = self._exchange('POST', self.urls['runsync'], body)
        if x['outcome'] == 'local_config':
            return {'outcome': C.LOCAL_CONFIG, 'submissions': 0, 'exchanges': [], 'status_polls': 0, 'cancels': 0,
                    'job_final': None, 'job_status': None, 'client_seconds': self.clock() - t_sub,
                    'detail': f'local configuration failure before transmission ({x["error"]}); nothing sent'}
        out = {'submissions': 1, 'exchanges': [x], 'http_status': x.get('http_status'), 'status_polls': 0,
               'cancels': 0, 'job_final': False, 'job_status': None}   # job_final: a terminal state was OBSERVED

        def done(**kw):
            out['client_seconds'] = self.clock() - t_sub
            return {**out, **kw}

        if x['outcome'] == 'timeout':
            return done(outcome=C.TIMEOUT, detail='client timeout; the job may still run server side (bounded by ttl)')
        if x['outcome'] != 'http':
            return done(outcome=C.HTTP, detail=f'communication error ({x.get("error")})')
        if x['http_status'] != 200:
            return done(outcome=C.HTTP, detail=f'HTTP {x["http_status"]} (retries 0)', raw_text=x.get('text'))
        js = x['json']
        if not isinstance(js, dict):
            return done(outcome=C.HTTP, detail='HTTP 200 without a JSON object', raw_text=x.get('text'))
        final = x
        job = js.get('id')
        out['job_id'] = job
        ttl = P.CHOICE_POLICY['ttl'] / 1000.0
        while js.get('status') in ('IN_QUEUE', 'IN_PROGRESS') and job:
            remaining = ttl - (self.clock() - t_sub)
            if out['status_polls'] >= self.max_status_polls or remaining <= 0:
                c = self._exchange('POST', self.urls['cancel'].format(job_id=job), {})
                out['exchanges'].append(c)
                out['cancels'] += 1
                out['cancel_response_status'] = (c.get('json') or {}).get('status')   # a cancel request, not a final state
                return done(outcome=C.TIMEOUT, detail=f'job {js.get("status")} after {out["status_polls"]} status '
                                                      'polls; one cancel sent; cost unknown', raw_text=final.get('text'))
            self.sleep(min(self.poll_interval, remaining))
            s = self._exchange('GET', self.urls['status'].format(job_id=job), timeout=min(self._timeout, remaining))
            out['exchanges'].append(s)
            out['status_polls'] += 1
            if s['outcome'] != 'http' or s.get('http_status') != 200 or not isinstance(s.get('json'), dict):
                return done(outcome=C.HTTP if s['outcome'] != 'timeout' else C.TIMEOUT,
                            detail=f'status poll failed ({s["outcome"]} {s.get("http_status")}); job result unknown',
                            raw_text=s.get('text'))
            final, js = s, s['json']
        raw = final.get('text')
        o = js.get('output') if isinstance(js.get('output'), dict) else {}
        usage = o.get('usage') if isinstance(o.get('usage'), dict) else None
        out.update(raw_text=raw, response_model=o.get('model'),
                   usage={k: _num(usage.get(k)) for k in ('input_tokens', 'output_tokens')} if usage else None,
                   server_timing={'delay_ms': _num(js.get('delayTime')), 'execution_ms': _num(js.get('executionTime')),
                                  'latency_ms': _num(o.get('latency_ms'))})
        st = js.get('status')
        out['job_status'] = st
        out['job_final'] = st in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT')
        if st == 'TIMED_OUT':
            return done(outcome=C.TIMEOUT, detail='job TIMED_OUT (server executionTimeout)')
        if st in ('FAILED', 'CANCELLED'):
            return done(outcome=C.BACKEND_ERROR, detail=f'job {st}: {str(js.get("error"))[:200]}')
        if st != 'COMPLETED':
            return done(outcome=C.HTTP, detail=f'unknown job status {str(st)[:40]}')
        if self.expected_model is not None and o.get('model') != self.expected_model:
            return done(outcome=C.CONTRACT, detail=f'output.model {o.get("model")!r} != {self.expected_model!r}')
        r = P.restore_choice(js, to_label, P.question_id(sem))
        out['probabilities'] = r.get('probabilities')
        if r['status'] == 'ok':
            return done(outcome='ok', label=r['label'], detail=None)
        return done(outcome=CHOICE_STATUS_TO_REASON[r['status']], detail=f'{r["status"]}: {r.get("why")}')

    def cost(self, call_records):
        secs = 0.0
        for c in call_records:
            if not c.submissions:
                continue
            ex = (c.server_timing or {}).get('execution_ms')
            if not isinstance(ex, (int, float)):
                return None
            secs += ex / 1000.0
        if secs == 0.0:
            return 0.0
        if self.usd_per_second is None:
            return None                            # no rate: cost unknown, not zero
        return round(secs * self.usd_per_second, 8)

    @property
    def cost_basis(self):
        return (f'ESTIMATE: server executionTime x ${self.usd_per_second}/s ({self.rate_note}). Excludes cold start, '
                'idle timeout and queue billing. Not the bill; the RunPod console is the source of truth.')
