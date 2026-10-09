# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Fake transports for model-free runs (``--transport fake``, tests, demos). Never used on the real path.

The fake judge reads only the judged answer of each slot (the last "### Assistant:" segment) and follows markers:
  [[GOOD]]       that answer wins (content based, so the verdict follows the response across position swaps)
  neither/both   opt_tie (explicit tie)
  FAKE:POSITION  always opt_1 (position bias -> inconsistent across directions)
  FAKE:FORMAT    Haiku: free text instead of an option id; choice endpoint: probabilities that do not sum to 1
  FAKE:TRUNC     Haiku: stop_reason max_tokens; choice endpoint: truncated flag
  FAKE:REFUSE    Haiku: stop_reason refusal; choice endpoint: handler error
  FAKE:HTTP500   HTTP 500
  FAKE:TIMEOUT   transport timeout
  FAKE:QUEUE     choice endpoint: runsync returns IN_QUEUE; the next status poll returns COMPLETED
  FAKE:LEAK      the fake server echoes the request auth header into the body (scrubbing check)
No network, no key, no model. Usage numbers are invented (chars / 4) and are not costs.
"""
from __future__ import annotations

import itertools
import json
import re

import httpx

from . import prompt as P

_TAG = re.compile(r'<(assistant_[12]_conversation)>\n(.*?)\n</\1>', re.S)


def judged_answer(conversation):
    return conversation.rsplit('### Assistant:\n', 1)[-1]


def decide(a1, a2):
    """-> ('choice', criterion id) or ('fault', marker)."""
    both = a1 + '\n' + a2
    for m in ('FAKE:TIMEOUT', 'FAKE:HTTP500', 'FAKE:TRUNC', 'FAKE:REFUSE', 'FAKE:FORMAT'):
        if m in both:
            return 'fault', m
    if 'FAKE:POSITION' in both:
        return 'choice', 'opt_1'
    g1, g2 = '[[GOOD]]' in a1, '[[GOOD]]' in a2
    if g1 and not g2:
        return 'choice', 'opt_1'
    if g2 and not g1:
        return 'choice', 'opt_2'
    return 'choice', 'opt_tie'


class _Fake:
    def __init__(self):
        self.requests = []          # (method, path)

    def transport(self):
        return httpx.MockTransport(self.handle)

    def count(self, method=None, contains=None):
        return sum(1 for m, p in self.requests if (method is None or m == method) and (contains is None or contains in p))


class FakeAnthropic(_Fake):
    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        body = json.loads(request.content)
        text = body['messages'][0]['content']
        conv = dict(_TAG.findall(text))
        a1, a2 = judged_answer(conv['assistant_1_conversation']), judged_answer(conv['assistant_2_conversation'])
        kind, val = decide(a1, a2)
        stop, out = 'end_turn', None
        if kind == 'fault':
            if val == 'FAKE:TIMEOUT':
                raise httpx.ReadTimeout('fake timeout', request=request)
            if val == 'FAKE:HTTP500':
                return httpx.Response(500, json={'type': 'error', 'error': {'type': 'api_error', 'message': 'fake'}})
            stop, out = {'FAKE:TRUNC': ('max_tokens', 'opt_'), 'FAKE:REFUSE': ('refusal', ''),
                         'FAKE:FORMAT': ('end_turn', 'I pick opt_1')}[val]
        else:
            out = val
        js = {'id': 'msg_fake', 'type': 'message', 'role': 'assistant', 'model': P.HAIKU_MODEL,
              'content': [{'type': 'text', 'text': out}], 'stop_reason': stop,
              'usage': {'input_tokens': len(text) // 4, 'output_tokens': 3, 'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0}}
        if 'FAKE:LEAK' in a1 + a2:
            js['echo'] = request.headers.get('x-api-key')
        return httpx.Response(200, json=js)


class FakeChoiceEndpoint(_Fake):
    def __init__(self):
        super().__init__()
        self.jobs = {}
        self.n = 0

    def _completed(self, job_id, state, leak=None):
        a1 = judged_answer(state['assistant_1_conversation'])
        a2 = judged_answer(state['assistant_2_conversation'])
        kind, val = decide(a1, a2)
        output = {'model': FAKE_CHOICE_MODEL, 'latency_ms': 800,
                  'usage': {'input_tokens': sum(len(v) for v in state.values()) // 4, 'output_tokens': 0}}
        if kind == 'fault' and val == 'FAKE:TRUNC':
            output['truncated'] = True
        elif kind == 'fault' and val == 'FAKE:REFUSE':
            output['error'] = 'fake handler error'
        else:
            if kind == 'fault':          # FAKE:FORMAT -> sum 0.5
                probs, choice = {'opt_1': 0.3, 'opt_2': 0.1, 'opt_tie': 0.1}, 'opt_1'
            else:
                probs = {c: 0.1 for c in P.CRITERIA_IDS}
                probs[val] = 0.8
                choice = val
            output['answers'] = {P.QUESTION_ID: {'type': 'choice', 'choice': choice, 'probabilities': probs}}
        js = {'id': job_id, 'status': 'COMPLETED', 'delayTime': 120, 'executionTime': 850, 'output': output}
        if leak:
            js['echo'] = leak
        return js

    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        path = request.url.path
        if '/runsync' in path:
            body = json.loads(request.content)
            state = body['input']['state']
            a = judged_answer(state['assistant_1_conversation']) + judged_answer(state['assistant_2_conversation'])
            if 'FAKE:TIMEOUT' in a:
                raise httpx.ReadTimeout('fake timeout', request=request)
            if 'FAKE:HTTP500' in a:
                return httpx.Response(500, text='fake upstream error')
            self.n += 1
            job = f'fake-job-{self.n}'
            leak = request.headers.get('authorization') if 'FAKE:LEAK' in a else None
            if 'FAKE:QUEUE' in a:
                self.jobs[job] = (state, leak)
                return httpx.Response(200, json={'id': job, 'status': 'IN_QUEUE'})
            return httpx.Response(200, json=self._completed(job, state, leak))
        if '/status/' in path:
            job = path.rsplit('/', 1)[-1]
            state, leak = self.jobs[job]
            return httpx.Response(200, json=self._completed(job, state, leak))
        if '/cancel/' in path:
            return httpx.Response(200, json={'id': path.rsplit('/', 1)[-1], 'status': 'CANCELLED'})
        return httpx.Response(404, text='unknown fake path')


_KEEP_RE = re.compile(r'\[\[KEEP=([0-9]*\.?[0-9]+)\]\]')


class FakeRelevanceEndpoint(_Fake):
    """Fake choice endpoint for the context filter (question id from the request; options opt_keep / opt_drop).

    Reads markers in the rendered document only (never the question):
      [[KEEP=p]]     P(keep) = p exactly, P(drop) = 1 - p; choice = the larger one (p = 0.5 -> exact tie)
      [[RELEVANT]]   P(keep) = 0.9; no marker -> P(keep) = 0.1
      FAKE:HTTP500 / FAKE:TIMEOUT   HTTP 500 / transport timeout (job never observed final)
      FAKE:FORMAT    probabilities that do not sum to 1 (format failure, job completed)
      FAKE:REFUSE    handler error (job completed)
      FAKE:QUEUE     runsync returns IN_QUEUE; the next status poll returns COMPLETED
      FAKE:NOTIMING  completed without delayTime / executionTime (cost unknown)
    Usage numbers are invented (chars / 4). Not a model; not a cost.
    """

    def __init__(self):
        super().__init__()
        self.jobs = {}
        self._ids = itertools.count(1)     # thread-safe job numbering (the filter may run calls concurrently)
        self.bodies = []            # request bodies, for wire checks in tests

    def _completed(self, job_id, body):
        qid = next(iter(body['input']['questions']))
        doc = body['input']['state']['document']
        output = {'model': FAKE_CHOICE_MODEL, 'latency_ms': 400,
                  'usage': {'input_tokens': sum(len(v) for v in body['input']['state'].values()) // 4,
                            'output_tokens': 0}}
        if 'FAKE:REFUSE' in doc:
            output['error'] = 'fake handler error'
        elif 'FAKE:FORMAT' in doc:
            output['answers'] = {qid: {'type': 'choice', 'choice': 'opt_keep',
                                       'probabilities': {'opt_keep': 0.3, 'opt_drop': 0.2}}}
        else:
            m = _KEEP_RE.search(doc)
            p = float(m.group(1)) if m else (0.9 if '[[RELEVANT]]' in doc else 0.1)
            probs = {'opt_keep': p, 'opt_drop': 1.0 - p}
            output['answers'] = {qid: {'type': 'choice', 'choice': 'opt_keep' if p >= 0.5 else 'opt_drop',
                                       'probabilities': probs}}
        js = {'id': job_id, 'status': 'COMPLETED', 'delayTime': 100, 'executionTime': 500, 'output': output}
        if 'FAKE:NOTIMING' in doc:
            del js['delayTime'], js['executionTime']
        return js

    def handle(self, request):
        self.requests.append((request.method, request.url.path))
        path = request.url.path
        if '/runsync' in path:
            body = json.loads(request.content)
            self.bodies.append(body)
            doc = body['input']['state']['document']
            if 'FAKE:TIMEOUT' in doc:
                raise httpx.ReadTimeout('fake timeout', request=request)
            if 'FAKE:HTTP500' in doc:
                return httpx.Response(500, text='fake upstream error')
            job = f'fake-job-{next(self._ids)}'
            if 'FAKE:QUEUE' in doc:
                self.jobs[job] = body
                return httpx.Response(200, json={'id': job, 'status': 'IN_QUEUE'})
            return httpx.Response(200, json=self._completed(job, body))
        if '/status/' in path:
            job = path.rsplit('/', 1)[-1]
            return httpx.Response(200, json=self._completed(job, self.jobs[job]))
        if '/cancel/' in path:
            return httpx.Response(200, json={'id': path.rsplit('/', 1)[-1], 'status': 'CANCELLED'})
        return httpx.Response(404, text='unknown fake path')


def fake_for(backend):
    return FakeAnthropic() if backend == 'haiku_direct' else FakeChoiceEndpoint()


FAKE_CHOICE_MODEL = 'fake-choice-model'      # the model name the fake choice endpoint reports

DUMMY_KEY = 'sk-fake-DUMMY-KEY-not-a-secret-0123456789'
