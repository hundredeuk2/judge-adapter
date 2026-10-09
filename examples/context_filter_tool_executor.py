# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Example: a function-call tool executor that filters search results with judge_adapter.context_filter.

Model-free: a mock search returns synthetic documents and the filter talks to the in-process fake choice endpoint
(markers such as [[KEEP=0.8]] in the synthetic text set the fake P(keep)). No key, no network, no agent SDK.

    python examples/context_filter_tool_executor.py

The executor answers tool calls shaped like the function calls of common chat APIs
({"id", "name", "arguments": "<json string>"}). `web_search` returns only the selected documents and lists the
omitted ids; the ORIGINAL documents stay in the executor's own store, and `get_document` restores one by id. Storing
and restoring originals is the caller's job; the filter never rewrites or keeps text.
"""
from __future__ import annotations

import json
import logging

from judge_adapter import AdapterConfig, build_backend, fakes
from judge_adapter.context_filter import ContextFilter, Document, FilterInputError, select_originals

SYNTHETIC_INDEX = [
    Document('s1', 'Festival overview', 'https://example.invalid/festival',
             'The fictional Lantern Festival is held every spring in the synthetic city of Norvale. [[KEEP=0.82]]'),
    Document('s2', 'Unrelated recipe', 'https://example.invalid/recipe',
             'A synthetic recipe for barley soup with no festival content. [[KEEP=0.04]]'),
    Document('s3', 'Travel notes', 'https://example.invalid/travel',
             'Synthetic travel notes mention Norvale trains and, briefly, a spring event. [[KEEP=0.31]]'),
    Document('s4', 'Broken page', 'https://example.invalid/broken',
             'This synthetic page makes the fake endpoint return malformed probabilities. FAKE:FORMAT'),
]


def mock_search(query, max_results=4):
    """Stand-in for a real search API: returns synthetic documents in a fixed rank order."""
    return list(SYNTHETIC_INDEX[:max_results])


class ToolExecutor:
    def __init__(self, context_filter):
        self.filter = context_filter
        self.originals = {}                       # caller-owned store: id -> original Document

    def execute(self, tool_call):
        args = json.loads(tool_call['arguments'] or '{}')
        if tool_call['name'] == 'web_search':
            content = self._web_search(args['query'])
        elif tool_call['name'] == 'get_document':
            d = self.originals.get(args['id'])
            content = {'error': 'unknown id'} if d is None else {'id': d.id, 'title': d.title, 'url': d.url,
                                                                 'text': d.text}
        else:
            content = {'error': f'unknown tool {tool_call["name"]}'}
        return {'tool_call_id': tool_call['id'], 'content': json.dumps(content, ensure_ascii=False)}

    def _web_search(self, query):
        docs = mock_search(query)
        for d in docs:
            self.originals[d.id] = d              # keep every original, selected or not
        try:
            res = self.filter.filter(query, docs)
        except FilterInputError as e:            # refused before any call: pass everything through, say so
            return {'results': [vars(d) for d in docs], 'filter': {'refused': e.problems}}
        kept = select_originals(docs, res)
        return {'results': [vars(d) for d in kept],
                'omitted_ids': [r.id for r in res.documents if r.decision == 'drop'],
                'filter': {'statuses': {r.id: r.status for r in res.documents}, 'mock': res.mock}}


def main():
    logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')    # httpx logs request URLs at INFO
    logging.getLogger('judge_adapter.context_filter').setLevel(logging.INFO)    # counts and sizes only
    cfg = AdapterConfig(transport='fake', backend='choice_endpoint', trace_dir='unused-trace')
    fake = fakes.FakeRelevanceEndpoint()
    backend = build_backend(cfg, transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY)
    flt = ContextFilter(cfg, backend, preset='v1-thr0.3-dev')          # explicit opt-in; there is no default policy
    ex = ToolExecutor(flt)
    out = ex.execute({'id': 'call_1', 'name': 'web_search',
                      'arguments': json.dumps({'query': 'Which synthetic city hosts the Lantern Festival?'})})
    print('[MOCK] web_search ->', out['content'])
    restored = ex.execute({'id': 'call_2', 'name': 'get_document', 'arguments': json.dumps({'id': 's2'})})
    print('[MOCK] get_document(s2) ->', restored['content'])
    print(f'[MOCK] fake endpoint requests: {fake.count("POST")} (not a model run)')
    flt.close()


if __name__ == '__main__':
    main()
