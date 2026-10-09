"""Provider responses are parsed raw; only output paths (error details, stored traces, CLI files, logs) are scrubbed.

Regression for a real failure mode: a URL-safe base64 blob (e.g. an ``encrypted_content`` field) that happens to
contain ``-sk-<8+ chars>`` was rewritten to ``[redacted]`` BEFORE json parsing, so the caller received a corrupted
value. Covers success, nested fields, parse failure, HTTP error, provider error text, transport exceptions and
repeated calls. Synthetic secrets only; fake transports only.
"""
from __future__ import annotations

import hashlib
import json
import logging

import httpx

from judge_adapter import AdapterConfig, JudgeAdapter, JudgeRequest, Response, build_backend, fakes
from judge_adapter import cli as J
from judge_adapter import context_filter as F
from judge_adapter import keys as K
from judge_adapter import prompt as P
from judge_adapter.context_filter import ContextFilter, Document

# Synthetic values only. SECRET is deliberately NOT key-like, so only the exact-secret rule can remove it.
SECRET = 'synthetic-TEST-secret-7f3a9c41'
KEYLIKE = 'sk-SYNTHETICkeylike0123456789'
# URL-safe base64-like blob with "-sk-..." inside by chance: the shape of the reported incident.
BLOB = 'gAAAAABnZx9_q-sk-A1b2C3d4E5f6G7h8_q-Lw9xR2yT=='


def _choice_output(qid, probs, choice, extra=None):
    out = {'model': fakes.FAKE_CHOICE_MODEL, 'latency_ms': 1, 'usage': {'input_tokens': 10, 'output_tokens': 0},
           'answers': {qid: {'type': 'choice', 'choice': choice, 'probabilities': probs}},
           'reasoning': {'items': [{'type': 'reasoning', 'encrypted_content': BLOB, 'note': f'id {KEYLIKE}'}]}}
    out.update(extra or {})
    return out


class ScriptedChoice:
    """A choice endpoint that answers each /runsync with the next scripted response (bytes kept for comparison)."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []          # exact response bytes, in order

    def transport(self):
        return httpx.MockTransport(self.handle)

    def handle(self, request):
        body = json.loads(request.content)
        qid = next(iter(body['input']['questions']))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        status, content = step(qid, len(self.sent) + 1)
        self.sent.append(content)
        return httpx.Response(status, content=content)


def ok_pair(qid, n):
    js = {'id': f'job-{n}-{KEYLIKE}', 'status': 'COMPLETED', 'delayTime': 1, 'executionTime': 2,
          'output': _choice_output(qid, {'opt_1': 0.8, 'opt_2': 0.1, 'opt_tie': 0.1}, 'opt_1')}
    return 200, json.dumps(js, ensure_ascii=False).encode('utf-8')


def req(rid='r1'):
    return JudgeRequest(request_id=rid, question_turns=('Q?',), response_a=Response('id-a', ('a',)),
                        response_b=Response('id-b', ('b',)))


def adapter(tmp_path, fake, mode='single'):
    cfg = AdapterConfig(transport='fake', backend='choice_endpoint', mode=mode, trace_dir=str(tmp_path / 't'))
    be = build_backend(cfg, transport=fake.transport(), key_provider=lambda: SECRET, sleep=lambda s: None)
    return JudgeAdapter(cfg, be), be


def assert_clean(text):
    assert SECRET not in text and KEYLIKE not in text and BLOB not in text
    assert K._KEYLIKE_RE.search(text) is None, text


def stored(tmp_path):
    return {n: (tmp_path / 't' / n).read_text(encoding='utf-8') for n in ('judge_raw.jsonl', 'judge_trace.jsonl')}


# (a) key-like substrings inside JSON values are parsed and returned byte-for-byte --------------------------------
def test_success_nested_keylike_values_preserved_byte_for_byte(tmp_path):
    fake = ScriptedChoice([ok_pair])
    _, be = adapter(tmp_path, fake)
    r = be.call(P.semantic(('Q?',), ('a',), ('b',), 1), {'opt_1': 'id-a', 'opt_2': 'id-b', 'opt_tie': 'tie'})
    assert r['outcome'] == 'ok' and r['label'] == 'id-a'
    assert r['raw_text'].encode('utf-8') == fake.sent[0]                      # the body as received
    js = r['exchanges'][0]['json']
    item = js['output']['reasoning']['items'][0]
    assert item['encrypted_content'] == BLOB and item['note'] == f'id {KEYLIKE}'   # nested values intact
    assert r['job_id'] == f'job-1-{KEYLIKE}'                                  # parsed field intact


def test_adapter_returns_raw_fields_but_stores_scrubbed(tmp_path):
    fake = ScriptedChoice([ok_pair])
    ad, _ = adapter(tmp_path, fake)
    res = ad.judge(req())
    assert res.status == 'ok' and res.calls[0].job_id == f'job-1-{KEYLIKE}'   # in-memory result: provider data
    files = stored(tmp_path)
    for text in files.values():
        assert_clean(text)
    assert '[redacted]' in files['judge_raw.jsonl'] and '[redacted]' in files['judge_trace.jsonl']
    rec = json.loads(files['judge_raw.jsonl'])['exchanges'][0]['text']
    assert res.calls[0].raw_output_sha256 == hashlib.sha256(rec.encode()).hexdigest()   # hash of the stored body


def test_repeated_calls_parse_raw_and_store_scrubbed(tmp_path):
    fake = ScriptedChoice([ok_pair])
    ad, _ = adapter(tmp_path, fake, mode='two_direction')
    out = [ad.judge(req(f'r{i}')) for i in range(3)]               # the key is known from the first call onward
    assert all(c.outcome == 'ok' for r in out for c in r.calls) and len(fake.sent) == 6   # always opt_1 -> tie
    assert [c.job_id for r in out for c in r.calls] == [f'job-{n}-{KEYLIKE}' for n in range(1, 7)]
    for text in stored(tmp_path).values():
        assert_clean(text)


# parse failure, HTTP error, provider error text, transport exception ----------------------------------------------
def test_parse_failure_body_is_not_parsed_and_never_stored_raw(tmp_path):
    body = f'<html>upstream said {SECRET} {KEYLIKE} {BLOB}</html>'.encode()
    fake = ScriptedChoice([lambda qid, n: (200, body)])
    ad, be = adapter(tmp_path, fake)
    r = be.call(P.semantic(('Q?',), ('a',), ('b',), 1), {'opt_1': 'id-a', 'opt_2': 'id-b', 'opt_tie': 'tie'})
    assert r['outcome'] == 'http' and 'without a JSON object' in r['detail']
    assert r['raw_text'].encode() == body                                    # in memory: as received
    res = ad.judge(req())
    assert res.failure['reason'] == 'http'
    assert_clean(json.dumps(res.to_dict()))
    for text in stored(tmp_path).values():
        assert_clean(text)


def test_http_error_body_with_secret_is_scrubbed_in_errors_and_traces(tmp_path):
    body = json.dumps({'error': {'message': f'bad key {SECRET}', 'hint': f'Bearer {KEYLIKE}', 'blob': BLOB}}).encode()
    fake = ScriptedChoice([lambda qid, n: (401, body)])
    ad, _ = adapter(tmp_path, fake)
    res = ad.judge(req())
    assert res.failure['reason'] == 'http' and res.calls[0].http_status == 401
    assert_clean(json.dumps(res.to_dict()))
    for text in stored(tmp_path).values():
        assert_clean(text)


def test_provider_error_text_is_scrubbed_in_detail(tmp_path):
    def failed(qid, n):
        return 200, json.dumps({'id': f'job-{n}', 'status': 'FAILED', 'delayTime': 1, 'executionTime': 2,
                                'error': f'worker crashed with {SECRET} and {KEYLIKE}'}).encode()

    def handler_err(qid, n):
        out = _choice_output(qid, {'opt_1': 0.8, 'opt_2': 0.1, 'opt_tie': 0.1}, 'opt_1',
                             {'error': f'handler saw {SECRET} {KEYLIKE}'})
        return 200, json.dumps({'id': f'job-{n}', 'status': 'COMPLETED', 'delayTime': 1, 'executionTime': 2,
                                'output': out}).encode()

    for step in (failed, handler_err):
        ad, _ = adapter(tmp_path / step.__name__, ScriptedChoice([step]))
        res = ad.judge(req())
        assert res.status == 'error' and '[redacted]' in res.failure['detail']
        assert_clean(json.dumps(res.to_dict()))
        for text in stored(tmp_path / step.__name__).values():
            assert_clean(text)


def test_transport_exception_message_is_scrubbed(tmp_path):
    fake = ScriptedChoice([httpx.ConnectError(f'proxy rejected Bearer {SECRET} / {KEYLIKE}')])
    ad, _ = adapter(tmp_path, fake)
    res = ad.judge(req())
    assert res.failure['reason'] == 'http' and 'ConnectError' in res.failure['detail']
    assert_clean(json.dumps(res.to_dict()))
    for text in stored(tmp_path).values():
        assert_clean(text)


def test_haiku_nested_keylike_preserved_and_http_error_scrubbed(tmp_path):
    def handle(request):
        if b'HTTPERR' in request.content:
            return httpx.Response(400, content=json.dumps({'type': 'error', 'error': {
                'type': 'invalid_request_error', 'message': f'{SECRET} {KEYLIKE} {BLOB}'}}).encode())
        return httpx.Response(200, content=json.dumps({
            'id': f'msg_{KEYLIKE}', 'type': 'message', 'model': P.HAIKU_MODEL, 'stop_reason': 'end_turn',
            'content': [{'type': 'text', 'text': 'opt_1'}, {'type': 'thinking', 'signature': BLOB}],
            'usage': {'input_tokens': 5, 'output_tokens': 3}}).encode())

    cfg = AdapterConfig(transport='fake', trace_dir=str(tmp_path / 't'))
    be = build_backend(cfg, transport=httpx.MockTransport(handle), key_provider=lambda: SECRET)
    ad = JudgeAdapter(cfg, be)
    sem = P.semantic(('Q?',), ('a',), ('b',), 1)
    r = be.call(sem, {'opt_1': 'id-a', 'opt_2': 'id-b', 'opt_tie': 'tie'})
    assert r['outcome'] == 'ok' and r['exchanges'][0]['json']['content'][1]['signature'] == BLOB
    assert r['exchanges'][0]['json']['id'] == f'msg_{KEYLIKE}'
    res = ad.judge(JudgeRequest(request_id='h1', question_turns=('Q?',), response_a=Response('id-a', ('HTTPERR',)),
                                response_b=Response('id-b', ('b',))))
    assert res.failure['reason'] == 'http'
    assert_clean(json.dumps(res.to_dict()))
    for text in stored(tmp_path).values():
        assert_clean(text)


# context filter and CLI output files ------------------------------------------------------------------------------
def test_context_filter_scores_from_raw_body_and_logs_stay_clean(caplog):
    def relevance(qid, n):
        p = 0.9 if n % 2 else 0.2
        js = {'id': f'job-{n}-{KEYLIKE}', 'status': 'COMPLETED', 'delayTime': 1, 'executionTime': 2,
              'output': _choice_output(qid, {'opt_keep': p, 'opt_drop': round(1 - p, 10)},
                                       'opt_keep' if p > 0.5 else 'opt_drop', {'model': f'served-{KEYLIKE}'})}
        return 200, json.dumps(js).encode()

    fake = ScriptedChoice([relevance])
    cfg = AdapterConfig(transport='fake', backend='choice_endpoint', trace_dir='unused-trace',
                        choice_usd_per_second=0.001)
    be = build_backend(cfg, transport=fake.transport(), key_provider=lambda: SECRET, sleep=lambda s: None)
    flt = ContextFilter(cfg, be, policy=F.threshold(0.5))
    docs = [Document(id=f'd{i}', title='t', url='https://example.invalid', text=f'text {i}') for i in range(3)]
    with caplog.at_level(logging.INFO, logger='judge_adapter.context_filter'):
        res = flt.filter('Synthetic question?', docs)
    assert [r.score for r in res.documents] == [0.9, 0.2, 0.9] and res.selected_ids == ['d0', 'd2']
    assert res.documents[0].provenance['response_model'] == f'served-{KEYLIKE}'   # in-memory: provider data
    logged = '\n'.join(r.getMessage() for r in caplog.records)
    assert logged and SECRET not in logged and K._KEYLIKE_RE.search(logged) is None


def test_cli_saved_files_are_scrubbed(tmp_path):
    p = tmp_path / 'in.jsonl'
    p.write_text(''.join(json.dumps({'id': f'r{i}', 'question_turns': ['Q?'],
                                     'response_a': {'id': 'id-a', 'turns': ['a']},
                                     'response_b': {'id': 'id-b', 'turns': ['b']}}) + '\n' for i in range(2)))
    fake = ScriptedChoice([ok_pair])
    s = J.run_file(p, tmp_path / 'out', transport='fake', backend='choice_endpoint', fake=fake,
                   key_provider=lambda: SECRET)
    assert s['counts']['ok'] == 2
    for f in ('results.jsonl', 'summary.json', 'gate.json', 'trace/judge_raw.jsonl', 'trace/judge_trace.jsonl'):
        assert_clean((tmp_path / 'out' / f).read_text(encoding='utf-8'))
