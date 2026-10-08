"""User rubric + per-turn reference, model-free. Wire bodies are captured from the fake
transport (the bytes the backend would send), not read from in-memory objects."""
from __future__ import annotations

import hashlib
import json
import socket
import tempfile

import httpx
import pytest

from judge_adapter import cli as J, fakes
from judge_adapter import AdapterConfig, JudgeAdapter, JudgeRequest, Response, build_backend
from judge_adapter import keys as K
from judge_adapter import prompt as P
from judge_adapter import user_prompt as U
from judge_adapter.gate import haiku_reservation_usd

RUBRIC = 'Prefer the answer that cites a primary source. Ignore length.'
REF2 = (None, 'The second answer should be 23 or 29.')


@pytest.fixture(autouse=True)
def no_key_no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('key read or network attempted')
    monkeypatch.setattr(K, 'dotenv_key', boom)
    monkeypatch.setattr(K, 'env_key', boom)
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


class Capture:
    """Wraps a fake server and keeps the exact POST bytes."""

    def __init__(self, backend):
        self.fake = fakes.fake_for(backend)
        self.bodies = []
        self.requests = self.fake.requests

    def transport(self):
        def handle(req):
            if req.method == 'POST' and ('/v1/messages' in req.url.path or '/runsync' in req.url.path):
                self.bodies.append(req.content)
            return self.fake.handle(req)
        return httpx.MockTransport(handle)

    def count(self, *a, **k):
        return self.fake.count(*a, **k)

    def json(self):
        return [json.loads(b) for b in self.bodies]


def judge(req, backend='haiku_direct', mode='two_direction'):
    cap = Capture(backend)
    cfg = AdapterConfig(transport='fake', backend=backend, mode=mode, trace_dir=tempfile.mkdtemp())
    ad = JudgeAdapter(cfg, build_backend(cfg, transport=cap.transport(), key_provider=lambda: fakes.DUMMY_KEY,
                                         sleep=lambda s: None))
    return ad.judge(req), cap


# --- (1) default byte parity -------------------------------------------------------------------------------------
GOLDEN_CASES = {
    't1': JudgeRequest('g1', ('Give one tip for a bug report.',), Response('a', ('Write good reports.',)),
                       Response('b', ('State steps, expected, actual.',))),
    't2': JudgeRequest('g2', ('Name a prime 10-20.', 'Now 20-30.'), Response('x', ('13.', '23.')),
                       Response('y', ('11.', '25.')), turn=2),
    'uni': JudgeRequest('g3', ('한국어 질문 "따옴표" \\ 백슬래시?',), Response('p', ('답 ✓\n둘째 줄',)), Response('q', ('',))),
}
# sha256 of the POST bodies (AB, BA) of the fixed default prompt, captured before rubric/reference support was added
GOLDEN = {
    'haiku_direct:t1': ['7b329c4073fee9efac9eed655bda3c90a01f01397063a16a66899af2e272b213',
                        '9653808c6be91da5e1644d451ba244739da1b1ab5d46a9da9b76e578d24d8e32'],
    'haiku_direct:t2': ['77e95b2baf40621bb976cf5c7e36af1b40cec5c0426b990ae84249e9f07837c5',
                        '7cf57930a2ffa08f9dc5bf6d6ef9bdb1ec1707fefb957055538f6e66e3ce5cf8'],
    'haiku_direct:uni': ['1976c93b59ab901c62fc8f1466a412ca80028c3d3a035cda481b149e33702e4a',
                         'ce5bc40bd6b97378b7ad92ec768c5642878a987838977402efd75752652d8272'],
    'choice_endpoint:t1': ['e00c9751d7b364c3af962ba3bd92a3e5dd26d3bc0d12e93ade7a84469bd3e7ff',
                '26d2d4d46cff33188670171e529f68a94366e34e3d6acf0a065bcd4dbaa2a355'],
    'choice_endpoint:t2': ['8e916f8eaaea7a5f471f5dc27aae1b00b49f9140fbbfc7296c15ecde05bc8a88',
                'aa303012e28d3d0b88080e19d17c5a15bb63eed232714d57f4299ae43754571b'],
    'choice_endpoint:uni': ['ecfaa1a0a5b9a2b0340a636d48a30041bc16e04c53a466f58614fadadc440189',
                 'd6e829f291b27f099f062b17a852ee5cec0689e8615e932574d60f2ddad34010'],
}


@pytest.mark.parametrize('key', sorted(GOLDEN))
def test_default_wire_bodies_are_byte_identical_to_before(key):
    backend, case = key.split(':')
    r, cap = judge(GOLDEN_CASES[case], backend)
    assert [hashlib.sha256(b).hexdigest() for b in cap.bodies] == GOLDEN[key]
    assert r.criteria_info['prompt_version'] == U.PROMPT_VERSION_DEFAULT and r.criteria_info['reference'] is None
    turn = GOLDEN_CASES[case].turn
    assert r.criteria_info['rubric'] == {'source': 'default', 'sha256': U.sha_text(P.INSTRUCTIONS[turn])}


# sha256 of the wire body of the original fixed-prompt renderer for this two-turn item (slot 1 = 'ry'), recorded
# when the renderer was copied into this package
FROZEN_RENDERER_SHA256 = {'haiku_direct': '51eb91c598dc5a9073d2f732bb8806a59700d927fbe41ce915a246cf7e4aca0e',
                          'choice_endpoint': 'e17c5f8cf81789b0833c2a3741cb712d681fc66aed745c86ffbb00db48a26f1e'}


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_default_wire_body_equals_frozen_renderer(backend):
    r, cap = judge(JudgeRequest('i', ('Q1', 'Q2'), Response('ry', ('Y1', 'Y2')), Response('rx', ('X1', 'X2')), turn=2),
                   backend, mode='single')
    assert hashlib.sha256(cap.bodies[0]).hexdigest() == FROZEN_RENDERER_SHA256[backend]


def test_cli_default_line_sends_the_same_bytes(tmp_path):
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps({'id': 'g1', 'question_turns': ['Give one tip for a bug report.'],
                             'response_a': {'id': 'a', 'turns': ['Write good reports.']},
                             'response_b': {'id': 'b', 'turns': ['State steps, expected, actual.']}}) + '\n')
    cap = Capture('haiku_direct')
    J.run_file(p, tmp_path / 'o', transport='fake', mode='two_direction', fake=cap)
    assert [hashlib.sha256(b).hexdigest() for b in cap.bodies] == GOLDEN['haiku_direct:t1']


# --- (2) rubric / reference on both wire bodies ------------------------------------------------------------------
def user_req(rubric=RUBRIC, ref=REF2, a=('13.', '23.'), b=('11.', '25.')):
    return JudgeRequest('u1', ('Name a prime 10-20.', 'Now 20-30.'), Response('x', a), Response('y', b), turn=2,
                        rubric=rubric, reference_turns=ref)


def test_haiku_wire_text_carries_rubric_and_reference_and_drops_no_field():
    r, cap = judge(user_req(), 'haiku_direct', mode='single')
    text = cap.json()[0]['messages'][0]['content']
    assert f'<{U.RUBRIC_FIELD}>\n{RUBRIC}\n</{U.RUBRIC_FIELD}>' in text
    assert f'<{U.REFERENCE_FIELD}>\n### Reference answer to user question 2:\n{REF2[1]}\n</{U.REFERENCE_FIELD}>' in text
    assert 'helpfulness, relevance' not in text            # the rubric replaces the general standard
    assert 'not one of the candidates' in text and P.HAIKU_FORMAT_LINE in text
    sem = U.semantic(('Name a prime 10-20.', 'Now 20-30.'), ('13.', '23.'), ('11.', '25.'), 2, RUBRIC, REF2)
    assert all(f'<{f}>' in text for f in sem['state'])     # every state field reaches the wire
    body = cap.json()[0]
    assert {k: body[k] for k in ('model', 'max_tokens', 'temperature')} == {
        'model': P.HAIKU_MODEL, 'max_tokens': P.HAIKU_MAX_TOKENS, 'temperature': P.HAIKU_TEMPERATURE}
    assert r.criteria_info['prompt_version'] == U.PROMPT_VERSION_USER


def test_choice_wire_body_carries_rubric_and_reference():
    r, cap = judge(user_req(), 'choice_endpoint', mode='single')
    body = cap.json()[0]
    st, q = body['input']['state'], body['input']['questions'][P.QUESTION_ID]
    assert st[U.RUBRIC_FIELD] == RUBRIC
    assert st[U.REFERENCE_FIELD] == f'### Reference answer to user question 2:\n{REF2[1]}'
    assert q['criteria'] == P.CRITERIA and q['type'] == 'choice' and body['policy'] == P.CHOICE_POLICY
    assert U.RUBRIC_FIELD in q['instructions'] and 'helpfulness, relevance' not in q['instructions']


def test_reference_only_keeps_general_standard_and_both_backends_get_the_same_semantics():
    rq = user_req(rubric=None, ref=('ref one', 'ref two'))
    _, ch = judge(rq, 'haiku_direct', mode='single')
    _, cd = judge(rq, 'choice_endpoint', mode='single')
    text, choice = ch.json()[0]['messages'][0]['content'], cd.json()[0]['input']
    assert 'helpfulness, relevance' in text and U.RUBRIC_FIELD not in choice['state']
    assert choice['questions'][P.QUESTION_ID]['instructions'] in text
    for f, v in choice['state'].items():
        assert f'<{f}>\n{v}\n</{f}>' in text                 # same fields, same values on both wires


def test_rubric_only_has_no_reference_field():
    _, cap = judge(user_req(ref=None), 'choice_endpoint', mode='single')
    st = cap.json()[0]['input']['state']
    assert U.REFERENCE_FIELD not in st and st[U.RUBRIC_FIELD] == RUBRIC


# --- (3) direction swap ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_swap_keeps_rubric_and_reference_and_restores_winner(backend):
    r, cap = judge(user_req(a=('13.', '23.'), b=('11.', '[[GOOD]] 29.')), backend)
    assert (r.status, r.verdict, r.winner_id) == ('ok', 'B', 'y')
    assert [c.slot_ids for c in r.calls] == [{'1': 'x', '2': 'y'}, {'1': 'y', '2': 'x'}]
    if backend == 'choice_endpoint':
        ab, ba = (b['input'] for b in cap.json())
        for f in (U.RUBRIC_FIELD, U.REFERENCE_FIELD, 'question'):
            assert ab['state'][f] == ba['state'][f]
        assert ab['questions'] == ba['questions']
        assert ab['state']['assistant_1_conversation'] == ba['state']['assistant_2_conversation']
    else:
        ab, ba = (b['messages'][0]['content'] for b in cap.json())
        for f in (U.RUBRIC_FIELD, U.REFERENCE_FIELD):
            pick = lambda t: t.split(f'<{f}>')[1].split(f'</{f}>')[0]  # noqa: E731
            assert pick(ab) == pick(ba)
        assert ab.split('\n\n')[0] == ba.split('\n\n')[0]     # same instructions


# --- (4) validation and limits -----------------------------------------------------------------------------------
def line(**extra):
    obj = {'id': 'r1', 'question_turns': ['Q1?', 'Q2?'], 'response_a': {'id': 'a', 'turns': ['a1', 'a2']},
           'response_b': {'id': 'b', 'turns': ['b1', 'b2']}}
    obj.update(extra)
    return obj


def errors(tmp_path, obj, backend='haiku_direct'):
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(obj) + '\n')
    with pytest.raises(J.InputRejected) as e:
        J.load_requests(p, backend)
    return [m for _, m in e.value.errors]


@pytest.mark.parametrize('extra,needle', [
    ({'rubric': 5}, 'rubric must be a string'),
    ({'rubric': '   '}, 'rubric is empty'),
    ({'rubric': '가' * 2667}, 'rubric is 8001 UTF-8 bytes > limit 8000'),
    ({'reference': 'R'}, 'reference must be a list'),
    ({'reference': ['R']}, 'reference has 1 entries but question_turns has 2'),
    ({'reference': [None, None]}, 'only null entries'),
    ({'reference': []}, 'empty list'),
    ({'reference': [None, '']}, 'reference[1] is empty'),
    ({'reference': [None, 3]}, 'reference[1] must be a string or null'),
    ({'reference': [None, 'x' * 20_001]}, 'reference[1] is 20001 UTF-8 bytes > limit 20000'),
    ({'instructions': 'Be strict.', 'rubric': 'R'}, "use 'rubric'"),
    ({'criteria': {'opt_1': 'x'}}, 'option ids'),
])
def test_rubric_reference_validation(tmp_path, extra, needle):
    errs = errors(tmp_path, line(**extra))
    assert any(needle in m for m in errs), errs


def test_rubric_inside_a_response_is_rejected(tmp_path):
    obj = line()
    obj['response_a']['reference'] = ['R1', 'R2']
    assert any('belongs at the top level' in m for m in errors(tmp_path, obj))


def test_limits_are_utf8_bytes_and_boundaries_pass(tmp_path):
    p = tmp_path / 'ok.jsonl'
    p.write_text(json.dumps(line(rubric='가' * 2666, reference=[None, 'x' * 20_000])) + '\n')
    (_, r), = J.load_requests(p)
    assert U.utf8_len(r.rubric) == 7998 and len(r.rubric) == 2666 and r.reference_turns == (None, 'x' * 20_000)


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_total_wire_body_limit(tmp_path, backend):
    big = 'y' * 199_000
    obj = line(question_turns=['Q?'], response_a={'id': 'a', 'turns': [big]}, response_b={'id': 'b', 'turns': ['b']})
    assert any('wire body' in m and 'limit 200000' in m for m in errors(tmp_path, obj, backend))


def test_adapter_refuses_ambiguous_or_oversized_user_fields_without_a_call():
    for rq in (JudgeRequest('r', ('q',), Response('a', ('x',)), Response('b', ('y',)), rubric='R', instructions='I'),
               JudgeRequest('r', ('q',), Response('a', ('x',)), Response('b', ('y',)), rubric='R' * 8001),
               JudgeRequest('r', ('q',), Response('a', ('x',)), Response('b', ('y',)), reference_turns=('a', 'b')),
               JudgeRequest('r', ('q',), Response('a', ('x',)), Response('b', ('y',)), reference=['host ref'],
                            rubric='R')):
        r, cap = judge(rq, mode='single')
        assert r.failure['reason'] == 'unsupported_input' and r.submissions == 0 and cap.bodies == []


# --- (5) reservation includes the extra input --------------------------------------------------------------------
def test_haiku_reservation_grows_with_rubric_and_reference(tmp_path):
    base = U.semantic(('Q1?', 'Q2?'), ('a1', 'a2'), ('b1', 'b2'), 2)
    user = U.semantic(('Q1?', 'Q2?'), ('a1', 'a2'), ('b1', 'b2'), 2, 'R' * 4000, (None, 'S' * 4000))
    r0, r1 = haiku_reservation_usd(P.haiku_body(base)), haiku_reservation_usd(P.haiku_body(user))
    assert r1 - r0 >= 8000 * P.HAIKU_PRICE_IN_PER_MTOK * 2 / 1e6      # extra bytes x price x safety multiplier
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(line(id='plain')) + '\n'
                 + json.dumps(line(id='big', rubric='R' * 4000, reference=[None, 'S' * 4000])) + '\n')
    cap = (r0 + r1) / 2                    # the plain request fits, the one with rubric + reference does not
    s = J.run_file(p, tmp_path / 'o', transport='fake', budget_cap_usd=cap, fake=fakes.FakeAnthropic())
    gate = json.loads((tmp_path / 'o' / 'gate.json').read_text())
    assert gate['log'][0]['reservation_usd'] == r0 and gate['log'][1]['decision'] == 'refused'
    assert s['counts']['unrun'] == 1 and 'cap' in s['limits']['stopped']


# --- (6) trace and CLI end to end --------------------------------------------------------------------------------
def test_cli_example_records_versions_and_hashes(tmp_path):
    ex = J.EXAMPLES_DIR / 'pairs.rubric_reference.example.jsonl'
    s = J.run_file(ex, tmp_path / 'o', transport='fake', mode='two_direction')
    rows = [json.loads(x) for x in (tmp_path / 'o' / 'results.jsonl').read_text().splitlines()]
    src = [json.loads(x) for x in ex.read_text().splitlines()]
    assert s['prompt_versions'] == {U.PROMPT_VERSION_USER: 3} and s['counts']['requests'] == 3
    for row, obj in zip(rows, src):
        ci = row['criteria_info']
        if 'rubric' in obj:
            assert ci['rubric'] == {'source': 'user', 'sha256': U.sha_text(obj['rubric']),
                                    'utf8_bytes': U.utf8_len(obj['rubric'])}
        else:
            assert ci['rubric']['source'] == 'default'
        if 'reference' in obj:
            want = hashlib.sha256(json.dumps(obj['reference'], ensure_ascii=False).encode('utf-8')).hexdigest()
            assert ci['reference']['sha256'] == want
        else:
            assert ci['reference'] is None
    trace = [json.loads(x) for x in (tmp_path / 'o' / 'trace' / 'judge_trace.jsonl').read_text().splitlines()]
    assert [t['criteria_info'] for t in trace] == [r['criteria_info'] for r in rows]
    assert fakes.DUMMY_KEY not in (tmp_path / 'o' / 'trace' / 'judge_raw.jsonl').read_text()


def test_validate_prints_prompt_version_and_body_size(capsys):
    ex = J.EXAMPLES_DIR / 'pairs.rubric_reference.example.jsonl'
    assert J.main(['validate', str(ex), '--backend', 'choice_endpoint']) == 0
    out = capsys.readouterr().out
    assert out.count(U.PROMPT_VERSION_USER) == 3 and 'rubric user' in out and 'reference turns [2]' in out
