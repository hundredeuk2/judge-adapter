"""OPTIONAL (extra "fastchat"): FastChat reference categories through an EXPLICIT reference mapping, model-free.

Host gen_judgment.py __main__ -> patched run_judge_pair -> adapter -> fake transport (wire bytes captured) -> host
result file. Network and key reads fail the test.

The fake judge reads only the marked answers, never the reference. A matching fake verdict in both host directions
shows that the IDs are restored correctly across the swap. It is NOT evidence that a real model is order invariant.

Skipped unless a FastChat source checkout at commit 587d5cfa is given in JUDGE_ADAPTER_FASTCHAT_ROOT: the demo
workdir and the MT-Bench reference used for comparison are read from that checkout (nothing is bundled). The
FastChat-free parts of the mapping are covered by test_reference_mapping.py.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import socket
import tempfile
from pathlib import Path

import httpx
import pytest

from judge_adapter import AdapterConfig, JudgeRequest, Response
from judge_adapter import fakes, fastchat_runner as RF
from judge_adapter import keys as K
from judge_adapter import user_prompt as U
from judge_adapter.hosts import fastchat as B

FC = os.environ.get('JUDGE_ADAPTER_FASTCHAT_ROOT')
if not FC or not (Path(FC) / 'fastchat' / 'llm_judge' / 'data' / 'mt_bench').is_dir():
    pytest.skip('needs a FastChat checkout in JUDGE_ADAPTER_FASTCHAT_ROOT (optional extra)', allow_module_level=True)
RF._host_path(FC)
pytest.importorskip('fastchat.llm_judge.common')
CONFIG_FAKE = Path(RF.__file__).resolve().parent / 'examples' / 'config.fake.json'

JUDGE = 'judge-adapter'
HOST_ARGS = ['--mode', 'pairwise-baseline', '--bench-name', RF.DEMO_BENCH, '--judge-model', JUDGE,
             '--baseline-model', 'model-y', '--model-list', 'model-x']
ANSWERS = {   # q111 is category "math" in the host data (reference judge pair-math-v1*)
    'model-x': {81: ['[[GOOD]] A Hawaii travel post by X.', 'All sentences start with A, by X.'],
                82: ['An email draft by X.', 'A self-critique by X.'],
                111: ['[[GOOD]] The area is 3, by X.', 'The circle area, by X.']},
    'model-y': {81: ['A Hawaii travel post by Y.', 'All sentences start with A, by Y.'],
                82: ['An email draft by Y.', 'A self-critique by Y.'],
                111: ['The area is 3, by Y.', 'The circle area, by Y.']},
}
REF_T1, REF_T2 = 'USER-REF-111-T1: the area is 3.', 'USER-REF-111-T2: radius sqrt(5), area 5*pi.'
REF_HEAD = '### Reference answer to user question'


@pytest.fixture(autouse=True)
def no_key_no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('key read or network attempted')
    monkeypatch.setattr(K, 'dotenv_key', boom)
    monkeypatch.setattr(K, 'env_key', boom)
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


class Capture:
    """Wraps a fake server and keeps the exact POST bytes (the bytes the backend would send)."""

    def __init__(self, backend):
        self.backend = backend
        self.fake = fakes.fake_for(backend)
        self.bodies = []

    def transport(self):
        def handle(req):
            if req.method == 'POST' and ('/v1/messages' in req.url.path or '/runsync' in req.url.path):
                self.bodies.append(req.content)
            return self.fake.handle(req)
        return httpx.MockTransport(handle)

    def reference_part(self, body):
        """The reference as it appears on the wire, or None."""
        js = json.loads(body)
        if self.backend == 'choice_endpoint':
            return js['input']['state'].get(U.REFERENCE_FIELD)
        m = re.search(rf'<{U.REFERENCE_FIELD}>\n(.*?)\n</{U.REFERENCE_FIELD}>', js['messages'][0]['content'], re.S)
        return m.group(1) if m else None


def write_refs(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return path


def ref_row(qid, turns):
    return {'question_id': qid, 'answer_id': f'user-ref-{qid}', 'model_id': 'user-reference',
            'choices': [{'index': 0, 'turns': list(turns)}], 'tstamp': 0.0}


def user_ref_file(tmp_path, turns=(REF_T1, REF_T2), qid=111, extra=()):
    return write_refs(tmp_path / 'my_refs.jsonl', [ref_row(qid, turns), *extra])


def run(tmp_path, backend='haiku_direct', host_args=HOST_ARGS, answers=ANSWERS, **ref):
    wd = tmp_path / 'w'
    if not wd.exists():
        RF.demo_workdir(wd, FC, answers=answers)
    cap = Capture(backend)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        adapter, _ = RF.judge(AdapterConfig(transport='fake', backend=backend), wd, host_args, fastchat_root=FC,
                              confirm=True,
                              transport=cap.transport(), key_provider=lambda: fakes.DUMMY_KEY, **ref)
    rows = [json.loads(l) for l in (wd / 'data' / RF.DEMO_BENCH / 'model_judgment' / f'{JUDGE}_pair.jsonl')
            .read_text().splitlines()]
    trace = [json.loads(l) for l in adapter.trace.trace.read_text().splitlines()]
    return wd, rows, trace, cap, adapter


def by_key(trace):
    """(question_id, turn, host model_a) -> trace row; model_a == model-x is g1, model-y is g2 (baseline mode)."""
    return {(t['host']['question_id'], int(t['request_id'].split(':')[2][1:]), t['host']['model_a']): t for t in trace}


def gpt4_ref_turns(qid=111):
    p = Path(FC) / 'fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines() if json.loads(l)['question_id'] == qid][0][
        'choices'][0]['turns']


# --- (1) the reference is on both backends' wire bodies, for reference categories only -------------------------------
@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_reference_on_both_backends_wire_and_only_for_reference_category(tmp_path, backend):
    _, rows, trace, cap, _ = run(tmp_path, backend, reference_file=user_ref_file(tmp_path))
    assert len(cap.bodies) == len(trace) == 12                 # 6 matches x 2 host directions, all sent
    refs = [cap.reference_part(b) for b in cap.bodies]
    with_ref = [r for r in refs if r is not None]
    assert len(with_ref) == 4                                   # q111 turn 1 and turn 2, g1 and g2
    t1 = f'{REF_HEAD} 1:\n{REF_T1}'
    t2 = f'{REF_HEAD} 1:\n{REF_T1}\n\n{REF_HEAD} 2:\n{REF_T2}'
    assert sorted(with_ref) == sorted([t1, t1, t2, t2])
    gpt4 = gpt4_ref_turns()                                     # the file staged under the judge name in the workdir
    for b in cap.bodies:
        js = json.loads(b)                                       # search the JSON-decoded wire strings
        flat = js['messages'][0]['content'] if backend == 'haiku_direct' else '\n'.join(js['input']['state'].values())
        assert gpt4[0][:60] not in flat and gpt4[1][:60] not in flat   # never inferred from --judge-model
        if REF_T1 not in flat:
            assert REF_HEAD not in flat and U.REFERENCE_FIELD.encode() not in b
    for t in trace:
        info = t['criteria_info']
        if t['host']['question_id'] == 111:
            assert t['host']['host_prompt_name'].startswith('pair-math-v1')
            assert info['prompt_version'] == U.PROMPT_VERSION_USER and t['submissions'] == 1
            turn = int(t['request_id'].split(':')[2][1:])
            assert info['reference']['turns_with_reference'] == list(range(1, turn + 1))
        else:
            assert info['prompt_version'] == U.PROMPT_VERSION_DEFAULT and info['reference'] is None
            assert 'reference' not in t['host']


# --- (2) both host directions: reference fixed, slots swapped, winner restored ---------------------------------------
@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_swap_keeps_reference_and_restores_winner(tmp_path, backend):
    _, rows, trace, cap, _ = run(tmp_path, backend, reference_file=user_ref_file(tmp_path))
    k = by_key(trace)
    for turn in (1, 2):
        g1, g2 = k[(111, turn, 'model-x')], k[(111, turn, 'model-y')]
        assert g1['calls'][0]['slot_ids'] == {'1': 'ans-model-x-q111', '2': 'ans-model-y-q111'}
        assert g2['calls'][0]['slot_ids'] == {'1': 'ans-model-y-q111', '2': 'ans-model-x-q111'}
        assert g1['criteria_info']['reference'] == g2['criteria_info']['reference']       # same sha both ways
        assert g1['host']['reference'] == g2['host']['reference']
    t1g1, t1g2 = k[(111, 1, 'model-x')], k[(111, 1, 'model-y')]
    assert t1g1['winner_id'] == t1g2['winner_id'] == 'ans-model-x-q111'
    assert (t1g1['verdict'], t1g2['verdict']) == ('A', 'B')
    w = {(r['question_id'], r['turn']): (r['g1_winner'], r['g2_winner']) for r in rows}
    assert w[(111, 1)] == ('model_1', 'model_1')        # host maps g1 A / g2 B back to model_1 = model-x
    assert w[(111, 2)] == ('tie', 'tie')
    # the reference block on the wire is identical in the two directions, while the conversations swap
    ref_bodies = [b for b in cap.bodies if cap.reference_part(b) is not None]
    for i in (0, 2):
        a, b = ref_bodies[i], ref_bodies[i + 1]
        assert cap.reference_part(a) == cap.reference_part(b) and a != b
    # the host stores the rendered adapter input (with the reference) as its prompt fields
    r111 = [r for r in rows if r['question_id'] == 111 and r['turn'] == 1][0]
    assert REF_T1 in r111['g1_user_prompt'] and REF_T1 in r111['g2_user_prompt']


# --- (3) non-reference categories: request bytes identical to the run without a mapping ------------------------------
def _pre_change_to_request(question, answer_a, answer_b, judge, ref_answer, multi_turn):
    """Frozen copy of hosts/fastchat.to_request before reference support."""
    turn = 2 if multi_turn else 1
    qid = question['question_id']
    ia, ib = B.response_id(answer_a, qid), B.response_id(answer_b, qid)
    ref = None
    if ref_answer is not None:
        ref = list(ref_answer['choices'][0]['turns'][:turn])
    return JudgeRequest(
        request_id=f'fastchat:q{qid}:t{turn}:{ia}|{ib}',
        question_turns=tuple(question['turns'][:turn]),
        response_a=Response(ia, tuple(answer_a['choices'][0]['turns'][:turn])),
        response_b=Response(ib, tuple(answer_b['choices'][0]['turns'][:turn])),
        turn=turn, reference=ref,
        host={'host': 'fastchat.llm_judge', 'question_id': qid, 'category': question.get('category'),
              'host_judge_label': judge.model_name, 'host_prompt_name': judge.prompt_template.get('name'),
              'host_prompt_replaced': True, 'model_a': answer_a.get('model_id'), 'model_b': answer_b.get('model_id')})


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_non_reference_bytes_identical_with_and_without_mapping(tmp_path, backend):
    _, _, trace0, cap0, _ = run(tmp_path / 'plain', backend)
    _, _, trace1, cap1, _ = run(tmp_path / 'mapped', backend, reference_file=user_ref_file(tmp_path))
    assert len(cap0.bodies) == 8                         # no mapping: q111 is unsupported_input, 0 calls (as before)
    non_ref = [b for b in cap1.bodies if cap1.reference_part(b) is None]
    assert non_ref == cap0.bodies                        # same bytes, same order
    ref0 = [t for t in trace0 if t['host']['question_id'] == 111]
    assert all(t['failure']['reason'] == 'unsupported_input' and t['calls'] == [] for t in ref0)
    strip = lambda ts: [{k: v for k, v in t.items() if k not in ('total_seconds', 'calls', 'usage', 'cost_estimate')}
                        for t in ts if t['host']['question_id'] != 111]
    assert strip(trace0) == strip(trace1)                # non-reference trace rows equal too (host metadata included)


def test_to_request_unchanged_without_reference():
    q = {'question_id': 81, 'category': 'writing', 'turns': ['Q1', 'Q2']}
    a = {'answer_id': 'a1', 'model_id': 'mx', 'choices': [{'turns': ['A1', 'A2']}]}
    b = {'answer_id': 'b1', 'model_id': 'my', 'choices': [{'turns': ['B1', 'B2']}]}
    judge = type('J', (), {'model_name': 'jl', 'prompt_template': {'name': 'pair-v2'}, 'ref_based': False})()
    host_ref = {'choices': [{'turns': ['R1', 'R2']}]}
    for mt in (False, True):
        for ref in (None, host_ref):
            assert B.to_request(q, a, b, judge, ref, mt) == _pre_change_to_request(q, a, b, judge, ref, mt)


# --- (4) refusals before the host calls the judge ---------------------------------------------------------------------
@pytest.fixture
def host_never_runs(monkeypatch):
    started = []
    monkeypatch.setattr(RF.runpy, 'run_module', lambda *a, **k: started.append(a))
    return started


@pytest.mark.parametrize('rows, needle', [
    ([ref_row(999, ['x'])], 'no reference row'),                                      # 111 missing
    ([ref_row(111, [REF_T1])], 'reference has 1 turns, question has 2'),              # turn mismatch (short)
    ([ref_row(111, [REF_T1, REF_T2, 'extra'])], 'reference has 3 turns'),             # turn mismatch (long)
    ([ref_row(111, [REF_T1, ''])], 'empty or not text'),
    ([ref_row(111, [REF_T1, None])], 'empty or not text'),
    ([ref_row(111, [REF_T1, 'x' * 20_001])], 'over 20000 UTF-8 bytes'),
    ([ref_row(111, [REF_T1, REF_T2]), ref_row(111, [REF_T1, REF_T2])], 'duplicate question_id'),
    ([{'question_id': 111, 'turns': [REF_T1, REF_T2]}], 'not a FastChat answer row'),
])
def test_bad_reference_refused_before_host(tmp_path, host_never_runs, rows, needle):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    before = sorted(p.relative_to(wd) for p in wd.rglob('*'))
    cap = Capture('haiku_direct')
    with pytest.raises(B.ReferenceMappingError, match=re.escape(needle)):
        RF.judge(AdapterConfig(transport='fake'), wd, HOST_ARGS, fastchat_root=FC, confirm=True,
                 transport=cap.transport(),
                 key_provider=lambda: fakes.DUMMY_KEY, reference_file=write_refs(tmp_path / 'r.jsonl', rows))
    assert host_never_runs == [] and cap.bodies == [] and cap.fake.requests == []
    assert sorted(p.relative_to(wd) for p in wd.rglob('*')) == before      # no trace dir, no host output


@pytest.mark.parametrize('kw, needle', [
    ({'reference_file': '/nonexistent/refs.jsonl'}, 'reference file not found'),
    ({'reference_model': 'no-such-model'}, 'reference file not found'),
    ({'reference_model': '../escape'}, 'plain file stem'),
    ({'reference_file': 'X', 'reference_model': 'gpt-4'}, 'not both'),
])
def test_mapping_option_errors(tmp_path, host_never_runs, kw, needle):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    with pytest.raises(B.ReferenceMappingError, match=re.escape(needle)):
        RF.judge(AdapterConfig(transport='fake'), wd, HOST_ARGS, fastchat_root=FC, confirm=True, **kw)
    assert host_never_runs == [] and not (wd / 'judge_adapter_trace').exists()


def test_cli_refuses_missing_reference_with_exit_2(tmp_path, host_never_runs, capsys):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    bad = write_refs(tmp_path / 'r.jsonl', [ref_row(111, [REF_T1])])
    cfg = CONFIG_FAKE
    assert RF.main(['judge', '--config', str(cfg), '--workdir', str(wd), '--fastchat-root', FC, '--confirm', '--reference-file', str(bad),
                    '--', *HOST_ARGS]) == 2
    assert 'refused: reference mapping refused before any call' in capsys.readouterr().err
    assert host_never_runs == [] and not (wd / 'judge_adapter_trace').exists()


def test_host_revision_mismatch_refused_before_host(tmp_path, host_never_runs, monkeypatch):
    RF._host_path(FC)
    import fastchat.llm_judge.common as common
    monkeypatch.setattr(common, 'NEED_REF_CATS', ['math', 'reasoning'])
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    with pytest.raises(B.ReferenceMappingError, match='differs from the supported revision'):
        RF.judge(AdapterConfig(transport='fake'), wd, HOST_ARGS, fastchat_root=FC, confirm=True,
                 reference_file=user_ref_file(tmp_path))
    assert host_never_runs == [] and common.run_judge_pair.__name__ == 'run_judge_pair' \
        and not getattr(common.run_judge_pair, 'judge_adapter', False)


def test_first_n_without_reference_questions_needs_no_row(tmp_path):
    other = write_refs(tmp_path / 'r.jsonl', [ref_row(999, ['x'])])
    _, rows, trace, cap, _ = run(tmp_path, reference_file=other, host_args=HOST_ARGS + ['--first-n', '2'])
    assert {r['question_id'] for r in rows} == {81, 82} and all(cap.reference_part(b) is None for b in cap.bodies)


# --- (5) explicit mapping instead of the judge-name lookup; trace record; staging cleanup ----------------------------
def test_reference_model_maps_the_named_file_not_the_judge_name(tmp_path):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    judge_named = wd / 'data' / RF.DEMO_BENCH / 'reference_answer' / f'{JUDGE}.jsonl'
    judge_named.write_text(json.dumps(ref_row(111, ['JUDGE-NAMED-DECOY-1', 'JUDGE-NAMED-DECOY-2'])) + '\n')
    _, _, trace, cap, adapter = run(tmp_path, reference_model='gpt-4')
    assert all(b'JUDGE-NAMED-DECOY' not in b for b in cap.bodies)
    refs = [cap.reference_part(b) for b in cap.bodies if cap.reference_part(b) is not None]
    assert len(refs) == 4 and all(gpt4_ref_turns()[0] in r for r in refs)     # JSON-decoded wire text
    rec = [json.loads(l) for l in (adapter.trace.dir / 'fastchat_reference_mapping.jsonl').read_text().splitlines()]
    assert len(rec) == 1 and rec[0]['source_kind'] == 'reference_model' and rec[0]['reference_model'] == 'gpt-4'
    gpt4_file = wd / 'data' / RF.DEMO_BENCH / 'reference_answer' / 'gpt-4.jsonl'
    assert rec[0]['source_sha256'] == hashlib.sha256(gpt4_file.read_bytes()).hexdigest()
    assert rec[0]['host_lookup_key'] == JUDGE and rec[0]['staged_as'].endswith(f'reference_answer/{JUDGE}.jsonl')
    assert rec[0]['ignored_workdir_reference_files'] == sorted([f'{JUDGE}.jsonl', 'gpt-4.jsonl'])
    assert rec[0]['inferred_from_judge_name'] is False and rec[0]['auto_fill'] is False
    assert rec[0]['supported_host_revision'] == B.SUPPORTED_HOST_REVISION
    assert rec[0]['host_template_replacement'] == {'pair-v2': 'default-fixed/v1',
                                                   'pair-v2-multi-turn': 'default-fixed/v1',
                                                   'pair-math-v1': 'user-criteria/v1',
                                                   'pair-math-v1-multi-turn': 'user-criteria/v1'}
    assert rec[0]['mapped_question_ids'] == [111] and rec[0]['turns_per_question'] == {'111': 2}
    t = [x for x in trace if x['host']['question_id'] == 111][0]
    assert t['host']['reference']['source_sha256'] == rec[0]['source_sha256']


def test_staging_leaves_user_workdir_and_reference_file_intact(tmp_path, monkeypatch):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    snap = lambda: {str(p.relative_to(wd)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in wd.rglob('*') if p.is_file() and 'model_judgment' not in p.parts
                    and 'judge_adapter_trace' not in p.parts}
    before = snap()
    refs = user_ref_file(tmp_path)
    ref_sha = hashlib.sha256(refs.read_bytes()).hexdigest()
    stages = []
    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(RF.tempfile, 'mkdtemp', lambda **k: stages.append(real_mkdtemp(**k)) or stages[-1])
    run(tmp_path, reference_file=refs)
    assert snap() == before and hashlib.sha256(refs.read_bytes()).hexdigest() == ref_sha
    assert len(stages) == 1 and not Path(stages[0]).exists()           # staged copy removed after the run


@pytest.mark.parametrize('mode_args', [['--mode', 'pairwise-all', '--model-list', 'model-x', 'model-y']])
def test_pairwise_all_with_reference(tmp_path, mode_args):
    args = ['--bench-name', RF.DEMO_BENCH, '--judge-model', JUDGE, *mode_args]
    _, rows, trace, cap, _ = run(tmp_path, host_args=args, reference_file=user_ref_file(tmp_path))
    assert len(rows) == 6 and sum(cap.reference_part(b) is not None for b in cap.bodies) == 4


# --- (6) end to end: host CLI entry -> adapter -> host result file -> host show_result -------------------------------
def test_cli_end_to_end_math_question_with_user_reference_file(tmp_path):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=ANSWERS)
    refs = user_ref_file(tmp_path)
    cfg = CONFIG_FAKE
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        assert RF.main(['judge', '--config', str(cfg), '--workdir', str(wd), '--fastchat-root', FC, '--confirm', '--reference-file',
                        str(refs), '--', *HOST_ARGS]) == 0
    summary = json.loads(out.getvalue().strip().splitlines()[-1])
    assert summary['fake_http_requests'] == 12 and summary['transport'] == 'fake'
    mapping = Path(summary['reference_mapping'])
    rec = json.loads(mapping.read_text().splitlines()[0])
    assert rec['source_path'] == str(refs.resolve()) and rec['source_kind'] == 'reference_file'
    rows = [json.loads(l) for l in (wd / 'data' / RF.DEMO_BENCH / 'model_judgment' / f'{JUDGE}_pair.jsonl')
            .read_text().splitlines()]
    r111 = {r['turn']: r for r in rows if r['question_id'] == 111}
    assert r111[1]['judge'] == [JUDGE, 'pair-math-v1'] and r111[2]['judge'] == [JUDGE, 'pair-math-v1-multi-turn']
    assert (r111[1]['g1_winner'], r111[1]['g2_winner']) == ('model_1', 'model_1')
    summ = json.loads(r111[1]['g1_judgment'])['judge_adapter']
    assert summ['status'] == 'ok' and summ['submissions'] == 1 and summ['transport'] == 'fake'
    assert REF_T2 in r111[2]['g1_user_prompt']
    show = io.StringIO()
    with contextlib.redirect_stdout(show):
        assert RF.main(['show', '--workdir', str(wd), '--fastchat-root', FC, '--', '--mode', 'pairwise-baseline', '--bench-name',
                        RF.DEMO_BENCH, '--judge-model', JUDGE, '--baseline-model', 'model-y']) == 0
    line = [l for l in show.getvalue().splitlines() if l.startswith('model-x')][0].split()
    assert line[1:4] == ['2', '0', '4']   # wins 81t1, 111t1; ties 82t1, 81t2, 82t2, 111t2 (fake markers only)
    for f in wd.rglob('*'):
        if f.is_file():
            assert fakes.DUMMY_KEY not in f.read_text(errors='replace'), f
