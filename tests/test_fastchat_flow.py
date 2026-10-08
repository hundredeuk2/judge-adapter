"""OPTIONAL host flow (extra "fastchat"), model-free: FastChat gen_judgment.py __main__ -> patched run_judge_pair ->
adapter -> fake transport -> host result file (and show_result.py). No network, no key, no model.

Skipped unless FastChat is importable: set JUDGE_ADAPTER_FASTCHAT_ROOT to a FastChat checkout at commit 587d5cfa, or
install the "fastchat" extra (demo_workdir then needs the llm_judge data files in that installation)."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json

import pytest

from judge_adapter import fakes, fastchat_runner as RF
from judge_adapter import AdapterConfig

from conftest import block_key_reads, fastchat_root

FC = fastchat_root()
if FC is False:
    pytest.skip('FastChat not available (optional extra)', allow_module_level=True)
RF._host_path(FC)
pytest.importorskip('fastchat.llm_judge.common')

HOST_ARGS = ['--mode', 'pairwise-baseline', '--bench-name', RF.DEMO_BENCH, '--judge-model', 'judge-adapter',
             '--baseline-model', 'model-y', '--model-list', 'model-x']


@pytest.fixture(autouse=True)
def no_key_read(monkeypatch):
    """Any key read during a fake run fails the test."""
    block_key_reads(monkeypatch)


def run(tmp_path, answers=None, **cfg):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC, answers=answers)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        adapter, fake = RF.judge(AdapterConfig(transport='fake', **cfg), wd, HOST_ARGS, fastchat_root=FC, confirm=True)
    host_rows = [json.loads(l) for l in (wd / 'data' / RF.DEMO_BENCH / 'model_judgment' / 'judge-adapter_pair.jsonl')
                 .read_text().splitlines()]
    trace = [json.loads(l) for l in adapter.trace.trace.read_text().splitlines()]
    raw = {r['call_id']: r for r in map(json.loads, adapter.trace.raw.read_text().splitlines())}
    return wd, host_rows, trace, raw, fake, out.getvalue() + err.getvalue()


def winners(host_rows):
    return {(r['question_id'], r['turn']): (r['g1_winner'], r['g2_winner']) for r in host_rows}


EXPECTED = {(81, 1): ('model_1', 'model_1'),    # content-based win; restored across the swap
            (82, 1): ('tie', 'tie'),            # explicit tie
            (111, 1): ('error', 'error'),       # reference-based judge -> explicit unsupported_input
            (81, 2): ('model_1', 'model_2'),    # position bias -> host sees an inconsistent pair
            (82, 2): ('error', 'error'),        # format error
            (111, 2): ('error', 'error')}


def test_default_backend_is_haiku_direct_and_host_file_written(tmp_path):
    _, host_rows, trace, _, fake, _ = run(tmp_path)
    assert isinstance(fake, fakes.FakeAnthropic)
    assert winners(host_rows) == EXPECTED
    assert {t['backend'] for t in trace} == {'haiku_direct'} and {t['model'] for t in trace} == {'claude-haiku-4-5-20251001'}
    assert len(trace) == 12                                  # 6 matches x 2 host directions
    assert fake.count('POST', '/v1/messages') == 8           # 4 reference-based calls never reach the transport
    assert all(json.loads(r['g1_judgment'])['judge_adapter']['backend'] == 'haiku_direct' for r in host_rows)


def test_explicit_switch_to_choice_endpoint(tmp_path):
    _, host_rows, trace, _, fake, _ = run(tmp_path, backend='choice_endpoint')
    assert isinstance(fake, fakes.FakeChoiceEndpoint)
    assert winners(host_rows) == EXPECTED
    assert {t['backend'] for t in trace} == {'choice_endpoint'} and {t['model'] for t in trace} == {'unspecified'}
    assert fake.count('POST', '/runsync') == 8 and fake.count(contains='/v1/messages') == 0
    ok = [t for t in trace if t['calls'] and t['calls'][0]['outcome'] == 'ok']
    assert all(t['calls'][0]['server_timing']['execution_ms'] == 850 for t in ok)
    assert all('executionTime' in t['cost_estimate']['basis'] for t in trace)


def test_id_restoration_both_host_directions(tmp_path):
    _, _, trace, _, _, _ = run(tmp_path)
    q81 = [t for t in trace if t['host']['question_id'] == 81 and t['request_id'].split(':')[2] == 't1']
    g1, g2 = q81
    assert g1['calls'][0]['slot_ids'] == {'1': 'ans-model-x-q81', '2': 'ans-model-y-q81'}
    assert g2['calls'][0]['slot_ids'] == {'1': 'ans-model-y-q81', '2': 'ans-model-x-q81'}
    assert g1['winner_id'] == g2['winner_id'] == 'ans-model-x-q81'
    assert (g1['verdict'], g2['verdict']) == ('A', 'B')


def test_tie_inconsistent_error_are_separate(tmp_path):
    _, host_rows, trace, _, _, _ = run(tmp_path)
    by = {(t['host']['question_id'], t['request_id'].split(':')[2], t['host']['model_a']): t for t in trace}
    tie = by[(82, 't1', 'model-x')]
    assert (tie['status'], tie['tie_kind'], tie['failure']) == ('tie', 'explicit', None)
    pos = [by[(81, 't2', 'model-x')], by[(81, 't2', 'model-y')]]
    assert [p['status'] for p in pos] == ['ok', 'ok'] and pos[0]['winner_id'] != pos[1]['winner_id']
    fmt = by[(82, 't2', 'model-x')]
    assert fmt['status'] == 'error' and fmt['failure']['reason'] == 'format_error' and fmt['verdict'] is None
    ref = by[(111, 't1', 'model-x')]
    assert ref['failure']['reason'] == 'unsupported_input' and ref['submissions'] == 0 and ref['calls'] == []
    # the host's own loader keeps them apart too: tie / inconsistent / error
    import fastchat.llm_judge.common as common  # noqa: E402 (host on sys.path after RF.judge)
    p = tmp_path / 'w' / 'data' / RF.DEMO_BENCH / 'model_judgment' / 'judge-adapter_pair.jsonl'
    loaded = common.load_pairwise_model_judgments(str(p))     # keyed by (judge label, host prompt name)
    w = {(j[1], k[0]): v['winners'][0] for j, games in loaded.items() for k, v in games.items()}
    assert w == {('pair-v2', 81): 'model_1', ('pair-v2', 82): 'tie', ('pair-math-v1', 111): 'error',
                 ('pair-v2-multi-turn', 81): 'inconsistent', ('pair-v2-multi-turn', 82): 'error',
                 ('pair-math-v1-multi-turn', 111): 'error'}


def test_retry_zero_on_http_timeout_and_format(tmp_path):
    answers = {'model-x': {81: ['FAKE:HTTP500 x', 'FAKE:TIMEOUT x'], 82: ['FAKE:FORMAT x', 'FAKE:TRUNC x']},
               'model-y': {81: ['y', 'y'], 82: ['y', 'FAKE:REFUSE y']}}
    _, host_rows, trace, _, fake, _ = run(tmp_path, answers=answers)
    assert all(r['g1_winner'] == r['g2_winner'] == 'error' for r in host_rows)
    assert fake.count() == len(trace) == 8                    # exactly one request per judge call
    assert all(t['submissions'] == 1 and t['retries'] == 0 for t in trace)
    reasons = {t['failure']['reason'] for t in trace}
    assert reasons == {'http', 'timeout', 'format_error', 'truncated'}   # 82 t2: TRUNC wins the marker order


def test_usage_time_and_raw_trace(tmp_path):
    _, _, trace, raw, _, _ = run(tmp_path)
    for t in trace:
        assert t['total_seconds'] >= 0
        for c in t['calls']:
            assert c['client_seconds'] is not None and c['client_seconds'] <= t['total_seconds']
            text = raw[c['call_id']]['exchanges'][-1]['text']
            assert c['raw_output_sha256'] == hashlib.sha256(text.encode()).hexdigest()
            assert c['raw_output_pointer'] == {'file': 'judge_raw.jsonl', 'call_id': c['call_id']}
        if t['calls']:
            assert t['usage']['input_tokens'] > 0
            assert t['cost_estimate']['mock'] is True and t['cost_estimate']['is_estimate'] is True


def test_failure_surfaces_through_host_show_result(tmp_path):
    wd, _, _, _, _, _ = run(tmp_path)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        RF.run_host_module('fastchat.llm_judge.show_result', wd,
                           ['--mode', 'pairwise-baseline', '--bench-name', RF.DEMO_BENCH, '--judge-model',
                            'judge-adapter', '--baseline-model', 'model-y'])
    line = [l for l in out.getvalue().splitlines() if l.startswith('model-x')][0].split()
    assert line[1:4] == ['1', '0', '2']     # win 1, loss 0, tie 2 (explicit + inconsistent); 3 error rows dropped


def test_no_key_in_any_output(tmp_path):
    answers = {'model-x': {81: ['FAKE:LEAK [[GOOD]] x', 'FAKE:LEAK x']}, 'model-y': {81: ['y', 'y']}}
    for backend in ('haiku_direct', 'choice_endpoint'):
        wd, _, _, _, _, printed = run(tmp_path / backend, answers=answers, backend=backend)
        assert fakes.DUMMY_KEY not in printed
        for f in wd.rglob('*'):
            if f.is_file():
                assert fakes.DUMMY_KEY not in f.read_text(errors='replace'), f
        assert '[redacted]' in (wd / 'judge_adapter_trace' / 'judge_raw.jsonl').read_text()


def test_cli_refuses_real_without_flag(tmp_path, capsys):
    cfg = tmp_path / 'real.json'
    cfg.write_text(json.dumps({'transport': 'real'}))
    assert RF.main(['judge', '--config', str(cfg), '--workdir', str(tmp_path)]) == 2


def test_host_parallel_threads_keep_prompt_and_trace_per_call(tmp_path):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        adapter, fake = RF.judge(AdapterConfig(transport='fake'), wd, HOST_ARGS + ['--parallel', '4'], fastchat_root=FC,
                                 confirm=True)
    rows = [json.loads(l) for l in (wd / 'data' / RF.DEMO_BENCH / 'model_judgment' / 'judge-adapter_pair.jsonl')
            .read_text().splitlines()]
    assert winners(rows) == EXPECTED and len(adapter.trace.trace.read_text().splitlines()) == 12
    for r in rows:
        if r['g1_winner'] != 'error':
            x = RF.DEMO_ANSWERS['model-x'][r['question_id']][r['turn'] - 1]
            assert x in r['g1_user_prompt'] and x in r['g2_user_prompt']


def test_relative_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    RF.demo_workdir('w', FC)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        adapter, _ = RF.judge(AdapterConfig(transport='fake'), 'w', HOST_ARGS, fastchat_root=FC, confirm=True)
    assert adapter.trace.trace == (tmp_path / 'w' / 'judge_adapter_trace' / 'judge_trace.jsonl').resolve()
    assert len(adapter.trace.trace.read_text().splitlines()) == 12
