"""Connection contract, model-free:
(1) the FastChat wrapper refuses host modes other than pairwise-baseline / pairwise-all BEFORE the host runs;
(2) a failure before any HTTP transmission (choice endpoint or rate unset, key provider, missing key, client init)
    is local_config with submissions 0, stops the run, and leaves no job reservation; transmitted failures keep theirs.
Network and key reads fail the test unless a test injects its own failing key provider.
The FastChat host tests in (1) are skipped when FastChat is not available (optional extra)."""
from __future__ import annotations

import contextlib
import io
import json
import socket
import sys

import httpx
import pytest

from judge_adapter import cli as J, fakes, fastchat_runner as RF
from judge_adapter import AdapterConfig, JudgeAdapter, JudgeRequest, Response, build_backend
from judge_adapter import backends as B
from judge_adapter import keys as K
from judge_adapter import core
from judge_adapter import prompt as P

from conftest import block_key_reads, fastchat_root

FC = fastchat_root()
needs_fastchat = pytest.mark.skipif(FC is False, reason='FastChat not available (optional extra)')

PAIR_ARGS = ['--bench-name', RF.DEMO_BENCH, '--judge-model', 'judge-adapter', '--baseline-model', 'model-y',
             '--model-list', 'model-x']


@pytest.fixture(autouse=True)
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


@pytest.fixture
def no_key(monkeypatch):
    block_key_reads(monkeypatch)


# --- (1) FastChat host mode guard ----------------------------------------------------------------------------------
@pytest.mark.parametrize('host_args', [
    ['--mode', 'single', *PAIR_ARGS],
    ['--mode', 'single', '--judge-model', 'gpt-4', '--model-list', 'model-x'],   # host's own judge client
    PAIR_ARGS,                                                                   # no --mode: host default single
    ['--mode=single', *PAIR_ARGS],
    ['--mode', 'pairwise-baseline', '--mode', 'single', *PAIR_ARGS],             # last one wins in argparse
    ['--mode', 'pairwise-baseline', '--unknown-flag', *PAIR_ARGS],
    ['--mod', 'single', *PAIR_ARGS],                                             # ambiguous prefix
])
def test_unsupported_host_mode_never_runs_the_host(tmp_path, monkeypatch, no_key, host_args):
    started = []
    monkeypatch.setattr(RF.runpy, 'run_module', lambda *a, **k: started.append(a))
    wd = tmp_path / 'w'
    wd.mkdir()
    cfg = AdapterConfig(transport='fake', trace_dir='judge_adapter_trace')
    with pytest.raises(RF.HostArgsRefused):
        RF.judge(cfg, wd, host_args, confirm=True)
    with pytest.raises(RF.HostArgsRefused):
        RF.run_host_module('fastchat.llm_judge.gen_judgment', wd, host_args, confirm=True)
    assert started == [] and not (wd / 'judge_adapter_trace').exists()
    cfg_file = tmp_path / 'c.json'
    cfg_file.write_text(json.dumps({'transport': 'fake'}))
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert RF.main(['judge', '--config', str(cfg_file), '--workdir', str(wd), '--confirm', '--', *host_args]) == 2
    assert 'refused' in err.getvalue() and started == []


def test_supported_modes_pass_the_check():
    assert RF.check_host_args(['--mode', 'pairwise-baseline', *PAIR_ARGS]) == 'pairwise-baseline'
    assert RF.check_host_args(['--mode=pairwise-all', '--bench-name', 'b', '--model-list', 'a', 'b']) == 'pairwise-all'


@needs_fastchat
def test_pairwise_all_runs_through_the_adapter(tmp_path, no_key):
    wd = tmp_path / 'w'
    RF.demo_workdir(wd, FC)
    args = ['--mode', 'pairwise-all', '--bench-name', RF.DEMO_BENCH, '--judge-model', 'judge-adapter',
            '--model-list', 'model-x', 'model-y']
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        adapter, fake = RF.judge(AdapterConfig(transport='fake'), wd, args, fastchat_root=FC, confirm=True)
    rows = (wd / 'data' / RF.DEMO_BENCH / 'model_judgment' / 'judge-adapter_pair.jsonl').read_text().splitlines()
    assert len(rows) == 6 and fake.count('POST', '/v1/messages') == 8


@needs_fastchat
def test_host_single_grading_is_refused_while_installed(tmp_path):
    RF._host_path(FC)
    from judge_adapter.hosts import fastchat as binding
    import fastchat.llm_judge.common as common
    original = common.run_judge_single
    ad = JudgeAdapter(AdapterConfig(transport='fake', trace_dir=str(tmp_path)), object())
    restore = binding.install(ad)
    try:
        with pytest.raises(RuntimeError, match='refused'):
            common.run_judge_single({}, {}, None)
    finally:
        restore()
    assert common.run_judge_single is original


# --- (2) failures before transmission ------------------------------------------------------------------------------
def req(rid, a='a', b='b'):
    return JudgeRequest(rid, ('Q?',), Response('id-a', (a,)), Response('id-b', (b,)))


class CountingMissingKey:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise K.CredentialMissing('ANTHROPIC_API_KEY line absent or empty in the env file')


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_missing_key_is_local_config_with_zero_submissions_and_no_second_key_read(tmp_path, backend):
    fake, kp = fakes.fake_for(backend), CountingMissingKey()
    cfg = AdapterConfig(transport='fake', backend=backend, mode='two_direction', trace_dir=str(tmp_path / 't'))
    ad = JudgeAdapter(cfg, build_backend(cfg, transport=fake.transport(), key_provider=kp, sleep=lambda s: None))
    r1, r2 = ad.judge(req('r1')), ad.judge(req('r2'))
    for r in (r1, r2):
        assert r.status == 'error' and r.failure['reason'] == 'local_config' and r.submissions == 0
        assert len(r.calls) == 1 and r.cost_estimate['usd'] == 0.0   # second direction not attempted
    assert 'absent or empty' in r1.failure['detail'] and kp.calls == 1 and fake.count() == 0


def test_client_init_error_is_local_config(tmp_path, monkeypatch):
    def broken_client(*a, **k):
        raise RuntimeError('client init failed')
    monkeypatch.setattr(B.httpx, 'Client', broken_client)
    cfg = AdapterConfig(transport='fake', trace_dir=str(tmp_path))
    fake = fakes.FakeAnthropic()
    ad = JudgeAdapter(cfg, build_backend(cfg, transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY))
    r = ad.judge(req('r1'))
    assert (r.failure['reason'], r.submissions) == ('local_config', 0) and 'client init' in r.failure['detail']
    assert fakes.DUMMY_KEY not in json.dumps(r.to_dict())


def test_transmitted_timeout_is_unchanged(tmp_path):
    cfg = AdapterConfig(transport='fake', trace_dir=str(tmp_path))
    ad = JudgeAdapter(cfg, build_backend(cfg, transport=fakes.FakeAnthropic().transport(),
                                         key_provider=lambda: fakes.DUMMY_KEY))
    r = ad.judge(req('r1', a='FAKE:TIMEOUT'))
    assert (r.failure['reason'], r.submissions, r.cost_estimate['usd']) == ('timeout', 1, None)


def _gate_cli(tmp_path, backend, kp, objs, mode='single'):
    p = tmp_path / 'in.jsonl'
    p.write_text(''.join(json.dumps(o) + '\n' for o in objs))
    fake = fakes.fake_for(backend)
    s = J.run_file(p, tmp_path / 'out', transport='fake', backend=backend, mode=mode, fake=fake, key_provider=kp,
                   budget_cap_usd=5.0)
    rows = [json.loads(x) for x in (tmp_path / 'out' / 'results.jsonl').read_text().splitlines()]
    gate = json.loads((tmp_path / 'out' / 'gate.json').read_text())
    return s, rows, gate, fake


def line(rid, a='a'):
    return {'id': rid, 'question_turns': ['Q?'], 'response_a': {'id': 'x', 'turns': [a]},
            'response_b': {'id': 'y', 'turns': ['b']}}


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_rungate_local_config_releases_reservation_and_stops(tmp_path, backend):
    kp = CountingMissingKey()
    s, rows, gate, fake = _gate_cli(tmp_path, backend, kp, [line('r1'), line('r2')], mode='two_direction')
    assert [r['outcome'] for r in rows] == ['error', 'unrun'] and rows[0]['failure']['reason'] == 'local_config'
    assert s['submissions']['actual'] == 0 and gate['submissions'] == 0 and fake.count() == 0 and kp.calls == 1
    assert gate['committed_usd'] == 0.0 and 'local configuration' in gate['stopped']
    assert gate['submitted_call_keys'] == []
    if backend == 'choice_endpoint':
        assert gate['choice_endpoint']['unsettled_jobs'] == [] and gate['choice_endpoint']['settled_seconds'] == 0.0
    else:
        assert gate['haiku']['held_usd'] == 0.0


def test_rungate_transmitted_choice_failure_keeps_its_reservation(tmp_path):
    s, rows, gate, _ = _gate_cli(tmp_path, 'choice_endpoint', lambda: fakes.DUMMY_KEY, [line('r1', a='FAKE:TIMEOUT'), line('r2')])
    assert rows[0]['failure']['reason'] == 'timeout' and rows[0]['submissions'] == 1
    assert len(gate['choice_endpoint']['unsettled_jobs']) == 1 and gate['committed_usd'] >= 661 * core.FAKE_CHOICE_USD_PER_SECOND and rows[1]['outcome'] == 'unrun'


def test_cli_real_run_with_missing_env_file_sends_nothing(tmp_path):
    """The real path with a nonexistent explicit env file: no key exists to read, sockets are blocked."""
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(line('r1')) + '\n' + json.dumps(line('r2')) + '\n')
    s = J.run_file(p, tmp_path / 'out', transport='real', allow_real=True, max_submissions=2, budget_cap_usd=0.01,
                   env_file=tmp_path / 'does-not-exist.env')
    assert s['counts']['error'] == 1 and s['counts']['unrun'] == 1 and s['error_reasons'] == {'local_config': 1}
    assert s['submissions']['actual'] == 0 and s['cost_estimate']['gate_committed_usd'] == 0.0
    assert 'not readable' in s['limits']['stopped']


def test_cli_real_run_without_key_env_var_sends_nothing(tmp_path):
    """No --env-file and no ANTHROPIC_API_KEY (removed by conftest): local_config, 0 submissions, run stops."""
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(line('r1')) + '\n' + json.dumps(line('r2')) + '\n')
    s = J.run_file(p, tmp_path / 'out', transport='real', allow_real=True, max_submissions=2, budget_cap_usd=0.01)
    assert s['error_reasons'] == {'local_config': 1} and s['counts']['unrun'] == 1
    assert s['submissions']['actual'] == 0 and s['cost_estimate']['gate_committed_usd'] == 0.0
    assert 'ANTHROPIC_API_KEY is not set' in s['limits']['stopped']


# --- (3) choice endpoint: explicit config / environment only --------------------------------------------------------
def test_no_endpoint_is_built_in():
    assert not [n for n in dir(P) if n.startswith('CHOICE_') and ('MODEL' in n or 'USD' in n or n.endswith('_ID'))]
    with pytest.raises(TypeError):
        P.choice_urls()                                         # no default argument
    with pytest.raises(ValueError):
        P.choice_urls('bad/../id')


def test_real_choice_without_endpoint_is_local_config_and_reads_no_key(tmp_path, no_key):
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', mode='two_direction', trace_dir=str(tmp_path / 't'),
                        choice_usd_per_second=0.01)
    be = build_backend(cfg)
    assert be.urls is None
    ad = JudgeAdapter(cfg, be)
    r1, r2 = ad.judge(req('r1')), ad.judge(req('r2'))
    for r in (r1, r2):
        assert r.failure['reason'] == 'local_config' and r.submissions == 0 and len(r.calls) == 1
        assert r.cost_estimate['usd'] == 0.0
    assert P.CHOICE_ENDPOINT_ENV in r1.failure['detail'] and be._http is None


def test_cli_real_choice_without_endpoint_sends_nothing(tmp_path, no_key):
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(line('r1')) + '\n' + json.dumps(line('r2')) + '\n')
    s = J.run_file(p, tmp_path / 'out', transport='real', backend='choice_endpoint', allow_real=True, max_submissions=2,
                   budget_cap_usd=5.0, choice_usd_per_second=0.001)
    gate = json.loads((tmp_path / 'out' / 'gate.json').read_text())
    assert s['error_reasons'] == {'local_config': 1} and s['counts']['unrun'] == 1
    assert s['submissions']['actual'] == 0 and gate['committed_usd'] == 0.0 and gate['choice_endpoint']['unsettled_jobs'] == []
    assert 'no choice endpoint ID configured' in s['limits']['stopped']


def test_endpoint_from_config_or_environment(monkeypatch):
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', choice_endpoint_id='cfg-endpoint')
    assert build_backend(cfg).urls['runsync'].startswith('https://api.runpod.ai/v2/cfg-endpoint/runsync')
    monkeypatch.setenv(P.CHOICE_ENDPOINT_ENV, 'env-endpoint')
    assert '/v2/env-endpoint/' in build_backend(AdapterConfig(transport='real', backend='choice_endpoint')).urls['status']
    assert '/v2/cfg-endpoint/' in build_backend(cfg).urls['status']           # explicit config wins
    fake = build_backend(AdapterConfig(transport='fake', backend='choice_endpoint'), transport=fakes.FakeChoiceEndpoint().transport())
    assert '/v2/fake-endpoint/' in fake.urls['runsync']                       # fake ignores the environment
    with pytest.raises(ValueError):
        build_backend(AdapterConfig(transport='real', backend='choice_endpoint', choice_endpoint_id='x/y'))


def test_key_comes_from_env_var_or_explicit_file_only(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(K, 'env_key', lambda name: seen.append(('env', name)) or 'k')
    monkeypatch.setattr(K, 'dotenv_key', lambda path, name: seen.append(('file', name)) or 'k')
    K.default_provider(None, 'ANTHROPIC_API_KEY')()
    K.default_provider(str(tmp_path / 'keys.env'), 'RUNPOD_API_KEY')()
    assert seen == [('env', 'ANTHROPIC_API_KEY'), ('file', 'RUNPOD_API_KEY')]


def test_real_choice_without_rate_is_refused_before_anything(tmp_path, no_key):
    p = tmp_path / 'in.jsonl'
    p.write_text(json.dumps(line('r1')) + '\n')
    with pytest.raises(J.RunRefused, match='--choice-usd-per-second'):
        J.run_file(p, tmp_path / 'out', transport='real', backend='choice_endpoint', allow_real=True,
                   max_submissions=1, budget_cap_usd=5.0, choice_endpoint_id='e1')
    assert not (tmp_path / 'out').exists()
    assert J.main(['run', str(p), '--out', str(tmp_path / 'out'), '--transport', 'real', '--backend', 'choice_endpoint',
                   '--allow-real', '--max-submissions', '1', '--budget-cap-usd', '5', '--choice-endpoint-id', 'e1']) == 3


def test_api_real_choice_without_rate_is_local_config_and_reads_no_key(tmp_path, no_key):
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', choice_endpoint_id='e1', trace_dir=str(tmp_path))
    be = build_backend(cfg)
    r = JudgeAdapter(cfg, be).judge(req('r1'))
    assert (r.failure['reason'], r.submissions, r.cost_estimate['usd']) == ('local_config', 0, 0.0)
    assert 'no per-second rate' in r.failure['detail'] and be._http is None
