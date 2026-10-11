"""davi on/off toggles (toggles.json), the control CLI (hooks/davi_toggle.py, bin/davi-toggle), the slash-command files,
and that every hook honours its own toggle. Model-free: fake endpoints only; network and key reads fail every test.

Covers: CLI argument validation (nothing written on a usage error), 0600 + atomic writes, concurrent changes from
separate processes all preserved (flock around the read-modify-write), corrupt / unreadable files (hooks fall back to
userConfig, status warns, the CLI refuses to modify and reports failure), write failures reported as failures,
precedence (global enabled=false > toggle > userConfig), the three toggles independent, status without secrets and
with non-runnable states, each launcher's fast path, and the UserPromptSubmit launcher (no stdout, no state when off)."""
from __future__ import annotations

import io
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from judge_adapter import fakes
from judge_adapter import plugin_hook as H
from judge_adapter import prompt_signal as PS
from judge_adapter import query_hook as QH
from judge_adapter import toggles as T

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'hooks' / 'davi_toggle.py'
KEY = 'rpa_TOGGLETESTKEY0123456789'
ENDPOINT = 'endpointsecret99'
SESSION = 'session-T'
QUERY = 'How do zebra lanterns glow in the synthetic festival'
NEUTRAL = 'Tell me how zebra lanterns glow at the synthetic festival, in detail please.'


@pytest.fixture(autouse=True)
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


def cfg_env(data, **opts):
    e = {'CLAUDE_PLUGIN_DATA': str(data), 'CLAUDE_PLUGIN_OPTION_RUNPOD_API_KEY': KEY,
         'CLAUDE_PLUGIN_OPTION_CHOICE_ENDPOINT_ID': ENDPOINT, 'CLAUDE_PLUGIN_OPTION_USD_PER_SECOND': '0.0004'}
    e.update({H.OPTION_ENV + k.upper(): v for k, v in opts.items() if v is not None})
    for k, v in opts.items():
        if v is None:
            e.pop(H.OPTION_ENV + k.upper(), None)
    return e


def cli(*argv, env=None, settings=None):
    out, err = io.StringIO(), io.StringIO()
    mod = _cli_module()
    args = list(argv) if settings is None else ['--settings', str(settings), *argv]
    rc = mod.main(args, env=env if env is not None else {}, out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


_CLI_MOD = None


def _cli_module():
    global _CLI_MOD
    if _CLI_MOD is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_davi_toggle_cli', CLI)
        _CLI_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CLI_MOD)
    return _CLI_MOD


# --- CLI validation and writes --------------------------------------------------------------------------------------
@pytest.mark.parametrize('argv', [
    [], ['gate'], ['gate', 'ON'], ['gate', 'yes'], ['gate', 'on', 'off'], ['gate', 'on;rm'], ['gates', 'on'],
    ['domain-mode', 'allow'], ['domain-mode'], ['status', 'x'], ['reset', 'now'], ['--data-dir'], ['--bogus', 'status'],
    ['filter', ''], ['enabled', 'off'], ['key', 'on']])
def test_cli_rejects_bad_arguments_and_writes_nothing(tmp_path, argv):
    rc, out, err = cli('--data-dir', str(tmp_path / 'd'), *argv) if argv[:1] != ['--data-dir'] else cli(*argv)
    assert rc == 2 and out == '' and 'usage:' in err
    assert not (tmp_path / 'd').exists()


def test_cli_set_and_reset_file_mode_and_content(tmp_path):
    d = tmp_path / 'd'
    for argv in (('gate', 'on'), ('filter', 'off'), ('domain', 'on'), ('domain-mode', 'input_only'), ('gate', 'off')):
        rc, out, err = cli('--data-dir', str(d), *argv)
        assert rc == 0 and err == '' and 'install-wide' in out
    f = d / T.FILE_NAME
    data = json.loads(f.read_text())
    assert data == {'v': 1, 'filter': False, 'gate': False, 'domain': True, 'domain_mode': 'input_only',
                    'updated': data['updated']}
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    assert not [x for x in os.listdir(d) if x.endswith('.tmp')]
    rc, out, _ = cli('--data-dir', str(d), 'reset')
    assert rc == 0 and 'removed' in out and not f.exists()
    rc, out, _ = cli('--data-dir', str(d), 'reset')
    assert rc == 0 and 'no toggle file' in out


def test_cli_data_dir_resolution(tmp_path):
    assert T.data_dir({}, str(tmp_path)) == str(tmp_path)
    assert T.data_dir({'CLAUDE_PLUGIN_DATA': str(tmp_path / 'p')}, '') == str(tmp_path / 'p')     # empty -> env


def test_bin_wrapper_runs_the_cli_and_is_executable(tmp_path):
    wrapper = ROOT / 'bin' / 'davi-toggle'
    assert os.access(wrapper, os.X_OK)
    r = subprocess.run([str(wrapper), '--data-dir', str(tmp_path / 'd'), 'gate', 'on'], capture_output=True,
                       text=True, timeout=30, env={'PATH': os.environ.get('PATH', ''), 'HOME': str(tmp_path)})
    assert r.returncode == 0 and 'gate set to on' in r.stdout
    # no --data-dir and no CLAUDE_PLUGIN_DATA (as in the Bash tool): the documented default under ~/.claude
    r = subprocess.run([str(wrapper), 'filter', 'off'], capture_output=True, text=True, timeout=30,
                       env={'PATH': os.environ.get('PATH', ''), 'HOME': str(tmp_path)})
    default = tmp_path / '.claude' / 'plugins' / 'data' / 'davi-judge-adapter' / T.FILE_NAME
    assert r.returncode == 0 and str(default) in r.stdout and json.loads(default.read_text())['filter'] is False
    r = subprocess.run([sys.executable, str(CLI), '--data-dir', str(tmp_path / 'd'), 'gate', 'maybe'],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and r.stdout == '' and 'usage:' in r.stderr


@pytest.mark.parametrize('content', ['{bad json', '[]', '{"gate": "yes"}', '{"gate": true, "runpod_api_key": "x"}',
                                     '{"domain_mode": "allow"}', '{"v": 2}', ' ' * 5000])
def test_corrupt_file_is_ignored_by_hooks_refused_by_cli_and_reset_removes_it(tmp_path, content):
    d = tmp_path / 'd'
    d.mkdir()
    f = d / T.FILE_NAME
    f.write_text(content)
    assert T.read(str(f)) == ({}, 'corrupt')
    eff = T.effective_from_env(cfg_env(d, search_gate_enabled='true'))
    assert eff.gate is True and eff.sources['gate'] == 'userConfig' and eff.toggle_status == 'corrupt'
    rc, out, err = cli('--data-dir', str(d), 'gate', 'off')
    assert rc == 1 and out == '' and 'FAILED' in err and 'reset' in err
    assert f.read_text() == content                                       # unchanged
    rc, out, _ = cli('--data-dir', str(d), 'status', settings=tmp_path / 'none.json')
    assert rc == 0 and 'toggle file is corrupt' in out
    assert cli('--data-dir', str(d), 'reset')[0] == 0 and not f.exists()


def test_write_failures_are_reported_as_failures(tmp_path, monkeypatch):
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    rc, out, err = cli('--data-dir', str(blocker / 'sub'), 'gate', 'on')
    assert rc == 1 and out == '' and 'FAILED' in err
    d = tmp_path / 'd'
    assert cli('--data-dir', str(d), 'gate', 'on')[0] == 0
    before = (d / T.FILE_NAME).read_text()
    monkeypatch.setattr(T.os, 'replace', lambda a, b: (_ for _ in ()).throw(OSError('disk full')))
    rc, out, err = cli('--data-dir', str(d), 'gate', 'off')
    assert rc == 1 and out == '' and 'FAILED' in err and 'nothing was changed' in err
    monkeypatch.undo()
    assert (d / T.FILE_NAME).read_text() == before
    assert not [x for x in os.listdir(d) if x.endswith('.tmp')]


def test_no_fcntl_refuses_writes(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'fcntl':
            raise ImportError('no fcntl')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(T.ToggleError):
        T.update(str(tmp_path / T.FILE_NAME), {'gate': True})
    assert not (tmp_path / T.FILE_NAME).exists()


CHILD = r'''
import os, sys, time
sys.path.insert(0, os.environ['SRC'])
from judge_adapter import toggles as T
path, go, key, final = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
while not os.path.exists(go):
    time.sleep(0.001)
vals = {'filter': [True, False], 'gate': [True, False], 'domain': [True, False], 'domain_mode': ['ask', 'input_only']}
for i in range(25):
    T.update(path, {key: vals[key][i % 2]})
T.update(path, {key: {'true': True, 'false': False}.get(final, final)})
print('ok')
'''


@pytest.mark.skipif(sys.platform.startswith('win'), reason='fcntl lock: POSIX only')
@pytest.mark.parametrize('round_', range(2))
def test_concurrent_changes_from_separate_processes_are_all_preserved(tmp_path, round_):
    script, go, path = tmp_path / 'child.py', tmp_path / 'go', tmp_path / 'd' / T.FILE_NAME
    script.write_text(CHILD)
    env = {**os.environ, 'SRC': str(ROOT / 'src')}
    jobs = [('filter', 'false'), ('gate', 'true'), ('domain', 'true'), ('domain_mode', 'input_only')] * 2
    procs = [subprocess.Popen([sys.executable, str(script), str(path), str(go), k, v], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for k, v in jobs]
    time.sleep(0.5)
    go.write_text('go')
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0 and out.strip() == 'ok', err
    data, status = T.read(str(path))
    assert status == 'ok'
    assert (data['filter'], data['gate'], data['domain'], data['domain_mode']) == (False, True, True, 'input_only')


# --- precedence -----------------------------------------------------------------------------------------------------
def _eff(cfg, toggles):
    return T.effective(lambda k: cfg.get(k), toggles)


def test_precedence_toggle_over_userconfig_and_global_off_over_all():
    e = _eff({}, {})
    assert (e.enabled, e.filter, e.gate, e.domain, e.domain_mode) == (True, True, False, False, 'ask')
    assert e.sources == {'filter': 'default', 'gate': 'default', 'domain': 'default', 'domain_mode': 'default'}
    e = _eff({'search_gate_enabled': 'true', 'domain_apply_mode': 'input_only'}, {})
    assert e.gate is True and e.sources['gate'] == 'userConfig' and e.domain_mode == 'input_only'
    e = _eff({'search_gate_enabled': 'true'}, {'gate': False})
    assert e.gate is False and e.sources['gate'] == 'toggle'
    assert any('overrides userConfig search_gate_enabled=true' in n for n in e.notes)
    e = _eff({'search_gate_enabled': 'false'}, {'gate': True, 'domain': True, 'domain_mode': 'input_only'})
    assert e.gate is True and e.domain is True and e.domain_mode == 'input_only'
    assert any('overrides userConfig search_gate_enabled=false' in n for n in e.notes)
    e = _eff({'enabled': 'false'}, {'filter': True, 'gate': True, 'domain': True})
    assert (e.filter, e.gate, e.domain) == (False, False, False)
    assert sum('enabled=false turns every feature off' in n for n in e.notes) == 3


def test_three_toggles_are_independent():
    for feat in T.FEATURES:
        for val in (True, False):
            e = _eff({'search_gate_enabled': 'true', 'domain_presets_enabled': 'true'}, {feat: val})
            for other in T.FEATURES:
                if other != feat:
                    assert getattr(e, other) is True, (feat, val, other)
            assert getattr(e, feat) is val


# --- hooks honour their own toggle -----------------------------------------------------------------------------------
def _set(data, **kw):
    T.update(os.path.join(str(data), T.FILE_NAME), kw)


def _qh_run(e, fake_gate=None, prompt=NEUTRAL, now=time.time, tmp=None):
    from test_query_hook import FakeChoice                    # same fake endpoint as the query-hook tests
    fake = fake_gate or FakeChoice(gate={'needed': 0.02, 'not_needed': 0.97, 'cannot_tell': 0.01},
                                   domain={'docs': 0.9, 'none': 0.1})
    if prompt is not None:
        PS.PromptState(PS.default_state_dir(e) if 'CLAUDE_PLUGIN_DATA' not in e else
                       os.path.join(e['CLAUDE_PLUGIN_DATA'], PS.STATE_DIRNAME)).on_prompt(SESSION, prompt)
    p = {'session_id': SESSION, 'hook_event_name': 'PreToolUse', 'tool_name': 'WebSearch', 'tool_input': {'query': QUERY}}
    out, err = io.StringIO(), io.StringIO()
    QH.main(stdin=io.StringIO(json.dumps(p)), stdout=out, stderr=err, env=e, transport=httpx.MockTransport(fake.handle),
            exit_fn=lambda c: None, now=now)
    return (json.loads(out.getvalue()) if out.getvalue() else None), json.loads(err.getvalue().strip().splitlines()[-1]), fake


def test_gate_toggle_off_overrides_userconfig_on(tmp_path):
    e = cfg_env(tmp_path, search_gate_enabled='true')
    _set(tmp_path, gate=False)
    out, log, fake = _qh_run(e)
    assert out is None and fake.count() == 0 and log['status'] == 'options_off'


def test_gate_toggle_on_overrides_userconfig_off_after_the_next_prompt(tmp_path):
    e = cfg_env(tmp_path, search_gate_enabled='false')
    PS.PromptState(os.path.join(str(tmp_path), PS.STATE_DIRNAME)).on_prompt(SESSION, NEUTRAL)  # old state
    _set(tmp_path, gate=True)                                     # off -> on after that state
    out, log, fake = _qh_run(e, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_predates_toggle'
    time.sleep(0.01)
    out, log, fake = _qh_run(e, prompt=NEUTRAL)                   # the next prompt writes a fresh state
    assert fake.count() == 1 and out['hookSpecificOutput']['permissionDecision'] == 'deny'


def test_domain_toggle_on_and_off(tmp_path):
    e = cfg_env(tmp_path)
    _set(tmp_path, domain=True, domain_mode='input_only')
    out, log, fake = _qh_run(e)
    assert log['domain']['status'] == 'applied' and 'gate' not in log
    assert out == {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'updatedInput': {
        'query': QUERY, 'allowed_domains': ['docs.python.org', 'developer.mozilla.org', 'learn.microsoft.com',
                                            'docs.github.com']}}}
    _set(tmp_path, domain=False)
    out, log, fake = _qh_run(cfg_env(tmp_path, domain_presets_enabled='true'))
    assert out is None and fake.count() == 0 and log['status'] == 'options_off'


def test_global_enabled_false_wins_over_toggles(tmp_path):
    _set(tmp_path, filter=True, gate=True, domain=True)
    out, log, fake = _qh_run(cfg_env(tmp_path, enabled='false'))
    assert out is None and fake.count() == 0 and log['status'] == 'disabled'
    resp = {'results': [{'title': 't', 'url': 'https://example.invalid/a', 'snippet': 'x [[KEEP=0.1]]'}]}
    out, log, _ = _post_run(cfg_env(tmp_path, enabled='false'), resp)
    assert out is None and log['status'] == 'disabled'


def _post_run(e, resp):
    fake = fakes.FakeRelevanceEndpoint()
    p = {'session_id': SESSION, 'hook_event_name': 'PostToolUse', 'tool_name': 'WebSearch',
         'tool_input': {'query': QUERY}, 'tool_response': resp}
    out, err = io.StringIO(), io.StringIO()
    H.main(stdin=io.StringIO(json.dumps(p)), stdout=out, stderr=err, env=e, transport=httpx.MockTransport(fake.handle),
           exit_fn=lambda c: None)
    return (json.loads(out.getvalue()) if out.getvalue() else None), json.loads(err.getvalue().strip()), fake


def test_filter_toggle_off_makes_post_tool_use_a_no_op_and_leaves_gate_alone(tmp_path):
    resp = {'results': [{'title': 't0', 'url': 'https://example.invalid/0', 'snippet': 'a [[KEEP=0.9]]'},
                        {'title': 't1', 'url': 'https://example.invalid/1', 'snippet': 'b [[KEEP=0.1]]'}]}
    e = cfg_env(tmp_path, search_gate_enabled='true')
    out, log, fake = _post_run(e, resp)
    assert log['status'] == 'filtered' and fake.count('POST') == 2              # baseline: the filter works
    _set(tmp_path, filter=False)
    out, log, fake = _post_run(e, resp)
    assert out is None and log['status'] == 'disabled' and fake.count('POST') == 0
    out, log, fake = _qh_run(e)                                                  # gate still on
    assert fake.count() == 1 and log['gate']['status'] == 'deny'
    _set(tmp_path, filter=True)
    out, log, fake = _post_run(e, resp)
    assert log['status'] == 'filtered'


def test_corrupt_toggle_file_hooks_use_userconfig(tmp_path):
    (tmp_path / T.FILE_NAME).write_text('{oops')
    out, log, fake = _qh_run(cfg_env(tmp_path, search_gate_enabled='true'))
    assert fake.count() == 1 and log['gate']['status'] == 'deny'


# --- launchers (real subprocesses) ------------------------------------------------------------------------------------
def _launch_env(data, **opts):
    e = {k: v for k, v in os.environ.items() if not k.startswith(('CLAUDE_PLUGIN_OPTION_', 'JUDGE_ADAPTER_HOOK_'))
         and k not in ('RUNPOD_API_KEY', 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID', 'PYTHONPATH')}
    e['CLAUDE_PLUGIN_ROOT'] = str(ROOT)
    e['CLAUDE_PLUGIN_DATA'] = str(data)
    e.update({H.OPTION_ENV + k.upper(): v for k, v in opts.items()})
    return e


def _spawn(script, e, stdin=None):
    """stdin None: the pipe is left open and never written (a fast exit must not wait for it)."""
    p = subprocess.Popen([sys.executable, str(ROOT / 'hooks' / script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env=e)
    if stdin is None:
        p.wait(timeout=30)
        out, err = p.stdout.read(), p.stderr.read()
        p.stdin.close()
        p.stdout.close()
        p.stderr.close()
        return p.returncode, out.decode(), err.decode()
    out, err = p.communicate(stdin.encode(), timeout=60)
    return p.returncode, out.decode(), err.decode()


def _pre_payload():
    return json.dumps({'session_id': SESSION, 'hook_event_name': 'PreToolUse', 'tool_name': 'WebSearch',
                       'tool_input': {'query': QUERY}})


def test_query_launcher_honours_gate_and_domain_toggles(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    _set(d, gate=False)
    assert _spawn('query_options_hook.py', _launch_env(d, search_gate_enabled='true')) == (0, '', '')
    _set(d, gate=True)
    rc, out, err = _spawn('query_options_hook.py', _launch_env(d), _pre_payload())
    assert rc == 0 and out == '' and json.loads(err.strip().splitlines()[-1])['gate']['status'] == \
        'keep_no_prompt_state'
    assert _spawn('query_options_hook.py', _launch_env(d, enabled='false')) == (0, '', '')
    _set(d, gate=False, domain=True)
    rc, out, err = _spawn('query_options_hook.py', _launch_env(d), _pre_payload())
    assert rc == 0 and json.loads(err.strip().splitlines()[-1])['status'] == 'missing_key'
    assert not (d / QH.STATE_DIRNAME).exists()


def test_filter_launcher_fast_exit_when_toggled_off(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    _set(d, filter=False)
    assert _spawn('context_filter_hook.py', _launch_env(d)) == (0, '', '')       # stdin never read
    _set(d, filter=True)
    p = json.dumps({'hook_event_name': 'PostToolUse', 'tool_name': 'mcp__tavily__tavily_search',
                    'tool_input': {'query': 'q'}, 'tool_response': [{'type': 'text', 'text':
                    'Detailed Results:\n\nTitle: t\nURL: https://example.invalid\nContent: s'}]})
    rc, out, err = _spawn('context_filter_hook.py', _launch_env(d), p)
    assert rc == 0 and out == '' and json.loads(err.strip().splitlines()[-1])['status'] == 'missing_key'


def _ups_payload(prompt, session=SESSION):
    return json.dumps({'session_id': session, 'hook_event_name': 'UserPromptSubmit', 'prompt': prompt,
                       'transcript_path': '/nonexistent/t.jsonl', 'cwd': '/tmp', 'permission_mode': 'default'})


def test_user_prompt_launcher_off_reads_nothing_writes_nothing(tmp_path):
    d = tmp_path / 'data'
    for e in (_launch_env(d), _launch_env(d, search_gate_enabled='false', domain_presets_enabled='true'),
              _launch_env(d, enabled='false', search_gate_enabled='true')):
        assert _spawn('user_prompt_hook.py', e) == (0, '', '')                   # stdin left open, never read
    assert not d.exists()
    d.mkdir()
    _set(d, gate=False)
    assert _spawn('user_prompt_hook.py', _launch_env(d, search_gate_enabled='true')) == (0, '', '')
    assert not (d / PS.STATE_DIRNAME).exists()


def test_user_prompt_launcher_on_sets_and_clears_without_stdout(tmp_path):
    d = tmp_path / 'data'
    e = _launch_env(d, search_gate_enabled='true')
    st = PS.PromptState(str(d / PS.STATE_DIRNAME))
    rc, out, err = _spawn('user_prompt_hook.py', e, _ups_payload('Using web search, confirm how many days there are '
                                                                 'in a week. Answer in one sentence.'))
    assert rc == 0 and out == '' and json.loads(err)['status'] == 'explicit_set'
    assert st.read(SESSION) == 'explicit'
    rc, out, err = _spawn('user_prompt_hook.py', e, _ups_payload('continue'))
    assert out == '' and json.loads(err)['status'] == 'follow_up_kept' and st.read(SESSION) == 'explicit'
    rc, out, err = _spawn('user_prompt_hook.py', e, _ups_payload(NEUTRAL))
    assert out == '' and json.loads(err)['status'] == 'follow_up_kept' and st.read(SESSION) == 'explicit'
    rc, out, err = _spawn('user_prompt_hook.py', e, _ups_payload('New question: ' + NEUTRAL))
    assert out == '' and json.loads(err)['status'] == 'cleared' and st.read(SESSION) == 'not_explicit'
    assert st.read('other-session') == 'absent'
    for bad in ('not json', '[]', json.dumps({'hook_event_name': 'UserPromptSubmit', 'prompt': 'x'})):
        rc, out, err = _spawn('user_prompt_hook.py', e, bad)
        assert rc == 0 and out == ''
    assert 'zebra' not in err and all('zebra' not in open(os.path.join(st.root, f)).read()
                                      for f in os.listdir(st.root) if f.endswith('.json'))


# --- status ---------------------------------------------------------------------------------------------------------
def test_status_never_prints_secrets_and_shows_sources(tmp_path):
    presets_file = tmp_path / 'my-secret-presets.json'
    presets_file.write_text(json.dumps({'mine': {'description': 'Mine.', 'include_domains': ['hidden.example']}}))
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'env': {'ANTHROPIC_API_KEY': 'sk-ant-SETTINGSSECRET'},
                                    'pluginConfigs': {'davi@judge-adapter': {'options': {
                                        'runpod_api_key': 'rpa_SAVEDSECRET', 'domain_presets': str(presets_file),
                                        'domain_presets_enabled': True, 'search_gate_threshold': 0.9}}}}))
    e = cfg_env(tmp_path / 'data')
    _set(tmp_path / 'data', gate=True)
    rc, out, err = cli('--data-dir', str(tmp_path / 'data'), 'status', env=e, settings=settings)
    assert rc == 0 and err == ''
    for secret in (KEY, ENDPOINT, 'rpa_SAVEDSECRET', 'sk-ant-SETTINGSSECRET', 'hidden.example', 'my-secret-presets'):
        assert secret not in out, secret
    assert 'key (runpod_api_key): set' in out and 'endpoint (choice_endpoint_id): set' in out
    assert 'gate    on   [toggle]' in out and 'domain  on   [userConfig]' in out
    assert 'search_gate_threshold: 0.9 [settings]' in out and 'domain presets: none, mine (yours)' in out
    assert 'install-wide' in out


def test_status_shows_non_runnable_states(tmp_path):
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps({'pluginConfigs': {'davi@judge-adapter': {'options': {
        'domain_presets': '{"x": {"description": "d", "include_domains": ["https://bad/path"]}}'}}}}))
    e = cfg_env(tmp_path / 'data', runpod_api_key=None, usd_per_second=None)
    _set(tmp_path / 'data', domain=True)
    rc, out, _ = cli('--data-dir', str(tmp_path / 'data'), 'status', env=e, settings=settings)
    assert rc == 0
    assert 'warning: domain presets are on but domain_presets is invalid' in out and 'https://bad' not in out
    assert 'key (runpod_api_key): missing' in out and 'warning: key missing' in out
    assert 'warning: usd_per_second missing' in out
    # no hook environment (as from the Bash tool): secrets are reported as not visible, not as missing
    rc, out, _ = cli('--data-dir', str(tmp_path / 'data'), 'status', env={}, settings=tmp_path / 'none.json')
    assert 'not visible here' in out and 'warning: key missing' not in out
    assert 'user settings file not found' in out
    # global switch off with a toggle on
    _set(tmp_path / 'data', gate=True)
    rc, out, _ = cli('--data-dir', str(tmp_path / 'data'), 'status', env=cfg_env(tmp_path / 'data', enabled='false'),
                     settings=tmp_path / 'none.json')
    assert 'gate    off' in out and 'enabled=false turns every feature off' in out


def test_set_with_global_off_says_it_has_no_effect(tmp_path):
    rc, out, _ = cli('--data-dir', str(tmp_path / 'd'), 'gate', 'on', env={'CLAUDE_PLUGIN_OPTION_ENABLED': 'false'})
    assert rc == 0 and 'has no effect until enabled is true' in out


# --- command files ----------------------------------------------------------------------------------------------------
def _frontmatter(path):
    text = path.read_text(encoding='utf-8')
    assert text.startswith('---\n')
    head, body = text[4:].split('\n---\n', 1)
    return dict(line.split(': ', 1) for line in head.splitlines()), body


COMMANDS = {'status': 'status', 'filter-on': 'filter on', 'filter-off': 'filter off', 'gate-on': 'gate on',
            'gate-off': 'gate off', 'domain-on': 'domain on', 'domain-off': 'domain off',
            'domain-mode-ask': 'domain-mode ask', 'domain-mode-input-only': 'domain-mode input_only'}
HOSTILE_ARGS = ['on; touch {s}', '$(touch {s})', '`touch {s}`', 'on && touch {s}', "'; touch {s}; '",
                '"; touch {s}; "', 'on\ntouch {s}', '!`touch {s}`', '${{IFS}}touch${{IFS}}{s}', 'on | touch {s}']


def inject_lines(text):
    """Every shell command Claude Code's ``!`` injection would run for this text: inline ``!`cmd` `` at the start of
    a line or after whitespace, and fenced blocks opened with ```! (skills docs, "Injecting dynamic context")."""
    import re
    cmds = [m.group(1) for m in re.finditer(r'(?:^|(?<=\s))!`([^`]*)`', text, flags=re.M)]
    cmds += [m.group(1) for m in re.finditer(r'^```!\n(.*?)^```', text, flags=re.M | re.S)]
    return cmds


def render(text, data_dir, args):
    """Claude Code's documented substitutions on the ORIGINAL file: ``$ARGUMENTS`` / ``$ARGUMENTS[N]`` / ``$N`` and the
    plugin path variables. (Unconsumed arguments are appended as ``ARGUMENTS: <value>``; the docs say ``!``
    substitution "runs once over the original file", so the appended text is not scanned here.)"""
    import re
    import shlex
    try:
        parts = shlex.split(args)
    except ValueError:
        parts = args.split()
    text = re.sub(r'\$ARGUMENTS\[(\d+)\]', lambda m: parts[int(m.group(1))] if int(m.group(1)) < len(parts) else '',
                  text)
    text = text.replace('$ARGUMENTS', args)
    text = re.sub(r'\$(\d+)', lambda m: parts[int(m.group(1))] if int(m.group(1)) < len(parts) else '', text)
    return text.replace('${CLAUDE_PLUGIN_DATA}', str(data_dir)).replace('${CLAUDE_PLUGIN_ROOT}', str(ROOT))


def test_command_files_are_fixed_and_least_privilege():
    assert sorted(p.stem for p in (ROOT / 'commands').iterdir()) == sorted(COMMANDS)
    for name, sub in COMMANDS.items():
        fm, body = _frontmatter(ROOT / 'commands' / f'{name}.md')
        assert set(fm) == {'description', 'allowed-tools', 'disable-model-invocation'}       # no argument-hint
        assert fm['allowed-tools'] == 'Bash(davi-toggle *)' and fm['disable-model-invocation'] == 'true'
        assert inject_lines(body) == [f'davi-toggle --data-dir "${{CLAUDE_PLUGIN_DATA}}" {sub}']
        assert 'Do not run any tool' in body
        text = (ROOT / 'commands' / f'{name}.md').read_text(encoding='utf-8')
        assert '$ARGUMENTS' not in text and not __import__('re').search(r'\$\d|\$[a-z_]', text)
        for cmd in inject_lines(text):              # the only expansion in a shell line is the plugin data path
            assert cmd.replace('${CLAUDE_PLUGIN_DATA}', '').count('$') == 0 and '`' not in cmd


def _run_rendered(cmd, tmp_path, hostile):
    env = {'PATH': f'{ROOT / "bin"}{os.pathsep}{os.environ.get("PATH", "")}', 'HOME': str(tmp_path),
           'ARGUMENTS': hostile}
    return subprocess.run(['/bin/bash', '-c', cmd], capture_output=True, text=True, env=env, timeout=30,
                          cwd=str(tmp_path))


@pytest.mark.parametrize('name', sorted(COMMANDS))
def test_hostile_arguments_never_execute(tmp_path, name):
    """Sentinel: render each command with hostile arguments (and a hostile ARGUMENTS environment variable), run the
    resulting injected shell lines; the harmless sentinel file must never appear."""
    text = (ROOT / 'commands' / f'{name}.md').read_text(encoding='utf-8')
    data = tmp_path / 'data'
    for i, tmpl in enumerate(HOSTILE_ARGS):
        sentinel = tmp_path / f'SENTINEL-{i}'
        hostile = tmpl.format(s=sentinel)
        cmds = inject_lines(render(text, data, hostile))
        assert cmds == [f'davi-toggle --data-dir "{data}" {COMMANDS[name]}']                # args never reach it
        for cmd in cmds:
            r = _run_rendered(cmd, tmp_path, hostile)
            assert r.returncode == 0, r.stderr
        assert not sentinel.exists()
    assert not list(tmp_path.glob('SENTINEL-*'))


def test_sentinel_harness_detects_an_argument_placeholder(tmp_path):
    """Negative control: the earlier form (``... gate $ARGUMENTS`` in the ``!`` line) DOES let a hostile argument run
    a command, which is why every command is now a fixed line."""
    old = '!`davi-toggle --data-dir "${CLAUDE_PLUGIN_DATA}" gate $ARGUMENTS`\n'
    sentinel = tmp_path / 'SENTINEL-old'
    (cmd,) = inject_lines(render(old, tmp_path / 'data', f'on; touch {sentinel}'))
    _run_rendered(cmd, tmp_path, '')
    assert sentinel.exists()


def test_manifest_hooks_include_user_prompt_submit():
    hooks = json.loads((ROOT / 'hooks' / 'hooks.json').read_text(encoding='utf-8'))['hooks']
    (entry,) = hooks['UserPromptSubmit']
    assert 'matcher' not in entry                                          # UserPromptSubmit has no matcher support
    (h,) = entry['hooks']
    assert h == {'type': 'command', 'command': 'python3', 'args': ['${CLAUDE_PLUGIN_ROOT}/hooks/user_prompt_hook.py'],
                 'timeout': 10}
    assert set(hooks) == {'PreToolUse', 'PostToolUse', 'UserPromptSubmit'}


@pytest.mark.parametrize('text', [
    '{}', '{"gate": true}', '{"domain": false, "domain_mode": "input_only", "filter": true, "gate": false, '
    '"updated": 1760190000.123, "v": 1}', ' {"gate":false,"v":1} ', '{"updated": 1e9}', '{"gate": true,}',
    '{"gate": "a,b"}', '{"gate": NaN}', '{"g\\u0061te": true}', '{"gate": true} x', '[true]', '{"gate": null}',
    '{"domain_mode": "ask"}', '{"gate": True}'])
def test_flat_parser_agrees_with_json_or_defers(text):
    flat = T._parse_flat(text)
    if flat is not None:
        assert flat == json.loads(text)
    try:
        expected = (T.validate(json.loads(text)), 'ok')
    except (ValueError, T.ToggleError):
        expected = ({}, 'corrupt')
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, T.FILE_NAME)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
        assert T.read(p) == expected


def test_data_dir_id_matches_between_cli_and_hook_logs(tmp_path):
    """The smoke compares the id that /davi:status (injected ${CLAUDE_PLUGIN_DATA}) prints with the id the hooks log
    from their CLAUDE_PLUGIN_DATA environment variable."""
    import re
    d = tmp_path / 'data'
    rc, out, _ = cli('--data-dir', str(d), 'gate', 'on')
    (cli_id,) = re.findall(r'data dir id: ([0-9a-f]{12})', out)
    rc, out, _ = cli('--data-dir', str(d), 'status', settings=tmp_path / 'none.json')
    assert re.findall(r'data dir id: ([0-9a-f]{12})', out) == [cli_id]
    rc, _, err = _spawn('user_prompt_hook.py', _launch_env(d), _ups_payload('Using web search, check this.'))
    assert json.loads(err)['data_dir_id'] == cli_id
    rc, _, err = _spawn('query_options_hook.py', _launch_env(d), _pre_payload())
    assert json.loads(err.strip().splitlines()[-1])['data_dir_id'] == cli_id
    assert str(d) not in err
