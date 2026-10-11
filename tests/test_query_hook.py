"""Optional query-stage options (PreToolUse search gate and domain presets), model-free: an in-process fake choice
endpoint only. Network and key reads fail every test.

Covers: both options off (no call, no state, no output; launcher exits before reading stdin), each option alone and
both together (fixed order), the documented PreToolUse output shapes, the full original input kept with only one domain
field added, existing domain fields / "none" / low probability leave the input unchanged, deny then same-query retry
(no output), session and tool isolation, TTL expiry, concurrency of the retry state, the never-deny guard, fail-open on
errors / timeouts / missing or invalid configuration, zero intervention for other tools, logs without query, URLs,
domains, config text or keys, and that "allow" is never emitted."""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from judge_adapter import plugin_hook as H
from judge_adapter import prompt_signal as PS
from judge_adapter import query_hook as QH
from judge_adapter import query_presets as Q

ROOT = Path(__file__).resolve().parents[1]
KEY = 'rpa_QUERYHOOKTESTKEY0123456789'
ENDPOINT = 'endpointsecret77'
QUERY = 'How do zebra lanterns glow in the synthetic festival'
TAVILY = 'mcp__tavily__tavily_search'
SESSION = 'session-A'


@pytest.fixture(autouse=True)
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


class FakeChoice:
    """Fake choice endpoint. Gate options arrive as opt_0 needed / opt_1 not_needed / opt_2 cannot_tell; domain
    options as '<preset name>: ...'. ``gate`` / ``domain``: label -> probability (choice = argmax). ``fail``: one of
    http500, timeout, format, slow."""

    def __init__(self, gate=None, domain=None, fail=None):
        self.gate = gate or {'needed': 0.9, 'not_needed': 0.05, 'cannot_tell': 0.05}
        self.domain = domain or {'none': 1.0}
        self.fail = fail
        self.bodies = []
        self.lock = threading.Lock()

    def count(self, qid=None):
        return sum(1 for b in self.bodies if qid is None or qid in b['input']['questions'])

    def handle(self, request):
        assert '/runsync' in request.url.path
        body = json.loads(request.content)
        with self.lock:
            self.bodies.append(body)
        if self.fail == 'http500':
            return httpx.Response(500, text='fake upstream error')
        if self.fail == 'timeout':
            raise httpx.ReadTimeout('fake timeout', request=request)
        if self.fail == 'slow':
            time.sleep(3)
        (qid, q), = body['input']['questions'].items()
        ids = list(q['criteria'])
        if qid == 'search_need':
            labels = ['needed', 'not_needed', 'cannot_tell']
            table = self.gate
        else:
            labels = [q['criteria'][i].split(':', 1)[0] for i in ids]
            table = self.domain
        probs = {i: float(table.get(lab, 0.0)) for i, lab in zip(ids, labels)}
        if self.fail == 'format':
            probs = {i: 0.2 for i in ids}
        choice = max(probs, key=probs.get)
        out = {'model': 'fake-choice', 'answers': {qid: {'type': 'choice', 'choice': choice, 'probabilities': probs}}}
        return httpx.Response(200, json={'id': 'job-1', 'status': 'COMPLETED', 'delayTime': 10, 'executionTime': 200,
                                         'output': out})


def env(**kw):
    e = {'CLAUDE_PLUGIN_OPTION_RUNPOD_API_KEY': KEY, 'CLAUDE_PLUGIN_OPTION_CHOICE_ENDPOINT_ID': ENDPOINT,
         'CLAUDE_PLUGIN_OPTION_USD_PER_SECOND': '0.0004'}
    for k, v in kw.items():
        name = H.OPTION_ENV + k.upper()
        if v is None:
            e.pop(name, None)
        else:
            e[name] = v
    return e


GATE_ON = {'search_gate_enabled': 'true'}
DOMAIN_ON = {'domain_presets_enabled': 'true'}


def payload(tool='WebSearch', query=QUERY, session=SESSION, **extra):
    tin = {'query': query, **extra}
    p = {'session_id': session, 'transcript_path': '/nonexistent/transcript.jsonl', 'cwd': '/tmp',
         'permission_mode': 'default', 'hook_event_name': 'PreToolUse', 'tool_name': tool, 'tool_input': tin,
         'tool_use_id': 'toolu_01'}
    if session is None:
        del p['session_id']
    return p


OUTPUTS = []      # every non-empty output of every run, checked for "allow" at the end
# A user prompt with no explicit-search signal: the gate may run after it (when no explicit flag is fresh).
NEUTRAL_PROMPT = 'Tell me how zebra lanterns glow at the synthetic festival, in detail please.'
PROMPT_DIR = None


@pytest.fixture(autouse=True)
def _prompt_dir(tmp_path_factory):
    global PROMPT_DIR
    PROMPT_DIR = str(tmp_path_factory.mktemp('prompt-state'))


def run(p, e, fake=None, state_dir=None, now=time.time, prompt=NEUTRAL_PROMPT):
    """``prompt``: the user prompt the UserPromptSubmit hook saw first for this session (None: no prompt state)."""
    fake = fake or FakeChoice()
    if prompt is not None and isinstance(p, dict) and isinstance(p.get('session_id'), str):
        PS.PromptState(PROMPT_DIR, now=now).on_prompt(p['session_id'], prompt, p.get('prompt_id'))
    out, err, exits = io.StringIO(), io.StringIO(), []
    rc = QH.main(stdin=io.StringIO(json.dumps(p)), stdout=out, stderr=err, env=e,
                 transport=httpx.MockTransport(fake.handle), exit_fn=exits.append, now=now,
                 state_dir=None if state_dir is None else str(state_dir), prompt_state_dir=PROMPT_DIR)
    assert rc == 0
    text = out.getvalue()
    log = json.loads(err.getvalue().strip().splitlines()[-1])
    if text:
        OUTPUTS.append(json.loads(text))
    return (json.loads(text) if text else None), log, fake, err.getvalue(), exits


# --- both off ------------------------------------------------------------------------------------------------------
def test_both_off_no_call_no_state_no_output(tmp_path):
    state = tmp_path / 'state'
    for e in (env(), env(search_gate_enabled='false', domain_presets_enabled='false'),
              env(enabled='false', **GATE_ON, **DOMAIN_ON)):
        out, log, fake, _, _ = run(payload(), e, state_dir=state)
        assert out is None and fake.count() == 0 and not state.exists()
        assert log['status'] in ('options_off', 'disabled')


def _launcher_env(**opts):
    e = {k: v for k, v in os.environ.items() if not k.startswith(('CLAUDE_PLUGIN_OPTION_', 'JUDGE_ADAPTER_HOOK_'))
         and k not in ('RUNPOD_API_KEY', 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID', 'PYTHONPATH')}
    e['CLAUDE_PLUGIN_ROOT'] = str(ROOT)
    e.update({H.OPTION_ENV + k.upper(): v for k, v in opts.items()})
    return e


def test_launcher_fast_exit_when_off_reads_nothing(tmp_path):
    """Both off: exit 0 with no stdout and no stderr, without reading stdin (stdin left open here) or importing the
    package (a broken src copy would fail)."""
    e = _launcher_env(search_gate_enabled='false')
    e['CLAUDE_PLUGIN_DATA'] = str(tmp_path / 'data')
    t = time.perf_counter()
    proc = subprocess.Popen([sys.executable, str(ROOT / 'hooks' / 'query_options_hook.py')], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=e)
    out, err = proc.communicate(timeout=30)       # stdin never written or closed before exit is not required
    assert proc.returncode == 0 and out == b'' and err == b''
    assert time.perf_counter() - t < 5
    assert not (tmp_path / 'data').exists()
    assert QH.LOG_TAG.startswith('davi-')


def test_launcher_subprocess_gate_on_without_key_fails_open(tmp_path):
    e = _launcher_env(search_gate_enabled='true')
    e['CLAUDE_PLUGIN_DATA'] = str(tmp_path / 'data')
    PS.PromptState(str(tmp_path / 'data' / PS.STATE_DIRNAME)).on_prompt(SESSION, NEUTRAL_PROMPT)
    r = subprocess.run([sys.executable, str(ROOT / 'hooks' / 'query_options_hook.py')], input=json.dumps(payload()),
                       capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 0 and r.stdout == ''
    assert json.loads(r.stderr.strip().splitlines()[-1])['status'] == 'missing_key'
    assert not (tmp_path / 'data' / QH.STATE_DIRNAME).exists()      # nothing denied -> no retry state written


# --- search gate ---------------------------------------------------------------------------------------------------
def deny_fake(p_not=0.95):
    return FakeChoice(gate={'needed': (1 - p_not) * 0.8, 'not_needed': p_not, 'cannot_tell': (1 - p_not) * 0.2})


def test_gate_deny_output_schema_and_wire_body(tmp_path):
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path)
    assert out == {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                          'permissionDecisionReason': Q.GATE_DENY_NOTICE}}
    assert 'retry the same query' in Q.GATE_DENY_NOTICE and 'not a verified judgment' in Q.GATE_DENY_NOTICE
    assert fake.count() == 1
    body = fake.bodies[0]
    assert body['input']['state'] == {'search_query': QUERY}            # the query only; no transcript, no user text
    q = body['input']['questions']['search_need']
    assert q['type'] == 'choice' and q['instructions'] == Q.GATE_PROMPTS['gate-v0-dev'].instructions
    assert list(q['criteria']) == ['opt_0', 'opt_1', 'opt_2']
    assert log['status'] == 'denied' and log['gate']['status'] == 'deny' and log['gate']['calls'] == 1
    assert log['gate']['probabilities'] == {'needed': 0.04, 'not_needed': 0.95, 'cannot_tell': 0.01}
    assert 'domain' not in log


@pytest.mark.parametrize('p_not,denied', [(0.85, True), (0.849, False), (0.5, False), (0.1, False)])
def test_gate_threshold(tmp_path, p_not, denied):
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(p_not), tmp_path)
    assert (out is not None) == denied and fake.count() == 1
    assert log['gate']['status'] == ('deny' if denied else 'keep')


def test_gate_custom_threshold(tmp_path):
    out, log, _, _, _ = run(payload(), env(search_gate_threshold='0.99', **GATE_ON), deny_fake(0.95), tmp_path)
    assert out is None and log['gate']['status'] == 'keep' and log['gate']['threshold'] == 0.99


def test_gate_cannot_tell_keeps_the_call(tmp_path):
    fake = FakeChoice(gate={'needed': 0.05, 'not_needed': 0.15, 'cannot_tell': 0.8})
    out, log, _, _, _ = run(payload(), env(**GATE_ON), fake, tmp_path)
    assert out is None and log['gate']['status'] == 'keep'


def test_retry_after_deny_outputs_nothing_and_isolation(tmp_path):
    e = env(**GATE_ON)
    out, _, fake, _, _ = run(payload(), e, deny_fake(), tmp_path)
    assert out['hookSpecificOutput']['permissionDecision'] == 'deny'
    # same session, same tool, same query after normalization: no call, no output (never an explicit allow)
    out, log, fake2, _, _ = run(payload(query='  how DO zebra lanterns   glow in the synthetic festival '), e,
                                deny_fake(), tmp_path)
    assert out is None and fake2.count() == 0 and log['status'] == 'retry_allowed'
    # another session: gated normally
    out, log, fake3, _, _ = run(payload(session='session-B'), e, deny_fake(), tmp_path)
    assert out is not None and fake3.count() == 1
    # same session, other tool: gated normally
    out, log, fake4, _, _ = run(payload(tool=TAVILY), e, deny_fake(), tmp_path)
    assert out is not None and fake4.count() == 1
    # a different query in the same session: gated normally
    out, log, fake5, _, _ = run(payload(query=QUERY + ' tonightless'), e, deny_fake(), tmp_path)
    assert fake5.count() == 1


def test_retry_record_expires_after_ttl(tmp_path):
    e, clock = env(**GATE_ON), [1_000_000.0]
    now = lambda: clock[0]               # noqa: E731
    run(payload(), e, deny_fake(), tmp_path, now=now)
    clock[0] += QH.GATE_RETRY_TTL_SECONDS - 1
    out, log, fake, _, _ = run(payload(), e, deny_fake(), tmp_path, now=now)
    assert out is None and log['status'] == 'retry_allowed' and fake.count() == 0
    clock[0] += 2                         # now past the TTL of the first deny
    out, log, fake, _, _ = run(payload(), e, deny_fake(), tmp_path, now=now)
    assert out is not None and fake.count() == 1


def test_state_holds_hashes_only(tmp_path):
    run(payload(), env(**GATE_ON), deny_fake(), tmp_path)
    assert sorted(f.rsplit('.', 1)[1] for f in os.listdir(tmp_path)) == ['json', 'lock']
    files = [f for f in os.listdir(tmp_path) if f.endswith('.json')]
    assert len(files) == 1 and files[0].endswith('.json') and len(files[0]) == 64 + 5
    content = (tmp_path / files[0]).read_text(encoding='utf-8')
    assert set(json.loads(content)) == {'v', 't'}
    for s in (QUERY, 'zebra', SESSION, 'WebSearch'):
        assert s not in content and s not in files[0]


def test_default_state_dir_uses_plugin_data(tmp_path):
    assert QH.default_state_dir({'CLAUDE_PLUGIN_DATA': str(tmp_path)}) == str(tmp_path / QH.STATE_DIRNAME)
    import tempfile
    assert os.path.dirname(QH.default_state_dir({})) == tempfile.gettempdir()


def test_retry_state_threads_one_claim_per_key(tmp_path):
    st = QH.RetryState(str(tmp_path))
    keys = [QH.RetryState.key(f's{i % 3}', 'WebSearch', f'q{i % 5}') for i in range(60)]   # 15 distinct keys
    results, lock = [], threading.Lock()

    def work(k):
        r = st.claim(k)
        with lock:
            results.append((k, r))
    threads = [threading.Thread(target=work, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r in (True, False) for _, r in results)
    for k in set(keys):
        assert sum(1 for kk, r in results if kk == k and r is True) == 1
        assert st.denied_recently(k) is True
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.tmp-')]          # no temp files left


def test_claim_after_fresh_record_from_elsewhere_outputs_nothing(tmp_path, monkeypatch):
    """The pre-check saw nothing, but another process recorded the deny before this one's claim: no output."""
    key = QH.RetryState.key(SESSION, 'WebSearch', QUERY)
    assert QH.RetryState(str(tmp_path)).claim(key) is True
    monkeypatch.setattr(QH.RetryState, 'denied_recently', lambda self, k: False)
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 1 and log['gate']['status'] == 'already_denied_concurrently'


def test_prune_keeps_fresh_and_removes_stale(tmp_path):
    clock = [1_000_000.0]
    st = QH.RetryState(str(tmp_path), now=lambda: clock[0])
    old, new = QH.RetryState.key('s', 't', 'old'), QH.RetryState.key('s', 't', 'new')
    assert st.claim(old) is True
    clock[0] += QH.GATE_RETRY_TTL_SECONDS + 1
    assert st.claim(new) is True                      # claim prunes the stale record, never the fresh one
    names = set(os.listdir(tmp_path))
    assert names == {new + '.json', new + '.lock'}
    assert st.claim(old) is True                      # the expired query can be denied again (TTL refresh)


def test_no_fcntl_never_denies(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'fcntl':
            raise ImportError('no fcntl')
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake_import)
    assert QH.RetryState(str(tmp_path)).claim('a' * 64) is None


def test_gate_without_session_abstains(tmp_path):
    out, log, fake, _, _ = run(payload(session=None), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'abstain_no_session'


def test_gate_state_unusable_never_denies(tmp_path, monkeypatch):
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), blocker / 'sub')     # cannot be read
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'abstain_state_unavailable'
    monkeypatch.setattr(QH.RetryState, 'claim', lambda self, key: None)                    # lock / write failed
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path / 'state')
    assert out is None and fake.count() == 1 and log['gate']['status'] == 'abstain_state_write_failed'
    monkeypatch.undo()
    blocker2 = tmp_path / 'file2'
    blocker2.write_text('x')
    assert QH.RetryState(str(blocker2)).claim('k' * 64) is None                             # real write failure


@pytest.mark.parametrize('query,extra,guard', [
    ('latest python release notes', {}, 'recency'),
    ('population of France 2026', {}, 'recency'),
    ('오늘 서울 날씨', {}, 'recency'),
    ('verify the boiling point of water source', {}, 'verification'),
    ('site:example.org zebra lanterns', {}, 'specific_source'),
    (QUERY, {'allowed_domains': ['example.org']}, 'domains_set'),
])
def test_gate_guard_keeps_without_a_call(tmp_path, query, extra, guard):
    out, log, fake, _, _ = run(payload(query=query, **extra), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 0 and log['gate'] == {'prompt': 'gate-v0-dev', 'status': 'keep_guard',
                                                                  'calls': 0, 'guard': guard}


def test_gate_guard_tavily_time_fields(tmp_path):
    out, log, fake, _, _ = run(payload(tool=TAVILY, time_range='week'), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 0 and log['gate']['guard'] == 'tavily_recency_field'
    out, log, fake, _, _ = run(payload(tool=TAVILY, start_date=''), env(**GATE_ON), deny_fake(), tmp_path)
    assert fake.count() == 1                     # an empty default date is not a recency signal


# --- domain presets ------------------------------------------------------------------------------------------------
def dom_fake(name, p=0.9):
    other = 'docs' if name == 'none' else 'none'
    return FakeChoice(domain={name: p, other: 1 - p})


def test_domain_websearch_applied_with_ask_and_full_input(tmp_path):
    p = payload(extra_field='kept')
    out, log, fake, _, _ = run(p, env(**DOMAIN_ON), dom_fake('docs'), tmp_path)
    hso = out['hookSpecificOutput']
    assert set(hso) == {'hookEventName', 'permissionDecision', 'permissionDecisionReason', 'updatedInput'}
    assert hso['hookEventName'] == 'PreToolUse' and hso['permissionDecision'] == 'ask'
    assert hso['updatedInput'] == {'query': QUERY, 'extra_field': 'kept',
                                   'allowed_domains': Q.EXAMPLE_DOMAIN_PRESETS['docs']['include_domains']}
    assert 'blocked_domains' not in hso['updatedInput']
    assert '"docs"' in hso['permissionDecisionReason'] and 'allowed_domains' in hso['permissionDecisionReason']
    assert p['tool_input'] == {'query': QUERY, 'extra_field': 'kept'}          # the input object is not mutated
    assert fake.count() == 1 and fake.count('domain_preset') == 1
    body = fake.bodies[0]
    assert body['input']['state'] == {'search_query': QUERY}
    crit = body['input']['questions']['domain_preset']['criteria']
    assert [c.split(':', 1)[0] for c in crit.values()] == ['none', 'docs', 'academic']
    assert log['status'] == 'domain_applied' and log['domain']['preset'] == 'docs' and log['domain']['p_top'] == 0.9
    assert log['domain']['source'] == 'examples' and 'gate' not in log


def test_domain_tavily_applied_keeps_every_other_field(tmp_path):
    tin = dict(search_depth='advanced', max_results=7, include_raw_content=False, country='Japan', exact_match=True)
    out, _, _, _, _ = run(payload(tool=TAVILY, **tin), env(**DOMAIN_ON), dom_fake('academic'), tmp_path)
    upd = out['hookSpecificOutput']['updatedInput']
    assert upd == {'query': QUERY, **tin, 'include_domains': Q.EXAMPLE_DOMAIN_PRESETS['academic']['include_domains']}


def test_domain_exclude_preset_maps_per_tool(tmp_path):
    presets = json.dumps({'no-forums': {'description': 'Skip forums.', 'exclude_domains': ['forum.example']}})
    e = env(domain_presets=presets, **DOMAIN_ON)
    out, log, _, _, _ = run(payload(), e, dom_fake('no-forums'), tmp_path)
    assert out['hookSpecificOutput']['updatedInput'] == {'query': QUERY, 'blocked_domains': ['forum.example']}
    assert log['domain']['source'] == 'user'
    out, _, _, _, _ = run(payload(tool=TAVILY), e, dom_fake('no-forums'), tmp_path)
    assert out['hookSpecificOutput']['updatedInput'] == {'query': QUERY, 'exclude_domains': ['forum.example']}


@pytest.mark.parametrize('tool,field,value', [
    ('WebSearch', 'allowed_domains', ['example.org']), ('WebSearch', 'blocked_domains', []),
    (TAVILY, 'include_domains', ['example.org']), (TAVILY, 'exclude_domains', [])])
def test_domain_existing_fields_untouched_without_a_call(tmp_path, tool, field, value):
    out, log, fake, _, _ = run(payload(tool=tool, **{field: value}), env(**DOMAIN_ON), dom_fake('docs'), tmp_path)
    assert out is None and fake.count() == 0 and log['domain']['status'] == 'existing_domains'


def test_domain_none_and_low_confidence_leave_input(tmp_path):
    out, log, fake, _, _ = run(payload(), env(**DOMAIN_ON), dom_fake('none', 0.95), tmp_path)
    assert out is None and log['domain']['status'] == 'none' and fake.count() == 1
    out, log, _, _, _ = run(payload(), env(**DOMAIN_ON), dom_fake('docs', 0.55), tmp_path)
    assert out is None and log['domain']['status'] == 'low_confidence' and log['domain']['preset'] == 'docs'
    out, log, _, _, _ = run(payload(), env(domain_min_confidence='0.5', **DOMAIN_ON), dom_fake('docs', 0.55), tmp_path)
    assert out is not None and log['domain']['status'] == 'applied'


def test_domain_input_only_mode_has_no_decision(tmp_path):
    out, _, _, _, _ = run(payload(), env(domain_apply_mode='input_only', **DOMAIN_ON), dom_fake('docs'), tmp_path)
    assert out == {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'updatedInput': {
        'query': QUERY, 'allowed_domains': Q.EXAMPLE_DOMAIN_PRESETS['docs']['include_domains']}}}


def test_domain_presets_from_file(tmp_path):
    f = tmp_path / 'presets.json'
    f.write_text(json.dumps({'none': {'description': 'Whole web.'},
                             'mine': {'description': 'Mine.', 'include_domains': ['a.example', 'b.example']}}))
    out, log, fake, _, _ = run(payload(), env(domain_presets=str(f), **DOMAIN_ON), dom_fake('mine'), tmp_path)
    assert out['hookSpecificOutput']['updatedInput']['allowed_domains'] == ['a.example', 'b.example']
    crit = list(fake.bodies[0]['input']['questions']['domain_preset']['criteria'].values())
    assert crit[0] == 'none: Whole web.' and len(crit) == 2


def test_domain_state_never_written(tmp_path):
    run(payload(), env(**DOMAIN_ON), dom_fake('docs'), tmp_path / 'state')
    assert not (tmp_path / 'state').exists()


# --- both options: fixed order -------------------------------------------------------------------------------------
def test_both_gate_deny_skips_domain(tmp_path):
    fake = FakeChoice(gate={'needed': 0.02, 'not_needed': 0.96, 'cannot_tell': 0.02}, domain={'docs': 0.9, 'none': 0.1})
    out, log, _, _, _ = run(payload(), env(**GATE_ON, **DOMAIN_ON), fake, tmp_path)
    assert out['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert fake.count() == 1 and fake.count('search_need') == 1
    assert log['domain']['status'] == 'skipped_gate_deny' and log['domain']['calls'] == 0


def test_both_gate_keep_then_domain(tmp_path):
    fake = FakeChoice(domain={'docs': 0.9, 'none': 0.1})
    out, log, _, _, _ = run(payload(), env(**GATE_ON, **DOMAIN_ON), fake, tmp_path)
    assert out['hookSpecificOutput']['permissionDecision'] == 'ask'
    assert [next(iter(b['input']['questions'])) for b in fake.bodies] == ['search_need', 'domain_preset']
    assert log['gate']['status'] == 'keep' and log['gate']['calls'] == 1
    assert log['domain']['status'] == 'applied' and log['domain']['calls'] == 1


def test_both_retry_outputs_nothing_and_skips_domain(tmp_path):
    e = env(**GATE_ON, **DOMAIN_ON)
    run(payload(), e, FakeChoice(gate={'needed': 0, 'not_needed': 1.0, 'cannot_tell': 0}), tmp_path)
    fake = FakeChoice(domain={'docs': 0.9, 'none': 0.1})
    out, log, _, _, _ = run(payload(), e, fake, tmp_path)
    assert out is None and fake.count() == 0 and log['domain']['status'] == 'skipped_retry'


def test_both_guard_then_domain(tmp_path):
    fake = FakeChoice(domain={'docs': 0.9, 'none': 0.1})
    out, log, _, _, _ = run(payload(query='latest pandas merge docs'), env(**GATE_ON, **DOMAIN_ON), fake, tmp_path)
    assert fake.count() == 1 and fake.count('domain_preset') == 1 and log['gate']['status'] == 'keep_guard'


# --- fail-open -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('fail', ['http500', 'timeout', 'format'])
@pytest.mark.parametrize('opts', [GATE_ON, DOMAIN_ON])
def test_endpoint_failures_fail_open(tmp_path, fail, opts):
    fake = FakeChoice(gate={'not_needed': 1.0}, domain={'docs': 1.0}, fail=fail)
    out, log, _, _, _ = run(payload(), env(**opts), fake, tmp_path)
    assert out is None and fake.count() == 1
    assert log['status'] == 'unchanged'
    assert not any(tmp_path.iterdir())


def test_wall_budget_fails_open(tmp_path):
    t = time.perf_counter()
    out, log, _, _, exits = run(payload(), env(wall_budget_seconds='1', **GATE_ON), FakeChoice(fail='slow'), tmp_path)
    assert time.perf_counter() - t < 2.5
    assert out is None and log['status'] == 'wall_budget_exceeded' and exits == [0]


@pytest.mark.parametrize('missing,status', [
    ({'runpod_api_key': None}, 'missing_key'), ({'choice_endpoint_id': None}, 'missing_endpoint'),
    ({'usd_per_second': None}, 'missing_rate')])
def test_missing_configuration_fails_open(tmp_path, missing, status):
    out, log, fake, _, _ = run(payload(), env(**GATE_ON, **DOMAIN_ON, **missing), FakeChoice(), tmp_path)
    assert out is None and fake.count() == 0 and log['status'] == status


@pytest.mark.parametrize('bad', [
    {'search_gate_enabled': 'maybe'}, {'search_gate_threshold': '0.3'}, {'domain_min_confidence': '2'},
    {'domain_apply_mode': 'allow'}, {'domain_presets': '{not json'}, {'domain_presets': '[]'},
    {'domain_presets': json.dumps({'x': {'description': 'd', 'include_domains': ['a.example'],
                                         'exclude_domains': ['b.example']}})},
    {'domain_presets': json.dumps({'none': {'description': 'd', 'include_domains': ['a.example']}})},
    {'domain_presets': json.dumps({'x': {'description': 'd', 'include_domains': ['https://a.example/path']}})},
    {'domain_presets': json.dumps({'x': {'description': 'd', 'include_domains': ['*.example.org']}})},
    {'domain_presets': json.dumps({'Bad Name': {'description': 'd', 'include_domains': ['a.example']}})},
    {'domain_presets': json.dumps({'none': {'description': 'only none'}})},
    {'domain_presets': json.dumps({f'p{i}': {'description': 'd', 'include_domains': ['a.example']}
                                   for i in range(QH.MAX_PRESETS + 1)})},
    {'domain_presets': '/nonexistent/secret-presets.json'},
    {'concurrency': 'many'}])
def test_config_errors_fail_open_without_leaking(tmp_path, bad):
    out, log, fake, err, _ = run(payload(), env(**{**GATE_ON, **DOMAIN_ON, **bad}), FakeChoice(), tmp_path)
    assert out is None and fake.count() == 0 and log['status'] == 'config_error'
    for s in ('a.example', 'b.example', 'secret-presets', 'Bad Name', KEY, ENDPOINT, QUERY):
        assert s not in err


def test_unexpected_exception_fails_open(tmp_path, monkeypatch):
    monkeypatch.setattr(QH, 'gate_guard', lambda q, t: 1 / 0)
    out, log, _, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and log['status'] == 'error' and log['detail'] == 'ZeroDivisionError'


def test_bad_input_and_other_events(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    assert QH.main(stdin=io.StringIO('not json'), stdout=out, stderr=err, env=env(**GATE_ON),
                   exit_fn=lambda c: None) == 0
    assert out.getvalue() == '' and json.loads(err.getvalue())['status'] == 'bad_input'
    out, log, fake, _, _ = run({**payload(), 'hook_event_name': 'PostToolUse'}, env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and log['status'] == 'not_pre_tool_use' and fake.count() == 0
    for tin in ({}, {'query': '   '}, {'query': 3}):
        out, log, fake, _, _ = run({**payload(), 'tool_input': tin}, env(**GATE_ON), deny_fake(), tmp_path)
        assert out is None and log['status'] == 'no_query' and fake.count() == 0
    out, log, fake, _, _ = run(payload(query='q' * (QH.MAX_QUERY_BYTES + 1)), env(**GATE_ON), deny_fake(), tmp_path)
    assert out is None and log['status'] == 'query_too_long' and fake.count() == 0


# --- zero intervention, logs, "allow" ------------------------------------------------------------------------------
OTHER_TOOLS = ('WebFetch', 'Read', 'Write', 'Edit', 'Bash', 'mcp__tavily__tavily_extract', 'mcp__tavily__tavily_crawl',
               'mcp__tavily__tavily_map', 'mcp__tavily__tavily_research', 'mcp__tavily__tavily_search_extra',
               'mcp__exa__web_search_exa', 'mcp__brave-search__brave_web_search', 'websearch', 'WebSearchX',
               'tavily_search', 'xmcp__tavily__tavily_search', 'mcp__tavily__tavily_search\n')


@pytest.mark.parametrize('tool', OTHER_TOOLS)
def test_zero_intervention_for_other_tools(tmp_path, tool):
    out, log, fake, _, _ = run(payload(tool=tool), env(**GATE_ON, **DOMAIN_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 0 and log['status'] == 'tool_not_enabled'


def test_mcp_tools_off_excludes_tavily(tmp_path):
    out, log, fake, _, _ = run(payload(tool=TAVILY), env(mcp_tools=f'{TAVILY}=off', **GATE_ON), deny_fake(), tmp_path)
    assert out is None and fake.count() == 0 and log['status'] == 'tool_not_enabled'


def test_logs_hold_no_query_urls_domains_or_keys(tmp_path):
    log_file = tmp_path / 'q.log'
    presets = json.dumps({'mine': {'description': 'Secret description words.', 'include_domains': ['hidden.example']}})
    e = env(log_file=str(log_file), domain_presets=presets, **GATE_ON, **DOMAIN_ON)
    fake = FakeChoice(domain={'mine': 0.8, 'none': 0.2})
    out, log, _, err, _ = run(payload(query=QUERY + ' https://url.example/x'), e, fake, tmp_path / 's')
    run(payload(query=QUERY + ' plain'), e, FakeChoice(gate={'not_needed': 1.0}), tmp_path / 's')
    stored = log_file.read_text(encoding='utf-8')
    for s in (KEY, ENDPOINT, QUERY, 'zebra', 'url.example', 'hidden.example', 'Secret description', SESSION,
              '/nonexistent/transcript', presets):
        assert s not in stored and s not in err, s
    assert set(log) >= {'event', 'status', 'tool', 'seconds', 'query_bytes', 'gate', 'domain'}
    assert log['domain']['preset'] == 'mine' and log['domain']['p_top'] == 0.8


def test_never_emits_allow():
    if not OUTPUTS:
        pytest.skip('collects the outputs of the tests above; run the whole module')
    for o in OUTPUTS:
        assert o['hookSpecificOutput'].get('permissionDecision') in ('deny', 'ask', None)
        assert set(o) == {'hookSpecificOutput'}


# --- manifests -----------------------------------------------------------------------------------------------------
def test_manifest_pre_tool_use_matchers_and_options():
    plugin = json.loads((ROOT / '.claude-plugin' / 'plugin.json').read_text(encoding='utf-8'))
    hooks = json.loads((ROOT / 'hooks' / 'hooks.json').read_text(encoding='utf-8'))
    pre = hooks['hooks']['PreToolUse']
    assert [e['matcher'] for e in pre] == ['WebSearch', H.KNOWN_TOOLS_MATCHER]
    for entry in pre:
        (h,) = entry['hooks']
        assert h['type'] == 'command' and h['command'] == 'python3'
        assert h['args'] == ['${CLAUDE_PLUGIN_ROOT}/hooks/query_options_hook.py'] and h['timeout'] > 60
    cfg = plugin['userConfig']
    assert cfg['domain_presets_enabled']['default'] is False and cfg['search_gate_enabled']['default'] is False
    assert cfg['domain_min_confidence']['default'] == H.Settings().domain_min_confidence
    assert cfg['search_gate_threshold']['default'] == H.Settings().search_gate_threshold
    assert cfg['domain_apply_mode']['default'] == H.Settings().domain_apply_mode
    assert 'gate_use_last_user_message' not in cfg        # the PreToolUse input has no user-message field
    assert plugin['version'] == __import__('judge_adapter').__version__   # manifest and package versions match


def test_postuse_settings_ignore_query_options():
    """The PostToolUse filter neither reads nor validates the query-stage options."""
    s = H.load_settings(env(search_gate_enabled='maybe', domain_presets='{bad', search_gate_threshold='9'))
    assert s.search_gate_enabled is False and s.domain_presets is None


def test_presets_module_marks_experimental():
    for p in list(Q.GATE_PROMPTS.values()) + list(Q.DOMAIN_PROMPTS.values()):
        assert p.note.startswith('EXPERIMENTAL')
    assert all(v['description'].startswith('EXAMPLE') for v in Q.EXAMPLE_DOMAIN_PRESETS.values())


# --- explicit user request (UserPromptSubmit prompt state) --------------------------------------------------------
A1_PROMPT = 'Using web search, confirm how many days there are in a week. Answer in one sentence.'
A1_QUERY = 'how many days in a week'


def ups(session, prompt, e=None, prompt_id=None, now=time.time, state_dir=None):
    """The UserPromptSubmit entry point (prompt_signal.main) as the launcher runs it when the gate is on."""
    p = {'session_id': session, 'transcript_path': '/nonexistent/transcript.jsonl', 'cwd': '/tmp',
         'permission_mode': 'default', 'hook_event_name': 'UserPromptSubmit', 'prompt': prompt}
    if prompt_id is not None:
        p['prompt_id'] = prompt_id
    err = io.StringIO()
    assert PS.main(stdin=io.StringIO(json.dumps(p)), stderr=err, env=e or env(**GATE_ON), now=now,
                   state_dir=state_dir or PROMPT_DIR) == 0
    return json.loads(err.getvalue().strip().splitlines()[-1]), err.getvalue()


def test_a1_explicit_request_is_kept_with_zero_endpoint_calls(tmp_path):
    """Regression for the real 2026-10-11 smoke: explicit 'Using web search' prompt, query rated 0.999 not needed."""
    log_ups, _ = ups(SESSION, A1_PROMPT)
    assert log_ups['status'] == 'explicit_set' and log_ups['signal'] is True
    out, log, fake, _, _ = run(payload(query=A1_QUERY), env(**GATE_ON), deny_fake(0.999), tmp_path, prompt=None)
    assert out is None and fake.count() == 0
    assert log['status'] == 'explicit_request_keep' and log['gate']['status'] == 'explicit_request_keep'
    assert log['gate']['calls'] == 0 and not any(tmp_path.iterdir())          # no deny record either


def test_prompt_without_signal_still_reaches_the_gate_decision(tmp_path):
    log_ups, _ = ups(SESSION, 'How many days are there in a week? Answer in one sentence for me.')
    assert log_ups['status'] == 'cleared' and log_ups['signal'] is False
    out, log, fake, _, _ = run(payload(query=A1_QUERY), env(**GATE_ON), deny_fake(0.999), tmp_path, prompt=None)
    assert fake.count() == 1 and out['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert log['gate']['status'] == 'deny'


@pytest.mark.parametrize('prompt', [
    A1_PROMPT, 'Please search the web for the boiling point of water and give me one line.',
    'Search for the definition of a prime number and explain it to me in two lines.',
    'Could you look it up and tell me the capital of Australia in a single sentence?',
    'Google the speed of light for me and give me the number with its usual units.',
    'Find it online: how many legs does a spider have? One sentence is enough for me.',
    'Use the search tool to tell me what the chemical symbol for gold is, briefly please.',
    'Give the answer with a cited source: what is the largest ocean on our planet Earth?',
    'What is the latest stable release of the Python language? Keep the answer short.',
    'WEB SEARCH: what is the tallest mountain on Earth? Answer briefly in one sentence.',
    '웹 검색으로 일주일이 며칠인지 확인하고 한 문장으로 답해 주세요. 부탁드립니다.',
    '웹검색 해서 물의 끓는점을 알려줘. 짧게 한 문장으로 대답해 주면 좋겠어요.',
    '지구에서 가장 높은 산이 어디인지 검색해서 한 문장으로 알려 주세요. 감사합니다.',
    '거미 다리가 몇 개인지 찾아봐 줘. 한 문장이면 충분하니까 짧게 부탁할게요.',
    '빛의 속도를 찾아줘. 숫자와 단위만 짧게 한 줄로 정리해서 보여 주면 됩니다.',
    '태평양의 면적을 출처와 함께 알려 주세요. 길게 설명하지 않아도 괜찮습니다.',
    '파이썬 최신 버전이 무엇인지 알려 주세요. 짧게 한 문장으로 대답해 주세요.',
    '오늘 날짜 기준으로 한국의 수도가 어디인지 한 문장으로 알려 주시겠어요?'])
def test_explicit_requests_english_and_korean_keep_without_a_call(tmp_path, prompt):
    assert PS.explicit_request(prompt) is True
    out, log, fake, _, _ = run(payload(query=A1_QUERY), env(**GATE_ON), deny_fake(0.999), tmp_path, prompt=prompt)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'explicit_request_keep'


@pytest.mark.parametrize('prompt', [
    'How many days are there in a week? Answer in one sentence for me.',
    '일주일은 며칠인지 한 문장으로 짧게 대답해 주세요. 설명은 필요 없어요.'])
def test_prompts_without_signal_are_not_flagged(prompt):
    assert PS.explicit_request(prompt) is False


def test_no_prompt_state_keeps_without_a_call(tmp_path):
    """E.g. the gate was just switched on (off -> on): no state for this session yet."""
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_no_prompt_state'
    assert log['status'] == 'unchanged'


def test_expired_prompt_state_keeps_without_a_call(tmp_path):
    clock = [1_000_000.0]
    now = lambda: clock[0]               # noqa: E731
    ups(SESSION, NEUTRAL_PROMPT, now=now)
    clock[0] += PS.PROMPT_STATE_TTL_SECONDS - 1
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, now=now, prompt=None)
    assert fake.count() == 1 and log['gate']['status'] == 'deny'
    clock[0] += 2
    out, log, fake, _, _ = run(payload(query=QUERY + ' other'), env(**GATE_ON), deny_fake(), tmp_path, now=now,
                               prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_expired'


def test_explicit_flag_expires_after_ttl_and_is_then_kept_anyway(tmp_path):
    clock = [1_000_000.0]
    now = lambda: clock[0]               # noqa: E731
    ups(SESSION, A1_PROMPT, now=now)
    clock[0] += PS.PROMPT_STATE_TTL_SECONDS + 1
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, now=now, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_expired'


@pytest.mark.parametrize('content', ['{not json', '[]', json.dumps({'v': 1, 't': 'x', 'explicit': False, 's': 'a'}),
                                     json.dumps({'v': 2, 't': 1.0, 'explicit': False, 's': 'a', 'p': None}), 'x' * 5000])
def test_unreadable_prompt_state_keeps_without_a_call(tmp_path, content):
    st = PS.PromptState(PROMPT_DIR)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with open(st.path(SESSION), 'w', encoding='utf-8') as f:
        f.write(content)
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_error'


def test_future_timestamp_counts_as_error(tmp_path):
    ups(SESSION, NEUTRAL_PROMPT, now=lambda: time.time() + PS.FUTURE_SKEW_SECONDS + 60)
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_error'


def test_new_session_and_session_mismatch_keep_without_a_call(tmp_path):
    ups('session-A', NEUTRAL_PROMPT)
    out, log, fake, _, _ = run(payload(session='session-B'), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_no_prompt_state'
    st = PS.PromptState(PROMPT_DIR)
    os.replace(st.path('session-A'), st.path('session-B'))         # a state that belongs to another session
    out, log, fake, _, _ = run(payload(session='session-B'), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_mismatch'


def test_sessions_do_not_share_state(tmp_path):
    ups('session-A', A1_PROMPT)
    ups('session-B', NEUTRAL_PROMPT)
    out, log, fake, _, _ = run(payload(session='session-A'), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 0 and log['gate']['status'] == 'explicit_request_keep'
    out, log, fake, _, _ = run(payload(session='session-B'), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 1 and log['gate']['status'] == 'deny'


@pytest.mark.parametrize('follow_up', [
    'continue', 'yes, go ahead', '계속', '그걸로 해줘', 'ok', '네 진행해 주세요',
    # ambiguous follow-ups longer than a few words (cleared under the earlier 40-character rule): kept now
    'Do the same thing for the number of hours in a day, please, and keep it brief.',
    'And how many weeks are there in a year? Same format as before, one sentence please.',
    'Thanks. Could you also tell me how many months have exactly thirty days in them?',
    '그럼 한 달은 며칠인지도 같은 방식으로 한 문장으로 짧게 알려줄 수 있을까요? 부탁해요.',
    '좋아요, 이번에는 하루가 몇 시간인지도 같은 형식으로 정리해서 답해 주세요. 감사합니다.'])
def test_follow_up_without_new_task_marker_keeps_an_explicit_flag(tmp_path, follow_up):
    assert not PS.explicit_request(follow_up) and not PS.starts_new_task(follow_up)
    ups(SESSION, A1_PROMPT)
    log_ups, _ = ups(SESSION, follow_up)
    assert log_ups['status'] == 'follow_up_kept'
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'explicit_request_keep'


def test_kept_flag_still_expires_at_the_original_ttl(tmp_path):
    clock = [1_000_000.0]
    now = lambda: clock[0]               # noqa: E731
    ups(SESSION, A1_PROMPT, now=now)
    clock[0] += PS.PROMPT_STATE_TTL_SECONDS - 10
    assert ups(SESSION, NEUTRAL_PROMPT, now=now)[0]['status'] == 'follow_up_kept'      # does not refresh the time
    clock[0] += 20
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, now=now, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_state_expired'
    assert ups(SESSION, NEUTRAL_PROMPT, now=now)[0]['status'] == 'cleared'             # expired flag: a new state


@pytest.mark.parametrize('prompt', [
    'New question: tell me how zebra lanterns glow at the synthetic festival, in detail.',
    'Different topic. Explain how zebra lanterns glow at the synthetic festival in detail.',
    '새 질문입니다. 축제에서 얼룩말 등불이 어떻게 빛나는지 자세히 설명해 주세요.',
    '다른 주제로 넘어가서, 축제의 얼룩말 등불이 어떻게 빛나는지 설명해 주세요.'])
def test_clear_new_task_marker_clears_the_flag(tmp_path, prompt):
    assert PS.starts_new_task(prompt) and not PS.explicit_request(prompt)
    ups(SESSION, A1_PROMPT)
    assert ups(SESSION, prompt)[0]['status'] == 'cleared'
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 1 and log['gate']['status'] == 'deny'


def test_no_earlier_flag_prompt_without_signal_writes_not_explicit(tmp_path):
    assert ups('session-F', 'continue')[0]['status'] == 'cleared'
    assert ups('session-G', NEUTRAL_PROMPT)[0]['status'] == 'cleared'


def test_lock_failure_never_leaves_a_not_explicit_state(tmp_path, monkeypatch):
    ups(SESSION, NEUTRAL_PROMPT)
    monkeypatch.setattr(PS.PromptState, '_lock', lambda self, s: None)
    assert ups(SESSION, 'Another plain prompt without any of the signal words, long enough.')[0]['status'] == \
        'lock_failed'
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 0 and log['gate']['status'] == 'keep_no_prompt_state'
    assert ups(SESSION, A1_PROMPT)[0]['status'] == 'lock_failed'                         # a signal is still recorded
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 0 and log['gate']['status'] == 'explicit_request_keep'


def test_concurrent_prompts_do_not_drop_a_flag(tmp_path):
    """Threads racing an explicit prompt against prompts without a signal: once the explicit prompt has been
    applied, no later no-signal prompt (without a new-task marker) can clear it."""
    import threading
    for _ in range(20):
        st = PS.PromptState(str(tmp_path / 'race'))
        st.forget(SESSION)
        st.on_prompt(SESSION, A1_PROMPT)
        ts = [threading.Thread(target=st.on_prompt, args=(SESSION, NEUTRAL_PROMPT)) for _ in range(8)]
        ts.append(threading.Thread(target=st.on_prompt, args=(SESSION, A1_PROMPT)))
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert st.read(SESSION) == 'explicit'


def test_prompt_id_must_match_when_both_carry_it(tmp_path):
    ups(SESSION, NEUTRAL_PROMPT, prompt_id='prompt-1')
    p = payload()
    p['prompt_id'] = 'prompt-2'
    out, log, fake, _, _ = run(p, env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_prompt_id_mismatch'
    p['prompt_id'] = 'prompt-1'
    out, log, fake, _, _ = run(p, env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 1 and log['gate']['status'] == 'deny'
    out, log, fake, _, _ = run(payload(query=QUERY + ' x'), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 1                       # no prompt_id in the PreToolUse input: session + age only


def test_prompt_state_write_failure_removes_old_state(tmp_path, monkeypatch):
    ups(SESSION, NEUTRAL_PROMPT)
    real = os.replace
    monkeypatch.setattr(PS.os, 'replace', lambda a, b: (_ for _ in ()).throw(OSError('disk full')))
    log_ups, _ = ups(SESSION, 'Something else entirely without any signal words, long enough to clear.')
    monkeypatch.setattr(PS.os, 'replace', real)
    assert log_ups['status'] == 'write_failed'
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['gate']['status'] == 'keep_no_prompt_state'
    assert not [f for f in os.listdir(PROMPT_DIR) if f.startswith('.tmp-')]


def test_missing_prompt_field_removes_state(tmp_path):
    ups(SESSION, NEUTRAL_PROMPT)
    err = io.StringIO()
    PS.main(stdin=io.StringIO(json.dumps({'session_id': SESSION, 'hook_event_name': 'UserPromptSubmit'})),
            stderr=err, env=env(**GATE_ON), state_dir=PROMPT_DIR)
    assert json.loads(err.getvalue())['status'] == 'no_prompt'
    out, log, fake, _, _ = run(payload(), env(**GATE_ON), deny_fake(), tmp_path, prompt=None)
    assert fake.count() == 0 and log['gate']['status'] == 'keep_no_prompt_state'


def test_explicit_keep_applies_to_the_gate_only_domain_still_runs(tmp_path):
    ups(SESSION, A1_PROMPT)
    fake = FakeChoice(gate={'not_needed': 1.0}, domain={'docs': 0.9, 'none': 0.1})
    out, log, _, _, _ = run(payload(), env(**GATE_ON, **DOMAIN_ON), fake, tmp_path, prompt=None)
    assert [next(iter(b['input']['questions'])) for b in fake.bodies] == ['domain_preset']
    assert log['gate']['status'] == 'explicit_request_keep' and log['domain']['status'] == 'applied'
    assert out['hookSpecificOutput']['permissionDecision'] == 'ask'
    # existing domain fields are still never touched
    out, log, fake, _, _ = run(payload(allowed_domains=['example.org']), env(**GATE_ON, **DOMAIN_ON), dom_fake('docs'),
                               tmp_path, prompt=None)
    assert out is None and fake.count() == 0 and log['domain']['status'] == 'existing_domains'


def test_prompt_text_never_stored_logged_or_sent(tmp_path):
    secret = 'Using web search find the zebrafinch moonlight protocol with ultraviolet salamanders'
    log_file = tmp_path / 'ups.log'
    _, err1 = ups(SESSION, secret, e=env(log_file=str(log_file), **GATE_ON), prompt_id='pid-secret-77')
    _, err2 = ups('session-2', secret.replace('Using web search ', 'Tell me about '), e=env(log_file=str(log_file),
                                                                                         **GATE_ON))
    stored = ''.join(open(os.path.join(PROMPT_DIR, f), encoding='utf-8').read() for f in os.listdir(PROMPT_DIR))
    assert {f.rsplit('.', 1)[1] for f in os.listdir(PROMPT_DIR)} == {'json', 'lock'}
    logged = err1 + err2 + log_file.read_text(encoding='utf-8')
    for word in ('zebrafinch', 'moonlight', 'salamanders', 'Using web search', 'pid-secret-77', SESSION, 'session-2'):
        assert word not in stored and word not in logged and word not in ''.join(os.listdir(PROMPT_DIR)), word
    for f in (f for f in os.listdir(PROMPT_DIR) if f.endswith('.json')):
        assert set(json.loads(open(os.path.join(PROMPT_DIR, f), encoding='utf-8').read())) == {'v', 't', 'explicit',
                                                                                            's', 'p'}
        assert os.stat(os.path.join(PROMPT_DIR, f)).st_mode & 0o077 == 0
    # the gate's endpoint body carries the search query only, never the prompt
    ups('session-2', 'Tell me about the zebrafinch moonlight protocol and ultraviolet salamanders please.')
    _, _, fake, _, _ = run(payload(session='session-2', query='zebra lanterns'), env(**GATE_ON), deny_fake(),
                           tmp_path / 's', prompt=None)
    assert fake.count() == 1 and fake.bodies[0]['input']['state'] == {'search_query': 'zebra lanterns'}
    assert 'zebrafinch' not in json.dumps(fake.bodies)
