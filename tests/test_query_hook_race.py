"""Search-gate retry state under REAL separate processes (subprocesses started together behind a file barrier).

- the full hook entry point in N processes, same session / tool / query, fake endpoint answering "not needed":
  at most one deny (exactly one here), the others print nothing;
- claims racing with a TTL refresh of an expired record and with concurrent pruners: one claim wins and the new,
  fresh record survives;
- pruners looping while new keys are claimed: every fresh record survives;
- different sessions: each gets its own deny.
No network, no keys; the fake endpoint is an in-process httpx.MockTransport inside each child."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from judge_adapter import query_hook as QH

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform.startswith('win'), reason='fcntl lock: POSIX only (Windows never denies)')

CHILD = r'''
import io, json, os, sys, time
sys.path.insert(0, os.environ['SRC'])
import httpx
from judge_adapter import query_hook as QH
role, state, go, arg = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
from judge_adapter import prompt_signal as PS
pdir = state + '-prompt'
if role == 'hook':
    PS.PromptState(pdir).record(arg, False)     # the user's prompt had no explicit-search signal
while not os.path.exists(go):
    time.sleep(0.001)
if role == 'hook':
    def handle(request):
        body = json.loads(request.content)
        (qid, q), = body['input']['questions'].items()
        ids = list(q['criteria'])
        probs = {ids[0]: 0.02, ids[1]: 0.97, ids[2]: 0.01}
        return httpx.Response(200, json={'id': 'j', 'status': 'COMPLETED', 'delayTime': 1, 'executionTime': 5,
                                         'output': {'model': 'fake', 'answers': {qid: {'type': 'choice',
                                         'choice': ids[1], 'probabilities': probs}}}})
    env = {'CLAUDE_PLUGIN_OPTION_RUNPOD_API_KEY': 'rpa_FAKE', 'CLAUDE_PLUGIN_OPTION_CHOICE_ENDPOINT_ID': 'fakeendpoint',
           'CLAUDE_PLUGIN_OPTION_USD_PER_SECOND': '0.0004', 'CLAUDE_PLUGIN_OPTION_SEARCH_GATE_ENABLED': 'true'}
    p = {'session_id': arg, 'hook_event_name': 'PreToolUse', 'tool_name': 'WebSearch',
         'tool_input': {'query': 'how do zebra lanterns glow'}}
    out, err = io.StringIO(), io.StringIO()
    QH.main(stdin=io.StringIO(json.dumps(p)), stdout=out, stderr=err, env=env,
            transport=httpx.MockTransport(handle), exit_fn=lambda c: None, state_dir=state, prompt_state_dir=pdir,
            toggle_file=os.path.join(pdir, 'no-toggles.json'))
    log = json.loads(err.getvalue().strip().splitlines()[-1])
    print(json.dumps({'out': out.getvalue(), 'gate': log.get('gate', {}).get('status')}))
elif role == 'claim':
    print(json.dumps({'claim': QH.RetryState(state).claim(arg)}))
elif role == 'prune':
    st, end = QH.RetryState(state), time.monotonic() + float(arg)
    n = 0
    while time.monotonic() < end:
        st.prune(); n += 1
    print(json.dumps({'prunes': n}))
'''


def launch(tmp_path, jobs):
    """jobs: [(role, arg)] -> parsed stdout per job, all released together."""
    script, go, state = tmp_path / 'child.py', tmp_path / 'go', tmp_path / 'state'
    script.write_text(CHILD)
    env = {**os.environ, 'SRC': str(ROOT / 'src')}
    for k in list(env):
        if k.startswith(('CLAUDE_PLUGIN_OPTION_', 'JUDGE_ADAPTER_HOOK_')):
            del env[k]
    procs = [subprocess.Popen([sys.executable, str(script), role, str(state), str(go), arg], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, env=env) for role, arg in jobs]
    time.sleep(0.8)                      # let every child import and wait at the barrier
    go.write_text('go')
    res = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        res.append(json.loads(out.strip().splitlines()[-1]))
    return res, state


@pytest.mark.parametrize('round_', range(3))
def test_full_hook_processes_same_key_at_most_one_deny(tmp_path, round_):
    res, state = launch(tmp_path, [('hook', 'session-race')] * 8)
    denies = [r for r in res if r['out']]
    assert len(denies) == 1
    assert json.loads(denies[0]['out'])['hookSpecificOutput']['permissionDecision'] == 'deny'
    assert all(r['gate'] in ('deny', 'already_denied_concurrently', 'retry_allowed') for r in res)
    assert len([f for f in os.listdir(state) if f.endswith('.json')]) == 1


def test_different_sessions_are_independent(tmp_path):
    res, state = launch(tmp_path, [('hook', f'session-{i}') for i in range(6)])
    assert all(r['out'] and r['gate'] == 'deny' for r in res)
    assert len([f for f in os.listdir(state) if f.endswith('.json')]) == 6


@pytest.mark.parametrize('round_', range(3))
def test_ttl_refresh_racing_with_claims_and_pruners_keeps_new_record(tmp_path, round_):
    key = QH.RetryState.key('s', 'WebSearch', 'expired query')
    state = tmp_path / 'state'
    state.mkdir()
    (state / (key + '.json')).write_text(json.dumps({'v': 1, 't': time.time() - QH.GATE_RETRY_TTL_SECONDS - 100}))
    res, _ = launch(tmp_path, [('claim', key)] * 6 + [('prune', '1.5')] * 3)
    claims = [r['claim'] for r in res if 'claim' in r]
    assert claims.count(True) == 1 and claims.count(False) == 5
    assert QH.RetryState(str(state)).denied_recently(key) is True          # the new record survived the pruners
    t = json.loads((state / (key + '.json')).read_text())['t']
    assert time.time() - t < 60


def test_pruners_never_remove_fresh_records(tmp_path):
    state = tmp_path / 'state'
    state.mkdir()
    stale = [QH.RetryState.key('s', 'WebSearch', f'stale {i}') for i in range(20)]
    for k in stale:
        (state / (k + '.json')).write_text(json.dumps({'v': 1, 't': time.time() - QH.GATE_RETRY_TTL_SECONDS - 100}))
    fresh = [QH.RetryState.key('s', 'WebSearch', f'fresh {i}') for i in range(12)]
    res, _ = launch(tmp_path, [('claim', k) for k in fresh] + [('claim', k) for k in stale[:4]] +
                    [('prune', '1.5')] * 3)
    assert all(r['claim'] is True for r in res if 'claim' in r)
    st = QH.RetryState(str(state))
    for k in fresh + stale[:4]:
        assert st.denied_recently(k) is True, k
    for k in stale[4:]:
        assert not (state / (k + '.json')).exists() or st.denied_recently(k) is False
