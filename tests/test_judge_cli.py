"""Model-free boundary checks of the user-input JSONL CLI (judge_adapter.cli). Fake transports only; no key, no network."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from judge_adapter import fakes
from judge_adapter import cli as J
from judge_adapter import keys as K
from judge_adapter import core
from judge_adapter import prompt as P
from judge_adapter.gate import (JOB_RESERVATION_SECONDS, IDLE_ALLOWANCE_SECONDS,
                                                  haiku_reservation_usd, haiku_usd)

EXAMPLE = J.EXAMPLES_DIR / 'pairs.example.jsonl'
MARKERS = J.EXAMPLES_DIR / 'pairs.fake_markers.jsonl'
PY = sys.executable
# the subprocess imports the same judge_adapter as this test run (installed wheel or src/ checkout)
SUB_ENV = {**os.environ, 'PYTHONPATH': os.pathsep.join(
    [str(Path(J.__file__).resolve().parents[1])] + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])}


def line(rid='r1', a='answer a', b='answer b', q=('Question?',), ida='id-a', idb='id-b', **extra):
    obj = {'id': rid, 'question_turns': list(q), 'response_a': {'id': ida, 'turns': [a] if isinstance(a, str) else a},
           'response_b': {'id': idb, 'turns': [b] if isinstance(b, str) else b}}
    obj.update(extra)
    return obj


def write(tmp_path, objs, name='in.jsonl'):
    p = tmp_path / name
    p.write_text(''.join((o if isinstance(o, str) else json.dumps(o)) + '\n' for o in objs), encoding='utf-8')
    return p


def run(tmp_path, objs, backend='haiku_direct', mode='single', out='out', **kw):
    fake = fakes.fake_for(backend)
    s = J.run_file(write(tmp_path, objs), tmp_path / out, transport='fake', backend=backend, mode=mode, fake=fake, **kw)
    rows = [json.loads(x) for x in (tmp_path / out / 'results.jsonl').read_text().splitlines()]
    return s, rows, fake


def errors_of(tmp_path, objs):
    with pytest.raises(J.InputRejected) as e:
        J.load_requests(write(tmp_path, objs))
    return e.value.errors


@pytest.fixture
def no_key_no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('key read or network attempted')
    monkeypatch.setattr(K, 'dotenv_key', boom)
    monkeypatch.setattr(K, 'env_key', boom)
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


# --- input validation --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('obj,needle', [
    ({k: v for k, v in line().items() if k != 'response_b'}, "missing field 'response_b'"),
    ({**line(), 'response_a': {'id': 'x'}}, 'missing field response_a.turns'),
    (line(ida='same', idb='same'), 'are the same'),
    (line(q=('Q1', 'Q2'), a=['a1', 'a2'], b=['b1']), 'turn count mismatch'),
    (line(extra_field=1), "unknown field 'extra_field'"),
    ({**line(), 'response_a': {'id': 'x', 'turns': ['a'], 'model': 'm'}}, "unknown field response_a.'model'"),
    (line(q=('Q1', 'Q2'), a=['a1', 'a2'], b=['b1', 'b2'], turn=1), 'judged turn is always the last turn'),
    (line(q=('Q1', 'Q2', 'Q3'), a=['1', '2', '3'], b=['1', '2', '3']), 'at most 2 are supported'),
    (line(q=('',)), 'question_turns[0] is empty'),
    (line(rid=''), 'id must be a non-empty string'),
    (line(ida='tie'), 'reserved'),
    (line(turn=True), 'turn must be an integer'),
    ({**line(), 'question_turns': 'Q?'}, 'must be a non-empty list'),
    ('{"id": "x", "id": "y"}', 'duplicate key'),
    ('{not json', 'not valid JSON'),
    ('[1, 2]', 'one JSON object'),
])
def test_each_input_error_class_is_rejected_with_line_number(tmp_path, obj, needle):
    errs = errors_of(tmp_path, [line('ok-0'), obj])
    assert any(no == 2 and needle in msg for no, msg in errs), errs
    assert all(no == 2 for no, _ in errs)


def test_duplicate_request_id_and_empty_line(tmp_path):
    errs = errors_of(tmp_path, [line('r1'), line('r2'), '', line('r1')])
    assert (3, 'empty line (every line must be one request)') in errs
    assert any(no == 4 and "duplicate request id 'r1' (first at line 1)" in m for no, m in errs)


@pytest.mark.parametrize('field,hint', [('criteria', 'option ids'), ('instructions', "use 'rubric'"),
                                        ('reference_answer', "did you mean 'reference'"), ('judge_prompt', 'fixed')])
def test_criteria_instructions_and_near_miss_fields_are_explicit_errors(tmp_path, field, hint):
    """rubric / reference are supported fields (tests/test_user_criteria.py); these names are not."""
    errs = errors_of(tmp_path, [line(**{field: 'Be strict.'})])
    assert any(f"field {field!r} is not supported" in m and hint in m for _, m in errs)
    nested = line()
    nested['response_b'][field] = 'gold'
    errs = errors_of(tmp_path, [nested])
    assert any(f"'response_b.{field}' is not supported" in m for _, m in errs)


def test_bad_file_sends_nothing_and_writes_nothing(tmp_path, no_key_no_net):
    p = write(tmp_path, [line('r1'), line('r1')])
    with pytest.raises(J.InputRejected):
        J.run_file(p, tmp_path / 'out', transport='fake')
    assert not (tmp_path / 'out').exists()
    assert J.main(['run', str(p), '--out', str(tmp_path / 'out'), '--transport', 'fake']) == 2


def test_turn_mapping(tmp_path):
    (no, r), = J.load_requests(write(tmp_path, [line(q=('Q1', 'Q2'), a=['a1', 'a2'], b=['b1', 'b2'])]))
    assert (r.turn, r.question_turns, r.response_a.turns, r.response_b.turns) == (2, ('Q1', 'Q2'), ('a1', 'a2'),
                                                                                  ('b1', 'b2'))
    sem = P.semantic(r.question_turns, r.response_a.turns, r.response_b.turns, r.turn)
    assert sem['state']['question'] == 'Q2' and 'a1' in sem['state']['assistant_1_conversation']


# --- backend, directions, verdicts ---------------------------------------------------------------------------------
def test_backend_default_and_explicit_choice_endpoint(tmp_path):
    s, rows, fake = run(tmp_path, [line()])
    assert s['backend'] == 'haiku_direct' and fake.count('POST', '/v1/messages') == 1
    s, rows, fake = run(tmp_path, [line()], backend='choice_endpoint', out='o2')
    assert s['backend'] == 'choice_endpoint' and fake.count('POST', '/runsync') == 1
    # the command line: no --backend -> haiku_direct; choice_endpoint only when named
    assert J.main(['run', str(EXAMPLE), '--out', str(tmp_path / 'c1'), '--transport', 'fake']) == 0
    assert json.loads((tmp_path / 'c1' / 'summary.json').read_text())['backend'] == 'haiku_direct'
    assert J.main(['run', str(EXAMPLE), '--out', str(tmp_path / 'c2'), '--transport', 'fake', '--backend', 'choice_endpoint']) == 0
    assert json.loads((tmp_path / 'c2' / 'summary.json').read_text())['model'] == 'unspecified'


def test_two_direction_restores_response_ids(tmp_path):
    s, rows, fake = run(tmp_path, [line(a='plain', b='[[GOOD]] better')], mode='two_direction')
    r = rows[0]
    assert (r['outcome'], r['verdict'], r['winner_id']) == ('ok', 'B', 'id-b')
    assert [c['slot_ids'] for c in r['calls']] == [{'1': 'id-a', '2': 'id-b'}, {'1': 'id-b', '2': 'id-a'}]
    assert [c['label'] for c in r['calls']] == ['id-b', 'id-b'] and fake.count() == 2


def test_tie_inconsistent_and_error_are_distinct(tmp_path):
    s, rows, _ = run(tmp_path, [line('t'), line('p', a='FAKE:POSITION x'), line('f', a='FAKE:FORMAT x')],
                     mode='two_direction')
    got = [(r['outcome'], r['tie_kind'], (r['failure'] or {}).get('reason')) for r in rows]
    assert got == [('tie', 'explicit', None), ('tie', 'inconsistent', None), ('error', None, 'format_error')]
    assert s['counts'] == {'requests': 3, 'ok': 0, 'tie': 2, 'tie_explicit': 1, 'tie_inconsistent': 1, 'error': 1,
                           'unrun': 0}


def test_second_direction_skipped_when_first_fails(tmp_path):
    s, rows, fake = run(tmp_path, [line(a='FAKE:FORMAT x')], mode='two_direction')
    assert rows[0]['submissions'] == 1 and rows[0]['failure']['direction'] == 'AB' and fake.count() == 1
    assert s['submissions']['planned_max'] == 2 and s['submissions']['actual'] == 1


# --- run-wide limits -----------------------------------------------------------------------------------------------
def test_max_submissions_is_run_wide_and_two_direction_reserves_two(tmp_path):
    s, rows, fake = run(tmp_path, [line('r1'), line('r2'), line('r3')], mode='two_direction', max_submissions=3)
    assert [r['outcome'] for r in rows] == ['tie', 'unrun', 'unrun']
    assert rows[1]['submissions'] == 0 and 'would be exceeded' in rows[1]['failure']['detail']
    assert fake.count() == 2 and s['submissions'] == {**s['submissions'], 'planned_max': 6, 'actual': 2}


def test_haiku_cap_is_run_wide_not_per_row(tmp_path):
    objs = [line(f'r{i}') for i in range(3)]
    reqs = J.load_requests(write(tmp_path, objs, 'probe.jsonl'))
    sems = [P.semantic(r.question_turns, r.response_a.turns, r.response_b.turns, 1) for _, r in reqs]
    res = haiku_reservation_usd(P.haiku_body(sems[0]))
    spent = haiku_usd({'input_tokens': len(P.haiku_text(sems[0])) // 4, 'output_tokens': 3})   # fake usage
    cap = res + 1.5 * spent      # per-row reset would admit all three; run-wide admits two
    s, rows, fake = run(tmp_path, objs, budget_cap_usd=cap)
    assert [r['outcome'] for r in rows] == ['tie', 'tie', 'unrun'] and fake.count() == 2
    assert 'cap' in s['limits']['stopped'] and s['limits']['budget_cap_usd'] == cap


def test_choice_cap_is_run_wide(tmp_path):
    rate = core.FAKE_CHOICE_USD_PER_SECOND                  # fake placeholder rate
    first = (JOB_RESERVATION_SECONDS + IDLE_ALLOWANCE_SECONDS) * rate     # 721 s: first job fits
    second = (0.12 + 0.85 + 1 + IDLE_ALLOWANCE_SECONDS + JOB_RESERVATION_SECONDS) * rate   # settled + idle + new
    cap = (first + second) / 2
    s, rows, fake = run(tmp_path, [line('r1'), line('r2')], backend='choice_endpoint', budget_cap_usd=cap, clock=lambda: 1000.0)
    assert [r['outcome'] for r in rows] == ['tie', 'unrun'] and fake.count('POST', '/runsync') == 1


def test_identical_content_different_ids_both_run(tmp_path):
    s, rows, fake = run(tmp_path, [line('same-1', b='[[GOOD]] b'), line('same-2', b='[[GOOD]] b')])
    assert [r['outcome'] for r in rows] == ['ok', 'ok'] and fake.count() == 2
    assert rows[0]['semantic_sha256'] == rows[1]['semantic_sha256']
    gate = json.loads((tmp_path / 'out' / 'gate.json').read_text())
    assert gate['submitted_call_keys'] == ['same-1:AB', 'same-2:AB']


def test_same_request_and_direction_never_submitted_twice(tmp_path):
    from judge_adapter import AdapterConfig, build_backend
    fake = fakes.FakeAnthropic()
    be = build_backend(AdapterConfig(transport='fake'), transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY)
    gate = J.RunGate(be, 'single', None, 10)
    (_, r), = J.load_requests(write(tmp_path, [line()]))
    gate.begin(r)
    sem = P.semantic(r.question_turns, r.response_a.turns, r.response_b.turns, 1)
    to = {'opt_1': 'id-a', 'opt_2': 'id-b', 'opt_tie': 'tie'}
    assert gate.call(sem, to)['outcome'] == 'ok'
    again = gate.call(sem, to)
    assert again['outcome'] == 'not_submitted' and again['submissions'] == 0 and fake.count() == 1


def test_failure_stop_leaves_later_requests_unrun_and_counted(tmp_path):
    s, rows, fake = run(tmp_path, [line('a', b='[[GOOD]] b'), line('h', a='FAKE:HTTP500'), line('c')])
    assert [r['outcome'] for r in rows] == ['ok', 'error', 'unrun']
    assert s['counts']['requests'] == 3 and s['error_reasons'] == {'http': 1}
    ce = s['cost_estimate']
    assert ce['usd_all_submitted_requests'] is None and ce['requests_with_unknown_cost'] == 1
    assert ce['includes_failed_requests'] and ce['is_estimate'] and ce['mock']


# --- real-run guards, output dir, keys, imports --------------------------------------------------------------------
@pytest.mark.parametrize('kw,missing', [({}, '--allow-real'), ({'allow_real': True}, '--max-submissions'),
                                        ({'allow_real': True, 'max_submissions': 2}, '--budget-cap-usd')])
def test_real_run_needs_allow_real_max_submissions_and_cap(tmp_path, no_key_no_net, kw, missing):
    with pytest.raises(J.RunRefused, match=missing):
        J.run_file(EXAMPLE, tmp_path / 'out', transport='real', **kw)
    assert not (tmp_path / 'out').exists()


def test_overwrite_refused_by_default_and_overwrite_removes_only_own_files(tmp_path):
    run(tmp_path, [line()])
    (tmp_path / 'out' / 'mine.txt').write_text('keep')
    with pytest.raises(J.RunRefused, match='already has results'):
        run(tmp_path, [line()])
    s, rows, _ = run(tmp_path, [line('x'), line('y')], overwrite=True)
    assert len(rows) == 2 and (tmp_path / 'out' / 'mine.txt').read_text() == 'keep'
    assert len((tmp_path / 'out' / 'trace' / 'judge_trace.jsonl').read_text().splitlines()) == 2


def test_no_key_read_or_network_in_validate_and_fake(tmp_path, no_key_no_net, capsys):
    assert J.main(['validate', str(EXAMPLE)]) == 0
    assert J.main(['validate', str(EXAMPLE), '--backend', 'choice_endpoint', '--mode', 'two_direction']) == 0
    for backend in ('haiku_direct', 'choice_endpoint'):
        assert J.main(['run', str(MARKERS), '--out', str(tmp_path / backend), '--transport', 'fake',
                       '--backend', backend, '--mode', 'two_direction']) == 0
    assert J.main(['run', str(EXAMPLE), '--out', str(tmp_path / 'r'), '--transport', 'real']) == 3
    assert 'No call made, no key read' in capsys.readouterr().out


def test_raw_outputs_have_no_key(tmp_path):
    s, rows, _ = run(tmp_path, [line(a='FAKE:LEAK x')])
    raw = (tmp_path / 'out' / 'trace' / 'judge_raw.jsonl').read_text()
    assert fakes.DUMMY_KEY not in raw and '[redacted]' in raw
    assert s['files']['raw_outputs']['sha256'] and rows[0]['calls'][0]['raw_output_sha256']


BLOCKER = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in ('torch', 'fastchat', 'transformers', 'accelerate'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, Block())
'''


def test_blocker_really_blocks_torch():
    p = subprocess.run([PY, '-B', '-c', BLOCKER + 'import torch'], capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and 'blocked: torch' in p.stderr


def test_cli_runs_without_torch_or_fastchat_from_another_cwd(tmp_path):
    """torch and fastchat imports are blocked; the CLI module is started from an unrelated directory."""
    code = BLOCKER + f'''
import runpy, json
sys.argv = ['judge-adapter', 'run', {str(MARKERS)!r}, '--out', {str(tmp_path / 'o')!r}, '--transport', 'fake',
            '--mode', 'two_direction']
try:
    runpy.run_module('judge_adapter.cli', run_name='__main__', alter_sys=True)
except SystemExit as e:
    assert e.code == 0, e.code
bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'fastchat', 'transformers', 'accelerate'))
print(json.dumps(bad))
'''
    p = subprocess.run([PY, '-B', '-c', code], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=SUB_ENV)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip().splitlines()[-1] == '[]'
    assert (tmp_path / 'o' / 'summary.json').exists()


def test_import_loads_no_torch_or_fastchat(tmp_path):
    code = ('import sys, json; import judge_adapter, judge_adapter.cli, judge_adapter.hosts; '
            'print(json.dumps(sorted(m for m in sys.modules if m.split(".")[0] in '
            '("torch", "fastchat", "transformers", "accelerate"))))')
    p = subprocess.run([PY, '-B', '-c', code], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=SUB_ENV)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == '[]'


def test_examples_command_lists_bundled_files(capsys):
    assert J.main(['examples']) == 0
    out = capsys.readouterr().out.split()
    assert {Path(x).name for x in out} >= {'pairs.example.jsonl', 'pairs.fake_markers.jsonl',
                                          'pairs.rubric_reference.example.jsonl', 'config.fake.json'}
    assert all(Path(x).is_file() for x in out)


# --- standard command end to end -----------------------------------------------------------------------------------
def test_standard_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / 'demo'
    assert J.main(['run', str(MARKERS), '--out', str(out), '--transport', 'fake', '--mode', 'two_direction']) == 0
    text = capsys.readouterr().out
    assert text.startswith('[MOCK]') and 'Not the bill' in text
    rows = [json.loads(x) for x in (out / 'results.jsonl').read_text().splitlines()]
    s = json.loads((out / 'summary.json').read_text())
    assert len(rows) == len(MARKERS.read_text().splitlines()) == s['counts']['requests'] == 8
    assert [r['outcome'] for r in rows] == ['ok', 'tie', 'tie', 'error', 'ok', 'ok', 'error', 'unrun']
    assert sum(s['counts'][k] for k in ('ok', 'tie', 'error', 'unrun')) == s['counts']['requests']
    assert s['mock'] is True and all(r['mock'] for r in rows) and s['cost_estimate']['mock'] is True
    assert s['submissions']['planned_max'] == 16 and s['submissions']['actual'] == s['submissions']['gate_counted'] == 12
    assert s['files']['results']['sha256'] and Path(s['files']['raw_outputs']['path']).is_absolute()
    assert 'success_rate' not in json.dumps(s)
    assert s['requests_with_unknown_usage'] == 1 and 'MOCK' in s['usage_note'] and 'MOCK' in s['mock_note']
    assert all(r['run_id'] == s['run_id'] and r['backend'] == 'haiku_direct' and r['transport'] == 'fake'
               for r in rows)
    raw = [json.loads(x) for x in (out / 'trace' / 'judge_raw.jsonl').read_text().splitlines()]
    assert raw and all(x['transport'] == 'fake' and x['run_id'] == s['run_id'] and 'mock_note' in x for x in raw)
