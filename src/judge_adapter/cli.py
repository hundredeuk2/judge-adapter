# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""JSONL judge CLI: your question + two responses -> the adapter -> results JSONL + summary.

No torch, no FastChat. Runs from any directory (paths are resolved to absolute paths).

  judge-adapter validate PAIRS.jsonl [--backend choice_endpoint] [--mode two_direction] [--max-submissions N]
  judge-adapter run PAIRS.jsonl --out DIR --transport fake [--backend choice_endpoint] [--mode two_direction]
  judge-adapter run PAIRS.jsonl --out DIR --transport real --allow-real --max-submissions N --budget-cap-usd X
  judge-adapter examples                       # paths of the bundled synthetic example files

Input: one JSON object per line (schema in README). The whole file is validated before any call; one bad line
rejects the whole file with line numbers. Optional ``rubric`` (string) and ``reference`` (one entry per question
turn) use the versioned user-criteria prompt (user_prompt.py); without them the fixed default prompt is sent byte for
byte. Options opt_1 / opt_2 / opt_tie, the strict parser and the generation settings are fixed. ``criteria`` /
``instructions`` are an explicit error (no custom option ids, no score scales, no alias). Size limits are UTF-8 bytes
(user_prompt.LIMITS); nothing is truncated.

Run-wide limits (never reset per request): --max-submissions and the client-side cost cap of gate.RunGate.
Retries 0, fallback 0. Import, validate and fake runs read no key and open no network connection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from collections import Counter
from pathlib import Path

from . import __version__

from . import contract as C
from . import prompt as P
from . import user_prompt as U
from .contract import AdapterConfig, JudgeRequest, Response
from .core import JudgeAdapter, TraceWriter, build_backend, scrubber_of
from .gate import JOB_RESERVATION_SECONDS, IDLE_ALLOWANCE_SECONDS, RunGate, haiku_reservation_usd  # noqa: F401

SCHEMA = 'judge-adapter-pairs/v2'       # v2: optional rubric + reference (v1 files stay valid)
TOP_FIELDS = ('id', 'question_turns', 'response_a', 'response_b', 'turn', 'rubric', 'reference')
REQUIRED_TOP = ('id', 'question_turns', 'response_a', 'response_b')
RESPONSE_FIELDS = ('id', 'turns')
UNSUPPORTED_EVAL_FIELDS = {
    'instructions': "use 'rubric' for your evaluation standard (no instructions alias, so there is no precedence "
                    "question); the comparison framing and output format are fixed",
    'criteria': 'custom option ids and score scales are not supported; options are fixed (opt_1 / opt_2 / opt_tie)',
    'judge_prompt': "the judge prompt is fixed; put your evaluation standard in 'rubric'",
    'system_prompt': "the judge prompt is fixed; put your evaluation standard in 'rubric'",
    'rubrics': "did you mean 'rubric' (one string)?",
    'references': "did you mean 'reference' (one entry per question turn)?",
    'reference_answer': "did you mean 'reference' (one entry per question turn)?",
    'ref_answer': "did you mean 'reference' (one entry per question turn)?",
}
MAX_TURNS = 2                                     # the fixed judge supports turn 1 and turn 2 only
OUTPUT_FILES = ('results.jsonl', 'summary.json', 'gate.json', 'trace/judge_trace.jsonl', 'trace/judge_raw.jsonl')
EXAMPLES_DIR = Path(__file__).resolve().parent / 'examples'
MOCK_NOTE = ('MOCK: fake transport. The fake verdict ignores answer quality; it follows test markers only '
             '([[GOOD]], FAKE:*; see judge_adapter.fakes). Usage numbers are invented (chars / 4). Not a judgment, not a cost.')


# --- input -----------------------------------------------------------------------------------------------------
class InputRejected(Exception):
    def __init__(self, errors):
        super().__init__(f'{len(errors)} input error(s)')
        self.errors = errors                       # [(line_no, message)]; line 0 = whole file


def _no_dup_keys(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError(f'duplicate key {k!r} in one object')
        seen[k] = v
    return seen


def _unsupported_msg(where, name):
    return f'field {where!r} is not supported: {UNSUPPORTED_EVAL_FIELDS[name]}'


def _check_turns(value, where, allow_empty):
    if not isinstance(value, list) or not value:
        return None, [f'{where} must be a non-empty list of strings']
    errs = []
    if len(value) > MAX_TURNS:
        errs.append(f'{where} has {len(value)} turns; at most {MAX_TURNS} are supported (nothing is cut)')
    for i, t in enumerate(value):
        if not isinstance(t, str):
            errs.append(f'{where}[{i}] must be a string')
        elif not allow_empty and not t.strip():
            errs.append(f'{where}[{i}] is empty')
    return (None if errs else len(value)), errs


def check_line(obj):
    """-> (list of error messages, JudgeRequest fields or None)."""
    if not isinstance(obj, dict):
        return ['each line must be one JSON object'], None
    errs = []
    for k in obj:
        if k in UNSUPPORTED_EVAL_FIELDS:
            errs.append(_unsupported_msg(k, k))
        elif k not in TOP_FIELDS:
            errs.append(f'unknown field {k!r} (allowed: {", ".join(TOP_FIELDS)})')
    errs += [f'missing field {k!r}' for k in REQUIRED_TOP if k not in obj]
    rid = obj.get('id')
    if 'id' in obj and not (isinstance(rid, str) and rid.strip()):
        errs.append('id must be a non-empty string')
    n_q = None
    if 'question_turns' in obj:
        n_q, e = _check_turns(obj['question_turns'], 'question_turns', allow_empty=False)
        errs += e
    n_r, ids = {}, {}
    for side in ('response_a', 'response_b'):
        if side not in obj:
            continue
        r = obj[side]
        if not isinstance(r, dict):
            errs.append(f'{side} must be an object {{"id": ..., "turns": [...]}}')
            continue
        for k in r:
            if k in UNSUPPORTED_EVAL_FIELDS:
                errs.append(_unsupported_msg(f'{side}.{k}', k))
            elif k in ('rubric', 'reference'):
                errs.append(f'{side}.{k}: {k!r} belongs at the top level of the line, not inside a response '
                            '(it is shared by both responses and never swapped)')
            elif k not in RESPONSE_FIELDS:
                errs.append(f'unknown field {side}.{k!r} (allowed: id, turns)')
        errs += [f'missing field {side}.{k}' for k in RESPONSE_FIELDS if k not in r]
        if 'id' in r:
            if not (isinstance(r['id'], str) and r['id'].strip()):
                errs.append(f'{side}.id must be a non-empty string')
            elif r['id'] == C.TIE:
                errs.append(f'{side}.id "tie" is reserved for the tie verdict')
            else:
                ids[side] = r['id']
        if 'turns' in r:
            n_r[side], e = _check_turns(r['turns'], f'{side}.turns', allow_empty=True)
            errs += e
    if len(ids) == 2 and ids['response_a'] == ids['response_b']:
        errs.append(f'response_a.id and response_b.id are the same ({ids["response_a"]!r}); they must differ')
    counts = [n_q, n_r.get('response_a'), n_r.get('response_b')]
    if None not in counts and len(set(counts)) != 1:
        errs.append(f'turn count mismatch: question_turns {n_q}, response_a.turns {counts[1]}, response_b.turns '
                    f'{counts[2]} (each response needs exactly one answer per question turn)')
    if 'turn' in obj:
        t = obj['turn']
        if not isinstance(t, int) or isinstance(t, bool):
            errs.append('turn must be an integer')
        elif n_q is not None and t != n_q:
            errs.append(f'turn {t} not allowed: the judged turn is always the last turn ({n_q}). To judge turn 1 '
                        'only, give one question turn and one answer per response (turns are never dropped)')
    rubric = obj.get('rubric')
    if 'rubric' in obj:
        if not isinstance(rubric, str):
            errs.append('rubric must be a string (omit the field to use the default standard)')
        elif not rubric.strip():
            errs.append('rubric is empty (omit the field to use the default standard)')
        elif U.utf8_len(rubric) > U.LIMITS['rubric_bytes']:
            errs.append(f'rubric is {U.utf8_len(rubric)} UTF-8 bytes > limit {U.LIMITS["rubric_bytes"]} '
                        '(not truncated; shorten it)')
    ref = obj.get('reference')
    if 'reference' in obj:
        if not isinstance(ref, list):
            errs.append('reference must be a list with one entry (string or null) per question turn')
        else:
            if n_q is not None and len(ref) != n_q:
                errs.append(f'reference has {len(ref)} entries but question_turns has {n_q} (one entry per question '
                            'turn; use null for a turn without a reference)')
            if ref and all(t is None for t in ref):
                errs.append('reference has only null entries (omit the field instead)')
            if not ref:
                errs.append('reference is an empty list (omit the field instead)')
            for i, t in enumerate(ref):
                if t is None:
                    continue
                if not isinstance(t, str):
                    errs.append(f'reference[{i}] must be a string or null')
                elif not t.strip():
                    errs.append(f'reference[{i}] is empty (use null for a turn without a reference)')
                elif U.utf8_len(t) > U.LIMITS['reference_entry_bytes']:
                    errs.append(f'reference[{i}] is {U.utf8_len(t)} UTF-8 bytes > limit '
                                f'{U.LIMITS["reference_entry_bytes"]} (not truncated; shorten it)')
    if errs:
        return errs, None
    a, b = obj['response_a'], obj['response_b']
    return [], dict(request_id=rid, question_turns=tuple(obj['question_turns']),
                    response_a=Response(a['id'], tuple(a['turns'])), response_b=Response(b['id'], tuple(b['turns'])),
                    turn=n_q, rubric=rubric, reference_turns=tuple(ref) if ref is not None else None)


def request_semantic(r, direction='AB'):
    s1, s2 = (r.response_a, r.response_b) if direction == 'AB' else (r.response_b, r.response_a)
    return U.semantic(r.question_turns, s1.turns, s2.turns, r.turn, r.rubric, r.reference_turns)


def wire_body_bytes(r, backend):
    """UTF-8 bytes of the JSON wire body exactly as the backend encodes it (both directions have the same size)."""
    sem = request_semantic(r)
    body = P.haiku_body(sem) if backend == 'haiku_direct' else P.choice_body(sem)
    return len(json.dumps(body, ensure_ascii=False).encode('utf-8'))


def load_requests(path, backend=C.DEFAULT_BACKEND):
    """Validate the WHOLE file before any call. -> [(line_no, JudgeRequest)]; raises InputRejected.
    The request size limit is checked on the wire body of ``backend``."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise InputRejected([(0, f'cannot read input file {str(p.resolve())!r}: {e.strerror}')]) from None
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise InputRejected([(0, f'input is not UTF-8 (byte offset {e.start})')]) from None
    text = text.removeprefix('﻿')
    lines = text.split('\n')
    if lines and lines[-1] == '':
        lines.pop()                                   # the newline that ends the last line
    if not lines:
        raise InputRejected([(0, 'input has no requests')])
    errors, out, seen = [], [], {}
    for no, line in enumerate(lines, 1):
        if not line.strip():
            errors.append((no, 'empty line (every line must be one request)'))
            continue
        try:
            obj = json.loads(line, object_pairs_hook=_no_dup_keys)
        except ValueError as e:
            errors.append((no, f'not valid JSON: {getattr(e, "msg", str(e))}'))
            continue
        errs, fields = check_line(obj)
        rid = obj.get('id') if isinstance(obj, dict) else None
        if isinstance(rid, str) and rid.strip():
            if rid in seen:
                errs = errs + [f'duplicate request id {rid!r} (first at line {seen[rid]})']
            else:
                seen[rid] = no
        if not errs:
            req = JudgeRequest(**fields, host={'source': 'judge_cli', 'input_line': no})
            n = wire_body_bytes(req, backend)
            if n > U.LIMITS['request_body_bytes']:
                errs = [f'request wire body for {backend} is {n} UTF-8 bytes > limit {U.LIMITS["request_body_bytes"]} '
                        '(not truncated; shorten the texts)']
        if errs:
            errors += [(no, m) for m in errs]
        else:
            out.append((no, req))
    if errors:
        raise InputRejected(errors)
    return out


def directions(mode):
    return ('AB',) if mode == 'single' else ('AB', 'BA')


def plan(requests, mode):
    calls = [f'{r.request_id}:{d}' for _, r in requests for d in directions(mode)]
    return {'requests': len(requests), 'mode': mode, 'planned_max_submissions': len(calls), 'call_keys': calls}


# --- run -------------------------------------------------------------------------------------------------------
class RunRefused(Exception):
    pass


class RunTraceWriter(TraceWriter):
    """The adapter's trace writer plus run_id / backend / transport on every raw record (CLI only; the adapter core
    and the FastChat path keep their own writer unchanged)."""

    def __init__(self, trace_dir, run_id, backend, transport, scrub=None):
        super().__init__(trace_dir, scrub)
        self.tag = {'run_id': run_id, 'backend': backend, 'transport': transport}
        if transport == 'fake':
            self.tag['mock_note'] = 'MOCK: in-process fake transport; the URLs below were never contacted'

    def write_raw(self, call_id, exchanges):
        self._append(self.raw, {**self.tag, 'call_id': call_id,
                                'exchanges': [{k: v for k, v in x.items() if k != 'json'} for x in exchanges]})
        return {'file': self.raw.name, 'call_id': call_id}


def _sha_file(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def _outcome(res):
    if res.status in (C.STATUS_OK, C.STATUS_TIE):
        return res.status
    if res.submissions == 0 and (res.failure or {}).get('reason') == C.NOT_SUBMITTED:
        return 'unrun'
    return 'error'


def prepare_out_dir(out_dir, overwrite):
    out = Path(out_dir).resolve()
    existing = [f for f in OUTPUT_FILES if (out / f).exists()]
    if existing and not overwrite:
        raise RunRefused(f'output dir {str(out)!r} already has results ({", ".join(existing)}); choose a new dir or '
                         'pass --overwrite (removes only these files)')
    for f in existing:
        (out / f).unlink()
    (out / 'trace').mkdir(parents=True, exist_ok=True)
    return out


def check_limits(transport, allow_real, max_submissions, budget_cap_usd, backend='haiku_direct',
                 choice_usd_per_second=None):
    if transport == 'real':
        need = [('--allow-real', allow_real), ('--max-submissions', max_submissions),
                ('--budget-cap-usd', budget_cap_usd)]
        if backend == 'choice_endpoint':
            need.append(('--choice-usd-per-second', choice_usd_per_second))     # no built-in rate
        missing = [n for n, ok in need if not ok]
        if missing:
            raise RunRefused(f'a real (paid) run needs {", ".join(missing)}')
    if choice_usd_per_second is not None and choice_usd_per_second <= 0:
        raise RunRefused('--choice-usd-per-second must be positive')
    if max_submissions is not None and max_submissions <= 0:
        raise RunRefused('--max-submissions must be positive')
    if budget_cap_usd is not None and budget_cap_usd <= 0:
        raise RunRefused('--budget-cap-usd must be positive')


def run_file(input_path, out_dir, *, transport, backend='haiku_direct', mode='single', allow_real=False,
             max_submissions=None, budget_cap_usd=None, overwrite=False, env_file=None, timeout_seconds=120.0,
             choice_endpoint_id=None, choice_usd_per_second=None, choice_expected_model=None, fake=None,
             key_provider=None, clock=time.perf_counter):
    """Validate, then judge every request once (sequentially). fake: an injected fake server (tests).
    env_file: optional explicit KEY=VALUE file for a real run; None -> environment variables."""
    if transport not in C.TRANSPORTS:
        raise RunRefused(f'--transport must be one of {C.TRANSPORTS}')
    check_limits(transport, allow_real, max_submissions, budget_cap_usd, backend, choice_usd_per_second)
    in_path = Path(input_path).resolve()
    requests = load_requests(in_path, backend)            # raises InputRejected before anything else
    pl = plan(requests, mode)
    out = prepare_out_dir(out_dir, overwrite)
    is_fake = transport == 'fake'
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + secrets.token_hex(3)
    if is_fake:
        from . import fakes                                 # model-free stand-in; never on the real path
        fake = fake or fakes.fake_for(backend)
        key_provider = key_provider or (lambda: fakes.DUMMY_KEY)
    elif fake is not None:
        raise RunRefused('a real run takes no fake server')
    cfg = AdapterConfig(transport=transport, backend=backend, mode=mode, trace_dir=str(out / 'trace'),
                        timeout_seconds=timeout_seconds, choice_endpoint_id=choice_endpoint_id,
                        choice_usd_per_second=choice_usd_per_second, choice_expected_model=choice_expected_model,
                        env_file=str(Path(env_file).resolve()) if env_file else None)
    be = build_backend(cfg, transport=fake.transport() if is_fake else None, key_provider=key_provider, clock=clock)
    gate = RunGate(be, mode, budget_cap_usd, max_submissions or pl['planned_max_submissions'], clock=clock)
    adapter = JudgeAdapter(cfg, gate, clock=clock)
    scrub = scrubber_of(be)          # saved files are scrubbed; in-memory results keep provider data as received
    adapter.trace = RunTraceWriter(cfg.trace_dir, run_id, be.name, transport, scrub)
    rows, t0 = [], clock()
    results_path = out / 'results.jsonl'
    interrupted = None
    with open(results_path, 'w', encoding='utf-8') as rf:
        try:
            for line_no, req in requests:
                gate.begin(req)
                res = adapter.judge(req)
                row = {'run_id': run_id, 'input_line': line_no, 'outcome': _outcome(res), 'mock': is_fake,
                       **({'mock_note': MOCK_NOTE} if is_fake else {}), **res.to_dict()}
                rows.append(row)
                rf.write(scrub.text(json.dumps(scrub.obj(row), ensure_ascii=False, default=str)) + '\n')
                rf.flush()
        except KeyboardInterrupt:
            interrupted = 'KeyboardInterrupt'
        finally:
            adapter.close()
            for line_no, req in requests[len(rows):]:
                row = {'run_id': run_id, 'input_line': line_no, 'outcome': 'unrun', 'mock': is_fake,
                       'request_id': req.request_id, 'backend': be.name, 'transport': transport,
                       'status': None, 'failure': {'reason': 'not_reached', 'detail': f'run interrupted '
                                                   f'({interrupted})', 'direction': None},
                       'submissions': 0, 'calls': [], 'cost_estimate': {'usd': 0.0, 'is_estimate': True}}
                rows.append(row)
                rf.write(json.dumps(row, ensure_ascii=False) + '\n')
    total = clock() - t0
    snap = gate.snapshot()
    (out / 'gate.json').write_text(scrub.text(json.dumps(scrub.obj(snap), ensure_ascii=False, indent=1, default=str))
                                   + '\n', encoding='utf-8')
    summary = summarize(rows, pl, snap, in_path, out, cfg, be, is_fake, total, interrupted, run_id)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return summary


def summarize(rows, pl, snap, in_path, out, cfg, backend, is_fake, total_seconds, interrupted=None, run_id=None):
    counts = Counter(r['outcome'] for r in rows)
    tie_kinds = Counter(r.get('tie_kind') for r in rows if r['outcome'] == 'tie')
    calls = [c for r in rows for c in r.get('calls') or []]
    submitted = [c for c in calls if c.get('submissions')]
    with_usage = [c for c in submitted if isinstance((c.get('usage') or {}).get('input_tokens'), int)
                  and isinstance((c.get('usage') or {}).get('output_tokens'), int)]
    if not submitted:
        usage = 'no submissions'
    elif not with_usage:
        usage = 'unsupported or unknown (no submitted call reported usage)'
    else:
        usage = {'input_tokens': sum(c['usage']['input_tokens'] for c in with_usage),
                 'output_tokens': sum(c['usage']['output_tokens'] for c in with_usage),
                 'calls_with_usage': len(with_usage), 'submitted_calls_without_usage': len(submitted) - len(with_usage)}
    unknown_usage_rows = sum(1 for r in rows if any(c.get('submissions') and c not in with_usage
                                                    for c in r.get('calls') or []))
    costs = [r['cost_estimate'].get('usd') for r in rows if r.get('submissions')]
    unknown = sum(1 for u in costs if u is None)
    known = round(sum(u for u in costs if u is not None), 8)
    mock = 'MOCK: fake transport; numbers are not costs. ' if is_fake else ''
    submissions = sum(r.get('submissions') or 0 for r in rows)
    return {
        'schema': SCHEMA, 'kind': 'MOCK run (fake transport, no model, no network)' if is_fake else 'REAL run',
        'run_id': run_id, 'mock': is_fake, **({'mock_note': MOCK_NOTE} if is_fake else {}), 'backend': backend.name, 'model': backend.model, 'mode': cfg.mode, 'transport': cfg.transport,
        'input': {'path': str(in_path), 'sha256': _sha_file(in_path), 'requests': len(rows)},
        'counts': {'requests': len(rows), 'ok': counts['ok'], 'tie': counts['tie'],
                   'tie_explicit': tie_kinds['explicit'], 'tie_inconsistent': tie_kinds['inconsistent'],
                   'error': counts['error'], 'unrun': counts['unrun']},
        'prompt_versions': dict(Counter((r.get('criteria_info') or {}).get('prompt_version') for r in rows
                                        if r.get('criteria_info'))),
        'error_reasons': dict(Counter(r['failure']['reason'] for r in rows if r['outcome'] == 'error')),
        'unrun_reasons': dict(Counter(r['failure']['reason'] for r in rows if r['outcome'] == 'unrun')),
        'submissions': {'planned_max': pl['planned_max_submissions'], 'actual': submissions,
                        'status_polls': sum(c.get('status_polls') or 0 for c in calls),
                        'cancels': sum(c.get('cancels') or 0 for c in calls), 'retries': 0, 'fallbacks': 0,
                        'gate_counted': snap['submissions']},
        'limits': {'max_submissions': snap['max_submissions'], 'budget_cap_usd': snap['cap_usd'],
                   'stopped': snap['stopped'], 'interrupted': interrupted},
        'usage': usage, 'requests_with_unknown_usage': unknown_usage_rows, **({'usage_note': 'MOCK: invented by the fake server, not real token counts'}
                           if is_fake else {}),
        'total_seconds': round(total_seconds, 3),
        'cost_estimate': {
            'is_estimate': True, 'mock': is_fake, 'not_the_bill': True, 'includes_failed_requests': True,
            'usd_all_submitted_requests': known if unknown == 0 else None,
            'usd_known_part': known, 'requests_with_unknown_cost': unknown,
            'gate_committed_usd': round(snap['committed_usd'], 8),
            'basis': mock + backend.cost_basis + ' gate_committed_usd adds reservations held for unknown usage and '
                     '(choice_endpoint) idle allowance and wall clock (see gate.json).'},
        'files': {
            'results': {'path': str(out / 'results.jsonl'), 'sha256': _sha_file(out / 'results.jsonl')},
            'raw_outputs': {'path': str(out / 'trace' / 'judge_raw.jsonl'),
                            'sha256': _sha_file(out / 'trace' / 'judge_raw.jsonl'),
                            'note': 'HTTP response bodies per call; keys and bearer tokens removed'
                                    + ('. MOCK: the URLs in this file were never contacted (in-process fake '
                                       'transport)' if is_fake else '')},
            'trace': {'path': str(out / 'trace' / 'judge_trace.jsonl'),
                      'sha256': _sha_file(out / 'trace' / 'judge_trace.jsonl')},
            'gate': {'path': str(out / 'gate.json')}},
        'note': 'No success rate is reported. Every request is counted, including errors and unrun requests.',
    }


def summary_text(s):
    c, sub, ce = s['counts'], s['submissions'], s['cost_estimate']
    usd = ce['usd_all_submitted_requests']
    usd_txt = f'${usd:.6f}' if usd is not None else (f'unknown ({ce["requests_with_unknown_cost"]} request(s) without '
                                                     f'cost; known part ${ce["usd_known_part"]:.6f})')
    lines = [
        ('[MOCK] ' if s['mock'] else '') + f'{s["kind"]}: backend {s["backend"]} ({s["model"]}), mode {s["mode"]}, '
        f'run {s["run_id"]}',
        *([s['mock_note']] if s['mock'] else []),
        f'requests {c["requests"]}: ok {c["ok"]}, tie {c["tie"]} (explicit {c["tie_explicit"]}, inconsistent '
        f'{c["tie_inconsistent"]}), error {c["error"]}, unrun {c["unrun"]}',
        f'error reasons {s["error_reasons"] or "-"}; unrun reasons {s["unrun_reasons"] or "-"}',
        f'prompt versions {s["prompt_versions"] or "-"}',
        f'submissions: planned max {sub["planned_max"]}, actual {sub["actual"]}; status polls {sub["status_polls"]}, '
        f'cancels {sub["cancels"]}, retries 0, fallbacks 0',
        f'limits: max_submissions {s["limits"]["max_submissions"]}, cap {s["limits"]["budget_cap_usd"]}, '
        f'stopped: {s["limits"]["stopped"] or "no"}',
        f'usage{" (MOCK, invented)" if s["mock"] else ""}: {s["usage"]}; requests with unknown usage '
        f'{s["requests_with_unknown_usage"]}',
        f'total time {s["total_seconds"]} s',
        f'cost ESTIMATE{" (MOCK, not a cost)" if s["mock"] else ""}, all submitted requests incl. failures: {usd_txt}; '
        f'gate committed estimate ${ce["gate_committed_usd"]:.6f}. Not the bill.',
        f'results {s["files"]["results"]["path"]}',
        f'raw outputs {s["files"]["raw_outputs"]["path"]} sha256 {s["files"]["raw_outputs"]["sha256"]}',
    ]
    return '\n'.join(lines)


# --- CLI -------------------------------------------------------------------------------------------------------
def print_errors(errors, stream=sys.stderr):
    print(f'input rejected: {len(errors)} error(s); nothing was sent', file=stream)
    for no, msg in errors:
        print(f'  line {no}: {msg}' if no else f'  file: {msg}', file=stream)


def cmd_validate(a):
    requests = load_requests(a.input, a.backend)
    pl = plan(requests, a.mode)
    print(f'input OK: {Path(a.input).resolve()} ({len(requests)} requests). No call made, no key read.')
    print(f'backend {a.backend}, mode {a.mode}: planned max submissions {pl["planned_max_submissions"]}')
    for no, r in requests:
        info = U.criteria_info(r.turn, r.rubric, r.reference_turns)
        ref = info['reference']
        print(f'  line {no}: id {r.request_id!r}, turns {r.turn}, A={r.response_a.response_id!r} '
              f'B={r.response_b.response_id!r}, calls {", ".join(directions(a.mode))}; prompt {info["prompt_version"]}, '
              f'rubric {info["rubric"]["source"]}, reference '
              f'{"turns " + str(ref["turns_with_reference"]) if ref else "none"}, body {wire_body_bytes(r, a.backend)} B')
    if a.max_submissions is not None and a.max_submissions < pl['planned_max_submissions']:
        per = len(directions(a.mode))
        print(f'note: --max-submissions {a.max_submissions} < planned {pl["planned_max_submissions"]}; at least '
              f'{a.max_submissions // per} request(s) run (more if some use fewer calls); requests that do not fit '
              'are reported as unrun')
    if a.backend == 'haiku_direct':
        res = sum(haiku_reservation_usd(P.haiku_body(request_semantic(r))) for _, r in requests) * len(
            directions(a.mode))
        print(f'haiku_direct conservative reservation for all planned calls: ${res:.6f} (an upper estimate used by '
              'the cap; not a prediction or a bill)')
    else:
        print(f'choice_endpoint: the cap reserves {JOB_RESERVATION_SECONDS:.0f} s x your --choice-usd-per-second '
              f'rate per open job, plus idle {IDLE_ALLOWANCE_SECONDS:.0f} s once (no built-in rate; a real run '
              'refuses to start without one)')
    return 0


def cmd_run(a):
    s = run_file(a.input, a.out, transport=a.transport, backend=a.backend, mode=a.mode, allow_real=a.allow_real,
                 max_submissions=a.max_submissions, budget_cap_usd=a.budget_cap_usd, overwrite=a.overwrite,
                 env_file=a.env_file, timeout_seconds=a.timeout_seconds, choice_endpoint_id=a.choice_endpoint_id,
                 choice_usd_per_second=a.choice_usd_per_second, choice_expected_model=a.choice_expected_model)
    print(summary_text(s))
    return 0


def example_files():
    return sorted(EXAMPLES_DIR.glob('*.json*'))


def cmd_examples(a):
    for p in example_files():
        print(p)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='judge-adapter', description='Judge your own response pairs (JSONL) with the '
                                 'fixed pairwise judge. Default backend haiku_direct; choice_endpoint only with --backend choice_endpoint.')
    ap.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('examples', help='print the paths of the bundled synthetic example files')
    for name in ('validate', 'run'):
        p = sub.add_parser(name)
        p.add_argument('input', help='JSONL file, one request per line')
        p.add_argument('--backend', choices=C.BACKENDS, default=C.DEFAULT_BACKEND)
        p.add_argument('--mode', choices=C.MODES, default='single')
        p.add_argument('--max-submissions', type=int, help='run-wide limit (required for real)')
        if name == 'run':
            p.add_argument('--out', required=True, help='local output dir (results, summary, raw outputs)')
            p.add_argument('--transport', choices=C.TRANSPORTS, required=True, help='fake (model-free) or real')
            p.add_argument('--allow-real', action='store_true', help='required for a real (paid) run')
            p.add_argument('--budget-cap-usd', type=float, help='run-wide client-side cap (required for real)')
            p.add_argument('--overwrite', action='store_true', help='replace this tool\'s result files in --out')
            p.add_argument('--env-file', help='optional KEY=VALUE file read only by a real run (no default; without it '
                           'the key comes from ANTHROPIC_API_KEY / RUNPOD_API_KEY in the environment)')
            p.add_argument('--choice-endpoint-id', help=f'RunPod endpoint ID for --backend choice_endpoint (else env '
                           f'{P.CHOICE_ENDPOINT_ENV}; unset -> local_config, nothing sent)')
            p.add_argument('--choice-usd-per-second', type=float, help='per-second rate of your choice endpoint, '
                           'used for the cost estimate and cap (required for a real choice_endpoint run; no default)')
            p.add_argument('--choice-expected-model', help='optional: expected output.model of your choice endpoint; '
                           'a different model is a contract failure (unset: not checked, only recorded)')
            p.add_argument('--timeout-seconds', type=float, default=120.0)
    a = ap.parse_args(argv)
    if a.cmd == 'examples':
        return cmd_examples(a)
    try:
        return cmd_validate(a) if a.cmd == 'validate' else cmd_run(a)
    except InputRejected as e:
        print_errors(e.errors)
        return 2
    except RunRefused as e:
        print(f'refused: {e}', file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())
