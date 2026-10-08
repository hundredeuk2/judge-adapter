# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
#
# OPTIONAL module (extra "fastchat"). Binding for FastChat llm_judge (https://github.com/lm-sys/FastChat, Apache
# License 2.0), written against commit 587d5cfa1609a43d192cedb8441cac3c17db105d. It contains no FastChat source code:
# it imports fastchat.llm_judge.common at install() time and replaces run_judge_pair / run_judge_single with
# functions that follow the same call signature and return contract. REFERENCE_CATEGORIES mirrors
# common.NEED_REF_CATS at that commit. See NOTICE.
"""FastChat llm_judge binding: replace ``fastchat.llm_judge.common.run_judge_pair`` at run time.

Supported host revision: lm-sys/FastChat commit 587d5cfa1609a43d192cedb8441cac3c17db105d (SUPPORTED_HOST_REVISION).
Other revisions are not checked; ``install()`` only refuses when the host's reference-category list differs.

Injection point (host source untouched):
  fastchat/llm_judge/common.py  play_a_match_pair() calls the module-global run_judge_pair() twice per match
  (g1: A=model_1, B=model_2; g2: swapped). Assigning a new function to ``common.run_judge_pair`` routes both calls
  through the adapter. The host keeps doing everything else: loading data, building matches, swapping positions,
  mapping A/B to model_1/model_2, writing data/<bench>/model_judgment/<judge>_pair.jsonl, show_result.py.

Return contract of run_judge_pair: (winner, user_prompt, judgment) with winner in "A" | "B" | "tie" | "error".
Adapter failures become the host's own "error" winner (show_result.py drops those rows). Nothing is retried.

Reference answers (host categories in common.NEED_REF_CATS: math, reasoning, coding, arena-hard-200):
- Host path at the pinned revision: gen_judgment.py loads ``data/<bench>/reference_answer/*.jsonl`` with
  load_model_answers (file stem = name) and looks up ``ref_answers[judge.model_name][question_id]``, i.e. the file
  named after ``--judge-model``. Reference judges use pair-math-v1 / pair-math-v1-multi-turn (ref_based=True).
  run_judge_pair formats ref_answer["choices"][0]["turns"][0] (and [1] when multi_turn) into the template.
- Without a ReferenceMap (default): the host reference is passed as JudgeRequest.reference, which the adapter refuses
  with unsupported_input (0 calls). The file found under the judge name is never used as a reference.
- With a ReferenceMap (built from a file the USER names explicitly, see load_reference_rows / build_reference_map):
  reference-category requests get ``reference_turns`` = the user's turns 1..turn, prompt user-criteria/v1 (user_prompt.PROMPT_VERSION_USER).
  The reference is not slot-specific, so the A/B swap of g1/g2 never moves it. Non-reference requests are built
  exactly as before (same JudgeRequest -> same wire bytes). Missing rows, turn-count mismatch and empty entries are
  refused before the host starts; nothing is filled in.
- Host templates are replaced, not reproduced: pair-v2* -> default-fixed/v1, pair-math-v1* -> user-criteria/v1
  (HOST_TEMPLATE_REPLACEMENT). Scores, explanations and the [[A]]/[[B]]/[[C]] format of the host are not used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .. import user_prompt as U
from ..contract import JudgeRequest, Response

SUPPORTED_HOST_REVISION = '587d5cfa1609a43d192cedb8441cac3c17db105d'
REFERENCE_CATEGORIES = ('math', 'reasoning', 'coding', 'arena-hard-200')   # common.NEED_REF_CATS at that revision
HOST_TEMPLATE_REPLACEMENT = {
    'pair-v2': U.PROMPT_VERSION_DEFAULT, 'pair-v2-multi-turn': U.PROMPT_VERSION_DEFAULT,
    'pair-math-v1': U.PROMPT_VERSION_USER, 'pair-math-v1-multi-turn': U.PROMPT_VERSION_USER,
}


class ReferenceMappingError(ValueError):
    """An explicit reference mapping cannot be used. Raised before the host starts; nothing is filled in."""


def response_id(answer, question_id):
    """Stable, host-owned id: the FastChat answer_id when present, else model_id + question_id."""
    aid = answer.get('answer_id')
    return str(aid) if aid else f'{answer.get("model_id")}:q{question_id}'


def to_request(question, answer_a, answer_b, judge, ref_answer, multi_turn, reference_turns=None, reference_host=None):
    """reference_turns None -> the request built before reference support (byte parity). Otherwise the user's
    reference for turns 1..turn goes to JudgeRequest.reference_turns and the host ref_answer is not forwarded."""
    turn = 2 if multi_turn else 1
    qid = question['question_id']
    ia, ib = response_id(answer_a, qid), response_id(answer_b, qid)
    host = {'host': 'fastchat.llm_judge', 'question_id': qid, 'category': question.get('category'),
            'host_judge_label': judge.model_name, 'host_prompt_name': judge.prompt_template.get('name'),
            'host_prompt_replaced': True, 'model_a': answer_a.get('model_id'), 'model_b': answer_b.get('model_id')}
    ref, ref_turns = None, None
    if reference_turns is not None:
        ref_turns = tuple(reference_turns[:turn])
        host['reference'] = dict(reference_host or {})
    elif ref_answer is not None:
        ref = list(ref_answer['choices'][0]['turns'][:turn])
    return JudgeRequest(
        request_id=f'fastchat:q{qid}:t{turn}:{ia}|{ib}',
        question_turns=tuple(question['turns'][:turn]),
        response_a=Response(ia, tuple(answer_a['choices'][0]['turns'][:turn])),
        response_b=Response(ib, tuple(answer_b['choices'][0]['turns'][:turn])),
        turn=turn, reference=ref, reference_turns=ref_turns, host=host)


def judgment_text(res):
    """What the host stores in g1_judgment / g2_judgment: a compact summary; raw output is in the trace."""
    return json.dumps({'judge_adapter': {
        'request_id': res.request_id, 'status': res.status, 'verdict': res.verdict, 'tie_kind': res.tie_kind,
        'failure': res.failure, 'backend': res.backend, 'model': res.model, 'mode': res.mode,
        'transport': res.transport, 'submissions': res.submissions,
        'raw_output_sha256': [c.raw_output_sha256 for c in res.calls]}}, ensure_ascii=False)


# --- explicit reference mapping -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceMap:
    """question_id -> the user's reference turns (one per question turn), plus the mapping record for the trace."""
    turns: dict
    record: dict = field(default_factory=dict)

    def host_record(self, question_id):
        """Per-request trace entry (host metadata only; never sent to the model)."""
        r = self.record
        return {'source_kind': r.get('source_kind'), 'source_path': r.get('source_path'),
                'source_sha256': r.get('source_sha256'), 'staged_as': r.get('staged_as'),
                'host_lookup_key': r.get('host_lookup_key'), 'question_id': question_id,
                'turns_sha256': hashlib.sha256(json.dumps(list(self.turns[question_id]), ensure_ascii=False)
                                               .encode('utf-8')).hexdigest()}


def load_reference_rows(data):
    """Parse a FastChat answer-format JSONL (bytes): one row per question_id with choices[0].turns (list of str).
    Same shape the host reads from reference_answer/<name>.jsonl. Duplicate question_ids are refused (the host
    would silently keep the last one)."""
    rows = {}
    for n, line in enumerate(data.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            qid = row['question_id']
            turns = row['choices'][0]['turns']
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise ReferenceMappingError(f'reference file line {n}: not a FastChat answer row with question_id and '
                                        f'choices[0].turns ({type(e).__name__})') from None
        if not isinstance(turns, list):
            raise ReferenceMappingError(f'reference file line {n}: choices[0].turns must be a list')
        if qid in rows:
            raise ReferenceMappingError(f'reference file line {n}: duplicate question_id {qid!r}')
        rows[qid] = turns
    return rows


def reference_questions(questions):
    return [q for q in questions if q.get('category') in REFERENCE_CATEGORIES]


def build_reference_map(questions, rows, record=None):
    """Check every reference-category question of the run against the user's rows -> ReferenceMap.

    Refused (ReferenceMappingError): a missing row, a turn count different from the question's, a turn that is not
    a non-empty string, a turn over the adapter's per-entry byte limit. Rows for other questions are ignored."""
    turns, problems = {}, []
    for q in reference_questions(questions):
        qid, n_q = q['question_id'], len(q['turns'])
        if qid not in rows:
            problems.append(f'question {qid!r} ({q["category"]}): no reference row')
            continue
        t = rows[qid]
        if len(t) != n_q:
            problems.append(f'question {qid!r}: reference has {len(t)} turns, question has {n_q}')
            continue
        bad = [i for i, x in enumerate(t, 1) if not isinstance(x, str) or not x.strip()]
        if bad:
            problems.append(f'question {qid!r}: reference turn(s) {bad} empty or not text')
            continue
        big = [i for i, x in enumerate(t, 1) if U.utf8_len(x) > U.LIMITS['reference_entry_bytes']]
        if big:
            problems.append(f'question {qid!r}: reference turn(s) {big} over {U.LIMITS["reference_entry_bytes"]} '
                            'UTF-8 bytes (not truncated)')
            continue
        turns[qid] = tuple(t)
    if problems:
        raise ReferenceMappingError('reference mapping refused before any call: ' + '; '.join(problems))
    rec = dict(record or {})
    rec['mapped_question_ids'] = sorted(turns, key=str)
    rec['turns_per_question'] = {str(k): len(v) for k, v in turns.items()}
    return ReferenceMap(turns=turns, record=rec)


def _reference_for(refmap, question, judge, ref_answer):
    """-> the user's reference turns for a reference judge, None for any other judge. Guards the host side."""
    is_ref_cat = question.get('category') in REFERENCE_CATEGORIES
    if bool(getattr(judge, 'ref_based', False)) != is_ref_cat:
        raise RuntimeError(f'judge adapter: host judge ref_based={judge.ref_based} but category '
                           f'{question.get("category")!r}; unsupported host revision?')
    if not is_ref_cat:
        return None
    qid = question['question_id']
    if qid not in refmap.turns:
        raise RuntimeError(f'judge adapter: no mapped reference for question {qid!r} (pre-run check bypassed)')
    mapped = refmap.turns[qid]
    host_turns = None if ref_answer is None else tuple(ref_answer['choices'][0]['turns'])
    if host_turns != mapped:
        raise RuntimeError(f'judge adapter: host reference for question {qid!r} differs from the mapped file '
                           '(staged file changed?)')
    return mapped


def make_run_judge_pair(adapter, reference_map=None):
    def run_judge_pair(question, answer_a, answer_b, judge, ref_answer, multi_turn=False):
        if reference_map is None:
            req = to_request(question, answer_a, answer_b, judge, ref_answer, multi_turn)
        else:
            ref = _reference_for(reference_map, question, judge, ref_answer)
            req = to_request(question, answer_a, answer_b, judge, None if ref is None else ref_answer, multi_turn,
                             reference_turns=ref,
                             reference_host=None if ref is None else reference_map.host_record(question['question_id']))
        res = adapter.judge(req)
        winner = res.verdict if res.status != 'error' else 'error'
        return winner, adapter.last_rendered_input or '', judgment_text(res)
    run_judge_pair.judge_adapter = True
    return run_judge_pair


def _refuse_single(*args, **kwargs):
    raise RuntimeError('judge adapter: host single-answer grading (run_judge_single) is not routed through the '
                       'adapter and is refused; use host --mode pairwise-baseline or pairwise-all')


def install(adapter, reference_map=None):
    """Patch the host boundary. Returns a function that restores the original.

    run_judge_single is replaced by a refusal while installed: the wrapper already rejects host --mode single before
    the host starts; this second guard also covers a pairwise run whose judge template has type "single".
    With a reference_map, the host's reference-category list must equal REFERENCE_CATEGORIES."""
    if adapter.cfg.mode != 'single':
        raise ValueError('FastChat already judges both positions (g1/g2); only adapter mode "single" is allowed '
                         '(host 2-direction x adapter 2-direction would double the calls)')
    import fastchat.llm_judge.common as common
    if reference_map is not None and tuple(common.NEED_REF_CATS) != REFERENCE_CATEGORIES:
        raise ReferenceMappingError(f'host NEED_REF_CATS {list(common.NEED_REF_CATS)} differs from the supported '
                                    f'revision {SUPPORTED_HOST_REVISION[:8]} {list(REFERENCE_CATEGORIES)}')
    original, original_single = common.run_judge_pair, common.run_judge_single
    common.run_judge_pair = make_run_judge_pair(adapter, reference_map)
    common.run_judge_single = _refuse_single

    def restore():
        common.run_judge_pair, common.run_judge_single = original, original_single
    return restore
