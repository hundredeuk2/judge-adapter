# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
#
# Portions of the judge wording in this file (INSTRUCTIONS / _COMMON) and the "### User:" / "### Assistant:"
# conversation rendering are adapted from the MT-Bench pairwise judge prompts of FastChat
# (https://github.com/lm-sys/FastChat, fastchat/llm_judge/data/judge_prompts.jsonl, prompts "pair-v2" and
# "pair-v2-multi-turn", commit 587d5cfa1609a43d192cedb8441cac3c17db105d), Copyright the FastChat authors,
# licensed under the Apache License, Version 2.0. See NOTICE.
# Modifications: reworded; assistants are referred to by state field name instead of "A"/"B"; the explanation and
# the "[[A]]"/"[[B]]"/"[[C]]" output format are removed; output is one option id (opt_1 / opt_2 / opt_tie); the
# question and conversations are passed as separate named fields.
"""Shared semantic input and the two wire formats (choice-endpoint structure / Haiku text). Stdlib only.

The default wording, parser and generation settings are fixed: they are not tuned on any result. haiku_text renders
every state field: for the fixed three-field state it iterates STATE_FIELDS in order; for a user-criteria state
(user_prompt.py) it renders the fields in state order, so no field (rubric, reference) is dropped.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

# --- semantic input (identical for both backends) -------------------------------------------------------------
QUESTION_ID = 'preference'
STATE_FIELDS = ('question', 'assistant_1_conversation', 'assistant_2_conversation')
CRITERIA = {
    'opt_1': "Assistant 1's answer is better.",
    'opt_2': "Assistant 2's answer is better.",
    'opt_tie': 'The two answers are equally good (tie).',
}
CRITERIA_IDS = tuple(CRITERIA)
_COMMON = ('Choose the assistant that follows the user\'s instructions and answers the user\'s question better, or '
           'choose a tie if the two answers are equally good. Consider helpfulness, relevance, accuracy, depth, '
           'creativity and level of detail. Do not let the order in which the answers are presented, the length of '
           'the answers, or the names of the assistants influence your decision. Be as objective as possible.')
INSTRUCTIONS = {
    1: ('Act as an impartial judge and evaluate the quality of the answers given by two AI assistants to the user '
        'question in the field "question". Each assistant\'s answer is in "assistant_1_conversation" and '
        '"assistant_2_conversation". ' + _COMMON),
    2: ('Act as an impartial judge and evaluate the quality of the answers given by two AI assistants. The fields '
        '"assistant_1_conversation" and "assistant_2_conversation" each hold one assistant\'s whole conversation with '
        'the user: two user questions and that assistant\'s two answers. Judge only the answers to the second user '
        'question, which is repeated in the field "question"; use the first exchange as context. ' + _COMMON),
}
HAIKU_FORMAT_LINE = 'Reply with exactly one option id (opt_1, opt_2 or opt_tie) and nothing else.'

# --- Haiku direct (8 tokens, temperature 0, no system prompt) -------------------------------------------------
HAIKU_MODEL = 'claude-haiku-4-5-20251001'
HAIKU_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_VERSION = '2023-06-01'
HAIKU_MAX_TOKENS = 8
HAIKU_TEMPERATURE = 0.0
HAIKU_BODY_KEYS = ('model', 'max_tokens', 'temperature', 'messages')
HAIKU_PRICE_IN_PER_MTOK = 1.00     # Anthropic list price for Claude Haiku 4.5 as checked 2026-09: $1 in / $5 out per
HAIKU_PRICE_OUT_PER_MTOK = 5.00    # MTok. Used for estimates only; check the current price list

# --- choice endpoint (a user-served choice-judge worker on RunPod serverless, /runsync) ------------------------
# Nothing endpoint-specific is built in: the endpoint ID comes only from AdapterConfig.choice_endpoint_id or the
# environment variable below, the expected model name only from AdapterConfig.choice_expected_model (optional), and
# the per-second rate only from AdapterConfig.choice_usd_per_second (required for a real run).
# Unset endpoint or rate -> every real call fails as local_config before any transmission.
CHOICE_ENDPOINT_ENV = 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID'
CHOICE_ENDPOINT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')
CHOICE_HOST = 'api.runpod.ai'
CHOICE_RUNSYNC_WAIT_MS = 110_000
CHOICE_POLICY = {'executionTimeout': 60_000, 'ttl': 600_000}
PROB_SUM_TOLERANCE = 0.01
PROB_VALUE_SLACK = 1e-9


def choice_urls(endpoint_id):
    if not isinstance(endpoint_id, str) or not CHOICE_ENDPOINT_ID_RE.match(endpoint_id):
        raise ValueError('choice endpoint ID must be 1-64 characters of A-Z a-z 0-9 _ -')
    base = f'https://{CHOICE_HOST}/v2/{endpoint_id}'
    return {'runsync': f'{base}/runsync?wait={CHOICE_RUNSYNC_WAIT_MS}', 'status': base + '/status/{job_id}',
            'cancel': base + '/cancel/{job_id}'}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def render_conversation(cv, turn):
    s = f'### User:\n{cv["q1"]}\n\n### Assistant:\n{cv["a1"]}'
    if turn == 2:
        s += f'\n\n### User:\n{cv["q2"]}\n\n### Assistant:\n{cv["a2"]}'
    return s


def semantic(question_turns, slot1_turns, slot2_turns, turn, instructions=None, criteria=None):
    """The ONE semantic input both backends receive. slotN_turns: that response's answers up to the judged turn."""
    def cv(ans):
        d = {'q1': question_turns[0], 'a1': ans[0]}
        if turn == 2:
            d.update(q2=question_turns[1], a2=ans[1])
        return d
    state = {'question': question_turns[turn - 1],
             'assistant_1_conversation': render_conversation(cv(slot1_turns), turn),
             'assistant_2_conversation': render_conversation(cv(slot2_turns), turn)}
    return {'turn': turn, 'instructions': instructions or INSTRUCTIONS[turn], 'state': state,
            'criteria': dict(criteria or CRITERIA)}


def question_id(sem):
    """The choice question id of a semantic input: 'preference' (pairwise; the key is absent from a pairwise semantic
    input, so pairwise bodies and semantic hashes are unchanged) or the id another task sets (context filter:
    'relevance')."""
    return sem.get('question_id', QUESTION_ID)


def choice_body(sem):
    return {'input': {'state': dict(sem['state']),
                      'questions': {question_id(sem): {'type': 'choice', 'instructions': sem['instructions'],
                                                       'criteria': dict(sem['criteria'])}}},
            'policy': dict(CHOICE_POLICY)}


def haiku_text(sem):
    parts = [sem['instructions'], '']
    fields = STATE_FIELDS if set(sem['state']) == set(STATE_FIELDS) else tuple(sem['state'])
    for f in fields:
        parts += [f'<{f}>', sem['state'][f], f'</{f}>', '']
    parts.append('Options (criterion id: description):')
    parts += [f'{cid}: {d}' for cid, d in sem['criteria'].items()]
    parts += ['', HAIKU_FORMAT_LINE]
    return '\n'.join(parts)


def haiku_body(sem):
    return {'model': HAIKU_MODEL, 'max_tokens': HAIKU_MAX_TOKENS, 'temperature': HAIKU_TEMPERATURE,
            'messages': [{'role': 'user', 'content': haiku_text(sem)}]}


# --- restoration / strict parse (statuses are mapped to adapter failure reasons in backends.py) ----------------
def _finite_prob(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and \
        0.0 <= x <= 1.0 + PROB_VALUE_SLACK


def restore_choice(resp, to_label, question_id=QUESTION_ID):
    """resp: JSON of a COMPLETED job. to_label: criterion id -> label (pairwise: response id or 'tie'; context filter:
    'keep' / 'drop'). question_id: the answers key to read (pairwise default 'preference')."""
    out = resp.get('output') if isinstance(resp, dict) else None
    if not isinstance(out, dict):
        return {'status': 'invalid', 'why': 'no output object'}
    if out.get('error'):
        return {'status': 'handler_error', 'why': str(out.get('error'))[:300]}
    if out.get('truncated') or out.get('is_truncated'):
        return {'status': 'truncated', 'why': 'truncated flag'}
    answers = out.get('answers')
    ans = answers.get(question_id) if isinstance(answers, dict) else None
    if not isinstance(ans, dict):
        return {'status': 'invalid', 'why': f'no answers.{question_id}'}
    if ans.get('truncated') or ans.get('error'):
        return {'status': 'truncated' if ans.get('truncated') else 'handler_error', 'why': 'answer-level flag'}
    probs = ans.get('probabilities')
    if ans.get('type') != 'choice' or not isinstance(probs, dict):
        return {'status': 'invalid', 'why': 'type != choice or probabilities not a dict'}
    if set(probs) != set(to_label):
        return {'status': 'invalid', 'why': f'probability ids {sorted(probs)} != {sorted(to_label)}'}
    if not all(_finite_prob(v) for v in probs.values()):
        return {'status': 'invalid', 'why': 'probability value not a finite number in [0, 1]'}
    total = float(sum(probs.values()))
    canon = {to_label[cid]: float(v) for cid, v in probs.items()}
    base = {'probabilities': canon, 'prob_sum': total, 'criterion_choice': ans.get('choice')}
    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        return {'status': 'invalid', 'why': f'probability sum {total}', **base}
    if ans.get('choice') not in to_label:
        return {'status': 'invalid', 'why': 'choice not a criterion id', **base}
    top = max(canon.values())
    tops = sorted(lab for lab, p in canon.items() if p == top)
    if len(tops) > 1:
        return {'status': 'prob_tie', 'why': f'exact maximum shared by {tops}; no decision (not read as tie)', **base}
    if tops[0] != to_label[ans['choice']]:
        return {'status': 'choice_argmax_mismatch', 'why': 'choice != argmax', **base}
    return {'status': 'ok', 'label': tops[0], **base}


def parse_haiku(stop_reason, text, to_label):
    if stop_reason == 'max_tokens':
        return {'status': 'truncated_max_tokens'}
    if stop_reason == 'refusal':
        return {'status': 'refused'}
    if stop_reason != 'end_turn':
        return {'status': 'unexpected_stop_reason'}
    t = (text or '').strip()
    if t in to_label:
        return {'status': 'ok', 'criterion': t, 'label': to_label[t]}
    return {'status': 'format_error'}
