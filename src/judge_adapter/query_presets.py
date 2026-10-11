# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Wording and example values of the optional query-stage options (PreToolUse), kept in one module for easy revision.

EXPERIMENTAL. Every string and value here is a first draft: untuned, unvalidated, never measured against a real model
or a labelled set. No quality claim is made for any of them. The thresholds in ``query_hook`` are pre-fixed policy
values, not calibrated ones, and a high probability from the served model is never treated as a correct judgment.

Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EXPERIMENTAL_NOTE = ('EXPERIMENTAL: untuned and unvalidated wording; the result depends on the model you serve. No '
                     'quality claim.')


@dataclass(frozen=True)
class ChoicePrompt:
    """One choice question: instructions plus fixed options. ``options`` maps a label to its option text; the wire
    option id is ``opt_<label>``."""
    name: str
    question_id: str
    instructions: str
    options: tuple = ()               # ((label, text), ...)
    note: str = EXPERIMENTAL_NOTE


# --- search gate ----------------------------------------------------------------------------------------------------
# Three options: the third one lets the model abstain when the query alone does not show whether a search is needed.
# The gate denies only when P(not_needed) >= search_gate_threshold; any other answer keeps the call.
GATE_PROMPTS = {
    'gate-v0-dev': ChoicePrompt(
        name='gate-v0-dev', question_id='search_need',
        instructions=(
            'An AI assistant is about to run the web search query in the field "search_query". Only the query is '
            'given; the user\'s message and the conversation are not. Decide whether running this search is needed. '
            'Choose "not needed" only when the query asks for stable, widely known general knowledge that does not '
            'depend on recent or changing information, on a specific website or document, or on verifying a fact. '
            'Choose "needed" when the answer may depend on recent or changing information, specific sources, facts '
            'that should be checked, or anything uncertain. Choose "cannot tell" when the query alone does not show '
            'whether a search is needed.'),
        options=(('needed', 'Needed: the search is needed or probably useful.'),
                 ('not_needed', 'Not needed: the query asks only for stable, widely known general knowledge.'),
                 ('cannot_tell', 'Cannot tell: the query alone does not show whether a search is needed.'))),
}
DEFAULT_GATE_PROMPT = 'gate-v0-dev'

# Fixed notice shown to Claude on a deny (permissionDecisionReason; the hooks reference: 'For "deny", shown to
# Claude'). It is a fixed string: the model's own reasoning is never shown or presented as fact.
GATE_DENY_NOTICE = (
    'davi search gate (experimental, optional): this search was skipped because an automated check rated it as '
    'probably unnecessary. This is not a verified judgment. If the user asked for a search, or the answer needs '
    'current information, a specific source or verification, retry the same query and it will run. Otherwise answer '
    'from the context you already have.')

# Keyword guard, checked locally before any endpoint call (no call is made when it matches): queries with recency,
# verification or specific-source keywords keep the call. A heuristic only, not exhaustive, and it does NOT preserve
# user intent: the query alone cannot identify an explicit user request to search (the PreToolUse input has no
# user-message field), which is why a denied query is always allowed on retry.
GATE_KEEP_PATTERNS = {
    'recency': re.compile(
        r'(?i)\b(19|20)\d{2}\b|\b(latest|newest|recent|recently|current|currently|today|tonight|yesterday|tomorrow|'
        r'this (week|month|year)|now|news|breaking|update[ds]?|release[ds]?|version|changelog|price[sd]?|stock|'
        r'weather|score[sd]?|schedule|deadline|live)\b|최신|최근|오늘|어제|현재|지금|뉴스|업데이트|출시|버전|가격|일정'),
    'verification': re.compile(
        r'(?i)\b(verify|verified|verification|fact[- ]?check|source[sd]?|citations?|cite|reference[sd]?|official|'
        r'documentation|docs|evidence|confirm|is it true|according to)\b|출처|확인|검증|공식|근거|사실'),
    'specific_source': re.compile(r'(?i)\bsite:|https?://|\bwww\.|\b[a-z0-9-]+\.(com|org|net|io|ai|dev|edu|gov)\b'),
}


# Explicit search request in the USER'S PROMPT (UserPromptSubmit hook, ``prompt_signal``). Checked on the prompt after
# NFKC normalization and case folding; any match marks the current prompt of the session as "explicit request", and
# the search gate then keeps every search of that prompt without an endpoint call. Deliberately broad: a false
# positive only keeps a search (the gate does nothing), a false negative can deny a search the user asked for. A
# phrasing that is NOT in this list is NOT evidence that the user did not ask for a search; adding the phrase of a
# known failure fixes that phrase only, not the problem in general. Fixed list, not tuned on any data.
EXPLICIT_SEARCH_PATTERNS = {
    # English: any form of "search" ("web search", "search the web", "search for", "use (the) search", "using web
    # search", also unrelated senses such as "binary search": accepted false positives).
    'search': re.compile(r'\bsearch(?:es|ed|ing)?\b|\bweb[\s-]?search'),
    'look_up': re.compile(r'\blook(?:s|ed|ing)?\s+(?:(?:it|this|that|them|these|those)\s+)?up\b|\blookup\b'),
    'google': re.compile(r'\bgoogl(?:e|es|ed|ing)\b|\bbing\b|\bduckduckgo\b'),
    'online': re.compile(r'\bonline\b|\binternet\b|\bon the web\b|\bbrowse\b|\bbrowsing\b'),
    'source': re.compile(r'\bcit(?:e|es|ed|ing|ation|ations)\b|\bsources?\b|\breferences?\b|\blinks?\b'),
    'verify': re.compile(r'\bverif(?:y|ies|ied|ication)\b|\bfact[\s-]?check|\bconfirm(?:s|ed|ing)?\b|\bdouble[\s-]?check'),
    'recency': re.compile(r'\b(?:latest|newest|current|currently|today|tonight|recent|recently|news|now|this '
                          r'(?:week|month|year)|up[\s-]to[\s-]date|as of)\b|\b(?:19|20)\d{2}\b'),
    # Korean: 웹 검색 / 검색해(서/줘) / 찾아봐 / 찾아줘 / 출처 / 최신 / 오늘 and close variants.
    'ko_search': re.compile(r'웹\s?검색|검색|서치|구글|인터넷|온라인|찾아\s?(?:봐|줘|서|보|볼|주)|찾아봐|찾아줘'),
    'ko_source': re.compile(r'출처|근거|인용|링크|레퍼런스|참고\s?자료'),
    'ko_verify': re.compile(r'확인|검증|팩트\s?체크'),
    'ko_recency': re.compile(r'최신|최근|오늘|현재|지금|요즘|뉴스|올해|이번\s?(?:주|달|해)'),
}
# A prompt longer than this many characters is treated as an explicit request without scanning (conservative).
EXPLICIT_SCAN_MAX_CHARS = 200_000
# Follow-up rule (conservative): while a fresh explicit-request flag exists for the session, a prompt WITHOUT a signal
# leaves the flag in place (it still expires after its TTL) -- whatever its length ("continue", "yes, go ahead", "계속",
# "그걸로", "now do the same for a month"). It clears the flag only if it clearly starts a new task: the normalized
# prompt begins with one of these fixed markers. Anything else, including an ambiguous follow-up, keeps the flag.
NEW_TASK_PREFIXES = ('new task', 'new question', 'new topic', 'different question', 'different topic',
                     'unrelated question', 'unrelated:', 'change of topic', 'changing the topic',
                     '새 작업', '새로운 작업', '새 질문', '새로운 질문', '다른 질문', '다른 주제', '주제를 바꿔', '주제 바꿔')


# --- domain presets -------------------------------------------------------------------------------------------------
DOMAIN_PROMPTS = {
    'domain-v0-dev': ChoicePrompt(
        name='domain-v0-dev', question_id='domain_preset',
        instructions=(
            'An AI assistant is about to run the web search query in the field "search_query". Choose the one search '
            'preset that fits the query best. A preset limits which websites the search can return. Choose "none" '
            'unless a restricted preset clearly suits the query: restricting a search can hide the answer.')),
}
DEFAULT_DOMAIN_PROMPT = 'domain-v0-dev'
NONE_PRESET = 'none'
NONE_DESCRIPTION = 'No restriction: search the whole web.'

# EXAMPLES only, used when the domain_presets option is empty. They are not the right domains for every query and are
# not a recommendation; replace them with your own lists. Each preset sets exactly one list (include or exclude).
EXAMPLE_DOMAIN_PRESETS = {
    'docs': {'description': 'EXAMPLE preset: official software documentation and reference sites.',
             'include_domains': ['docs.python.org', 'developer.mozilla.org', 'learn.microsoft.com',
                                 'docs.github.com']},
    'academic': {'description': 'EXAMPLE preset: scholarly sources (preprints, papers, abstracts).',
                 'include_domains': ['arxiv.org', 'aclanthology.org', 'openreview.net', 'pubmed.ncbi.nlm.nih.gov']},
}
