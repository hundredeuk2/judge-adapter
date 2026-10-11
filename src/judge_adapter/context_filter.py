# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Context / retrieval filter: one question + documents -> per-document P(keep), a keep/drop decision and the
selected ids, in the original order. Reuses the ``choice_endpoint`` backend (one choice call per document, question
id ``relevance``, options ``opt_keep`` / ``opt_drop``) and its strict probability restoration.

Fixed on purpose:
- **An explicit policy is required.** ``ContextFilter`` has no default policy (like ``AdapterConfig.transport``):
  pass ``policy=threshold(t)``, ``policy=top_k(k)``, ``policy=disabled()`` or a named ``preset``. ``disabled()``
  makes zero model calls and keeps every document.
- **Nothing is dropped silently.** One output row per input document, in input order. A document that fails (HTTP,
  timeout, format, budget stop, over-limit when flagged) is kept (fail-open) and its status says why. Input with
  missing / duplicate ids or over-limit sizes is refused before any call (``FilterInputError``); nothing is truncated.
- **Fail-open is per document.** An error or an unusable answer (format failure, handler error, truncation, an
  unexpected client exception) keeps that one document (status ``error``); the other documents are decided normally.
  The whole ``filter()`` call raises, with no result, only for invalid input (``FilterInputError``, before any call),
  or when a caller-supplied ``tokenizer`` / ``logger`` raises. An exact 0.5 / 0.5 answer keeps its score.
- **Zero kept is a normal result.** With a threshold policy every scored document can be dropped: ``selected_ids`` is
  then ``[]`` (not an error). There is no fallback document, minimum keep count or re-search; the caller decides what
  to do with an empty selection. One observed run (HLE, 90 English questions, GPT-5.2 agent, 5 documents per search,
  preset ``v1-thr0.3-dev``) dropped all 5 documents in 25 of 72 searches: a per-search count from a single run, not a
  general rate.
- **Opt-in input reduction, no quality or total-cost claim.** In that single run, on the 86 questions where all
  compared settings finished, agent input tokens were 31.4% lower than keeping the top 3 documents and 44.9% lower
  than keeping all 5. (An earlier figure, "about 44% lower than top-3 summed over all 90 questions", counted 4
  interrupted trajectories as 0 tokens and is withdrawn.) Quality improvement and equivalence were not established
  (difference within uncertainty). No total-cost advantage was established: on those 86 questions the filtered
  setting cost 4.5% more than top-3 because generator output grew (+6.7% output tokens). Not a guarantee. The
  filter-call price is the rate you configure, separate from the model you actually serve.
- **Originals stay with the caller.** The filter returns ids, never rewritten text. The caller stores the original
  documents and restores them by id (``select_originals``).
- **Run-wide limits.** One ``ContextFilter`` is one run: its client-side cost cap and submission limit are not reset
  per ``filter()`` call. Unknown usage (no server timing), an unsettled job, a contract failure, a local configuration
  failure, the cap or the submission limit stops further submissions; later documents are ``not_scored`` and kept.
- **Concurrency 1 by default.** ``concurrency > 1`` is opt-in (bounded thread pool, atomic reservations). No latency
  improvement is claimed.
- **Logs hold counts and sizes only.** No question, document text, ids or keys. Byte / token reductions are LOCAL
  ESTIMATES of document text, not billed usage, and are never converted into money.

P(keep) is the probability the served model returns for ``opt_keep``. It is not a calibrated or validated
probability that a document is useful.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import string
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field

from . import contract as C
from . import prompt as P
from .core import scrubber_of
from .gate import IDLE_ALLOWANCE_SECONDS, JOB_RESERVATION_SECONDS, JobTime
from .user_prompt import LIMITS as _PAIR_LIMITS
from .user_prompt import sha_text

LOG = logging.getLogger('judge_adapter.context_filter')

QUESTION_ID = 'relevance'
OPTION_IDS = ('opt_keep', 'opt_drop')
TO_LABEL = {'opt_keep': 'keep', 'opt_drop': 'drop'}
STATE_FIELDS = ('question', 'document')
TEMPLATE_FIELDS = ('title', 'url', 'text')

STATUS_OK, STATUS_ERROR, STATUS_NOT_SCORED = 'ok', 'error', 'not_scored'
# not_scored reasons
POLICY_DISABLED, OVER_LIMIT, BUDGET_STOP = 'policy_disabled', 'over_limit', 'not_submitted'
REDUCTION_KIND = ('LOCAL ESTIMATE of document text reduction (input documents vs selected documents); not billed '
                  'usage, not all-turn cached / uncached / output usage, not a cost saving')


class FilterInputError(ValueError):
    """The input was refused before any call (0 submissions). ``problems`` lists every problem found."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__('; '.join(self.problems))


# --- prompt ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RelevancePrompt:
    """The relevance question. ``document_template`` may use only {title}, {url}, {text} and must use {text}."""
    version: str
    instructions: str
    keep: str                          # option text of opt_keep
    drop: str                          # option text of opt_drop
    document_template: str

    def __post_init__(self):
        for name in ('version', 'instructions', 'keep', 'drop', 'document_template'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'RelevancePrompt.{name} must be a non-empty string')
        names = [f for _, f, _, _ in string.Formatter().parse(self.document_template) if f is not None]
        bad = [f for f in names if f not in TEMPLATE_FIELDS]
        if bad or 'text' not in names:
            raise ValueError(f'document_template may use only {{title}}, {{url}}, {{text}} and must use {{text}} '
                             f'(found {names})')

    def render(self, doc):
        return self.document_template.format(title=doc.title, url=doc.url, text=doc.text)

    def info(self):
        return {'version': self.version, 'question_id': QUESTION_ID, 'option_ids': list(OPTION_IDS),
                'instructions_sha256': sha_text(self.instructions), 'keep_sha256': sha_text(self.keep),
                'drop_sha256': sha_text(self.drop), 'document_template_sha256': sha_text(self.document_template)}


# V1 relevance wording (English), copied verbatim from the development experiment that produced the preset below.
V1_INSTRUCTIONS = ('You are given a question and one web search document below. Decide whether this document '
                   'provides a direct answer to the question or intermediate evidence needed to find the answer. '
                   'Being on the same topic as the question is not enough on its own. The correct answer to the '
                   'question is not given.')
V1_KEEP = 'Provides: this document contains a direct answer or needed intermediate evidence.'
V1_DROP = 'Does not provide: this document contains neither a direct answer nor needed intermediate evidence.'
# The document template the development experiment sent (its title label is Korean for "Title").
V1_DEV_DOCUMENT_TEMPLATE = '제목: {title}\nURL: {url}\n\n{text}'

DEFAULT_PROMPT = RelevancePrompt(version='relevance/v1', instructions=V1_INSTRUCTIONS, keep=V1_KEEP, drop=V1_DROP,
                                 document_template='Title: {title}\nURL: {url}\n\n{text}')


# --- policies ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    kind: str                          # 'threshold' | 'top_k' | 'disabled'
    value: object = None

    def describe(self):
        return self.kind if self.value is None else f'{self.kind}({self.value})'


def threshold(t):
    """Keep a scored document iff P(keep) >= t (t in [0, 1]; not a calibrated confidence). Unscored -> keep."""
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or not 0.0 <= t <= 1.0:
        raise ValueError('threshold t must be a number in [0, 1]')
    return Policy('threshold', float(t))


def top_k(k):
    """Keep the k scored documents with the highest P(keep); exact ties -> earlier input position first.
    Unscored documents are kept on top of the k (fail-open)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('top_k k must be an integer >= 1')
    return Policy('top_k', k)


def disabled():
    """No model call; every document is kept with status not_scored (reason policy_disabled)."""
    return Policy('disabled')


# --- limits ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FilterLimits:
    """Pre-call limits in UTF-8 bytes. Nothing is truncated. Over max_docs / question / body: the input is refused.
    A document over a per-document limit: refused (``oversize='reject'``, default) or kept unscored with status
    not_scored / over_limit (``oversize='flag'``)."""
    max_docs: int = 50                 # documents per filter() call
    max_question_bytes: int = 8_000
    max_title_bytes: int = 1_000
    max_url_bytes: int = 2_048
    max_text_bytes: int = 20_000
    max_body_bytes: int = _PAIR_LIMITS['request_body_bytes']     # one choice-endpoint wire body (same as pairwise)
    oversize: str = 'reject'           # 'reject' | 'flag'

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == 'oversize':
                if v not in ('reject', 'flag'):
                    raise ValueError("oversize must be 'reject' or 'flag'")
            elif isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f'{f.name} must be a positive integer')


# --- presets -----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    name: str
    prompt: RelevancePrompt
    policy: Policy
    limits: FilterLimits
    note: str


PRESETS = {
    'v1-thr0.3-dev': Preset(
        name='v1-thr0.3-dev',
        prompt=RelevancePrompt(version='relevance/v1-dev', instructions=V1_INSTRUCTIONS, keep=V1_KEEP, drop=V1_DROP,
                               document_template=V1_DEV_DOCUMENT_TEMPLATE),
        policy=threshold(0.3),
        limits=FilterLimits(max_title_bytes=300, max_url_bytes=500, max_text_bytes=8_000),
        note=('Development candidate validated once on 25 questions; not a quality guarantee. Fixes only the wording, '
              'document template, threshold and the size range it was tried on; the result depends on the model '
              'you serve.')),
}


# --- input -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Document:
    id: str                            # caller-owned, stable, unique within one filter() call; never sent
    title: str
    url: str
    text: str


def _b(s):
    return len(s.encode('utf-8'))


def _as_doc(i, d, problems):
    if isinstance(d, Document):
        vals = dataclasses.asdict(d)
    elif isinstance(d, Mapping):
        vals = {k: d.get(k) for k in ('id', 'title', 'url', 'text')}
    else:
        problems.append(f'documents[{i}]: not a Document or mapping')
        return None
    ok = True
    if not isinstance(vals['id'], str) or not vals['id']:
        problems.append(f'documents[{i}]: id missing or not a non-empty string')
        ok = False
    for k in ('title', 'url'):
        if not isinstance(vals[k], str):
            problems.append(f'documents[{i}]: {k} must be a string (may be empty)')
            ok = False
    if not isinstance(vals['text'], str) or not vals['text']:
        problems.append(f'documents[{i}]: text missing or empty')
        ok = False
    return Document(**vals) if ok else None


# --- budget ledger -----------------------------------------------------------------------------------------------
class _Ledger:
    """Run-wide CLIENT-SIDE estimate and stop rules for one choice endpoint (same estimate as RunGate: wall clock since
    the first submission + idle allowance + full reservation of every unsettled job). Thread-safe: every admission
    is one atomic check-and-reserve. NOT an account billing cap."""

    def __init__(self, rate, cap_usd, max_submissions, clock):
        self.lock = threading.Lock()
        self.job_time = JobTime(rate or 0.0, clock)
        self.cap = float('inf') if cap_usd is None else float(cap_usd)
        self.max_submissions = max_submissions
        self.admitted = 0                  # reserved submissions (in flight + done)
        self.submissions = self.status_polls = self.cancels = 0
        self.stopped = None                # {'code', 'detail'}
        self.unsettled = []
        self.n = 0

    def _stop(self, code, detail):
        if self.stopped is None:
            self.stopped = {'code': code, 'detail': detail}

    def admit(self):
        """-> (label, None) if reserved, else (None, why)."""
        with self.lock:
            if self.stopped:
                return None, f'stopped earlier ({self.stopped["code"]})'
            if self.max_submissions is not None and self.admitted + 1 > self.max_submissions:
                self._stop('max_submissions', f'max_submissions {self.max_submissions} reached')
                return None, self.stopped['detail']
            idle = 0.0 if self.job_time.first_submit is not None else IDLE_ALLOWANCE_SECONDS
            need = self.job_time.usd(JOB_RESERVATION_SECONDS + idle)
            if need > self.cap + 1e-12:
                self._stop('budget_cap', f'estimate with reservation ${need:.6f} > cap ${self.cap:.6f}')
                return None, self.stopped['detail']
            self.n += 1
            label = f'job-{self.n}'
            first_before = self.job_time.first_submit
            self.job_time.reserve(label)
            self.admitted += 1
            return (label, first_before), None

    def settle(self, ticket, r):
        label, first_before = ticket
        with self.lock:
            if r.get('outcome') == C.LOCAL_CONFIG and not r.get('submissions'):
                self.job_time.unsettled.pop(label, None)        # nothing was transmitted: release, hold nothing
                self.admitted -= 1
                if first_before is None and not self.job_time.unsettled and self.job_time.settled_seconds == 0.0 \
                        and self.submissions == 0:
                    self.job_time.first_submit = None
                self._stop('local_config', str(r.get('detail')))
                return
            self.submissions += r.get('submissions', 0)
            self.status_polls += r.get('status_polls', 0)
            self.cancels += r.get('cancels', 0)
            if r.get('job_final'):
                self.job_time.settle(label, r.get('server_timing'))
                if not isinstance((r.get('server_timing') or {}).get('execution_ms'), (int, float)):
                    self._stop('usage_unknown', 'job final without server execution time; cost unknown (full '
                                                'reservation charged)')
            else:
                self.unsettled.append({'label': label, 'job_observed_final': False,
                                       'unsettled_reservation_seconds': JOB_RESERVATION_SECONDS,
                                       'cancel_sent': bool(r.get('cancels'))})
                self._stop('unsettled_job', f'job not observed final ({r.get("outcome")}); its full reservation '
                                            'stays held')
            if r.get('outcome') == C.CONTRACT:
                self._stop('contract', 'contract failure')

    def snapshot(self):
        with self.lock:
            return {'kind': 'CLIENT-SIDE ESTIMATE for this ContextFilter run; NOT the bill and NOT an account billing '
                            'cap', 'cap_usd': None if self.cap == float('inf') else self.cap,
                    'max_submissions': self.max_submissions, 'submissions': self.submissions,
                    'status_polls': self.status_polls, 'cancels': self.cancels, 'retries': 0,
                    'committed_usd': self.job_time.usd(), 'usd_per_second_configured': self.job_time.rate,
                    'job_reservation_seconds': JOB_RESERVATION_SECONDS, 'idle_allowance_seconds': IDLE_ALLOWANCE_SECONDS,
                    'settled_seconds': self.job_time.settled_seconds, 'unsettled_jobs': list(self.unsettled),
                    'stopped': dict(self.stopped) if self.stopped else None}


# --- results -----------------------------------------------------------------------------------------------------
@dataclass
class DocResult:
    index: int                         # input position
    id: str                            # caller id, restored (never sent to the model)
    score: float | None                # P(keep) from the served model; None if not scored / failed
    status: str                        # ok | error | not_scored
    reason: str | None                 # error / not_scored reason; None when ok
    decision: str                      # keep | drop
    decision_basis: str                # threshold | top_k | fail_open | policy_disabled
    provenance: dict = field(default_factory=dict)


@dataclass
class FilterResult:
    documents: list                    # DocResult per input document, input order
    selected_ids: list                 # kept ids, input order; [] when nothing is kept (no substitute document)
    policy: dict
    usage: dict
    cost_estimate: dict
    timing: dict
    budget: dict
    reduction: dict                    # LOCAL ESTIMATE; also the default log record
    mock: bool

    def to_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def select_originals(documents, result):
    """The caller's own document objects for ``result.selected_ids``, in input order (no copy, no rewrite)."""
    by_index = {r.index: r for r in result.documents}
    if len(documents) != len(by_index):
        raise ValueError('documents do not match the result (different count)')
    keep = set(result.selected_ids)
    out = []
    for i, d in enumerate(documents):
        did = d.id if isinstance(d, Document) else d['id']
        if by_index[i].id != did:
            raise ValueError(f'documents[{i}] id differs from the result')
        if did in keep:
            out.append(d)
    return out


# --- filter ------------------------------------------------------------------------------------------------------
class ContextFilter:
    """See the module docstring. ``cfg`` / ``backend`` as for ``JudgeAdapter`` with ``backend='choice_endpoint'``.

    Exactly one of ``policy`` / ``preset`` is required. A preset fixes its prompt and policy (``prompt`` / ``policy``
    cannot be combined with it); ``limits`` defaults to the preset's and may be overridden (recorded in the result). A real transport needs ``budget_cap_usd`` and
    ``max_submissions`` unless the policy is ``disabled()``.
    ``tokenizer``: optional object with ``name``, ``version`` and ``count(text) -> int`` for the local reduction log.

    ``filter()`` is fail-open per document (an error or unusable answer keeps that document) and may return an empty
    ``selected_ids`` under a threshold policy; the caller decides what to do then. Scores and provenance hold the
    provider's data as received; only error ``detail`` strings are scrubbed (see the backends module).
    """

    def __init__(self, cfg, backend, *, policy=None, preset=None, prompt=None, limits=None, budget_cap_usd=None,
                 max_submissions=None, concurrency=1, tokenizer=None, clock=time.perf_counter, logger=LOG):
        if (policy is None) == (preset is None):
            raise ValueError('ContextFilter needs exactly one of policy=threshold(t) / top_k(k) / disabled() or '
                             f'preset=<name> (no default policy). Presets: {sorted(PRESETS)}')
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f'unknown preset {preset!r}; known: {sorted(PRESETS)}')
            if prompt is not None:
                raise ValueError('a preset fixes its prompt; pass policy=... and prompt=... instead of a preset')
            ps = PRESETS[preset]
            policy, prompt, limits = ps.policy, ps.prompt, limits or ps.limits
        if not isinstance(policy, Policy):
            raise ValueError('policy must be threshold(t), top_k(k) or disabled()')
        if cfg.backend != 'choice_endpoint' or getattr(backend, 'name', None) != 'choice_endpoint':
            raise ValueError('the context filter needs the choice_endpoint backend (P(keep) comes from its '
                             'probabilities); other backends are not supported')
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError('concurrency must be an integer >= 1 (default 1)')
        if budget_cap_usd is not None and not budget_cap_usd > 0:
            raise ValueError('budget_cap_usd must be positive')
        if max_submissions is not None and (isinstance(max_submissions, bool) or not isinstance(max_submissions, int)
                                            or max_submissions < 1):
            raise ValueError('max_submissions must be a positive integer')
        if cfg.transport == 'real' and policy.kind != 'disabled' and (budget_cap_usd is None or max_submissions is None):
            raise ValueError('a real transport needs budget_cap_usd and max_submissions (run-wide, client-side)')
        if tokenizer is not None and not (isinstance(getattr(tokenizer, 'name', None), str) and
                                          callable(getattr(tokenizer, 'count', None))):
            raise ValueError('tokenizer needs a name (str), a version and count(text) -> int')
        self.cfg, self.backend, self.policy = cfg, backend, policy
        self.preset = preset
        self.prompt = prompt or DEFAULT_PROMPT
        self.limits = limits or FilterLimits()
        self.concurrency = concurrency
        self.tokenizer = tokenizer
        self.clock = clock
        self.logger = logger
        self.ledger = _Ledger(getattr(backend, 'usd_per_second', None), budget_cap_usd, max_submissions, clock)
        self._calls = 0
        self._calls_lock = threading.Lock()

    # input checks: everything is validated before any call; one problem refuses the whole input
    def _validate(self, question, documents):
        problems = []
        if not isinstance(question, str) or not question.strip():
            problems.append('question must be a non-empty string')
        elif _b(question) > self.limits.max_question_bytes:
            problems.append(f'question {_b(question)} bytes > max_question_bytes {self.limits.max_question_bytes}')
        if isinstance(documents, (str, bytes, Mapping)) or not hasattr(documents, '__iter__'):
            raise FilterInputError(problems + ['documents must be a list'])
        documents = list(documents)
        if len(documents) > self.limits.max_docs:
            problems.append(f'{len(documents)} documents > max_docs {self.limits.max_docs} (not truncated)')
        docs, seen, over = [], {}, {}
        for i, d in enumerate(documents):
            doc = _as_doc(i, d, problems)
            docs.append(doc)
            if doc is None:
                continue
            if doc.id in seen:
                problems.append(f'documents[{i}]: duplicate id (also documents[{seen[doc.id]}])')
            seen.setdefault(doc.id, i)
            why = []
            for k, lim in (('title', self.limits.max_title_bytes), ('url', self.limits.max_url_bytes),
                           ('text', self.limits.max_text_bytes)):
                if _b(getattr(doc, k)) > lim:
                    why.append(f'{k} {_b(getattr(doc, k))} bytes > max_{k}_bytes {lim}')
            if not why and isinstance(question, str):
                body = _b(json.dumps(P.choice_body(self._semantic(question, doc)), ensure_ascii=False))
                if body > self.limits.max_body_bytes:
                    why.append(f'wire body {body} bytes > max_body_bytes {self.limits.max_body_bytes}')
            if why:
                if self.limits.oversize == 'flag':
                    over[i] = '; '.join(why)
                else:
                    problems.append(f'documents[{i}]: ' + '; '.join(why) + " (not truncated; oversize='flag' keeps "
                                    'it unscored instead)')
        if problems:
            raise FilterInputError(problems)
        return docs, over

    def _semantic(self, question, doc):
        return {'turn': 1, 'instructions': self.prompt.instructions,
                'state': {'question': question, 'document': self.prompt.render(doc)},
                'criteria': {'opt_keep': self.prompt.keep, 'opt_drop': self.prompt.drop}, 'question_id': QUESTION_ID}

    def _call(self, call_no, i, question, doc, ticket):
        sem = self._semantic(question, doc)
        try:
            r = self.backend.call(sem, dict(TO_LABEL))
        except Exception as e:  # noqa: BLE001 - an unexpected client error: treated as unsettled, never retried
            r = {'outcome': C.HTTP, 'detail': f'unexpected client error {type(e).__name__}', 'submissions': 1,
                 'job_final': False}
        self.ledger.settle(ticket, r)
        rec = C.CallRecord(call_id=f'{call_no}:{i}', direction='point', slot_ids={}, outcome=r['outcome'],
                           label=r.get('label'), detail=r.get('detail'), submissions=r.get('submissions', 0),
                           status_polls=r.get('status_polls', 0), cancels=r.get('cancels', 0),
                           http_status=r.get('http_status'), client_seconds=r.get('client_seconds'),
                           server_timing=r.get('server_timing'), usage=r.get('usage'),
                           response_model=r.get('response_model'), job_status=r.get('job_status'),
                           job_final=r.get('job_final'))
        probs = r.get('probabilities') if r['outcome'] in ('ok', C.NO_DECISION) else None
        p = (probs or {}).get('keep')
        score = float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) else None
        return {'record': rec, 'score': score, 'semantic_sha256': P.sha(sem),
                # hash of the scrubbed body (as before the raw-parse fix); the score comes from the raw body
                'raw_output_sha256': sha_text(scrubber_of(self.backend).text(r['raw_text']))
                if r.get('raw_text') is not None else None,
                'cost_usd': self.backend.cost([rec]) if rec.submissions else 0.0}

    def _score(self, call_no, question, todo):
        """todo: [(index, Document)] -> {index: call dict or ('not_submitted', why)}"""
        out = {}
        if self.concurrency == 1:
            for i, doc in todo:
                ticket, why = self.ledger.admit()
                if ticket is None:
                    out[i] = ('not_submitted', why)
                    continue
                out[i] = self._call(call_no, i, question, doc, ticket)
            return out
        pending = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for i, doc in todo:
                while len(pending) >= self.concurrency:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for f in done:
                        out[pending.pop(f)] = f.result()
                ticket, why = self.ledger.admit()             # atomic; refused after any stop
                if ticket is None:
                    out[i] = ('not_submitted', why)
                    continue
                pending[ex.submit(self._call, call_no, i, question, doc, ticket)] = i
            for f in list(pending):                           # in-flight requests are always settled
                out[pending.pop(f)] = f.result()
        return out

    def filter(self, question, documents):
        t0 = self.clock()
        docs, over = self._validate(question, documents)
        with self._calls_lock:
            self._calls += 1
            call_no = self._calls
        rows = [None] * len(docs)
        if self.policy.kind == 'disabled':
            calls = {}
        else:
            todo = [(i, d) for i, d in enumerate(docs) if i not in over]
            calls = self._score(call_no, question, todo)
        for i, d in enumerate(docs):
            prov = {'prompt': self.prompt.version, 'question_id': QUESTION_ID, 'backend': self.backend.name,
                    'policy': self.policy.describe(), 'preset': self.preset}
            if self.policy.kind == 'disabled':
                rows[i] = DocResult(i, d.id, None, STATUS_NOT_SCORED, POLICY_DISABLED, 'keep', 'policy_disabled',
                                    {**prov, 'submissions': 0})
            elif i in over:
                rows[i] = DocResult(i, d.id, None, STATUS_NOT_SCORED, OVER_LIMIT, 'keep', 'fail_open',
                                    {**prov, 'submissions': 0, 'detail': over[i]})
            elif i not in calls:                              # defensive: never drop a document without a row
                rows[i] = DocResult(i, d.id, None, STATUS_NOT_SCORED, 'no_result', 'keep', 'fail_open',
                                    {**prov, 'submissions': 0})
            elif isinstance(calls[i], tuple):
                rows[i] = DocResult(i, d.id, None, STATUS_NOT_SCORED, BUDGET_STOP, 'keep', 'fail_open',
                                    {**prov, 'submissions': 0, 'detail': calls[i][1]})
            else:
                c = calls[i]
                rec = c['record']
                status = STATUS_OK if c['score'] is not None else STATUS_ERROR
                prov.update(call_id=rec.call_id, model_outcome=rec.outcome, detail=rec.detail,
                            response_model=rec.response_model, submissions=rec.submissions,
                            status_polls=rec.status_polls, cancels=rec.cancels, http_status=rec.http_status,
                            job_status=rec.job_status, job_final=rec.job_final, client_seconds=rec.client_seconds,
                            server_timing=rec.server_timing, usage=rec.usage, cost_usd_estimate=c['cost_usd'],
                            semantic_sha256=c['semantic_sha256'], raw_output_sha256=c['raw_output_sha256'])
                rows[i] = DocResult(i, d.id, c['score'], status, None if status == STATUS_OK else rec.outcome,
                                    'keep', 'fail_open', prov)
        self._decide(rows)
        selected = [r.id for r in rows if r.decision == 'keep']
        return self._result(rows, docs, selected, t0)

    def _decide(self, rows):
        pol = self.policy
        if pol.kind == 'disabled':
            return
        scored = [r for r in rows if r.score is not None]
        if pol.kind == 'threshold':
            for r in scored:
                r.decision, r.decision_basis = ('keep' if r.score >= pol.value else 'drop'), 'threshold'
        else:
            ranked = sorted(scored, key=lambda r: (-r.score, r.index))     # stable: earlier input wins exact ties
            top = {r.index for r in ranked[:pol.value]}
            for r in scored:
                r.decision, r.decision_basis = ('keep' if r.index in top else 'drop'), 'top_k'

    def _result(self, rows, docs, selected, t0):
        recs = [r.provenance for r in rows if r.provenance.get('submissions')]
        usage_rows = [p.get('usage') for p in recs]
        usage = {'input_tokens': sum((u or {}).get('input_tokens') or 0 for u in usage_rows),
                 'output_tokens': sum((u or {}).get('output_tokens') or 0 for u in usage_rows),
                 'complete': all(isinstance((u or {}).get('input_tokens'), int) for u in usage_rows),
                 'submissions': sum(p['submissions'] for p in recs),
                 'note': 'filter model calls only (server-reported tokens where present); not the host model usage'}
        costs = [p.get('cost_usd_estimate') for p in recs]
        fake = self.cfg.transport == 'fake'
        cost = {'usd': None if any(c is None for c in costs) else round(sum(costs), 8), 'is_estimate': True,
                'mock': fake, 'basis': ('MOCK: fake transport; not a cost. ' if fake else '') + self.backend.cost_basis,
                'note': 'filter calls of this filter() call only; failed calls included'}
        execs = [(p.get('server_timing') or {}).get('execution_ms') for p in recs]
        timing = {'total_seconds': self.clock() - t0,
                  'client_seconds_sum': sum(p.get('client_seconds') or 0.0 for p in recs),
                  'server_execution_ms_sum': None if any(not isinstance(e, (int, float)) for e in execs) else sum(execs),
                  'concurrency': self.concurrency, 'note': 'measured only; no latency improvement is claimed'}
        reduction = self._reduction(rows, docs, selected)
        res = FilterResult(documents=rows, selected_ids=selected,
                           policy={'policy': self.policy.describe(), 'preset': self.preset,
                                   'preset_note': PRESETS[self.preset].note if self.preset else None,
                                   'prompt': self.prompt.info(), 'limits': dataclasses.asdict(self.limits)},
                           usage=usage, cost_estimate=cost, timing=timing, budget=self.ledger.snapshot(),
                           reduction=reduction, mock=fake)
        if self.logger is not None:
            self.logger.info('context_filter %s', json.dumps(reduction, ensure_ascii=False, sort_keys=True))
        return res

    def _reduction(self, rows, docs, selected):
        keep = set(selected)
        sel = [d for d in docs if d.id in keep]
        out = {'kind': REDUCTION_KIND, 'docs_in': len(docs), 'docs_out': len(sel),
               'text_bytes_in': sum(_b(d.text) for d in docs), 'text_bytes_out': sum(_b(d.text) for d in sel),
               'status_counts': {s: sum(r.status == s for r in rows) for s in (STATUS_OK, STATUS_ERROR,
                                                                              STATUS_NOT_SCORED)},
               'stopped': (self.ledger.stopped or {}).get('code'), 'mock': self.cfg.transport == 'fake',
               'tokens_supported': self.tokenizer is not None, 'tokens': None,
               'billed_savings': 'not computed (local text estimate only)'}
        if self.tokenizer is not None:
            tk = self.tokenizer
            counts = {d.id: int(tk.count(d.text)) for d in docs}
            out['tokens'] = {'tokenizer': tk.name, 'tokenizer_version': str(getattr(tk, 'version', None)),
                             'text_tokens_in': sum(counts.values()), 'text_tokens_out': sum(counts[d.id] for d in sel)}
        return out

    def close(self):
        self.backend.close()
