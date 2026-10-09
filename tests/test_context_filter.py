"""Context filter, model-free: synthetic documents and an in-process fake choice endpoint only.

Covers: id / option / order mapping, the threshold boundary, stable ties, missing / error handling (fail-open, honest
empty result), run-wide stops, opt-in bounded concurrency with fake barriers (atomic reservations, no submission after a
stop, in-flight settlement, order restoration), zero model calls without an active policy, the local reduction log.
Network and key reads fail every test."""
from __future__ import annotations

import json
import logging
import socket
import threading
import time

import httpx
import pytest

from judge_adapter import AdapterConfig, build_backend, fakes
from judge_adapter import context_filter as F
from judge_adapter import prompt as P
from judge_adapter.context_filter import ContextFilter, Document, FilterInputError, FilterLimits
from judge_adapter.user_prompt import sha_text

from conftest import block_key_reads

Q = 'Which synthetic city hosts the fictional festival?'


@pytest.fixture(autouse=True)
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


def fake_cfg(**kw):
    return AdapterConfig(transport='fake', backend='choice_endpoint', trace_dir='unused-trace', **kw)


def make(policy=None, preset=None, handler=None, fake=None, clock=lambda: 0.0, **kw):
    fake = fake or fakes.FakeRelevanceEndpoint()
    cfg = fake_cfg(choice_usd_per_second=0.001)
    tr = httpx.MockTransport(handler or fake.handle)
    be = build_backend(cfg, transport=tr, key_provider=lambda: fakes.DUMMY_KEY, clock=clock, sleep=lambda s: None)
    flt = ContextFilter(cfg, be, policy=policy, preset=preset, clock=clock, **kw)
    return flt, fake


def doc(i, text, title='Synthetic title', url='https://example.invalid/doc'):
    return Document(id=f'doc-{i}', title=title, url=f'{url}/{i}', text=text)


def docs_with(*texts):
    return [doc(i, t) for i, t in enumerate(texts)]


def rows(res):
    return [(r.id, r.score, r.status, r.reason, r.decision) for r in res.documents]


# --- mapping -------------------------------------------------------------------------------------------------------
def test_wire_body_options_ids_and_order():
    flt, fake = make(F.threshold(0.5))
    ds = docs_with('alpha [[KEEP=0.9]]', 'beta [[KEEP=0.1]]', 'gamma [[KEEP=0.6]]')
    res = flt.filter(Q, ds)
    assert [r.id for r in res.documents] == ['doc-0', 'doc-1', 'doc-2']
    assert [r.index for r in res.documents] == [0, 1, 2]
    assert res.selected_ids == ['doc-0', 'doc-2']
    assert [r.score for r in res.documents] == [0.9, 0.1, 0.6]
    assert len(fake.bodies) == 3
    for body, d in zip(fake.bodies, ds):
        q = body['input']['questions']
        assert list(q) == ['relevance']
        assert list(q['relevance']['criteria']) == ['opt_keep', 'opt_drop']
        assert q['relevance']['criteria'] == {'opt_keep': F.DEFAULT_PROMPT.keep, 'opt_drop': F.DEFAULT_PROMPT.drop}
        assert set(body['input']['state']) == {'question', 'document'}
        assert body['input']['state']['question'] == Q
        assert body['input']['state']['document'] == f'Title: {d.title}\nURL: {d.url}\n\n{d.text}'
        assert d.id not in json.dumps(body)                    # caller ids are never sent
    assert all(r.provenance['question_id'] == 'relevance' for r in res.documents)


def test_probability_keys_are_read_by_option_id_not_position():
    def handler(request):
        body = json.loads(request.content)
        js = {'id': 'j', 'status': 'COMPLETED', 'delayTime': 1, 'executionTime': 2,
              'output': {'answers': {'relevance': {'type': 'choice', 'choice': 'opt_drop',
                                                   'probabilities': {'opt_drop': 0.8, 'opt_keep': 0.2}}}}}
        assert 'relevance' in body['input']['questions']
        return httpx.Response(200, json=js)
    flt, _ = make(F.threshold(0.3), handler=handler)
    res = flt.filter(Q, docs_with('x'))
    assert rows(res) == [('doc-0', 0.2, 'ok', None, 'drop')]


def test_pairwise_body_unchanged_by_question_id_support():
    sem = P.semantic(('q',), ('a',), ('b',), 1)
    assert list(P.choice_body(sem)['input']['questions']) == ['preference']
    assert 'question_id' not in sem


def test_mapping_input_and_select_originals():
    flt, _ = make(F.threshold(0.3))
    originals = [{'id': 'm1', 'title': '', 'url': '', 'text': '[[KEEP=0.8]]', 'extra': 1},
                 {'id': 'm2', 'title': '', 'url': '', 'text': '[[KEEP=0.1]]'}]
    res = flt.filter(Q, originals)
    kept = F.select_originals(originals, res)
    assert kept == [originals[0]] and kept[0] is originals[0]


# --- threshold boundary, ties, empty -------------------------------------------------------------------------------
@pytest.mark.parametrize('kw', [{'policy': F.threshold(0.3)}, {'preset': 'v1-thr0.3-dev'}])
def test_threshold_boundary(kw):
    flt, _ = make(**kw)
    res = flt.filter(Q, docs_with('[[KEEP=0.2999]]', '[[KEEP=0.3]]', '[[KEEP=0.3001]]'))
    assert [r.decision for r in res.documents] == ['drop', 'keep', 'keep']
    assert res.selected_ids == ['doc-1', 'doc-2']


def test_exact_tie_probability_is_a_score():
    flt, _ = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.5]]'))
    r = res.documents[0]
    assert (r.score, r.status, r.decision) == (0.5, 'ok', 'keep')
    assert r.provenance['model_outcome'] == 'no_decision'


def test_top_k_stable_ties_by_input_order():
    flt, _ = make(F.top_k(2))
    res = flt.filter(Q, docs_with('[[KEEP=0.7]]', '[[KEEP=0.9]]', '[[KEEP=0.7]]', '[[KEEP=0.7]]'))
    assert res.selected_ids == ['doc-0', 'doc-1']
    assert [r.decision_basis for r in res.documents] == ['top_k'] * 4


def test_top_k_failed_documents_kept_on_top():
    flt, _ = make(F.top_k(1))
    res = flt.filter(Q, docs_with('[[KEEP=0.4]]', 'FAKE:FORMAT', '[[KEEP=0.6]]'))
    assert res.selected_ids == ['doc-1', 'doc-2']
    assert res.documents[1].decision_basis == 'fail_open'


def test_nothing_selected_is_an_empty_list():
    flt, _ = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.1]]', '[[KEEP=0.2]]'))
    assert res.selected_ids == [] and len(res.documents) == 2
    assert res.reduction['docs_out'] == 0 and res.reduction['text_bytes_out'] == 0


def test_no_documents_no_calls():
    flt, fake = make(F.threshold(0.3))
    res = flt.filter(Q, [])
    assert res.selected_ids == [] and res.documents == [] and fake.requests == []


# --- missing / error -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('bad, expect', [
    ([{'title': '', 'url': '', 'text': 'x'}], 'id missing'),
    ([{'id': '', 'title': '', 'url': '', 'text': 'x'}], 'id missing'),
    ([doc(0, 'x'), Document('doc-0', '', '', 'y')], 'duplicate id'),
    ([Document('a', '', '', '')], 'text missing'),
    ([Document('a', None, '', 'x')], 'title must be a string'),
    (['just a string'], 'not a Document'),
])
def test_bad_input_refused_before_any_call(bad, expect):
    flt, fake = make(F.threshold(0.3))
    with pytest.raises(FilterInputError) as e:
        flt.filter(Q, bad)
    assert expect in str(e.value) and fake.requests == []


def test_bad_question_refused():
    flt, fake = make(F.threshold(0.3))
    with pytest.raises(FilterInputError):
        flt.filter('  ', docs_with('x'))
    assert fake.requests == []


def test_format_and_handler_errors_fail_open_and_continue():
    flt, fake = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('FAKE:FORMAT', 'FAKE:REFUSE', '[[KEEP=0.1]]'))
    assert rows(res) == [('doc-0', None, 'error', 'format_error', 'keep'),
                         ('doc-1', None, 'error', 'backend_error', 'keep'),
                         ('doc-2', 0.1, 'ok', None, 'drop')]
    assert res.selected_ids == ['doc-0', 'doc-1'] and fake.count('POST') == 3 and res.budget['stopped'] is None


@pytest.mark.parametrize('marker, reason, code', [('FAKE:HTTP500', 'http', 'unsettled_job'),
                                                  ('FAKE:TIMEOUT', 'timeout', 'unsettled_job')])
def test_unsettled_job_stops_further_submissions(marker, reason, code):
    flt, fake = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.9]]', marker, '[[KEEP=0.9]]', '[[KEEP=0.1]]'))
    assert rows(res) == [('doc-0', 0.9, 'ok', None, 'keep'), ('doc-1', None, 'error', reason, 'keep'),
                         ('doc-2', None, 'not_scored', 'not_submitted', 'keep'),
                         ('doc-3', None, 'not_scored', 'not_submitted', 'keep')]
    assert fake.count('POST', '/runsync') == 2
    assert res.budget['stopped']['code'] == code and res.budget['unsettled_jobs']
    assert res.reduction['stopped'] == code
    again = flt.filter(Q, docs_with('[[KEEP=0.9]]'))               # run-wide: a stop is not reset per call
    assert again.documents[0].status == 'not_scored' and fake.count('POST', '/runsync') == 2


def test_unknown_usage_stops_after_settling():
    flt, fake = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.9]] FAKE:NOTIMING', '[[KEEP=0.9]]'))
    assert res.documents[0].status == 'ok' and res.documents[1].reason == 'not_submitted'
    assert res.budget['stopped']['code'] == 'usage_unknown' and res.cost_estimate['usd'] is None
    assert fake.count('POST', '/runsync') == 1


def test_queue_poll_is_not_a_resubmission():
    flt, fake = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.8]] FAKE:QUEUE'))
    assert res.documents[0].score == 0.8 and res.documents[0].provenance['status_polls'] == 1
    assert res.usage['submissions'] == 1 and fake.count('GET', '/status/') == 1


def test_max_submissions_is_run_wide():
    flt, fake = make(F.threshold(0.3), max_submissions=2)
    res = flt.filter(Q, docs_with('[[KEEP=0.9]]', '[[KEEP=0.9]]', '[[KEEP=0.9]]'))
    assert [r.status for r in res.documents] == ['ok', 'ok', 'not_scored']
    assert res.budget['stopped']['code'] == 'max_submissions'
    res2 = flt.filter(Q, docs_with('[[KEEP=0.9]]'))
    assert res2.selected_ids == ['doc-0'] and res2.documents[0].status == 'not_scored'
    assert fake.count('POST') == 2


def test_budget_cap_refuses_before_sending():
    flt, fake = make(F.threshold(0.3), budget_cap_usd=0.5)     # first job needs (661 + 60) s x $0.001 = $0.721
    res = flt.filter(Q, docs_with('[[KEEP=0.9]]'))
    assert res.documents[0].reason == 'not_submitted' and fake.requests == []
    assert res.budget['stopped']['code'] == 'budget_cap'


def test_local_config_failure_sends_nothing(monkeypatch):
    block_key_reads(monkeypatch)
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', trace_dir='unused')   # no endpoint, no rate
    flt = ContextFilter(cfg, build_backend(cfg), policy=F.threshold(0.3), budget_cap_usd=1.0, max_submissions=5)
    res = flt.filter(Q, docs_with('x', 'y'))
    assert rows(res) == [('doc-0', None, 'error', 'local_config', 'keep'),
                         ('doc-1', None, 'not_scored', 'not_submitted', 'keep')]
    assert res.budget['stopped']['code'] == 'local_config' and res.budget['submissions'] == 0
    assert res.budget['committed_usd'] == 0.0


# --- limits --------------------------------------------------------------------------------------------------------
def test_oversize_rejected_by_default_and_never_truncated():
    flt, fake = make(F.threshold(0.3), limits=FilterLimits(max_text_bytes=10))
    with pytest.raises(FilterInputError) as e:
        flt.filter(Q, docs_with('short', 'x' * 11))
    assert 'documents[1]' in str(e.value) and fake.requests == []


def test_oversize_flag_keeps_unscored():
    flt, fake = make(F.threshold(0.3), limits=FilterLimits(max_text_bytes=10, oversize='flag'))
    res = flt.filter(Q, docs_with('[[KEEP=0.1]]'[:10], 'x' * 11))
    assert rows(res)[1] == ('doc-1', None, 'not_scored', 'over_limit', 'keep')
    assert fake.count('POST') == 1 and fake.bodies[0]['input']['state']['document'].endswith('[[KEEP=0.1')


def test_max_docs_refused():
    flt, fake = make(F.threshold(0.3), limits=FilterLimits(max_docs=2))
    with pytest.raises(FilterInputError):
        flt.filter(Q, docs_with('a', 'b', 'c'))
    assert fake.requests == []


def test_preset_limits_are_the_tried_range():
    flt, fake = make(preset='v1-thr0.3-dev')
    with pytest.raises(FilterInputError):
        flt.filter(Q, docs_with('x' * 8_001))
    flt2, _ = make(preset='v1-thr0.3-dev', limits=FilterLimits(max_text_bytes=8_000, oversize='flag'))
    assert flt2.filter(Q, docs_with('x' * 8_001)).documents[0].reason == 'over_limit'


# --- policy required / disabled ------------------------------------------------------------------------------------
def test_policy_is_required():
    cfg = fake_cfg()
    be = build_backend(cfg, transport=fakes.FakeRelevanceEndpoint().transport(), key_provider=lambda: fakes.DUMMY_KEY)
    with pytest.raises(ValueError, match='exactly one'):
        ContextFilter(cfg, be)
    with pytest.raises(ValueError, match='exactly one'):
        ContextFilter(cfg, be, policy=F.threshold(0.3), preset='v1-thr0.3-dev')
    with pytest.raises(ValueError, match='unknown preset'):
        ContextFilter(cfg, be, preset='best')
    with pytest.raises(ValueError, match='fixes its prompt'):
        ContextFilter(cfg, be, preset='v1-thr0.3-dev', prompt=F.DEFAULT_PROMPT)


def test_disabled_makes_zero_model_calls_even_on_real_transport(monkeypatch):
    block_key_reads(monkeypatch)
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', trace_dir='unused')
    flt = ContextFilter(cfg, build_backend(cfg), policy=F.disabled())
    res = flt.filter(Q, docs_with('a', 'b'))
    assert res.selected_ids == ['doc-0', 'doc-1']
    assert [(r.status, r.reason, r.decision_basis) for r in res.documents] == [
        ('not_scored', 'policy_disabled', 'policy_disabled')] * 2
    assert res.usage['submissions'] == 0 and res.budget['submissions'] == 0


def test_disabled_fake_zero_requests():
    flt, fake = make(F.disabled())
    flt.filter(Q, docs_with('[[KEEP=0.1]]'))
    assert fake.requests == []


def test_real_transport_needs_caps():
    cfg = AdapterConfig(transport='real', backend='choice_endpoint', trace_dir='unused')
    with pytest.raises(ValueError, match='budget_cap_usd and max_submissions'):
        ContextFilter(cfg, build_backend(cfg), policy=F.threshold(0.3))


def test_other_backend_refused():
    cfg = AdapterConfig(transport='fake', trace_dir='unused')
    be = build_backend(cfg, transport=fakes.FakeAnthropic().transport(), key_provider=lambda: fakes.DUMMY_KEY)
    with pytest.raises(ValueError, match='choice_endpoint'):
        ContextFilter(cfg, be, policy=F.threshold(0.3))


@pytest.mark.parametrize('bad', [lambda: F.threshold(1.5), lambda: F.threshold(-0.1), lambda: F.threshold(True),
                                 lambda: F.top_k(0), lambda: F.top_k(1.0)])
def test_policy_arguments_checked(bad):
    with pytest.raises(ValueError):
        bad()


def test_preset_strings_pinned():
    ps = F.PRESETS['v1-thr0.3-dev']
    assert ps.policy == F.threshold(0.3)
    assert sha_text(ps.prompt.instructions) == 'f49efe79aace208495862f07ff9caa70a12f29f3ad7821b55f8e0fdf565d4ee1'
    assert sha_text(ps.prompt.keep) == '67762929dc6913c457f45518c934c43e99d525fef3a6d0b27d3946a0b192af92'
    assert sha_text(ps.prompt.drop) == 'f45693c58465fdc00165ed79f9c5bce4ada3b0d75f6b7c92b97d54422fad7de6'
    assert 'not a quality guarantee' in ps.note and '25 questions' in ps.note


# --- logging -------------------------------------------------------------------------------------------------------
class _Tok:
    name, version = 'whitespace-test', '0'

    @staticmethod
    def count(text):
        return len(text.split())


def test_reduction_log_counts_only(caplog):
    flt, _ = make(F.threshold(0.3), tokenizer=_Tok())
    secret_text = 'unique-original-text-marker [[KEEP=0.9]]'
    with caplog.at_level(logging.INFO, logger='judge_adapter.context_filter'):
        res = flt.filter(Q, docs_with(secret_text, 'dropped words here [[KEEP=0.1]]'))
    logged = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'docs_in' in logged and 'LOCAL ESTIMATE' in logged
    for s in (Q, 'unique-original-text-marker', 'doc-0', fakes.DUMMY_KEY):
        assert s not in logged
    red = res.reduction
    assert (red['docs_in'], red['docs_out']) == (2, 1)
    assert red['text_bytes_in'] > red['text_bytes_out'] > 0
    assert red['tokens'] == {'tokenizer': 'whitespace-test', 'tokenizer_version': '0', 'text_tokens_in': 6,
                             'text_tokens_out': 2}
    assert 'usd' not in json.dumps(red) and red['billed_savings'].startswith('not computed')


def test_no_tokenizer_no_token_counts():
    flt, _ = make(F.threshold(0.3))
    red = flt.filter(Q, docs_with('[[KEEP=0.9]]')).reduction
    assert red['tokens'] is None and red['tokens_supported'] is False


def test_result_labels_cost_as_estimate_and_mock():
    flt, _ = make(F.threshold(0.3))
    res = flt.filter(Q, docs_with('[[KEEP=0.9]]'))
    assert res.cost_estimate['is_estimate'] is True and res.cost_estimate['mock'] is True
    assert res.cost_estimate['basis'].startswith('MOCK')
    assert res.mock is True and 'no latency improvement' in res.timing['note']
    json.dumps(res.to_dict())


# --- concurrency (fake barriers) -----------------------------------------------------------------------------------
class InFlight:
    def __init__(self):
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *a):
        with self.lock:
            self.now -= 1


def test_default_concurrency_is_one():
    fake, inflight = fakes.FakeRelevanceEndpoint(), InFlight()

    def handler(request):
        with inflight:
            time.sleep(0.005)
            return fake.handle(request)
    flt, _ = make(F.threshold(0.3), handler=handler, fake=fake)
    assert flt.concurrency == 1
    flt.filter(Q, docs_with(*['[[KEEP=0.9]]'] * 4))
    assert inflight.peak == 1


def test_bounded_concurrency_really_overlaps_and_restores_order():
    fake = fakes.FakeRelevanceEndpoint()
    barrier = threading.Barrier(3, timeout=5)          # passes only if 3 requests are in flight at once
    order = []
    done = {i: threading.Event() for i in range(3)}

    def handler(request):
        d = json.loads(request.content)['input']['state']['document']
        i = int(d.split('#')[1].split()[0])
        if i < 3:
            barrier.wait()
            if i < 2:                                    # finish in reverse order: 2, 1, 0
                assert done[i + 1].wait(5)
        resp = fake.handle(request)
        order.append(i)
        if i < 3:
            done[i].set()
        return resp
    flt, _ = make(F.threshold(0.3), handler=handler, fake=fake, concurrency=3)
    ds = docs_with('#0 [[KEEP=0.9]]', '#1 [[KEEP=0.1]]', '#2 [[KEEP=0.5]]', '#3 [[KEEP=0.8]]')
    res = flt.filter(Q, ds)
    assert [i for i in order if i < 3] == [2, 1, 0]          # completion order differs from input order
    assert [r.id for r in res.documents] == ['doc-0', 'doc-1', 'doc-2', 'doc-3']
    assert [r.score for r in res.documents] == [0.9, 0.1, 0.5, 0.8]
    assert res.selected_ids == ['doc-0', 'doc-2', 'doc-3']


def test_concurrent_reservations_are_atomic():
    """Cap fits exactly two open reservations: (60 + 2 x 661) s x $0.001 = $1.382 <= 1.5 < $2.043 for a third."""
    fake = fakes.FakeRelevanceEndpoint()
    barrier = threading.Barrier(2, timeout=5)
    release = threading.Event()
    flt, _ = make(F.threshold(0.3), fake=fake, concurrency=4, budget_cap_usd=1.5,
                  handler=lambda req: (barrier.wait(), release.wait(5), fake.handle(req))[2])
    admit = flt.ledger.admit
    attempts = []

    def counting_admit():
        out = admit()
        attempts.append(out[0] is not None)
        if len(attempts) == 3:                           # the third admission was decided while 2 jobs are open
            release.set()
        return out
    flt.ledger.admit = counting_admit
    res = flt.filter(Q, docs_with(*['[[KEEP=0.9]]'] * 6))
    assert attempts[:3] == [True, True, False]
    assert [r.status for r in res.documents] == ['ok', 'ok'] + ['not_scored'] * 4
    assert fake.count('POST', '/runsync') == 2
    assert res.budget['stopped']['code'] == 'budget_cap' and res.budget['submissions'] == 2
    assert res.selected_ids == [f'doc-{i}' for i in range(6)]


def test_stop_during_concurrency_settles_in_flight_and_submits_nothing_new():
    fake = fakes.FakeRelevanceEndpoint()
    barrier = threading.Barrier(3, timeout=5)
    holder = {}

    def handler(request):
        d = json.loads(request.content)['input']['state']['document']
        barrier.wait()
        if 'FAKE:HTTP500' not in d:                      # in-flight successes finish only after the stop
            deadline = time.monotonic() + 5
            while holder['flt'].ledger.stopped is None and time.monotonic() < deadline:
                time.sleep(0.001)
        return fake.handle(request)
    flt, _ = make(F.threshold(0.3), handler=handler, fake=fake, concurrency=3)
    holder['flt'] = flt
    res = flt.filter(Q, docs_with('[[KEEP=0.9]]', 'FAKE:HTTP500', '[[KEEP=0.1]]', '[[KEEP=0.9]]', '[[KEEP=0.9]]'))
    assert rows(res) == [('doc-0', 0.9, 'ok', None, 'keep'), ('doc-1', None, 'error', 'http', 'keep'),
                         ('doc-2', 0.1, 'ok', None, 'drop'),
                         ('doc-3', None, 'not_scored', 'not_submitted', 'keep'),
                         ('doc-4', None, 'not_scored', 'not_submitted', 'keep')]
    assert fake.count('POST', '/runsync') == 3
    snap = res.budget
    assert snap['stopped']['code'] == 'unsettled_job' and snap['submissions'] == 3
    assert snap['settled_seconds'] > 0 and len(snap['unsettled_jobs']) == 1


def test_concurrency_argument_checked():
    with pytest.raises(ValueError):
        make(F.threshold(0.3), concurrency=0)


# --- the one integration example ------------------------------------------------------------------------------------
def test_tool_executor_example_runs(capsys):
    import runpy
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'examples' / 'context_filter_tool_executor.py'
    runpy.run_path(str(path), run_name='__main__')
    out = capsys.readouterr().out
    assert '"omitted_ids": ["s2"]' in out and 'Unrelated recipe' in out and 'not a model run' in out
