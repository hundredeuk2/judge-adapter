"""Adapter-level checks that the FastChat flow does not reach: two-direction mode, choice-endpoint polling/cancel, reason
mapping, config guards. Model-free (fake transports)."""
from __future__ import annotations

import httpx
import pytest

from judge_adapter import fakes
from judge_adapter import AdapterConfig, JudgeAdapter, JudgeRequest, Response, build_backend


def req(a_text, b_text, rid='r1'):
    return JudgeRequest(request_id=rid, question_turns=('Q1?',), response_a=Response('id-a', (a_text,)),
                        response_b=Response('id-b', (b_text,)))


def make(tmp_path, backend='haiku_direct', mode='two_direction', fake=None, sleep=lambda s: None, **kw):
    fake = fake or fakes.fake_for(backend)
    cfg = AdapterConfig(transport='fake', backend=backend, mode=mode, trace_dir=str(tmp_path / 't'), **kw)
    be = build_backend(cfg, transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY, sleep=sleep)
    return JudgeAdapter(cfg, be), fake


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_two_direction_restores_ids(tmp_path, backend):
    ad, fake = make(tmp_path, backend)
    r = ad.judge(req('b', '[[GOOD]] a'))
    assert (r.status, r.verdict, r.winner_id, r.tie_kind) == ('ok', 'B', 'id-b', None)
    assert [c.direction for c in r.calls] == ['AB', 'BA'] and [c.label for c in r.calls] == ['id-b', 'id-b']
    assert r.submissions == 2 and fake.count() == 2


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_two_direction_inconsistent_is_tie_recorded_separately(tmp_path, backend):
    ad, _ = make(tmp_path, backend)
    r = ad.judge(req('FAKE:POSITION a', 'b'))
    assert (r.status, r.verdict, r.tie_kind, r.winner_id, r.failure) == ('tie', 'tie', 'inconsistent', None, None)
    e = ad.judge(req('a', 'b', 'r2'))
    assert (e.status, e.tie_kind) == ('tie', 'explicit')


@pytest.mark.parametrize('backend', ['haiku_direct', 'choice_endpoint'])
def test_failure_stops_before_second_direction(tmp_path, backend):
    ad, fake = make(tmp_path, backend)
    r = ad.judge(req('FAKE:FORMAT a', 'b'))
    assert r.status == 'error' and r.verdict is None and r.failure['reason'] == 'format_error'
    assert r.failure['direction'] == 'AB' and r.submissions == 1 and fake.count() == 1


@pytest.mark.parametrize('marker,reason', [('FAKE:TRUNC', 'truncated'), ('FAKE:REFUSE', 'refused'),
                                           ('FAKE:HTTP500', 'http'), ('FAKE:TIMEOUT', 'timeout')])
def test_haiku_reason_mapping(tmp_path, marker, reason):
    ad, fake = make(tmp_path, mode='single')
    r = ad.judge(req(marker, 'b'))
    assert r.failure['reason'] == reason and fake.count() == 1 and r.retries == 0


@pytest.mark.parametrize('marker,reason', [('FAKE:TRUNC', 'truncated'), ('FAKE:REFUSE', 'backend_error'),
                                           ('FAKE:HTTP500', 'http'), ('FAKE:TIMEOUT', 'timeout')])
def test_choice_reason_mapping(tmp_path, marker, reason):
    ad, fake = make(tmp_path, backend='choice_endpoint', mode='single')
    r = ad.judge(req(marker, 'b'))
    assert r.failure['reason'] == reason and fake.count('POST', '/runsync') == 1


def test_choice_queue_polls_same_job_without_resubmitting(tmp_path):
    ad, fake = make(tmp_path, backend='choice_endpoint', mode='single')
    r = ad.judge(req('FAKE:QUEUE [[GOOD]] a', 'b'))
    c = r.calls[0]
    assert r.status == 'ok' and r.winner_id == 'id-a'
    assert (c.submissions, c.status_polls, c.cancels) == (1, 1, 0) and fake.count('POST', '/runsync') == 1


def test_choice_unresolved_job_is_cancelled_once_and_fails_as_timeout(tmp_path):
    class Stuck(fakes.FakeChoiceEndpoint):
        def handle(self, request):
            self.requests.append((request.method, request.url.path))
            if '/cancel/' in request.url.path:
                return httpx.Response(200, json={'status': 'CANCELLED'})
            return httpx.Response(200, json={'id': 'job-1', 'status': 'IN_PROGRESS'})
    ad, fake = make(tmp_path, backend='choice_endpoint', mode='single', fake=Stuck(), choice_max_status_polls=3)
    r = ad.judge(req('a', 'b'))
    c = r.calls[0]
    assert r.failure['reason'] == 'timeout' and (c.submissions, c.status_polls, c.cancels) == (1, 3, 1)
    assert fake.count('POST', '/runsync') == 1 and r.cost_estimate['usd'] is None   # unknown, not zero


def test_choice_wrong_model_is_contract_failure_only_when_expected_model_is_set(tmp_path):
    class Wrong(fakes.FakeChoiceEndpoint):
        def _completed(self, job_id, state, leak=None):
            js = super()._completed(job_id, state, leak)
            js['output']['model'] = 'other-model'
            return js
    ad, _ = make(tmp_path / 'set', backend='choice_endpoint', mode='single', fake=Wrong(),
                 choice_expected_model=fakes.FAKE_CHOICE_MODEL)
    r = ad.judge(req('a', 'b'))
    assert r.failure['reason'] == 'contract' and r.model == fakes.FAKE_CHOICE_MODEL
    ad, _ = make(tmp_path / 'unset', backend='choice_endpoint', mode='single', fake=Wrong())
    r = ad.judge(req('a', 'b'))
    assert r.status == 'tie' and r.model == 'unspecified' and r.calls[0].response_model == 'other-model'


def test_choice_matching_expected_model_passes(tmp_path):
    ad, _ = make(tmp_path, backend='choice_endpoint', mode='single', choice_expected_model=fakes.FAKE_CHOICE_MODEL)
    r = ad.judge(req('[[GOOD]] a', 'b'))
    assert r.status == 'ok' and r.calls[0].response_model == fakes.FAKE_CHOICE_MODEL


def test_fake_choice_rate_is_a_labeled_placeholder(tmp_path):
    from judge_adapter import core
    ad, _ = make(tmp_path, backend='choice_endpoint', mode='single')
    assert ad.backend.usd_per_second == core.FAKE_CHOICE_USD_PER_SECOND
    assert 'PLACEHOLDER' in ad.judge(req('a', 'b')).cost_estimate['basis']


def test_real_choice_backend_has_no_built_in_rate_or_model():
    be = build_backend(AdapterConfig(transport='real', backend='choice_endpoint', choice_endpoint_id='e1'))
    assert be.usd_per_second is None and be.expected_model is None and be.model == 'unspecified'
    with pytest.raises(ValueError):
        build_backend(AdapterConfig(transport='real', backend='choice_endpoint', choice_usd_per_second=-1.0))


def test_unsupported_inputs_are_refused_without_a_call(tmp_path):
    ad, fake = make(tmp_path, mode='single')
    same = JudgeRequest('r', ('q',), Response('x', ('a',)), Response('x', ('b',)))
    short = JudgeRequest('r', ('q',), Response('x', ('a',)), Response('y', ('b',)), turn=2)
    crit = JudgeRequest('r', ('q',), Response('x', ('a',)), Response('y', ('b',)), criteria={'a': 'b'})
    for rq in (same, short, crit):
        r = ad.judge(rq)
        assert r.failure['reason'] == 'unsupported_input' and r.submissions == 0
    assert fake.count() == 0


def test_config_guards():
    with pytest.raises(TypeError):
        AdapterConfig()                                      # transport is required
    assert AdapterConfig(transport='fake').backend == 'haiku_direct'
    with pytest.raises(ValueError):
        AdapterConfig(transport='fake', backend='gpt')
    with pytest.raises(ValueError):
        build_backend(AdapterConfig(transport='fake'))       # fake needs an injected transport
    with pytest.raises(ValueError):
        build_backend(AdapterConfig(transport='real'), transport=httpx.MockTransport(lambda r: None))


def test_real_backend_construction_reads_no_key(tmp_path):
    calls = []
    be = build_backend(AdapterConfig(transport='real'), key_provider=lambda: calls.append(1) or 'k')
    assert calls == [] and be._http is None                  # key is read only at the first request
