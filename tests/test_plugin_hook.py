"""Claude Code plugin hook (PostToolUse), model-free: an in-process fake choice endpoint only.

Covers: the hook I/O contract and the documented output schema, parsing of plausible search-result shapes, keep/drop
replacement in the same shape, fail-open (unknown shape, errors, per-request timeout, wall budget, missing key /
endpoint / rate, config errors, disabled), the all-dropped default, MCP tools off by default, logs without text or
keys, and that the plugin manifests agree with the names module. Network and key reads fail every test."""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from judge_adapter import fakes
from judge_adapter import plugin_hook as H
from judge_adapter import plugin_names as N

ROOT = Path(__file__).resolve().parents[1]
KEY = 'rpa_PLUGINHOOKTESTKEY0123456789'
ENDPOINT = 'endpointsecret42'
QUERY = 'Which synthetic city hosts the zebra lantern festival?'


@pytest.fixture(autouse=True)
def no_net(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('network attempted')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(socket, 'create_connection', boom)
    monkeypatch.setattr(socket, 'getaddrinfo', boom)


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


def payload(resp, tool='WebSearch', query=QUERY):
    return {'session_id': 's1', 'transcript_path': '/tmp/t.jsonl', 'cwd': '/tmp', 'permission_mode': 'default',
            'hook_event_name': 'PostToolUse', 'tool_name': tool, 'tool_input': {'query': query},
            'tool_response': resp, 'tool_use_id': 'toolu_01', 'duration_ms': 12}


def item(i, marker, key='snippet'):
    return {'title': f'Secret title {i}', 'url': f'https://example.invalid/secret-{i}',
            key: f'Synthetic body text number {i} about zebras. {marker}'}


def run(p, e=None, fake=None, handler=None):
    """Run the hook entry point. -> (stdout object or None, log record, fake)."""
    fake = fake or fakes.FakeRelevanceEndpoint()
    out, err = io.StringIO(), io.StringIO()
    exits = []
    rc = H.main(stdin=io.StringIO(json.dumps(p)), stdout=out, stderr=err, env=env() if e is None else e,
                transport=httpx.MockTransport(handler or fake.handle), exit_fn=exits.append)
    assert rc == 0
    text = out.getvalue()
    log = json.loads(err.getvalue().strip().splitlines()[-1])
    return (json.loads(text) if text else None), log, fake, text, err.getvalue(), exits


# --- output contract -----------------------------------------------------------------------------------------------
def test_filtered_output_schema_and_same_shape():
    resp = {'query': QUERY, 'results': [item(0, '[[KEEP=0.9]]'), item(1, '[[KEEP=0.1]]'), item(2, '[[KEEP=0.3]]')],
            'durationSeconds': 1.5}
    out, log, fake, text, _, _ = run(payload(resp))
    assert text.strip().startswith('{') and text.strip().endswith('}')      # one JSON object, nothing around it
    assert set(out) == {'hookSpecificOutput'}
    hso = out['hookSpecificOutput']
    assert set(hso) == {'hookEventName', 'updatedToolOutput', 'additionalContext'}
    assert hso['hookEventName'] == 'PostToolUse'
    assert hso['additionalContext'] == 'context filter (v1-thr0.3-dev) kept 2/3 results'
    upd = hso['updatedToolOutput']
    assert set(upd) == set(resp) and upd['query'] == QUERY and upd['durationSeconds'] == 1.5
    assert upd['results'] == [resp['results'][0], resp['results'][2]]          # threshold 0.3 keeps 0.3
    assert fake.count('POST') == 3
    assert log['status'] == 'filtered' and log['results_in'] == 3 and log['results_out'] == 2
    assert resp['results'][1]['title'] == 'Secret title 1'                     # the input object is not mutated


def test_preset_wire_body_uses_query_and_dev_template():
    resp = {'results': [item(0, '[[KEEP=0.9]]')]}
    _, _, fake, _, _, _ = run(payload(resp))
    body = fake.bodies[0]
    assert body['input']['state']['question'] == QUERY
    assert body['input']['state']['document'].startswith('제목: Secret title 0\nURL: https://example.invalid/secret-0')


# --- shapes --------------------------------------------------------------------------------------------------------
def _ids(lst):
    return [x['url'][-1] for x in lst]


@pytest.mark.parametrize('key', ['snippet', 'text', 'content', 'description'])
def test_flat_results_text_field_variants(key):
    resp = {'results': [item(0, '[[KEEP=0.9]]', key), item(1, '[[KEEP=0.05]]', key)]}
    out, _, _, _, _, _ = run(payload(resp))
    assert _ids(out['hookSpecificOutput']['updatedToolOutput']['results']) == ['0']


def test_top_level_list():
    resp = [item(0, '[[KEEP=0.05]]'), item(1, '[[KEEP=0.8]]')]
    out, _, _, _, _, _ = run(payload(resp))
    assert _ids(out['hookSpecificOutput']['updatedToolOutput']) == ['1']


def test_nested_tool_use_blocks_keep_commentary_strings():
    resp = {'query': QUERY, 'results': [
        {'tool_use_id': 'srvtoolu_1', 'content': [item(0, '[[KEEP=0.9]]'), item(1, '[[KEEP=0.1]]')]},
        'Commentary text between result blocks.',
        {'tool_use_id': 'srvtoolu_2', 'content': [item(2, '[[KEEP=0.2]]'), item(3, '[[KEEP=0.7]]')]}],
        'durationSeconds': 2.0}
    out, log, _, _, _, _ = run(payload(resp))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    assert upd['results'][1] == 'Commentary text between result blocks.'
    assert upd['results'][0]['tool_use_id'] == 'srvtoolu_1' and _ids(upd['results'][0]['content']) == ['0']
    assert _ids(upd['results'][2]['content']) == ['3']
    assert out['hookSpecificOutput']['additionalContext'].endswith('kept 2/4 results')
    assert log['parsed_as'] == 'object'


def test_web_results_wrapper():
    resp = {'type': 'search', 'web': {'results': [item(0, '[[KEEP=0.9]]', 'description'),
                                                  item(1, '[[KEEP=0.1]]', 'description')]}}
    out, _, _, _, _, _ = run(payload(resp))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    assert upd['type'] == 'search' and _ids(upd['web']['results']) == ['0']


def test_json_string_response_stays_a_string():
    inner = {'results': [item(0, '[[KEEP=0.1]]'), item(1, '[[KEEP=0.9]]')]}
    out, log, _, _, _, _ = run(payload(json.dumps(inner)))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    assert isinstance(upd, str) and _ids(json.loads(upd)['results']) == ['1']
    assert log['parsed_as'] == 'json_string'


def test_json_in_mcp_text_is_not_handled_on_this_branch():
    """Only the Tavily text format is supported; a JSON text block (even for an mcp_tools entry) is left alone."""
    inner = {'results': [item(0, '[[KEEP=0.9]]', 'content'), item(1, '[[KEEP=0.1]]', 'content')]}
    resp = [{'type': 'text', 'text': json.dumps(inner)}]
    tool = 'mcp__search-server__web_search'
    out, log, fake, raw, _, _ = run(payload(resp, tool=tool))
    assert raw == '' and log['status'] == 'tool_not_enabled' and fake.count() == 0
    out, log, fake, raw, _, _ = run(payload(resp, tool=tool), e=env(mcp_tools=f'other_tool, {tool}'))
    assert raw == '' and log['status'] == 'unrecognized_shape' and fake.count() == 0


def test_results_without_text_are_passed_through_unscored():
    resp = {'results': [{'title': 'Link only', 'url': 'https://example.invalid/a'}, item(1, '[[KEEP=0.1]]'),
                        item(2, '[[KEEP=0.9]]')]}
    out, log, fake, _, _, _ = run(payload(resp))
    assert fake.count('POST') == 2
    assert [x['url'] for x in out['hookSpecificOutput']['updatedToolOutput']['results']] == [
        'https://example.invalid/a', 'https://example.invalid/secret-2']
    assert log['results_with_text'] == 2


def test_link_only_results_change_nothing():
    resp = {'query': QUERY, 'results': [{'tool_use_id': 'x', 'content': [{'title': 't', 'url': 'https://e.invalid'}]},
                                        'summary text']}
    out, log, fake, _, _, _ = run(payload(resp))
    assert out is None and log['status'] == 'no_text_in_results' and fake.count() == 0


def test_max_docs_scores_only_the_first_results():
    resp = {'results': [item(i, '[[KEEP=0.1]]') for i in range(4)]}
    out, log, fake, _, _, _ = run(payload(resp), e=env(max_docs='2'))
    assert fake.count('POST') == 2
    assert _ids(out['hookSpecificOutput']['updatedToolOutput']['results']) == ['2', '3']
    assert log['results_scored'] == 2


def test_concurrency_option():
    resp = {'results': [item(i, '[[KEEP=0.9]]' if i % 2 else '[[KEEP=0.1]]') for i in range(6)]}
    out, log, fake, _, _, _ = run(payload(resp), e=env(concurrency='3'))
    assert _ids(out['hookSpecificOutput']['updatedToolOutput']['results']) == ['1', '3', '5']
    assert fake.count('POST') == 6


# --- fail-open -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('resp', [
    {'unexpected': 'shape'}, 'plain text result', 42, None, [], {'results': []},
    {'results': [{'foo': 'bar'}]}, [{'type': 'text', 'text': 'not json'}],
])
def test_unknown_shape_leaves_result_unchanged(resp):
    out, log, fake, text, _, _ = run(payload(resp))
    assert text == '' and out is None and fake.count() == 0
    assert log['status'] in ('unrecognized_shape', 'no_text_in_results')


@pytest.mark.parametrize('missing,status', [
    ({'runpod_api_key': None}, 'missing_key'), ({'choice_endpoint_id': None}, 'missing_endpoint'),
    ({'usd_per_second': None}, 'missing_rate')])
def test_missing_configuration_sends_nothing(missing, status):
    resp = {'results': [item(0, '[[KEEP=0.1]]')]}
    out, log, fake, text, _, _ = run(payload(resp), e=env(**missing))
    assert text == '' and fake.count() == 0 and log['status'] == status


def test_package_env_names_are_accepted_as_fallbacks():
    e = {'RUNPOD_API_KEY': KEY, 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID': ENDPOINT, N.ENV_PREFIX + 'USD_PER_SECOND': '0.0004'}
    out, log, _, _, _, _ = run(payload({'results': [item(0, '[[KEEP=0.9]]'), item(1, '[[KEEP=0.1]]')]}), e=e)
    assert log['status'] == 'filtered'


def test_errors_keep_documents_and_unchanged_when_nothing_dropped():
    resp = {'results': [item(0, 'FAKE:HTTP500'), item(1, '[[KEEP=0.9]]')]}
    out, log, _, text, _, _ = run(payload(resp))
    assert text == '' and log['status'] == 'all_kept'


def test_format_error_document_is_kept_others_decided():
    resp = {'results': [item(0, '[[KEEP=0.1]]'), item(1, 'FAKE:FORMAT')]}
    out, log, _, _, _, _ = run(payload(resp))
    assert _ids(out['hookSpecificOutput']['updatedToolOutput']['results']) == ['1']
    assert log['status_counts'] == {'ok': 1, 'error': 1, 'not_scored': 0}


def test_request_timeout_fails_open():
    resp = {'results': [item(0, 'FAKE:TIMEOUT'), item(1, '[[KEEP=0.1]]')]}
    out, log, fake, text, _, _ = run(payload(resp))
    # the timed-out job is unsettled -> later submissions stop -> both kept -> nothing to change
    assert text == '' and log['status'] == 'all_kept' and fake.count('POST') == 1


def test_wall_budget_fails_open_without_output():
    fake = fakes.FakeRelevanceEndpoint()

    def slow(request):
        time.sleep(3)
        return fake.handle(request)
    t = time.perf_counter()
    out, log, _, text, _, exits = run(payload({'results': [item(0, '[[KEEP=0.1]]')]}), e=env(wall_budget_seconds='1'),
                                      handler=slow)
    assert time.perf_counter() - t < 2.5
    assert text == '' and log['status'] == 'wall_budget_exceeded' and exits == [0]


def test_unexpected_exception_fails_open(monkeypatch):
    monkeypatch.setattr(H, 'parse_response', lambda r: 1 / 0)
    out, log, _, text, _, _ = run(payload({'results': [item(0, '[[KEEP=0.1]]')]}))
    assert text == '' and log['status'] == 'error' and log['detail'] == 'ZeroDivisionError'


@pytest.mark.parametrize('bad', [{'concurrency': 'many'}, {'concurrency': '0'}, {'on_all_dropped': 'hide'},
                                 {'enabled': 'maybe'}, {'wall_budget_seconds': '600'}])
def test_config_error_fails_open(bad):
    out, log, fake, text, _, _ = run(payload({'results': [item(0, '[[KEEP=0.1]]')]}), e=env(**bad))
    assert text == '' and log['status'] == 'config_error' and fake.count() == 0


def test_unknown_preset_fails_open():
    out, log, fake, text, _, _ = run(payload({'results': [item(0, '[[KEEP=0.1]]')]}), e=env(preset='nope'))
    assert text == '' and log['status'] == 'unknown_preset' and fake.count() == 0


def test_disabled_flag_and_other_events():
    p = payload({'results': [item(0, '[[KEEP=0.1]]')]})
    out, log, fake, text, _, _ = run(p, e=env(enabled='false'))
    assert text == '' and log['status'] == 'disabled' and fake.count() == 0
    out, log, _, text, _, _ = run({**p, 'hook_event_name': 'PreToolUse'})
    assert text == '' and log['status'] == 'not_post_tool_use'
    out, log, _, text, _, _ = run({**p, 'tool_input': {}})
    assert text == '' and log['status'] == 'no_query'


def test_invalid_stdin_fails_open():
    out, err = io.StringIO(), io.StringIO()
    assert H.main(stdin=io.StringIO('not json'), stdout=out, stderr=err, env=env(), exit_fn=lambda c: None) == 0
    assert out.getvalue() == '' and json.loads(err.getvalue())['status'] == 'bad_input'


# --- all dropped ---------------------------------------------------------------------------------------------------
def test_all_dropped_default_keeps_original_with_note():
    resp = {'results': [item(0, '[[KEEP=0.1]]'), item(1, '[[KEEP=0.2]]')]}
    out, log, _, _, _, _ = run(payload(resp))
    assert out == {'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': 'context filter (v1-thr0.3-dev) kept 0/2 results for this search; the original results '
                             'are shown unchanged.'}}
    assert log['status'] == 'all_dropped_kept_original'


def test_all_dropped_drop_all_mode():
    resp = {'query': QUERY, 'results': [item(0, '[[KEEP=0.1]]'), item(1, '[[KEEP=0.2]]')]}
    out, log, _, _, _, _ = run(payload(resp), e=env(on_all_dropped='drop_all'))
    hso = out['hookSpecificOutput']
    assert hso['updatedToolOutput'] == {'query': QUERY, 'results': []}
    assert hso['additionalContext'] == 'context filter (v1-thr0.3-dev) kept 0/2 results'
    assert log['status'] == 'filtered'


# --- logs ----------------------------------------------------------------------------------------------------------
def test_logs_hold_no_text_query_urls_or_keys(tmp_path):
    log_file = tmp_path / 'hook.log'
    resp = {'results': [item(0, '[[KEEP=0.9]]'), item(1, '[[KEEP=0.1]]'), item(2, 'FAKE:HTTP500')]}
    out, log, _, _, err, _ = run(payload(resp), e=env(log_file=str(log_file)))
    stored = log_file.read_text(encoding='utf-8')
    assert stored.strip() == err.strip()
    for s in (KEY, ENDPOINT, QUERY, 'zebra', 'Secret title', 'example.invalid', 'Synthetic body'):
        assert s not in err, s
    assert log['text_bytes_in'] > log['text_bytes_out'] > 0
    assert log['response_bytes_in'] > log['response_bytes_out'] > 0
    assert set(log) >= {'event', 'status', 'tool', 'seconds', 'shape', 'results_in', 'results_out'}


def test_httpx_logger_kept_quiet():
    run(payload({'results': [item(0, '[[KEEP=0.9]]')]}))
    import logging
    assert logging.getLogger('httpx').getEffectiveLevel() >= logging.WARNING


# --- launcher and manifests ----------------------------------------------------------------------------------------
def test_launcher_subprocess_fails_open_without_key():
    """The real entry point as Claude Code runs it (exec form). No key -> returns before any network use."""
    p = payload({'results': [item(0, '[[KEEP=0.1]]')]})
    e = {k: v for k, v in os.environ.items() if not k.startswith(('CLAUDE_PLUGIN_OPTION_', N.ENV_PREFIX))
         and k not in ('RUNPOD_API_KEY', 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID', 'PYTHONPATH')}
    e['CLAUDE_PLUGIN_ROOT'] = str(ROOT)
    r = subprocess.run([sys.executable, str(ROOT / 'hooks' / 'context_filter_hook.py')], input=json.dumps(p),
                       capture_output=True, text=True, env=e, timeout=60)
    assert r.returncode == 0 and r.stdout == ''
    assert json.loads(r.stderr.strip().splitlines()[-1])['status'] == 'missing_key'


def test_manifests_match_names_module_and_options():
    plugin = json.loads((ROOT / '.claude-plugin' / 'plugin.json').read_text(encoding='utf-8'))
    market = json.loads((ROOT / '.claude-plugin' / 'marketplace.json').read_text(encoding='utf-8'))
    hooks = json.loads((ROOT / 'hooks' / 'hooks.json').read_text(encoding='utf-8'))
    assert plugin['name'] == N.PLUGIN_NAME and market['name'] == N.MARKETPLACE_NAME
    assert market['plugins'] == [{**market['plugins'][0], 'name': N.PLUGIN_NAME, 'source': './'}]
    assert 'hooks' not in plugin                       # hooks/hooks.json loads by default; declaring it would duplicate
    cfg = plugin['userConfig']
    assert cfg['runpod_api_key']['sensitive'] is True and cfg['choice_endpoint_id']['sensitive'] is True
    read_by_hook = {f.name for f in H.dataclasses.fields(H.Settings)} | {'runpod_api_key', 'choice_endpoint_id'}
    assert set(cfg) == read_by_hook
    assert cfg['preset']['default'] == H.DEFAULT_PRESET
    post = hooks['hooks']['PostToolUse']
    assert [e['matcher'] for e in post] == [H.KNOWN_TOOLS_MATCHER]      # built-in WebSearch is not matched
    for entry in post:
        h = entry['hooks'][0]
        assert h['type'] == 'command' and h['command'] == 'python3'
        assert h['args'] == ['${CLAUDE_PLUGIN_ROOT}/hooks/context_filter_hook.py']
        assert h['timeout'] > 60                       # above the largest allowed wall budget


# --- search MCP formats (fixtures rebuilt from the servers' formatters; synthetic content) ----------------------------
import re  # noqa: E402


def tavily_text(results, answer=None, request_id=None, images=None, npm_0_2_22=False):
    """Python port of tavily-mcp formatResults (src/format-results.ts at 1c1d54c; npm 0.2.22 build when flagged)."""
    out = []
    if request_id and not npm_0_2_22:
        out.append(f'Request ID: {request_id}')
    if answer:
        out.append(f'Answer: {answer}')
    out.append('Detailed Results:')
    for r in results:
        if npm_0_2_22:
            out.append(f"\nTitle: {r['title']}")
        else:
            out.append('')
            if r.get('title'):
                out.append(f"Title: {r['title']}")
        if r.get('id'):
            out.append(f"ID: {r['id']}")
        out.append(f"URL: {r['url']}")
        out.append(f"Content: {r['content']}")
        if r.get('raw_content'):
            out.append(f"Raw Content: {r['raw_content']}")
        if r.get('favicon'):
            out.append(f"Favicon: {r['favicon']}")
    if images:
        out.append('\nImages:')
        for i, im in enumerate(images):
            if isinstance(im, str):
                out.append(f'\n[{i + 1}] URL: {im}')
            else:
                out.append(f"\n[{i + 1}] URL: {im['url']}")
                if im.get('description'):
                    out.append(f"   Description: {im['description']}")
    return '\n'.join(out)


def tav(i, marker, **kw):
    return {'title': f'Secret title {i}', 'url': f'https://example.invalid/secret-{i}',
            'content': f'Synthetic snippet {i} about zebras.\nSecond line. {marker}', **kw}


TAVILY_TOOL = 'mcp__tavily__tavily_search'


def test_tavily_list_blocks_filtered_header_answer_images_kept():
    results = [tav(0, '[[KEEP=0.9]]', id='id-0', favicon='https://example.invalid/f.ico'),
               tav(1, '[[KEEP=0.1]]', id='id-1'), tav(2, '[[KEEP=0.5]]', raw_content='Raw page [[KEEP=0.0]] text')]
    images = ['https://example.invalid/a.png', {'url': 'https://example.invalid/b.png', 'description': 'Synthetic'}]
    text = tavily_text(results, answer='Synthetic answer.', request_id='req-1', images=images)
    resp = [{'type': 'text', 'text': text}]
    out, log, fake, _, err, _ = run(payload(resp, tool=TAVILY_TOOL))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    assert isinstance(upd, list) and len(upd) == 1 and upd[0]['type'] == 'text'
    assert upd[0]['text'] == tavily_text([results[0], results[2]], answer='Synthetic answer.', request_id='req-1',
                                         images=images)
    assert out['hookSpecificOutput']['additionalContext'].endswith('kept 2/3 results')
    assert fake.count('POST') == 3
    docs = [b['input']['state']['document'] for b in fake.bodies]
    assert all('Synthetic answer' not in d and 'Images' not in d and 'Raw page' not in d for d in docs)
    assert docs[0].startswith('제목: Secret title 0\nURL: https://example.invalid/secret-0\n\nSynthetic snippet 0')
    assert log['format'] == 'tavily' and log['parsed_as'] == 'tavily_text'
    assert 'zebra' not in err and 'Synthetic answer' not in err


@pytest.mark.parametrize('wrap', ['list', 'object', 'string'])
def test_tavily_npm_release_format_in_each_response_shape(wrap):
    results = [tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.8]]')]
    text = tavily_text(results, npm_0_2_22=True)
    resp = {'list': [{'type': 'text', 'text': text}],
            'object': {'content': [{'type': 'text', 'text': text}], 'isError': False},
            'string': text}[wrap]
    out, _, _, _, _, _ = run(payload(resp, tool='mcp__my-search__tavily_search'))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    want = tavily_text([results[1]], npm_0_2_22=True)
    if wrap == 'list':
        assert upd == [{'type': 'text', 'text': want}]
    elif wrap == 'object':
        assert upd == {'content': [{'type': 'text', 'text': want}], 'isError': False}
    else:
        assert upd == want


def test_tavily_all_dropped_default_and_drop_all():
    results = [tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.2]]')]
    resp = [{'type': 'text', 'text': tavily_text(results, answer='A.')}]
    out, log, _, _, _, _ = run(payload(resp, tool=TAVILY_TOOL))
    assert 'updatedToolOutput' not in out['hookSpecificOutput'] and log['status'] == 'all_dropped_kept_original'
    out, _, _, _, _, _ = run(payload(resp, tool=TAVILY_TOOL), e=env(on_all_dropped='drop_all'))
    assert out['hookSpecificOutput']['updatedToolOutput'] == [{'type': 'text', 'text': 'Answer: A.\nDetailed Results:'}]


@pytest.mark.parametrize('text', [
    'Tavily API error: synthetic', 'Detailed Results:', 'Answer: x\nDetailed Results:\n\nNot a result line',
    'Detailed Results:\n\nTitle: t\nno url line here'])
def test_tavily_unrecognized_or_empty_fails_open(text):
    out, log, fake, raw, _, _ = run(payload([{'type': 'text', 'text': text}], tool=TAVILY_TOOL))
    assert raw == '' and fake.count() == 0 and log['status'] == 'unrecognized_shape'


def test_known_tool_matcher_is_anchored():
    m = re.compile(H.KNOWN_TOOLS_MATCHER)
    for name in ('mcp__tavily__tavily_search', 'mcp__tavily-local__tavily_search', 'mcp__plugin_x_tavily__tavily_search'):
        assert m.search(name), name
    for name in OTHER_TOOLS + ('mcp__tavily__tavily-search', 'mcp__a__b__tavily_search_x', 'mcp____tavily_search',
                               'mcp__ta.vily__tavily_search'):
        assert not m.search(name), name
    assert H.tool_format('mcp__tavily__tavily_search\n', H.Settings()) is None


OTHER_TOOLS = ('WebSearch', 'WebFetch', 'Read', 'Write', 'Edit', 'Bash', 'mcp__tavily__tavily_extract',
               'mcp__tavily__tavily_crawl', 'mcp__tavily__tavily_map', 'mcp__tavily__tavily_search_extra',
               'mcp__exa__web_search_exa', 'mcp__exa__web_search_advanced_exa', 'mcp__brave-search__brave_web_search',
               'mcp__github__search_code', 'mcp__github__search_repositories', 'mcp__filesystem__read_file',
               'mcp__filesystem__write_file', 'xmcp__tavily__tavily_search', 'tavily_search')


@pytest.mark.parametrize('tool', [t for t in OTHER_TOOLS if t != 'WebSearch'])
def test_zero_intervention_for_other_tools(tool):
    """Tavily-formatted text with markers that would be dropped: any other tool name gets no call and no output."""
    text = tavily_text([tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.9]]')])
    for resp in ([{'type': 'text', 'text': text}], {'results': [item(0, '[[KEEP=0.1]]'), item(1, '[[KEEP=0.9]]')]}):
        out, log, fake, raw, _, _ = run(payload(resp, tool=tool))
        assert raw == '' and out is None and fake.count() == 0 and log['status'] == 'tool_not_enabled'


def test_is_error_and_structured_content_pass_through():
    text = tavily_text([tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.9]]')])
    for resp, status in (({'content': [{'type': 'text', 'text': text}], 'isError': True}, 'tool_error_passthrough'),
                         ({'content': [{'type': 'text', 'text': text}], 'structuredContent': {'results': []}},
                          'structured_content_passthrough')):
        out, log, fake, raw, _, _ = run(payload(resp, tool=TAVILY_TOOL))
        assert raw == '' and fake.count() == 0 and log['status'] == status
    resp = {'content': [{'type': 'text', 'text': text}], 'isError': False}       # isError false: filtered
    out, log, _, _, _, _ = run(payload(resp, tool=TAVILY_TOOL))
    assert log['status'] == 'filtered' and out['hookSpecificOutput']['updatedToolOutput']['isError'] is False


def test_other_blocks_kept_and_only_tavily_block_edited():
    text = tavily_text([tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.9]]')], answer='Synthetic answer.')
    img = {'type': 'image', 'data': 'AAAA', 'mimeType': 'image/png'}
    res = {'type': 'resource', 'resource': {'uri': 'file:///synthetic', 'text': 'synthetic'}}
    note = {'type': 'text', 'text': 'Unrelated synthetic note.'}
    resp = [note, {'type': 'text', 'text': text}, img, res]
    out, _, _, _, _, _ = run(payload(resp, tool=TAVILY_TOOL))
    upd = out['hookSpecificOutput']['updatedToolOutput']
    assert upd[0] == note and upd[2] == img and upd[3] == res
    assert upd[1]['text'] == tavily_text([tav(1, '[[KEEP=0.9]]')], answer='Synthetic answer.')
    assert set(out['hookSpecificOutput']) == {'hookEventName', 'updatedToolOutput', 'additionalContext'}   # never both


def test_mcp_tools_override_off_and_extra_name():
    text = tavily_text([tav(0, '[[KEEP=0.1]]'), tav(1, '[[KEEP=0.9]]')])
    p = payload([{'type': 'text', 'text': text}], tool=TAVILY_TOOL)
    out, log, fake, raw, _, _ = run(p, e=env(mcp_tools=f'{TAVILY_TOOL}=off'))
    assert raw == '' and log['status'] == 'tool_not_enabled' and fake.count() == 0
    p2 = payload([{'type': 'text', 'text': text}], tool='mcp__corp__internal_search')
    out, log, _, raw, _, _ = run(p2)
    assert raw == '' and log['status'] == 'tool_not_enabled'
    out, log, _, _, _, _ = run(p2, e=env(mcp_tools='mcp__corp__internal_search=tavily'))
    assert log['status'] == 'filtered' and log['format'] == 'tavily'
    out, log, _, _, _, _ = run(p2, e=env(mcp_tools='mcp__corp__internal_search'))       # bare name = tavily
    assert log['status'] == 'filtered' and log['parsed_as'] == 'tavily_text'


def test_mcp_tools_bad_format_is_config_error():
    out, log, _, raw, _, _ = run(payload([], tool=TAVILY_TOOL), e=env(mcp_tools='x=yaml'))
    assert raw == '' and log['status'] == 'config_error'


def test_builtin_websearch_observed_shape_passes_through():
    """Observed once in a real run (key names only): results = [{tool_use_id, content: [{title, url} x N]}, str, ...],
    plus durationSeconds and searchCount. Synthetic titles / URLs here. No per-result text -> no filter call."""
    resp = {'query': QUERY, 'results': [
        {'tool_use_id': 'srvtoolu_A', 'content': [{'title': f'Synthetic {i}', 'url': f'https://example.invalid/{i}'}
                                                  for i in range(10)]},
        'Synthetic model-written summary of the first search.',
        {'tool_use_id': 'srvtoolu_B', 'content': [{'title': f'Synthetic {i}', 'url': f'https://example.invalid/b{i}'}
                                                  for i in range(9)]},
        'Another synthetic summary.'], 'durationSeconds': 3.2, 'searchCount': 2}
    out, log, fake, raw, err, _ = run(payload(resp))
    assert raw == '' and fake.count() == 0
    assert log['status'] == 'no_text_in_results' and log['results_in'] == 19 and log['results_with_text'] == 0
    first = log['shape']['results']['of'][0]
    assert log['shape']['results']['list'] == 4 and 'str' in log['shape']['results']['of']
    assert first == {'content': {'list': 10, 'of': [{'title': 'str', 'url': 'str'}]}, 'tool_use_id': 'str'}
    assert 'Synthetic' not in err and 'example.invalid' not in err
