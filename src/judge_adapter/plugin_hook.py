# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Claude Code PostToolUse hook: filter web search results through ``ContextFilter`` before the model sees them.

Run as ``python -m judge_adapter.plugin_hook`` (the plugin's ``hooks/context_filter_hook.py`` launcher does this with
the plugin's own ``src`` on ``sys.path``). Reads one hook input JSON object on stdin and prints at most one JSON
object on stdout. Always exits 0.

Target: the Tavily MCP search tool only (tavily-mcp npm 0.2.22, tool ``tavily_search``; source checked at
github.com/tavily-ai/tavily-mcp commit 1c1d54c2a619afe52544f775c8d82a56ac6d8bb5 and in the published 0.2.22 build). The
plugin's matcher in hooks/hooks.json must equal ``KNOWN_TOOLS_MATCHER``: any server alias, exact tool name. Every
other tool (other search servers, read / write tools, Claude Code's built-in WebSearch) gets zero intervention. One
observed run showed the built-in WebSearch returns only ``{title, url}`` per result, so there is nothing to score.
The ``mcp_tools`` option can switch the Tavily tool off or name another tool that emits the same Tavily text
(``name`` or ``name=tavily|off``).

Tavily result format (``formatResults``): one text block ``[Request ID: ..]\n[Answer: ..]\nDetailed Results:`` then
per result ``\n\nTitle: ..\n[ID: ..]\nURL: ..\nContent: ..[\nRaw Content: ..][\nFavicon: ..]``, then an optional
``\n\nImages:`` section. Only the ``Content`` snippet is scored. The header, the ``Answer:`` summary and the image
section are kept byte for byte (so the answer can still reflect dropped results); no text is generated or summarised.
A response with ``isError: true`` or with ``structuredContent`` is passed through unchanged (structuredContent could
not be kept consistent with an edited text block).

Hook contract (https://code.claude.com/docs/en/hooks, checked 2026-10-11):
- Input on stdin includes ``session_id``, ``cwd``, ``hook_event_name`` ("PostToolUse"), ``tool_name``,
  ``tool_input``, ``tool_response`` and ``tool_use_id`` (MCP tools also ``mcp_server``). The question is
  ``tool_input.query``. The docs do not document the shape of an MCP tool's ``tool_response``; the hook accepts a
  list of content blocks, an object with a ``content`` list, or a plain string, and writes back the same shape.
  Checked against the docs served on 2026-10-11 with Claude Code CLI 2.1.285 installed.
- Output, quoted from the docs' PostToolUse decision control example::

      {
        "hookSpecificOutput": {
          "hookEventName": "PostToolUse",
          "additionalContext": "Additional information for Claude",
          "updatedToolOutput": { ... }
        }
      }

  Field table: "``updatedToolOutput`` | Replaces the tool's output with the provided value before it is sent to
  Claude. The value must match the tool's output shape" and "``updatedMCPToolOutput`` | Replaces the output for MCP
  tools only. Prefer ``updatedToolOutput``, which works for all tools". Warning: "For built-in tools, a value that
  doesn't match the tool's output schema is ignored and the original output is used. MCP tool output is passed
  through without schema validation." So this hook uses ``updatedToolOutput`` and rebuilds the exact input shape,
  removing only dropped results (an MCP replacement is not validated, so a wrong shape would reach the model).
- Exit 0 with nothing on stdout is a success that changes nothing. A hook cancelled at its ``timeout`` has its output
  discarded. Both leave the tool result unchanged.

Behaviour (fail-open): any unrecognised result shape, missing key / endpoint / rate, disabled flag, input refusal,
error, timeout or the per-call wall budget prints nothing, so Claude sees the original result. Documents whose
filter call fails are kept (``ContextFilter`` is fail-open per document). When every result is dropped the default
(``on_all_dropped=keep_original``) leaves the result unchanged and adds a one-line note; ``drop_all`` removes them.

Logs: one JSON line per call on stderr (and optionally appended to ``log_file``) with counts, bytes, timings, status
and the structural shape of the tool response (key names and types only). Never the query, document text, titles,
URLs, keys or the endpoint ID.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass

from . import plugin_names as N

OPTION_ENV = 'CLAUDE_PLUGIN_OPTION_'           # Claude Code exports every userConfig option as this + KEY.upper()
KEY_ENVS = (OPTION_ENV + 'RUNPOD_API_KEY', 'RUNPOD_API_KEY')
ENDPOINT_ENVS = (OPTION_ENV + 'CHOICE_ENDPOINT_ID', 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID')
DEFAULT_PRESET = 'v1-thr0.3-dev'
ALL_DROPPED_MODES = ('keep_original', 'drop_all')
REQUEST_TIMEOUT_SECONDS = 15.0                 # per HTTP request; also capped by the wall budget
POLL_INTERVAL_SECONDS = 1.0

# Field names tried when reading a result item (first non-empty string wins).
URL_KEYS = ('url', 'link', 'href', 'source_url')
TITLE_KEYS = ('title', 'name')
TEXT_KEYS = ('text', 'snippet', 'content', 'description', 'body', 'summary', 'page_content', 'extract',
             'raw_content')
LIST_KEYS = ('results', 'items', 'documents', 'organic', 'organic_results', 'hits', 'search_results', 'data',
             'sources')
QUERY_KEYS = ('query', 'q', 'search_query')

# Verified search tool (tavily-mcp 0.2.22 registers only "tavily_search"). hooks/hooks.json uses this verbatim.
TAVILY_TOOL = 'tavily_search'
KNOWN_TOOLS_MATCHER = '^mcp__[A-Za-z0-9_-]+__' + TAVILY_TOOL + '$'
FORMATS = ('tavily', 'off')


class ConfigError(ValueError):
    pass


# --- settings ------------------------------------------------------------------------------------------------------
def option(env, key):
    """userConfig option ``key``: CLAUDE_PLUGIN_OPTION_<KEY>, then JUDGE_ADAPTER_HOOK_<KEY>. Empty -> None."""
    for name in (OPTION_ENV + key.upper(), N.ENV_PREFIX + key.upper()):
        v = env.get(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _present(env, names):
    """Name of the first set variable (presence only; the value is not returned)."""
    for n in names:
        v = env.get(n)
        if isinstance(v, str) and v.strip():
            return n
    return None


def _bool(v, default):
    if v is None:
        return default
    s = v.strip().lower()
    if s in ('1', 'true', 'yes', 'on'):
        return True
    if s in ('0', 'false', 'no', 'off'):
        return False
    raise ConfigError('enabled must be true or false')


def _num(v, name, default, lo, hi, cast=float):
    if v is None:
        return default
    try:
        x = float(v)
    except ValueError:
        raise ConfigError(f'{name} must be a number') from None
    if cast is int:
        if x != int(x):
            raise ConfigError(f'{name} must be an integer')
        x = int(x)
    if not lo <= x <= hi:
        raise ConfigError(f'{name} must be in [{lo}, {hi}]')
    return x


@dataclass(frozen=True)
class Settings:
    enabled: bool = True
    preset: str = DEFAULT_PRESET
    usd_per_second: float | None = None
    concurrency: int = 1
    max_docs: int = 10
    wall_budget_seconds: float = 20.0
    on_all_dropped: str = 'keep_original'
    mcp_tools: tuple = ()                 # ((tool name, format), ...)
    log_file: str | None = None
    # Optional query-stage options (PreToolUse, hooks/query_options_hook.py), all off by default. Parsed only by
    # ``query_hook.load_query_settings``; ``load_settings`` below leaves them at these defaults, so the PostToolUse
    # filter neither reads nor validates them.
    domain_presets_enabled: bool = False
    domain_presets: object = None         # parsed presets (tuple) after load_query_settings; raw option otherwise
    domain_min_confidence: float = 0.6
    domain_apply_mode: str = 'ask'
    search_gate_enabled: bool = False
    search_gate_threshold: float = 0.85


def load_settings(env):
    preset = option(env, 'preset') or DEFAULT_PRESET
    mode = option(env, 'on_all_dropped') or 'keep_original'
    if mode not in ALL_DROPPED_MODES:
        raise ConfigError(f'on_all_dropped must be one of {ALL_DROPPED_MODES}')
    rate = option(env, 'usd_per_second')
    rate = None if rate is None else _num(rate, 'usd_per_second', None, 1e-12, 1.0)
    tools = []
    for entry in (option(env, 'mcp_tools') or '').split(','):
        name, _, fmt = entry.strip().partition('=')
        name, fmt = name.strip(), (fmt.strip().lower() or 'tavily')
        if not name:
            continue
        if fmt not in FORMATS:
            raise ConfigError(f'mcp_tools format must be one of {FORMATS}')
        tools.append((name, fmt))
    return Settings(enabled=_bool(option(env, 'enabled'), True), preset=preset, usd_per_second=rate,
                    concurrency=_num(option(env, 'concurrency'), 'concurrency', 1, 1, 8, int),
                    max_docs=_num(option(env, 'max_docs'), 'max_docs', 10, 1, 50, int),
                    wall_budget_seconds=_num(option(env, 'wall_budget_seconds'), 'wall_budget_seconds', 20.0, 1, 60),
                    on_all_dropped=mode,
                    mcp_tools=tuple(tools),
                    log_file=option(env, 'log_file'))


# --- result parsing ------------------------------------------------------------------------------------------------
def _first_str(d, keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def as_item(el):
    """A result item: a dict with a URL or a text field (but not a bare ``{"type": "text", "text": ...}`` content
    block). -> {'title', 'url', 'text'} or None."""
    if not isinstance(el, dict):
        return None
    url, text = _first_str(el, URL_KEYS), _first_str(el, TEXT_KEYS)
    if url is None and (text is None or el.get('type') == 'text'):   # an MCP text content block is not a result
        return None
    return {'title': _first_str(el, TITLE_KEYS) or '', 'url': url or '', 'text': text or ''}


def _flat(lst, path):
    """Every element is a result item -> [group]."""
    if not lst:
        return None
    items = [as_item(el) for el in lst]
    if any(it is None for it in items):
        return None
    return [(path, list(enumerate(items)))]


def _nested(lst, path):
    """Elements are strings (commentary) or dicts whose ``content`` list holds result items, e.g.
    ``[{"tool_use_id": ..., "content": [{"title", "url"}, ...]}, "text"]``. -> one group per content list."""
    groups, n_items = [], 0
    for i, el in enumerate(lst):
        if isinstance(el, str):
            continue
        if not isinstance(el, dict) or not isinstance(el.get('content'), list) or as_item(el) is not None:
            return None
        if not el['content']:
            continue
        g = _flat(el['content'], path + (i, 'content'))
        if g is None:
            return None
        groups += g
        n_items += len(g[0][1])
    return groups if n_items else None


def _locate(obj, path=(), depth=0):
    if isinstance(obj, list):
        return _flat(obj, path) or _nested(obj, path)
    if isinstance(obj, dict) and depth < 2:
        for k in LIST_KEYS:
            if isinstance(obj.get(k), list):
                g = _locate(obj[k], path + (k,), depth + 1)
                if g:
                    return g
        if isinstance(obj.get('web'), dict):                      # e.g. {"web": {"results": [...]}}
            return _locate(obj['web'], path + ('web',), depth + 1)
    return None


def _get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def _set(obj, path, value):
    if not path:
        return value
    _get(obj, path[:-1])[path[-1]] = value
    return obj


def _text_blocks(obj):
    """MCP-style text content blocks whose text may be JSON: list of {"type": "text", "text": ...} or
    {"content": [blocks]}. -> [(path, text)]"""
    blocks, base = (obj, ()) if isinstance(obj, list) else (obj.get('content'), ('content',)) \
        if isinstance(obj, dict) else (None, ())
    if not isinstance(blocks, list):
        return []
    return [(base + (i, 'text'), b['text']) for i, b in enumerate(blocks)
            if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str)]


@dataclass
class Parsed:
    shape: str                       # 'object' | 'json_string' | 'json_text_block'
    groups: list                     # [(path, [(index, item)])], paths relative to ``inner``
    inner: object                    # the object the groups point into
    wrapper_path: tuple | None = None   # where the JSON string sits in the original (json_* shapes)

    def documents(self):
        """[(doc_id, item)] in result order."""
        out = []
        for gi, (_, items) in enumerate(self.groups):
            out += [(f'r{gi}.{i}', it) for i, it in items]
        return out

    def rebuild(self, original, dropped_ids):
        """A deep copy of ``original`` with the dropped items removed from their lists; everything else unchanged."""
        inner = copy.deepcopy(self.inner)
        for gi, (path, _) in enumerate(self.groups):
            lst = _get(inner, path)
            kept = [el for i, el in enumerate(lst) if f'r{gi}.{i}' not in dropped_ids]
            inner = _set(inner, path, kept)
        if self.shape == 'object':
            return inner
        text = json.dumps(inner, ensure_ascii=False)
        return text if self.shape == 'json_string' else _set(copy.deepcopy(original), self.wrapper_path, text)


def parse_response(resp):
    """Recognise a few plausible search-result shapes. -> Parsed or None (unknown shape: leave the result alone)."""
    g = _locate(resp)
    if g:
        return Parsed('object', g, resp)
    if isinstance(resp, str):
        try:
            inner = json.loads(resp)
        except ValueError:
            return None
        g = _locate(inner) if isinstance(inner, (dict, list)) else None
        return Parsed('json_string', g, inner, ()) if g else None
    for path, text in _text_blocks(resp):
        try:
            inner = json.loads(text)
        except ValueError:
            continue
        g = _locate(inner) if isinstance(inner, (dict, list)) else None
        if g:
            return Parsed('json_text_block', g, inner, path)
    return None


# --- MCP text format (Tavily) -------------------------------------------------------------------------------------
class TextView:
    """The text blocks of an MCP tool response and a writer that rebuilds the SAME shape.

    Accepted: a list of content blocks (``{"type": "text", "text": ...}``; other block types are kept), an object with
    such a ``content`` list (other keys kept), or a plain string (one block)."""

    def __init__(self, resp):
        self.resp = resp
        if isinstance(resp, str):
            self.blocks, self.idx = None, [None]
            self.texts = [resp]
            return
        blocks = resp if isinstance(resp, list) else resp.get('content') if isinstance(resp, dict) else None
        if not isinstance(blocks, list):
            raise ValueError('no content blocks')
        self.blocks = blocks
        self.idx = [i for i, b in enumerate(blocks)
                    if isinstance(b, dict) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
        if not self.idx:
            raise ValueError('no text blocks')
        self.texts = [blocks[i]['text'] for i in self.idx]

    @property
    def is_string(self):
        return self.blocks is None

    def write(self, new_texts):
        """new_texts aligned with ``self.texts``; None removes that block (string shape: becomes '')."""
        if self.is_string:
            return new_texts[0] or ''
        repl = dict(zip(self.idx, new_texts))
        out = []
        for i, b in enumerate(self.blocks):
            if i in repl:
                if repl[i] is not None:
                    nb = copy.deepcopy(b)
                    nb['text'] = repl[i]
                    out.append(nb)
            else:
                out.append(copy.deepcopy(b))
        if isinstance(self.resp, list):
            return out
        res = copy.deepcopy(self.resp)
        res['content'] = out
        return res


@dataclass
class TextParsed:
    shape: str                        # 'tavily_text'
    view: TextView
    docs: list                        # [(doc_id, item)]
    rebuild_texts: object             # callable(dropped_ids) -> new texts aligned with view.texts

    def documents(self):
        return list(self.docs)

    def rebuild(self, original, dropped_ids):
        return self.view.write(self.rebuild_texts(dropped_ids))


_TAVILY_HEAD = re.compile(r'(?m)^Detailed Results:$')
_TAVILY_IMAGES = re.compile(r'\n\nImages:(?:\n\n\[\d+\] URL: [^\n]*(?:\n   Description: [^\n]*)?)+\Z')


def _field(lines, prefix):
    return lines[0][len(prefix):] if lines and lines[0].startswith(prefix) else None


def _tavily_block(block):
    """``Title: ..\n[ID: ..]\nURL: ..\n[Content: ..][\nRaw Content: ..][\nFavicon: ..]`` -> item or None."""
    lines = block.split('\n')
    title = _field(lines, 'Title: ')
    if title is None:
        return None
    rest = lines[1:]
    if _field(rest, 'ID: ') is not None:
        rest = rest[1:]
    url = _field(rest, 'URL: ')
    if url is None:
        return None
    body = '\n'.join(rest[1:])
    text = ''
    if body.startswith('Content: '):
        text = body[len('Content: '):]
        for marker in ('\nRaw Content: ', '\nFavicon: '):      # score the snippet only, not raw page text
            j = text.find(marker)
            if j >= 0:
                text = text[:j]
    elif body and not body.startswith(('Raw Content: ', 'Favicon: ')):
        return None
    return {'title': title, 'url': url, 'text': text.strip()}


def parse_tavily(view):
    """tavily-mcp ``formatResults`` text (search tool). Non-result parts are kept verbatim."""
    for bi, text in enumerate(view.texts):
        m = _TAVILY_HEAD.search(text)
        if m is None:
            continue
        head, body = text[:m.end()], text[m.end():]
        t = _TAVILY_IMAGES.search(body)
        tail = body[t.start():] if t else ''
        body = body[:t.start()] if t else body
        if not body:
            return None                                         # no results
        if not body.startswith('\n\n'):
            return None
        blocks = re.split(r'\n\n(?=Title: )', body[2:])
        items = [_tavily_block(b) for b in blocks]
        if any(it is None for it in items):
            return None
        docs = [(f'b{bi}.{k}', it) for k, it in enumerate(items)]

        def rebuild(dropped, bi=bi, head=head, blocks=blocks, tail=tail):
            new = list(view.texts)
            new[bi] = head + ''.join('\n\n' + b for k, b in enumerate(blocks) if f'b{bi}.{k}' not in dropped) + tail
            return new
        return TextParsed('tavily_text', view, docs, rebuild)
    return None




def tool_format(tool, settings):
    """'tavily' for the verified Tavily tool or an mcp_tools entry, 'json' for the built-in WebSearch (only reachable
    through a user-added hook; passed through), None (zero intervention) for everything else."""
    for name, fmt in settings.mcp_tools:
        if name == tool:
            return None if fmt == 'off' else fmt
    if isinstance(tool, str) and re.fullmatch(KNOWN_TOOLS_MATCHER, tool):   # fullmatch: Python '$' allows a trailing \n
        return 'tavily'
    if tool == 'WebSearch':
        return 'json'
    return None


def passthrough_reason(resp):
    """Envelope states that must not be edited: an MCP error result, or structuredContent next to the text."""
    if isinstance(resp, dict):
        if resp.get('isError') is True:
            return 'tool_error_passthrough'
        if 'structuredContent' in resp or 'structured_content' in resp:
            return 'structured_content_passthrough'
    return None


def parse_for(fmt, resp):
    if fmt == 'json':
        return parse_response(resp)
    try:
        view = TextView(resp)
    except ValueError:
        return None
    return parse_tavily(view)


def shape_of(obj, depth=0):
    """Structural signature for logs: key names (truncated) and types only; never values."""
    if isinstance(obj, dict):
        if depth >= 5:
            return 'dict'
        keys = sorted(obj)[:20]
        return {str(k)[:40]: shape_of(obj[k], depth + 1) for k in keys}
    if isinstance(obj, list):
        kinds = []
        for el in obj[:3]:
            s = shape_of(el, depth + 1)
            if s not in kinds:
                kinds.append(s)
        return {'list': len(obj), 'of': kinds}
    if isinstance(obj, bool):
        return 'bool'
    if isinstance(obj, (int, float)):
        return 'num'
    if isinstance(obj, str):
        return 'str'
    return 'null' if obj is None else type(obj).__name__


def _b(s):
    return len(s.encode('utf-8'))


def _jbytes(o):
    return _b(o) if isinstance(o, str) else _b(json.dumps(o, ensure_ascii=False))


# --- output --------------------------------------------------------------------------------------------------------
def hook_output(updated=None, context=None):
    """The documented PostToolUse JSON output. ``updated`` None -> no replacement."""
    out = {'hookEventName': 'PostToolUse'}
    if updated is not None:
        out['updatedToolOutput'] = updated
    if context is not None:
        out['additionalContext'] = context
    return {'hookSpecificOutput': out}


# --- processing ----------------------------------------------------------------------------------------------------
def process(payload, settings, env, transport=None, clock=time.perf_counter, sleep=time.sleep):
    """-> (output dict or None, log record). Never raises for expected failures; callers still guard."""
    t0 = clock()
    log = {'event': N.LOG_TAG, 'status': None}

    def done(status, out=None, **kw):
        log.update(status=status, seconds=round(clock() - t0, 4), **kw)
        return out, log

    if not isinstance(payload, dict):
        return done('bad_input')
    if payload.get('hook_event_name', 'PostToolUse') != 'PostToolUse':
        return done('not_post_tool_use')
    tool = payload.get('tool_name')
    log['tool'] = tool if isinstance(tool, str) else None
    if not settings.enabled:
        return done('disabled')
    fmt = tool_format(tool, settings)
    if fmt is None:
        return done('tool_not_enabled')
    log['format'] = fmt
    tin = payload.get('tool_input') if isinstance(payload.get('tool_input'), dict) else {}
    query = next((tin[k] for k in QUERY_KEYS if isinstance(tin.get(k), str) and tin[k].strip()), None)
    resp = payload.get('tool_response')
    log['shape'] = shape_of(resp)
    if query is None:
        return done('no_query')
    why = passthrough_reason(resp)
    if why is not None:
        return done(why)
    parsed = parse_for(fmt, resp)
    if parsed is None:
        return done('unrecognized_shape')
    docs = parsed.documents()
    with_text = [(i, it) for i, it in docs if it['text']]
    log.update(results_in=len(docs), results_with_text=len(with_text), parsed_as=parsed.shape)
    if not with_text:
        return done('no_text_in_results')
    if _present(env, KEY_ENVS) is None:
        return done('missing_key')
    endpoint_var = _present(env, ENDPOINT_ENVS)
    if endpoint_var is None:
        return done('missing_endpoint')
    if settings.usd_per_second is None:
        return done('missing_rate')

    from .contract import AdapterConfig
    from .context_filter import PRESETS, ContextFilter, Document, FilterInputError
    from .core import build_backend
    from .keys import CredentialMissing

    if settings.preset not in PRESETS:
        return done('unknown_preset')
    scored = with_text[:settings.max_docs]
    wall = settings.wall_budget_seconds
    cfg = AdapterConfig(transport='real' if transport is None else 'fake', backend='choice_endpoint',
                        trace_dir='unused', timeout_seconds=min(REQUEST_TIMEOUT_SECONDS, wall),
                        choice_endpoint_id=env[endpoint_var].strip(), choice_poll_interval_seconds=POLL_INTERVAL_SECONDS,
                        choice_max_status_polls=max(1, int(wall / POLL_INTERVAL_SECONDS)),
                        choice_usd_per_second=settings.usd_per_second)
    key_var = _present(env, KEY_ENVS)

    def key_provider():                 # lazy: read only at the first real request, never logged
        v = env.get(key_var, '').strip()
        if not v:
            raise CredentialMissing(f'{key_var} is not set')
        return v

    backend = build_backend(cfg, transport=transport, key_provider=key_provider, clock=clock, sleep=sleep)
    # Client-side estimate cap per hook call: enough to admit every scored document at full reservation (the
    # ContextFilter ledger reserves 661 s per open job + 60 s idle); a hook call is one ContextFilter run.
    cap = settings.usd_per_second * (len(scored) * 661 + 60 + wall) * 1.0001
    limits = dataclasses.replace(PRESETS[settings.preset].limits, oversize='flag', max_docs=max(1, len(scored)))
    flt = ContextFilter(cfg, backend, preset=settings.preset, limits=limits, budget_cap_usd=cap,
                        max_submissions=len(scored), concurrency=settings.concurrency, logger=None, clock=clock)
    try:
        res = flt.filter(query, [Document(i, it['title'], it['url'], it['text']) for i, it in scored])
    except FilterInputError:
        return done('input_refused')
    finally:
        flt.close()
    dropped = {r.id for r in res.documents if r.decision == 'drop'}
    n, k = len(docs), len(docs) - len(dropped)
    log.update(results_scored=len(scored), results_out=k, status_counts=res.reduction['status_counts'],
               stopped=res.reduction['stopped'], filter_submissions=res.budget['submissions'],
               text_bytes_in=sum(_b(it['text']) for _, it in docs),
               text_bytes_out=sum(_b(it['text']) for i, it in docs if i not in dropped),
               response_bytes_in=_jbytes(resp))
    if not dropped:
        return done('all_kept')
    label = f'{N.CONTEXT_LABEL} ({settings.preset})'
    if k == 0 and settings.on_all_dropped == 'keep_original':
        log['response_bytes_out'] = log['response_bytes_in']
        return done('all_dropped_kept_original', hook_output(
            context=f'{label} kept 0/{n} results for this search; the original results are shown unchanged.'))
    updated = parsed.rebuild(resp, dropped)
    log['response_bytes_out'] = _jbytes(updated)
    return done('filtered', hook_output(updated=updated, context=f'{label} kept {k}/{n} results'))


# --- entry point ---------------------------------------------------------------------------------------------------
def _emit_log(rec, settings, stderr):
    line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
    try:
        print(line, file=stderr)
    except Exception:  # noqa: BLE001
        pass
    path = getattr(settings, 'log_file', None)
    if path:
        try:
            with open(os.path.expanduser(path), 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError:
            pass


def run_with_budget(fn, seconds):
    """Run ``fn`` in a daemon thread. -> (finished, value). An exception counts as finished with value None."""
    box = {}

    def target():
        try:
            box['value'] = fn()
        except Exception as e:  # noqa: BLE001 - fail-open
            box['error'] = type(e).__name__
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return False, None
    return True, box.get('value', ('error', box.get('error')))


def main(stdin=None, stdout=None, stderr=None, env=None, transport=None, exit_fn=os._exit, clock=time.perf_counter,
         sleep=time.sleep, toggle_file=None):
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    env = os.environ if env is None else env
    logging.getLogger('httpx').setLevel(logging.WARNING)       # httpx logs request URLs (endpoint ID) at INFO
    settings, rec = None, {'event': N.LOG_TAG}
    try:
        settings = load_settings(env)
        # The filter toggle (toggles.json, "filter": false) turns only this filter off; the global enabled=false
        # already does. A missing or corrupt toggle file changes nothing here.
        from . import toggles as T
        if T.read(toggle_file or T.toggle_path(env))[0].get('filter') is False:
            settings = dataclasses.replace(settings, enabled=False)
        payload = json.loads(stdin.read())
    except ConfigError as e:
        rec.update(status='config_error', detail=str(e))
        payload = None
    except Exception as e:  # noqa: BLE001
        rec.update(status='bad_input', detail=type(e).__name__)
        payload = None
    timed_out = False
    if payload is not None:
        t0 = clock()
        ok, value = run_with_budget(lambda: process(payload, settings, env, transport, clock, sleep),
                                    settings.wall_budget_seconds)
        if not ok:
            timed_out = True
            rec.update(status='wall_budget_exceeded', seconds=round(clock() - t0, 4),
                       tool=payload.get('tool_name') if isinstance(payload, dict) else None)
        elif isinstance(value, tuple) and len(value) == 2 and value[0] == 'error':
            rec.update(status='error', detail=value[1])
        else:
            out, rec = value
            if out is not None:
                try:
                    stdout.write(json.dumps(out, ensure_ascii=False))
                    stdout.flush()
                except Exception as e:  # noqa: BLE001
                    rec.update(status='write_error', detail=type(e).__name__)
    _emit_log(rec, settings, stderr)
    if timed_out:
        exit_fn(0)        # in-flight requests are abandoned; nothing was printed, so the result stays unchanged
    return 0


if __name__ == '__main__':
    main()
    sys.exit(0)
