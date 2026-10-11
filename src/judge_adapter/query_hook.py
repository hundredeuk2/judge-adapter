# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Claude Code PreToolUse hook: two OPTIONAL query-stage options for web searches, both OFF by default. EXPERIMENTAL.

Run by ``hooks/query_options_hook.py`` (which exits at once, before reading stdin, when both options are off). Reads one
hook input JSON object on stdin, prints at most one JSON object on stdout, always exits 0.

Tools: the built-in ``WebSearch`` and the Tavily MCP tool ``tavily_search`` (tavily-mcp npm 0.2.22, matcher
``plugin_hook.KNOWN_TOOLS_MATCHER``). Every other tool gets zero intervention.

Options (fixed order: the gate runs first; on a deny the domain decision is skipped):

- ``search_gate``: one choice call "is this search needed?" over the options needed / not_needed / cannot_tell (prompt
  ``gate-v0-dev``). Only the query is sent: the PreToolUse input has no user-message field and this hook never reads
  the transcript file. Deny only when P(not_needed) >= ``search_gate_threshold`` (pre-fixed policy value, default
  0.85, not calibrated; a high probability is not treated as a correct judgment). A local guard (recency,
  verification, a specific source; Tavily date / time-range fields; domain fields already set) keeps the call without
  any endpoint call; it is a heuristic and does not identify an explicit user request to search. Loop prevention: a
  denied (session, tool, normalized query) is recorded as a SHA-256 hash only, with a TTL, under a per-key
  inter-process lock (``RetryState.claim``); at most one deny per key within the TTL, and the same query in the same
  session then gets no output at all (normal flow). Missing session id, lock or I/O problems: the gate never denies.
  Explicit user request (``prompt_signal``, UserPromptSubmit hook): before anything else the gate reads the session's
  prompt state. ``explicit`` (the user's prompt matched an explicit-search pattern) -> status
  ``explicit_request_keep``: no endpoint call, no output. No state (e.g. the gate was just switched on), an
  unreadable state, a session mismatch, a state older than ``prompt_signal.PROMPT_STATE_TTL_SECONDS`` or a prompt_id
  mismatch, or a state written before the last toggle change -> the search is kept too (``keep_*``), with no
  endpoint call. Only a fresh "no explicit request" state
  lets the gate ask the endpoint. The prompt text never reaches this hook or the endpoint.
- ``domain_presets``: one choice call that picks ONE preset from a user-defined fixed list (prompt
  ``domain-v0-dev``); ``none`` (no restriction) is always available. Only the preset's own fixed domain list is
  applied; nothing the model writes is executed. Applied only when the input has none of the tool's domain fields
  and the top probability is >= ``domain_min_confidence`` (default 0.6). The query and every other field stay
  unchanged.

Hook contract (https://code.claude.com/docs/en/hooks and https://code.claude.com/docs/en/hooks-guide, read
2026-10-11; Claude Code CLI 2.1.285 installed):

- Input: common fields ``session_id``, ``transcript_path``, ``cwd``, ``permission_mode``, ``hook_event_name``, plus
  ``tool_name``, ``tool_input``, ``tool_use_id``. WebSearch ``tool_input`` (Tool input table): "``query`` | string |
  Search query", "``allowed_domains`` | array | Optional: only include results from these domains",
  "``blocked_domains`` | array | Optional: exclude results from these domains". No user-message field is listed.
- Output (PreToolUse decision control): ``hookSpecificOutput`` with ``hookEventName``, ``permissionDecision``
  (allow / deny / ask / defer), ``permissionDecisionReason``, ``updatedInput``, ``additionalContext``.
  * ``permissionDecisionReason``: "For ``"ask"``, shown to the user in the permission prompt. When Claude Code denies
    the call in a ``-p`` run where no one can answer that prompt, Claude reads the reason in the tool result instead.
    For ``"deny"``, shown to Claude."  The guide: "With ``"deny"``, Claude Code cancels the tool call and feeds
    ``permissionDecisionReason`` back to Claude."
  * ``updatedInput``: "Replaces the entire input object, so include unchanged fields alongside modified ones. Claude
    Code evaluates permission rules ... against the input your hook returns ... Combine with ``"allow"`` to
    auto-approve, or ``"ask"`` to show the modified input to the user."
  * "``"allow"`` skips the permission prompt" -- this hook NEVER emits ``"allow"`` (it could skip the user's own
    prompt). Deny rules: "Deny and ask rules are still evaluated regardless of what the hook returns"; precedence
    "``deny`` > ``defer`` > ``ask`` > ``allow``".
  * No decision: "Exit code 0 with no output means the hook has no decision to report, so the tool call continues
    through the normal permission flow. The hook can deny the call, but staying silent doesn't approve it." Every
    fail-open path and the retry after a deny print nothing.
  * Timeout: "A timed-out ``command`` ... hook doesn't block the tool call. The call continues through the normal
    permission flow".
- Applying a preset: the reference documents ``updatedInput`` together with ``"allow"`` or ``"ask"``. Since
  ``"allow"`` is never emitted, the default ``domain_apply_mode=ask`` returns ``permissionDecision: "ask"`` with the
  full ``updatedInput``: the user sees and confirms the modified input (and in a ``-p`` run with no one to answer,
  Claude Code denies the call). ``domain_apply_mode=input_only`` returns ``updatedInput`` with no
  ``permissionDecision``; the reference does not document that combination, so whether Claude Code applies it is
  NOT verified.
- WebSearch domain rule: the Claude API web search tool docs say "Provide ``allowed_domains`` or ``blocked_domains``,
  not both. If a request includes both, the API returns a 400 error." Each preset therefore sets exactly one list.
- Tavily ``tavily_search`` 0.2.22 input schema (published build, ``build/index.js``): ``include_domains`` / ``exclude_domains``
  arrays of strings ("A list of domains to specifically include in the search results", "List of domains to
  specifically exclude"), next to ``query``, ``search_depth``, ``topic``, ``time_range``, ``start_date``,
  ``end_date``, ``max_results``, ``include_images``, ``include_image_descriptions``, ``include_raw_content``,
  ``country``, ``include_favicon``, ``exact_match``. The server's ``DEFAULT_PARAMETERS`` environment variable
  overrides any of these, including the domain lists.

Logs: one JSON line per call (stderr, optionally ``log_file``): status, tool name, per option status / calls / seconds /
rounded probabilities / chosen preset name, query byte count. Never the query, user text, URLs, domain lists, keys or
the endpoint ID.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass

from . import plugin_hook as H
from . import plugin_names as N
from . import prompt_signal as PS
from . import query_presets as Q
from . import toggles as T

LOG_TAG = f'{N.PLUGIN_NAME}-query-options-hook'
APPLY_MODES = ('ask', 'input_only')
MAX_QUERY_BYTES = 2_000            # longer queries: no decision (nothing is truncated or sent)
GATE_RETRY_TTL_SECONDS = 600.0     # a denied query is allowed again in the same session for this long
MAX_PRESETS = 8                    # user presets, excluding "none"
MAX_DOMAINS_PER_PRESET = 20
MAX_PRESETS_FILE_BYTES = 65_536
STATE_DIRNAME = 'search-gate-state'

DOMAIN_FIELDS = {'websearch': ('allowed_domains', 'blocked_domains'),     # (include, exclude) per tool
                 'tavily': ('include_domains', 'exclude_domains')}
ALL_DOMAIN_FIELDS = tuple(f for pair in DOMAIN_FIELDS.values() for f in pair)
TAVILY_RECENCY_FIELDS = ('time_range', 'start_date', 'end_date')

_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_-]{0,31}$')
_DOMAIN_RE = re.compile(r'^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)+$')


ConfigError = H.ConfigError

# Gate status when the prompt state (``prompt_signal``) cannot show a fresh "no explicit request" for this session:
# the search is kept without any endpoint call.
PROMPT_STATE_KEEP = {'absent': 'keep_no_prompt_state', 'error': 'keep_prompt_state_error',
                     'mismatch': 'keep_prompt_state_mismatch', 'expired': 'keep_prompt_state_expired',
                     'prompt_mismatch': 'keep_prompt_id_mismatch', 'predates_toggle': 'keep_prompt_state_predates_toggle'}


# --- settings ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainPreset:
    name: str
    description: str
    include: tuple = ()
    exclude: tuple = ()


@dataclass(frozen=True)
class PresetSet:
    source: str                    # 'user' | 'examples'
    presets: tuple                 # DomainPreset, "none" first


def _flag(v, name):
    if v is None:
        return False
    s = v.strip().lower()
    if s in ('1', 'true', 'yes', 'on'):
        return True
    if s in ('0', 'false', 'no', 'off'):
        return False
    raise ConfigError(f'{name} must be true or false')


def parse_domain_presets(raw):
    """``raw``: None (built-in EXAMPLES), a JSON object, or a path to a file holding one. Messages never include
    preset names, domains or the path."""
    source = 'user'
    if raw is None:
        data, source = Q.EXAMPLE_DOMAIN_PRESETS, 'examples'
    else:
        text = raw
        if not raw.lstrip().startswith('{'):
            try:
                with open(os.path.expanduser(raw), 'rb') as f:
                    blob = f.read(MAX_PRESETS_FILE_BYTES + 1)
            except OSError:
                raise ConfigError('domain_presets file could not be read') from None
            if len(blob) > MAX_PRESETS_FILE_BYTES:
                raise ConfigError(f'domain_presets file is over {MAX_PRESETS_FILE_BYTES} bytes')
            text = blob.decode('utf-8', 'replace')
        try:
            data = json.loads(text)
        except ValueError:
            raise ConfigError('domain_presets is not valid JSON') from None
    if not isinstance(data, dict):
        raise ConfigError('domain_presets must be a JSON object {name: {description, include_domains | '
                          'exclude_domains}}')
    presets, none_desc = [], Q.NONE_DESCRIPTION
    for i, (name, spec) in enumerate(data.items()):
        where = f'domain_presets entry {i + 1}'
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ConfigError(f'{where}: name must be 1-32 characters of a-z 0-9 _ - (starting with a letter or digit)')
        if not isinstance(spec, dict) or not set(spec) <= {'description', 'include_domains', 'exclude_domains'}:
            raise ConfigError(f'{where}: must be an object with description and include_domains or exclude_domains')
        desc = spec.get('description')
        if not isinstance(desc, str) or not desc.strip() or len(desc) > 200:
            raise ConfigError(f'{where}: description must be a non-empty string of at most 200 characters')
        lists = {}
        for k in ('include_domains', 'exclude_domains'):
            v = spec.get(k)
            if v is None:
                continue
            if not isinstance(v, list) or not v or len(v) > MAX_DOMAINS_PER_PRESET or \
                    not all(isinstance(d, str) and _DOMAIN_RE.match(d) for d in v):
                raise ConfigError(f'{where}: {k} must be a non-empty list of at most {MAX_DOMAINS_PER_PRESET} bare '
                                  'lower-case domains (no scheme, path or wildcard)')
            lists[k] = tuple(v)
        if name == Q.NONE_PRESET:
            if lists:
                raise ConfigError(f'{where}: the "none" preset cannot list domains')
            none_desc = desc.strip()
            continue
        if len(lists) != 1:
            raise ConfigError(f'{where}: set exactly one of include_domains or exclude_domains (the WebSearch API '
                              'rejects both in one request)')
        presets.append(DomainPreset(name, desc.strip(), lists.get('include_domains', ()),
                                    lists.get('exclude_domains', ())))
    if not presets:
        raise ConfigError('domain_presets needs at least one preset besides "none"')
    if len(presets) > MAX_PRESETS:
        raise ConfigError(f'domain_presets allows at most {MAX_PRESETS} presets besides "none"')
    return PresetSet(source, (DomainPreset(Q.NONE_PRESET, none_desc),) + tuple(presets))


def load_query_settings(env, toggle_file=None):
    """Shared options via ``plugin_hook.load_settings`` plus the query-stage options. Raises ConfigError.

    The toggle file (``toggles.toggle_path(env)`` unless ``toggle_file`` is given) overrides ``search_gate_enabled``,
    ``domain_presets_enabled`` and ``domain_apply_mode`` when it sets them; a missing or corrupt file changes nothing.
    The global ``enabled=false`` still turns both options off (``process`` checks it first)."""
    s = H.load_settings(env)
    o = lambda k: H.option(env, k)            # noqa: E731
    dpe = _flag(o('domain_presets_enabled'), 'domain_presets_enabled')
    sge = _flag(o('search_gate_enabled'), 'search_gate_enabled')
    mode = (o('domain_apply_mode') or 'ask').lower()
    toggles, _ = T.read(toggle_file or T.toggle_path(env))
    sge = toggles.get('gate', sge)
    dpe = toggles.get('domain', dpe)
    mode = toggles.get('domain_mode', mode)
    if mode not in APPLY_MODES:
        raise ConfigError(f'domain_apply_mode must be one of {APPLY_MODES}')
    return dataclasses.replace(
        s, domain_presets_enabled=dpe, search_gate_enabled=sge, domain_apply_mode=mode,
        domain_min_confidence=H._num(o('domain_min_confidence'), 'domain_min_confidence', 0.6, 0.0, 1.0),
        search_gate_threshold=H._num(o('search_gate_threshold'), 'search_gate_threshold', 0.85, 0.5, 1.0),
        domain_presets=parse_domain_presets(o('domain_presets')) if dpe else None)


def tool_kind(tool, settings):
    """'websearch' | 'tavily' | None (zero intervention). Only the two tools whose input schema was checked."""
    if tool == 'WebSearch':
        return 'websearch'
    if isinstance(tool, str) and re.fullmatch(H.KNOWN_TOOLS_MATCHER, tool):
        if any(name == tool and fmt == 'off' for name, fmt in settings.mcp_tools):
            return None
        return 'tavily'
    return None


# --- retry state (hashes only) -------------------------------------------------------------------------------------
def normalize_query(q):
    return ' '.join(unicodedata.normalize('NFKC', q).casefold().split())


def default_state_dir(env):
    data = env.get('CLAUDE_PLUGIN_DATA')
    if isinstance(data, str) and data.strip():
        return os.path.join(data, STATE_DIRNAME)
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    return os.path.join(tempfile.gettempdir(), f'{N.PLUGIN_NAME}-{STATE_DIRNAME}-{uid}')


class RetryState:
    """Deny records for the search gate: one file per denied (session, tool, normalized query), named by the SHA-256 of
    that triple, holding only the deny time. A record older than the TTL counts as absent.

    Concurrency (several hook processes can run at once):
    - ``denied_recently`` is a read-only pre-check (records are replaced atomically with os.replace, so a reader never
      sees a partial file). It never deletes anything.
    - ``claim`` is the only path that may lead to a deny. It runs the final "is there a fresh record?" check and the
      write as one short inter-process critical section under ``fcntl.flock`` on a per-key lock file
      (``<key>.lock``). The lock is taken only after the endpoint has answered and is never held during a call.
    - Pruning takes the same per-key lock (non-blocking; busy keys are skipped) and re-checks inside it that the
      record is stale before removing it, so a fresh record is never removed. A lock file is unlinked only while held;
      after acquiring, a process checks that the path still names the inode it locked (else it retries), so an
      unlinked lock file can never admit two holders.
    - No ``fcntl`` (Windows), a lock wait over ``LOCK_WAIT_SECONDS`` or any I/O error: ``claim`` returns None and the
      caller does not deny.
    """
    LOCK_WAIT_SECONDS = 1.0
    PRUNE_LIMIT = 200

    def __init__(self, root, ttl=GATE_RETRY_TTL_SECONDS, now=time.time):
        self.root, self.ttl, self.now = root, ttl, now

    @staticmethod
    def key(session, tool, query):
        return hashlib.sha256('\0'.join((session, tool, normalize_query(query))).encode('utf-8')).hexdigest()

    def _path(self, key):
        return os.path.join(self.root, key + '.json')

    def _lock_path(self, key):
        return os.path.join(self.root, key + '.lock')

    def _read(self, key):
        """-> 'absent' | 'fresh' | 'stale' | None (state unusable). An unreadable record counts as fresh."""
        try:
            f = open(self._path(key), encoding='utf-8')
        except FileNotFoundError:
            return 'absent'
        except OSError:
            return None
        try:
            with f:
                t = json.load(f).get('t')
        except (OSError, ValueError, AttributeError):
            return 'fresh'
        if not isinstance(t, (int, float)) or self.now() - t <= self.ttl:
            return 'fresh'
        return 'stale'

    def denied_recently(self, key):
        """Read-only pre-check. True / False, or None if the state could not be read (the caller then never denies)."""
        r = self._read(key)
        return None if r is None else r == 'fresh'

    def _acquire(self, key, blocking):
        """-> an fd holding an exclusive flock on <key>.lock, or None (busy, timeout, no fcntl, I/O error)."""
        try:
            import fcntl
        except ImportError:
            return None
        lock = self._lock_path(key)
        deadline = time.monotonic() + (self.LOCK_WAIT_SECONDS if blocking else 0.0)
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
            except OSError:
                return None
            try:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            os.close(fd)
                            return None
                        time.sleep(0.005)
                try:
                    same = os.fstat(fd).st_ino == os.stat(lock).st_ino
                except FileNotFoundError:
                    same = False
                if same:
                    return fd
                os.close(fd)               # the lock file was unlinked by a pruner while we waited: retry
            except OSError:
                os.close(fd)
                return None

    def _write(self, key):
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix='.tmp-', suffix='.json')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump({'v': 1, 't': self.now()}, f)
            os.replace(tmp, self._path(key))
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def claim(self, key):
        """Final check + record under the per-key lock. -> True (recorded now: the caller may deny), False (a fresh
        record already exists, e.g. written by another process: do not deny), None (lock / I/O problem: do not
        deny)."""
        try:
            os.makedirs(self.root, mode=0o700, exist_ok=True)
        except OSError:
            return None
        fd = self._acquire(key, blocking=True)
        if fd is None:
            return None
        try:
            state = self._read(key)
            if state is None:
                return None
            if state == 'fresh':
                return False
            self._write(key)
            result = True
        except OSError:
            return None
        finally:
            os.close(fd)                   # closing the fd releases the flock
        self.prune(skip=key)
        return result

    def prune(self, skip=None):
        """Remove stale records (and their lock files), each under its own lock and re-checked inside it. Leftover
        temp files older than the TTL are removed too. Never blocks; best effort."""
        try:
            entries = []
            with os.scandir(self.root) as it:
                for n, e in enumerate(it):
                    if n >= self.PRUNE_LIMIT:
                        break
                    entries.append(e.name)
        except OSError:
            return
        cutoff = self.now() - self.ttl
        for name in entries:
            path = os.path.join(self.root, name)
            if name.startswith('.tmp-'):
                try:
                    if os.stat(path).st_mtime < cutoff:
                        os.remove(path)
                except OSError:
                    pass
                continue
            if not name.endswith('.json'):
                continue
            key = name[:-5]
            if key == skip or not re.fullmatch(r'[0-9a-f]{64}', key) or self._read(key) != 'stale':
                continue
            fd = self._acquire(key, blocking=False)
            if fd is None:
                continue
            try:
                if self._read(key) == 'stale':          # re-check inside the lock: a fresh record is never removed
                    os.remove(self._path(key))
                    os.remove(self._lock_path(key))      # unlinked while held; waiters re-check the inode
            except OSError:
                pass
            finally:
                os.close(fd)


# --- decisions -----------------------------------------------------------------------------------------------------
def gate_guard(query, tin):
    """A category name if the call must be kept without asking the endpoint, else None."""
    if any(k in tin for k in ALL_DOMAIN_FIELDS):
        return 'domains_set'
    if any(isinstance(tin.get(k), str) and tin[k].strip() for k in TAVILY_RECENCY_FIELDS):
        return 'tavily_recency_field'
    for name, pat in Q.GATE_KEEP_PATTERNS.items():
        if pat.search(query):
            return name
    return None


class _Caller:
    """Lazily built backend + one run-wide ledger shared by every decision of this hook call: at most ``n_calls``
    submissions and the same client-side estimate cap formula as the PostToolUse hook."""

    def __init__(self, settings, env, n_calls, transport, clock, sleep):
        self.settings, self.env, self.n_calls = settings, env, n_calls
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.backend = self.ledger = None

    def _setup(self):
        from .context_filter import _Ledger
        from .contract import AdapterConfig
        from .core import build_backend
        from .keys import CredentialMissing
        s, env = self.settings, self.env
        wall = s.wall_budget_seconds
        endpoint_var = H._present(env, H.ENDPOINT_ENVS)
        key_var = H._present(env, H.KEY_ENVS)
        cfg = AdapterConfig(transport='real' if self.transport is None else 'fake', backend='choice_endpoint',
                            trace_dir='unused', timeout_seconds=min(H.REQUEST_TIMEOUT_SECONDS, wall),
                            choice_endpoint_id=env[endpoint_var].strip(),
                            choice_poll_interval_seconds=H.POLL_INTERVAL_SECONDS,
                            choice_max_status_polls=max(1, int(wall / H.POLL_INTERVAL_SECONDS)),
                            choice_usd_per_second=s.usd_per_second)

        def key_provider():             # lazy: read only at the first real request, never logged
            v = env.get(key_var, '').strip()
            if not v:
                raise CredentialMissing(f'{key_var} is not set')
            return v
        self.backend = build_backend(cfg, transport=self.transport, key_provider=key_provider, clock=self.clock,
                                     sleep=self.sleep)
        cap = s.usd_per_second * (self.n_calls * 661 + 60 + wall) * 1.0001
        self.ledger = _Ledger(self.backend.usd_per_second, cap, self.n_calls, self.clock)

    def ask(self, prompt, query, options):
        """options: [(label, text)] -> {'outcome', 'probs' (label -> p) or None, 'label', 'calls', 'server_ms'}"""
        if self.backend is None:
            self._setup()
        ticket, why = self.ledger.admit()
        if ticket is None:
            return {'outcome': 'budget_stop', 'probs': None, 'label': None, 'calls': 0, 'server_ms': None}
        crit = {f'opt_{i}': text for i, (_, text) in enumerate(options)}
        to_label = {f'opt_{i}': label for i, (label, _) in enumerate(options)}
        sem = {'turn': 1, 'instructions': prompt.instructions, 'state': {'search_query': query}, 'criteria': crit,
               'question_id': prompt.question_id}
        try:
            r = self.backend.call(sem, to_label)
        except Exception as e:  # noqa: BLE001 - unexpected client error: unsettled, never retried
            r = {'outcome': 'http', 'detail': type(e).__name__, 'submissions': 1, 'job_final': False}
        self.ledger.settle(ticket, r)
        probs = r.get('probabilities') if r.get('outcome') in ('ok', 'no_decision') else None
        return {'outcome': r.get('outcome'), 'probs': probs if isinstance(probs, dict) else None,
                'label': r.get('label') if r.get('outcome') == 'ok' else None, 'calls': r.get('submissions', 0),
                'server_ms': (r.get('server_timing') or {}).get('execution_ms')}

    def close(self):
        if self.backend is not None:
            self.backend.close()


def _rounded(probs):
    return None if probs is None else {k: round(float(v), 3) for k, v in probs.items()}


def domain_options(preset_set):
    out = []
    for p in preset_set.presets:
        text = f'{p.name}: {p.description}'
        if p.include:
            text += ' Only these sites: ' + ', '.join(p.include) + '.'
        elif p.exclude:
            text += ' Excludes these sites: ' + ', '.join(p.exclude) + '.'
        out.append((p.name, text))
    return out


def apply_preset(kind, tin, preset):
    """The full original tool_input with only the tool's one domain field for this preset added."""
    inc, exc = DOMAIN_FIELDS[kind]
    new = dict(tin)
    if preset.include:
        new[inc] = list(preset.include)
    else:
        new[exc] = list(preset.exclude)
    return new, (inc if preset.include else exc)


def deny_output():
    return {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': 'deny',
                                   'permissionDecisionReason': Q.GATE_DENY_NOTICE}}


def preset_output(updated, preset, field, mode):
    hso = {'hookEventName': 'PreToolUse'}
    if mode == 'ask':
        hso['permissionDecision'] = 'ask'
        n = len(preset.include or preset.exclude)
        hso['permissionDecisionReason'] = (f'davi domain preset "{preset.name}" (experimental, optional) sets {field} '
                                           f'({n} domains) on this search. Approve to run it with this restriction.')
    hso['updatedInput'] = updated
    return {'hookSpecificOutput': hso}


# --- processing ----------------------------------------------------------------------------------------------------
def process(payload, settings, env, transport=None, clock=time.perf_counter, sleep=time.sleep, now=time.time,
            state_dir=None, prompt_state_dir=None, toggle_file=None):
    """-> (output dict or None, log record). Gate first, then domain presets."""
    t0 = clock()
    log = {'event': LOG_TAG, 'status': None}

    def done(status, out=None):
        log.update(status=status, seconds=round(clock() - t0, 4))
        return out, log

    if not isinstance(payload, dict):
        return done('bad_input')
    if payload.get('hook_event_name', 'PreToolUse') != 'PreToolUse':
        return done('not_pre_tool_use')
    tool = payload.get('tool_name')
    log['tool'] = tool if isinstance(tool, str) else None
    if not settings.enabled:
        return done('disabled')
    if not (settings.search_gate_enabled or settings.domain_presets_enabled):
        return done('options_off')
    kind = tool_kind(tool, settings)
    if kind is None:
        return done('tool_not_enabled')
    tin = payload.get('tool_input')
    if not isinstance(tin, dict):
        return done('no_query')
    query = tin.get('query')
    if not isinstance(query, str) or not query.strip():
        return done('no_query')
    qbytes = len(query.encode('utf-8'))
    log['query_bytes'] = qbytes
    if qbytes > MAX_QUERY_BYTES:
        return done('query_too_long')

    gate = {'prompt': Q.DEFAULT_GATE_PROMPT, 'status': None, 'calls': 0} if settings.search_gate_enabled else None
    dom = {'prompt': Q.DEFAULT_DOMAIN_PROMPT, 'status': None, 'calls': 0} if settings.domain_presets_enabled else None
    if gate is not None:
        log['gate'] = gate
    if dom is not None:
        log['domain'] = dom

    # Local checks first: they decide which endpoint calls are needed at all.
    state = key = None
    if gate is not None:
        session = payload.get('session_id')
        prompt_state = None
        if isinstance(session, str) and session.strip():
            toggled_at = T.read(toggle_file or T.toggle_path(env))[0].get('updated')
            prompt_state = PS.PromptState(prompt_state_dir or PS.default_state_dir(env), now=now).read(
                session, payload.get('prompt_id'), not_before=toggled_at)
        if not isinstance(session, str) or not session.strip():
            gate['status'] = 'abstain_no_session'
        elif prompt_state == 'explicit':
            gate['status'] = 'explicit_request_keep'          # the user's prompt asked for a search: no call
        elif prompt_state != 'not_explicit':
            gate['status'] = PROMPT_STATE_KEEP[prompt_state]  # no usable prompt state: keep, no call
        else:
            state = RetryState(state_dir or default_state_dir(env), now=now)
            key = RetryState.key(session, tool, query)
            recent = state.denied_recently(key)
            if recent is None:
                gate['status'] = 'abstain_state_unavailable'
            elif recent:
                gate['status'] = 'retry_allowed'
                if dom is not None:
                    dom['status'] = 'skipped_retry'
                return done('retry_allowed')          # no output at all: normal permission flow
            else:
                guard = gate_guard(query, tin)
                if guard is not None:
                    gate.update(status='keep_guard', guard=guard)
    if dom is not None:
        inc, exc = DOMAIN_FIELDS[kind]
        if inc in tin or exc in tin:
            dom['status'] = 'existing_domains'
    pending = [x for x in (gate, dom) if x is not None and x['status'] is None]
    if not pending:
        return done('explicit_request_keep' if gate is not None and gate['status'] == 'explicit_request_keep'
                    else 'unchanged')
    if H._present(env, H.KEY_ENVS) is None:
        return done('missing_key')
    if H._present(env, H.ENDPOINT_ENVS) is None:
        return done('missing_endpoint')
    if settings.usd_per_second is None:
        return done('missing_rate')

    caller = _Caller(settings, env, len(pending), transport, clock, sleep)
    try:
        if gate is not None and gate['status'] is None:
            t = clock()
            prompt = Q.GATE_PROMPTS[Q.DEFAULT_GATE_PROMPT]
            r = caller.ask(prompt, query, list(prompt.options))
            p = _rounded(r['probs'])
            gate.update(calls=r['calls'], seconds=round(clock() - t, 4), outcome=r['outcome'], probabilities=p,
                        threshold=settings.search_gate_threshold, server_ms=r['server_ms'])
            p_not = (r['probs'] or {}).get('not_needed')
            if r['outcome'] != 'ok' or p_not is None:
                gate['status'] = 'abstain_no_answer'
            elif p_not >= settings.search_gate_threshold:
                claimed = state.claim(key)          # short per-key lock, taken only after the endpoint answered
                if claimed is False:
                    gate['status'] = 'already_denied_concurrently'     # another process denied it: no output
                elif claimed is None:
                    gate['status'] = 'abstain_state_write_failed'       # cannot prevent a deny loop: never deny
                else:
                    gate['status'] = 'deny'
                    if dom is not None and dom['status'] is None:
                        dom['status'] = 'skipped_gate_deny'
                    return done('denied', deny_output())
            else:
                gate['status'] = 'keep'
        if dom is not None and dom['status'] is None:
            t = clock()
            pset = settings.domain_presets
            dom.update(source=pset.source, n_presets=len(pset.presets), apply_mode=settings.domain_apply_mode)
            r = caller.ask(Q.DOMAIN_PROMPTS[Q.DEFAULT_DOMAIN_PROMPT], query, domain_options(pset))
            top = max(r['probs'].values()) if r['probs'] else None
            dom.update(calls=r['calls'], seconds=round(clock() - t, 4), outcome=r['outcome'],
                       p_top=None if top is None else round(float(top), 3), min_confidence=settings.domain_min_confidence,
                       server_ms=r['server_ms'])
            if r['outcome'] != 'ok' or r['label'] is None:
                dom['status'] = 'no_answer'
            else:
                dom['preset'] = r['label']
                if r['label'] == Q.NONE_PRESET:
                    dom['status'] = 'none'
                elif top < settings.domain_min_confidence:
                    dom['status'] = 'low_confidence'
                else:
                    preset = next(p for p in pset.presets if p.name == r['label'])
                    updated, field = apply_preset(kind, tin, preset)
                    dom['status'] = 'applied'
                    return done('domain_applied', preset_output(updated, preset, field, settings.domain_apply_mode))
    finally:
        caller.close()
    return done('unchanged')


# --- entry point ---------------------------------------------------------------------------------------------------
def main(stdin=None, stdout=None, stderr=None, env=None, transport=None, exit_fn=os._exit, clock=time.perf_counter,
         sleep=time.sleep, now=time.time, state_dir=None, prompt_state_dir=None, toggle_file=None):
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    env = os.environ if env is None else env
    logging.getLogger('httpx').setLevel(logging.WARNING)       # httpx logs request URLs (endpoint ID) at INFO
    settings, rec = None, {'event': LOG_TAG}
    try:
        settings = load_query_settings(env, toggle_file)
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
        ok, value = H.run_with_budget(
            lambda: process(payload, settings, env, transport, clock, sleep, now, state_dir, prompt_state_dir,
                            toggle_file),
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
    try:
        rec['data_dir_id'] = T.data_dir_id(T.data_dir(env))
    except Exception:  # noqa: BLE001
        pass
    H._emit_log(rec, settings, stderr)
    if timed_out:
        exit_fn(0)        # nothing was printed: the tool call goes through the normal permission flow
    return 0


if __name__ == '__main__':
    main()
    sys.exit(0)
