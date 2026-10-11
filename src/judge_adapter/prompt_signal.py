# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""UserPromptSubmit hook for the search gate: notes whether the user's prompt explicitly asks for a search. Stdlib only.

Why: the PreToolUse input of WebSearch / ``tavily_search`` has no user-message field, so the gate alone cannot see
an explicit request. A real smoke (2026-10-11) showed the failure: the prompt "Using web search, confirm how many days
there are in a week. Answer in one sentence." led to a WebSearch "how many days in a week", which the gate rated
P(not_needed)=0.999 and denied. This hook runs before Claude handles the prompt and leaves a small per-session state
that the gate reads.

Hook contract (https://code.claude.com/docs/en/hooks, UserPromptSubmit, read 2026-10-11, Claude Code CLI 2.1.285):

- Input: "In addition to the common input fields, UserPromptSubmit hooks receive: ``prompt`` | The user's prompt
  text". JSON example: ``{"session_id": "abc123", "prompt_id": "550e8400-...", "transcript_path": ...,
  "cwd": ..., "permission_mode": "default", "hook_event_name": "UserPromptSubmit", "prompt": "..."}``. Common
  fields: "``session_id`` | Current session identifier", "``prompt_id`` | UUID identifying the user prompt currently
  being processed. ... Absent until the first user input".
- Output: "Exit 0 with no output means the hook has no decision and the prompt continues normally." But: "Exit 0
  with no JSON output: the prompt proceeds normally. Plain-text stdout is added as context for Claude." So this hook
  NEVER writes to stdout (anything there would reach Claude); it logs to stderr only and always exits 0. It never
  blocks a prompt ("Exit 2: blocks the prompt" -- never used).
- "UserPromptSubmit has no matcher support -- the hook always fires on every occurrence of the event."

Behaviour (only when the search gate is effectively on; the launcher ``hooks/user_prompt_hook.py`` exits before
reading stdin otherwise, with no state and no output):

- The prompt is checked locally against ``query_presets.EXPLICIT_SEARCH_PATTERNS`` (fixed list, English and
  Korean, case-insensitive after NFKC). The prompt text is never stored, logged or sent anywhere; this module makes
  no network call and reads no transcript.
- State: one file per session, ``<sha256(session_id)>.json`` under ``${CLAUDE_PLUGIN_DATA}/prompt-state/`` (or the
  system temp directory), holding only ``{"v", "t" (time), "explicit" (bool), "s" (session hash prefix), "p"
  (prompt_id hash prefix or null)}``. Atomic writes (temp file + ``os.replace``), 0600.
- Every prompt updates the session's state:
  * signal found -> ``explicit: true`` (time = now);
  * no signal while a fresh ``explicit: true`` state exists -> left unchanged (the earlier request stays protected
    until its TTL), unless the prompt clearly starts a new task (it begins with one of
    ``query_presets.NEW_TASK_PREFIXES``). Ambiguous follow-ups ("continue", "yes, go ahead", "계속", "그걸로", "now do
    the same for a month") therefore keep the protection;
  * otherwise (no signal and no fresh flag, or a clear new task) -> ``explicit: false`` (time = now): the gate may
    run for this prompt's searches.
  * Write or lock failure: the old file is removed if possible (a signal is still written if it can be), so the gate
    finds no usable state and keeps every search.
- Granularity: there is no documented turn id in the PreToolUse input other than the common ``prompt_id`` field.
  When both the state and the PreToolUse input carry a prompt_id, the gate requires them to match (else it keeps
  the search); when either lacks it, the state is matched by session and age only (``PROMPT_STATE_TTL_SECONDS``),
  which is a time window, not an exact turn.
- Toggles: while the gate is off this hook writes nothing, so a state can be stale after the gate is switched on
  again. The gate therefore ignores a "no explicit request" state written before the toggle file's ``updated`` time
  (any toggle change), and keeps the search until the next prompt writes a new state.
- Concurrency: each prompt's read-decide-write runs under a per-session ``fcntl.flock`` (``<hash>.lock``, waited at
  most 1 s), so a concurrent prompt without a signal cannot overwrite a flag it did not see; each write is one atomic
  replace, so the gate (which reads without the lock) sees the old or the new state, never a partial one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import unicodedata

from . import plugin_names as N
from . import query_presets as Q

LOG_TAG = f'{N.PLUGIN_NAME}-user-prompt-hook'
STATE_DIRNAME = 'prompt-state'
PROMPT_STATE_TTL_SECONDS = 1800.0      # 30 min: older state counts as absent (the gate then keeps the search)
FUTURE_SKEW_SECONDS = 300.0            # a time this far in the future is treated as unreadable
PRUNE_AGE_SECONDS = 86_400.0
PRUNE_LIMIT = 200
MAX_STATE_BYTES = 1024


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def explicit_request(prompt):
    """True if the prompt matches any explicit-search pattern, or is too long to scan (conservative)."""
    if len(prompt) > Q.EXPLICIT_SCAN_MAX_CHARS:
        return True
    t = normalize(prompt)
    return any(p.search(t) for p in Q.EXPLICIT_SEARCH_PATTERNS.values())


def starts_new_task(prompt):
    """True only if the normalized prompt begins with a fixed new-task marker (``NEW_TASK_PREFIXES``)."""
    t = normalize(prompt)
    return any(t.startswith(p) for p in Q.NEW_TASK_PREFIXES)


def _h(s, n=None):
    d = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return d if n is None else d[:n]


def default_state_dir(env):
    data = env.get('CLAUDE_PLUGIN_DATA')
    if isinstance(data, str) and data.strip():
        return os.path.join(data, STATE_DIRNAME)
    uid = os.getuid() if hasattr(os, 'getuid') else 'user'
    return os.path.join(tempfile.gettempdir(), f'{N.PLUGIN_NAME}-{STATE_DIRNAME}-{uid}')


class PromptState:
    """Per-session explicit-request state. ``read`` never raises; ``record`` / ``on_prompt`` never raise."""

    def __init__(self, root, ttl=PROMPT_STATE_TTL_SECONDS, now=time.time):
        self.root, self.ttl, self.now = root, ttl, now

    def path(self, session):
        return os.path.join(self.root, _h(session) + '.json')

    def _load(self, session):
        """-> (status, data). status: 'absent' | 'error' | 'ok'."""
        try:
            with open(self.path(session), 'rb') as f:
                blob = f.read(MAX_STATE_BYTES + 1)
        except FileNotFoundError:
            return 'absent', None
        except OSError:
            return 'error', None
        try:
            if len(blob) > MAX_STATE_BYTES:
                raise ValueError('too large')
            d = json.loads(blob.decode('utf-8'))
            if not (isinstance(d, dict) and d.get('v') == 1 and isinstance(d.get('explicit'), bool)
                    and isinstance(d.get('t'), (int, float)) and not isinstance(d.get('t'), bool)
                    and isinstance(d.get('s'), str) and (d.get('p') is None or isinstance(d.get('p'), str))):
                raise ValueError('schema')
        except (ValueError, UnicodeDecodeError):
            return 'error', None
        return 'ok', d

    def read(self, session, prompt_id=None, not_before=None):
        """For the gate. -> 'explicit' | 'not_explicit' (the gate may run) | 'absent' | 'error' | 'mismatch' |
        'expired' | 'prompt_mismatch' | 'predates_toggle'. Everything except 'not_explicit' means: keep the search,
        no endpoint call. ``not_before``: the toggle file's ``updated`` time; a state written before the last toggle
        change may come from a time the hook was off (it writes nothing then), so it is not trusted."""
        status, d = self._load(session)
        if status != 'ok':
            return status
        if d['s'] != _h(session, 32):
            return 'mismatch'
        age = self.now() - d['t']
        if age < -FUTURE_SKEW_SECONDS:
            return 'error'
        if age > self.ttl:
            return 'expired'
        if d['explicit']:
            return 'explicit'
        if isinstance(not_before, (int, float)) and d['t'] < not_before:
            return 'predates_toggle'
        if isinstance(prompt_id, str) and prompt_id and d['p'] is not None and d['p'] != _h(prompt_id, 32):
            return 'prompt_mismatch'
        return 'not_explicit'

    def record(self, session, explicit, prompt_id=None):
        """Atomic write. -> True, or False on failure (the old file is then removed if possible)."""
        data = {'v': 1, 't': self.now(), 'explicit': bool(explicit), 's': _h(session, 32),
                'p': _h(prompt_id, 32) if isinstance(prompt_id, str) and prompt_id else None}
        tmp = None
        try:
            os.makedirs(self.root, mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix='.tmp-', suffix='.json')
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, self.path(session))
            tmp = None
        except OSError:
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            try:
                os.remove(self.path(session))
            except OSError:
                pass
            return False
        self.prune(skip=self.path(session))
        return True

    LOCK_WAIT_SECONDS = 1.0

    def _lock(self, session):
        """-> fd holding an exclusive flock on the session's lock file, or None (no fcntl, timeout, I/O error)."""
        try:
            import fcntl
            os.makedirs(self.root, mode=0o700, exist_ok=True)
            fd = os.open(self.path(session)[:-5] + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
        except (ImportError, OSError):
            return None
        deadline = time.monotonic() + self.LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(0.005)
            except OSError:
                os.close(fd)
                return None

    def on_prompt(self, session, prompt, prompt_id=None):
        """Apply one new user prompt. -> (status, signal). status: 'explicit_set' | 'follow_up_kept' | 'cleared' |
        'write_failed' | 'lock_failed'."""
        signal = explicit_request(prompt)
        fd = self._lock(session)
        if fd is None:                      # cannot decide safely: protect (no fresh "not explicit" state remains)
            if not (signal and self.record(session, True, prompt_id)):
                self.forget(session)
            return 'lock_failed', signal
        try:
            if not signal and self.read(session) == 'explicit' and not starts_new_task(prompt):
                return 'follow_up_kept', False
            ok = self.record(session, signal, prompt_id)
        finally:
            os.close(fd)
        if not ok:
            return 'write_failed', signal
        return ('explicit_set' if signal else 'cleared'), signal

    def forget(self, session):
        """Remove the session's state (the gate then keeps every search). Best effort."""
        try:
            os.remove(self.path(session))
        except OSError:
            pass

    def prune(self, skip=None):
        """Best effort: remove session files and temp files older than a day."""
        cutoff = self.now() - PRUNE_AGE_SECONDS
        try:
            with os.scandir(self.root) as it:
                names = [e.path for n, e in zip(range(PRUNE_LIMIT), it)]
        except OSError:
            return
        for p in names:
            if p == skip or not p.endswith('.json'):
                continue
            try:
                if os.stat(p).st_mtime < cutoff:
                    os.remove(p)
                    if not os.path.basename(p).startswith('.tmp-'):
                        os.remove(p[:-5] + '.lock')      # the session's lock file goes with its day-old state
            except OSError:
                pass


# --- entry point ---------------------------------------------------------------------------------------------------
def _emit(rec, env, stderr):
    line = json.dumps(rec, sort_keys=True)
    try:
        print(line, file=stderr)
    except Exception:  # noqa: BLE001
        pass
    for p in ('CLAUDE_PLUGIN_OPTION_', N.ENV_PREFIX):
        path = env.get(p + 'LOG_FILE')
        if isinstance(path, str) and path.strip():
            try:
                with open(os.path.expanduser(path.strip()), 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
            except OSError:
                pass
            break


def main(stdin=None, stderr=None, env=None, now=time.time, state_dir=None):
    """Reads one UserPromptSubmit input. Writes NOTHING to stdout. Logs status / signal / prompt byte count only.
    Always returns 0. The caller (launcher) has already checked that the search gate is effectively on."""
    stdin, stderr = stdin or sys.stdin, stderr or sys.stderr
    env = os.environ if env is None else env
    rec = {'event': LOG_TAG}
    try:
        payload = json.loads(stdin.read())
        if not isinstance(payload, dict):
            raise ValueError('not an object')
    except Exception as e:  # noqa: BLE001
        rec.update(status='bad_input', detail=type(e).__name__)
        _emit(rec, env, stderr)
        return 0
    try:
        if payload.get('hook_event_name', 'UserPromptSubmit') != 'UserPromptSubmit':
            rec['status'] = 'not_user_prompt_submit'
        elif not isinstance(payload.get('session_id'), str) or not payload['session_id'].strip():
            rec['status'] = 'no_session'          # the gate finds no state for its session and keeps searches
        elif not isinstance(payload.get('prompt'), str):
            PromptState(state_dir or default_state_dir(env), now=now).forget(payload['session_id'])
            rec['status'] = 'no_prompt'           # state removed: the gate keeps this prompt's searches
        else:
            prompt = payload['prompt']
            st = PromptState(state_dir or default_state_dir(env), now=now)
            status, signal = st.on_prompt(payload['session_id'], prompt, payload.get('prompt_id'))
            rec.update(status=status, signal=signal, prompt_bytes=len(prompt.encode('utf-8')))
    except Exception as e:  # noqa: BLE001 - never block or disturb the prompt
        rec.update(status='error', detail=type(e).__name__)
        if isinstance(payload.get('session_id'), str) and payload['session_id'].strip():
            try:
                PromptState(state_dir or default_state_dir(env), now=now).forget(payload['session_id'])
            except Exception:  # noqa: BLE001
                pass
    try:
        from . import toggles as T
        rec['data_dir_id'] = T.data_dir_id(T.data_dir(env))
    except Exception:  # noqa: BLE001
        pass
    _emit(rec, env, stderr)
    return 0
