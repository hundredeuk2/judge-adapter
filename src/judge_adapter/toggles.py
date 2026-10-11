# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""On/off toggles for the davi plugin's three features, kept in one small user-local file. Standard library only.

The file ``toggles.json`` lives in the plugin's data directory (``${CLAUDE_PLUGIN_DATA}``; Claude Code resolves it to
``~/.claude/plugins/data/<id>/``, ``<id>`` = ``davi@judge-adapter`` with every character other than a letter, digit,
``_`` or ``-`` replaced by ``-``, see https://code.claude.com/docs/en/plugins/components "Reference plugin paths and
store data"). Without ``CLAUDE_PLUGIN_DATA`` the documented default ``~/.claude/plugins/data/davi-judge-adapter`` is
used. One file per plugin data directory: the toggles are **install-wide**, shared by every session (and every
project) that uses this data directory, including sessions that are already running (from their next hook call).

Content (every key optional; a missing key means "use the userConfig value")::

    {"v": 1, "filter": true|false, "gate": true|false, "domain": true|false,
     "domain_mode": "ask"|"input_only", "updated": <unix time>}

Precedence, per feature, the same in every hook and in ``status``:

1. userConfig ``enabled=false`` (the global switch) turns ALL three features off; a toggle never overrides it.
2. Otherwise the toggle file, when it sets the key.
3. Otherwise userConfig / environment (``CLAUDE_PLUGIN_OPTION_<KEY>``, then ``JUDGE_ADAPTER_HOOK_<KEY>``):
   filter -> on (it has no own userConfig flag besides ``enabled``), gate -> ``search_gate_enabled``, domain ->
   ``domain_presets_enabled``, domain_mode -> ``domain_apply_mode``.

The three features are independent: the filter toggle affects only the PostToolUse filter, the gate toggle only the
search gate (and the UserPromptSubmit hook that serves it), the domain toggle only the domain presets.

A missing file means "no toggles". A file that cannot be read or does not match the schema (bad JSON, unknown key,
wrong type, over 4 KiB) is ignored by the hooks (userConfig values apply) and reported by ``status``; the CLI refuses
to modify it until ``reset`` removes it. Keys, the endpoint ID and the rate are never stored here (userConfig only).

Writes: read-modify-write under an exclusive ``fcntl.flock`` on ``toggles.json.lock`` (so concurrent changes are not
lost), then an atomic ``os.replace`` of a 0600 temp file in the same directory. No ``fcntl`` (Windows): writes are
refused with an error; reads still work.
"""
from __future__ import annotations

import os
import time

from . import plugin_names as N

FILE_NAME = 'toggles.json'
LOCK_SUFFIX = '.lock'
MAX_BYTES = 4096
FEATURES = ('filter', 'gate', 'domain')
MODES = ('ask', 'input_only')
ALLOWED_KEYS = frozenset(FEATURES + ('domain_mode', 'v', 'updated'))
LOCK_WAIT_SECONDS = 5.0
OPTION_PREFIXES = ('CLAUDE_PLUGIN_OPTION_', N.ENV_PREFIX)
_TRUE, _FALSE = ('1', 'true', 'yes', 'on'), ('0', 'false', 'no', 'off')
# userConfig flag per feature (None: the feature has no own flag; on unless the global switch is off).
USERCONFIG_FLAG = {'filter': None, 'gate': 'search_gate_enabled', 'domain': 'domain_presets_enabled'}


class ToggleError(Exception):
    """A toggle file problem the CLI must report as a failure (never as success)."""


def default_data_dir(home=None):
    pid = ''.join(c if c.isalnum() or c in '_-' else '-' for c in N.INSTALL_ID)
    return os.path.join(home or os.path.expanduser('~'), '.claude', 'plugins', 'data', pid)


def data_dir(env, override=None):
    """``override`` (CLI ``--data-dir``) if non-empty, else ``CLAUDE_PLUGIN_DATA``, else the documented default."""
    for v in (override, env.get('CLAUDE_PLUGIN_DATA')):
        if isinstance(v, str) and v.strip():
            return os.path.expanduser(v.strip())
    return default_data_dir()


def data_dir_id(path):
    """Secret-free identifier of a data directory (first 12 hex digits of the SHA-256 of its resolved path), logged by
    the hooks and printed by the CLI so a smoke can check that both use the same directory."""
    import hashlib
    return hashlib.sha256(os.path.realpath(path).encode('utf-8')).hexdigest()[:12]


def toggle_path(env, override=None):
    return os.path.join(data_dir(env, override), FILE_NAME)


def validate(data):
    """-> the dict if it matches the schema, else raises ToggleError (message without file content)."""
    if not isinstance(data, dict):
        raise ToggleError('toggle file is not a JSON object')
    unknown = set(data) - ALLOWED_KEYS
    if unknown:
        raise ToggleError(f'toggle file has {len(unknown)} unknown key(s)')
    if data.get('v', 1) != 1:
        raise ToggleError('toggle file has an unsupported version')
    for k in FEATURES:
        if k in data and not isinstance(data[k], bool):
            raise ToggleError(f'toggle "{k}" must be true or false')
    if 'domain_mode' in data and data['domain_mode'] not in MODES:
        raise ToggleError(f'toggle "domain_mode" must be one of {MODES}')
    if 'updated' in data and (isinstance(data['updated'], bool) or not isinstance(data['updated'], (int, float))):
        raise ToggleError('toggle "updated" must be a number')
    return data


def _parse_flat(text):
    """The flat form ``json.dump`` writes here (``{"key": true|false|"word"|number, ...}``) without importing json
    (its import costs a few ms in every launcher start). Anything else -> None (the caller then uses json)."""
    t = text.strip()
    if len(t) < 2 or t[0] != '{' or t[-1] != '}':
        return None
    body, out = t[1:-1].strip(), {}
    if not body:
        return out
    for item in body.split(','):
        k, sep, v = item.partition(':')
        k, v = k.strip(), v.strip()
        if not sep or len(k) < 3 or k[0] != '"' or k[-1] != '"' or not k[1:-1].replace('_', '').isalnum():
            return None
        if v in ('true', 'false'):
            val = v == 'true'
        elif len(v) >= 2 and v[0] == '"' and v[-1] == '"' and v[1:-1].replace('_', '').isalnum():
            val = v[1:-1]
        else:
            try:
                val = float(v) if any(c in v for c in '.eE') else int(v)
            except ValueError:
                return None
            if val != val or val in (float('inf'), float('-inf')):
                return None
        out[k[1:-1]] = val
    return out


def read(path):
    """-> (toggles dict, status): ('absent' | 'ok' | 'corrupt' | 'unreadable'). Never raises; dict is {} unless ok."""
    try:
        with open(path, 'rb') as f:
            blob = f.read(MAX_BYTES + 1)
    except FileNotFoundError:
        return {}, 'absent'
    except OSError:
        return {}, 'unreadable'
    if len(blob) > MAX_BYTES:
        return {}, 'corrupt'
    try:
        text = blob.decode('utf-8')
        data = _parse_flat(text)
        if data is None:
            import json             # only for files outside the flat form this module writes
            data = json.loads(text)
        return validate(data), 'ok'
    except (ValueError, UnicodeDecodeError, ToggleError):
        return {}, 'corrupt'


# --- effective settings ----------------------------------------------------------------------------------------------
def option(env, key):
    """userConfig option as the hooks see it: CLAUDE_PLUGIN_OPTION_<KEY>, then JUDGE_ADAPTER_HOOK_<KEY>."""
    for p in OPTION_PREFIXES:
        v = env.get(p + key.upper())
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def flag_value(v):
    """'true' / 'false' / None (unset) / 'invalid' -- lenient: the hooks report invalid values themselves."""
    if v is None:
        return None
    s = str(v).strip().lower()
    return 'true' if s in _TRUE else 'false' if s in _FALSE else 'invalid'


class Effective:
    """enabled (global userConfig switch; an invalid value counts as on here, the hooks report it), filter, gate,
    domain (bool), domain_mode (str), sources (feature -> 'toggle' | 'userConfig' | 'default'), notes (conflicts to
    show in status; never secret values), toggle_status. A plain class: the launchers' fast path imports this module,
    so it avoids heavier imports (dataclasses, tempfile)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def effective(config, toggles, toggle_status='ok'):
    """``config``: callable key -> raw userConfig value or None. ``toggles``: validated dict."""
    enabled = flag_value(config('enabled')) != 'false'
    eff = {'enabled': enabled, 'sources': {}, 'notes': [], 'toggle_status': toggle_status}
    for feat in FEATURES:
        key = USERCONFIG_FLAG[feat]
        cfg = flag_value(config(key)) if key else None
        base = cfg == 'true' if key else True
        if feat in toggles:
            val, src = toggles[feat], 'toggle'
            if key and cfg is not None and (cfg == 'true') != val:
                eff['notes'].append(f'{feat}: the toggle ({"on" if val else "off"}) overrides userConfig {key}={cfg}')
        else:
            val, src = base, ('userConfig' if cfg is not None else 'default')
        if not enabled and val:
            eff['notes'].append(f'{feat}: {"toggle" if src == "toggle" else "setting"} is on, but userConfig '
                                'enabled=false turns every feature off')
        eff[feat] = enabled and val
        eff['sources'][feat] = src
    mode_cfg = config('domain_apply_mode')
    if 'domain_mode' in toggles:
        eff['domain_mode'], eff['sources']['domain_mode'] = toggles['domain_mode'], 'toggle'
        if mode_cfg is not None and mode_cfg.strip().lower() != toggles['domain_mode']:
            eff['notes'].append(f'domain_mode: the toggle ({toggles["domain_mode"]}) overrides userConfig '
                                f'domain_apply_mode={mode_cfg.strip().lower()}')
    else:
        eff['domain_mode'] = (mode_cfg or 'ask').strip().lower()
        eff['sources']['domain_mode'] = 'userConfig' if mode_cfg is not None else 'default'
    return Effective(**eff)


def effective_from_env(env, path=None):
    """What the hooks use: the toggle file at ``path`` (default: from ``env``) over the env userConfig values."""
    toggles, status = read(path or toggle_path(env))
    return effective(lambda k: option(env, k), toggles, status)


# --- writes ---------------------------------------------------------------------------------------------------------
def _lock(path):
    try:
        import fcntl
    except ImportError:
        raise ToggleError('file locking (fcntl) is not available on this platform; toggles cannot be changed '
                          'safely here') from None
    fd = os.open(path + LOCK_SUFFIX, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise ToggleError('the toggle file is locked by another process; try again') from None
            time.sleep(0.01)


def _atomic_write(path, data):
    import json
    import tempfile
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.toggles-', suffix='.tmp')
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def update(path, changes, now=time.time):
    """Read-modify-write under the lock. ``changes``: {feature: bool, 'domain_mode': str}. -> the new dict.
    Raises ToggleError (bad value, corrupt / unreadable existing file, lock or write failure)."""
    validate(dict(changes))
    if not changes:
        raise ToggleError('nothing to change')
    try:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    except OSError as e:
        raise ToggleError(f'cannot create the data directory ({type(e).__name__})') from None
    try:
        fd = _lock(path)
    except OSError as e:
        raise ToggleError(f'cannot lock the toggle file ({type(e).__name__})') from None
    try:
        current, status = read(path)
        if status in ('corrupt', 'unreadable'):
            raise ToggleError(f'the existing toggle file is {status}; nothing was changed. Run "reset" to remove it '
                              '(the userConfig values then apply)')
        new = {**current, **changes, 'v': 1, 'updated': now()}
        validate(new)
        try:
            _atomic_write(path, new)
        except OSError as e:
            raise ToggleError(f'writing the toggle file failed ({type(e).__name__}); nothing was changed') from None
        check, status = read(path)
        if status != 'ok' or any(check.get(k) != v for k, v in changes.items()):
            raise ToggleError('the toggle file could not be read back after writing')
        return check
    finally:
        os.close(fd)


def reset(path):
    """Remove the toggle file under the lock. -> True if a file was removed, False if there was none."""
    if not os.path.isdir(os.path.dirname(path)):
        return False
    try:
        fd = _lock(path)
    except OSError as e:
        raise ToggleError(f'cannot lock the toggle file ({type(e).__name__})') from None
    try:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            raise ToggleError(f'removing the toggle file failed ({type(e).__name__})') from None
    finally:
        os.close(fd)
