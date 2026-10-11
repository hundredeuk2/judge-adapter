#!/usr/bin/env python3
# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""davi on/off control: show the effective settings and switch the three features. Standard library only.

Usage (also run by the plugin's /davi:* commands through ``bin/davi-toggle``)::

    python3 hooks/davi_toggle.py [--data-dir DIR] [--settings FILE] status
    python3 hooks/davi_toggle.py [--data-dir DIR] filter|gate|domain on|off
    python3 hooks/davi_toggle.py [--data-dir DIR] domain-mode ask|input_only
    python3 hooks/davi_toggle.py [--data-dir DIR] reset

It edits ONLY ``<data dir>/toggles.json`` (see ``judge_adapter/toggles.py``): ``--data-dir`` if given and non-empty,
else ``CLAUDE_PLUGIN_DATA``, else ``~/.claude/plugins/data/davi-judge-adapter``. The file is install-wide: a change
applies to every session that uses this data directory, including running ones, from their next hook call. Keys, the
endpoint ID and the rate are userConfig-only and never written here.

``status`` reads, besides the toggle file: the userConfig values the environment carries (``CLAUDE_PLUGIN_OPTION_*``,
``JUDGE_ADAPTER_HOOK_*``; hooks get these, the Bash tool does not), then the NON-sensitive values Claude Code saved
under ``pluginConfigs["davi@judge-adapter"].options`` in the user settings file (``$CLAUDE_CONFIG_DIR/settings.json``
or ``~/.claude/settings.json``; only that block is used and only the plugin's non-sensitive option names), then the
manifest defaults. Sensitive values (key, endpoint ID) live in secure storage and reach only hook processes: status
reports them as "set" / "missing" when the environment shows them, else as "not visible here". It never prints a key,
the endpoint ID, a domain list or a presets file path.

Arguments are validated strictly (exact lower-case words). Exit codes: 0 done, 2 usage error (nothing changed), 1 the
change could not be made (nothing changed; the message says why). A failure is never reported as success.
"""
import importlib
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PKG = os.path.join(ROOT, 'src', 'judge_adapter')
FEATURE_WORDS = {'filter': 'filter', 'gate': 'gate', 'domain': 'domain'}
LABEL = {'filter': 'filter (PostToolUse, Tavily results)', 'gate': 'search gate (PreToolUse + UserPromptSubmit)',
         'domain': 'domain presets (PreToolUse)'}
SCOPE = ('install-wide: shared by every session and project that uses this plugin data directory, including '
         'sessions already running (from their next hook call)')
USAGE = ('usage: davi_toggle.py [--data-dir DIR] [--settings FILE] status\n'
         '       davi_toggle.py [--data-dir DIR] filter|gate|domain on|off\n'
         '       davi_toggle.py [--data-dir DIR] domain-mode ask|input_only\n'
         '       davi_toggle.py [--data-dir DIR] reset')


def _load(name):
    if '_davi_stub' not in sys.modules:
        stub = types.ModuleType('_davi_stub')
        stub.__path__ = [SRC_PKG]
        sys.modules['_davi_stub'] = stub
    return importlib.import_module('_davi_stub.' + name)


class UsageError(Exception):
    pass


def parse_args(argv):
    """-> (opts, command, value). Raises UsageError."""
    opts, rest, i = {'data_dir': None, 'settings': None}, [], 0
    while i < len(argv):
        a = argv[i]
        if a in ('--data-dir', '--settings'):
            if i + 1 >= len(argv):
                raise UsageError(f'{a} needs a value')
            opts[a[2:].replace('-', '_')] = argv[i + 1]
            i += 2
            continue
        if a.startswith('-'):
            raise UsageError(f'unknown option {a!r}')
        rest.append(a)
        i += 1
    if not rest:
        raise UsageError('missing command')
    cmd, args = rest[0], rest[1:]
    if cmd in ('status', 'reset'):
        if args:
            raise UsageError(f'"{cmd}" takes no argument')
        return opts, cmd, None
    if cmd in FEATURE_WORDS:
        if len(args) != 1 or args[0] not in ('on', 'off'):
            raise UsageError(f'"{cmd}" needs exactly one argument: on or off')
        return opts, cmd, args[0] == 'on'
    if cmd == 'domain-mode':
        if len(args) != 1 or args[0] not in ('ask', 'input_only'):
            raise UsageError('"domain-mode" needs exactly one argument: ask or input_only')
        return opts, cmd, args[0]
    raise UsageError(f'unknown command {cmd!r}')


# --- userConfig view for status --------------------------------------------------------------------------------------
def manifest():
    try:
        with open(os.path.join(ROOT, '.claude-plugin', 'plugin.json'), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def settings_path(override, env):
    if override:
        return os.path.expanduser(override)
    base = env.get('CLAUDE_CONFIG_DIR')
    base = os.path.expanduser(base) if isinstance(base, str) and base.strip() else os.path.expanduser('~/.claude')
    return os.path.join(base, 'settings.json')


def saved_options(path, install_id, allowed):
    """Non-sensitive saved option values from ``pluginConfigs[install_id].options`` -> ({key: str}, note)."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}, 'user settings file not found'
    except (OSError, ValueError):
        return {}, 'user settings file could not be read'
    block = (((data.get('pluginConfigs') or {}) if isinstance(data, dict) else {}).get(install_id) or {})
    opts = block.get('options') if isinstance(block, dict) else None
    out = {}
    for k, v in (opts or {}).items() if isinstance(opts, dict) else ():
        if k in allowed and isinstance(v, (str, int, float, bool)):
            out[k] = (('true' if v else 'false') if isinstance(v, bool) else str(v)).strip()
    return {k: v for k, v in out.items() if v}, None


class Config:
    """Explicit userConfig values: environment first (as hooks see them), then saved settings. None = not set."""

    def __init__(self, env, saved):
        self.env, self.saved, self.T = env, saved, _load('toggles')

    def __call__(self, key):
        v = self.T.option(self.env, key)
        return v if v is not None else self.saved.get(key)

    def source(self, key):
        return 'env' if self.T.option(self.env, key) is not None else 'settings' if key in self.saved else 'default'


def _shown(v):
    return ('on' if v else 'off') if isinstance(v, bool) else str(v)


def _onoff(b):
    return 'on ' if b else 'off'


def status(opts, env, out):
    T, H, QH = _load('toggles'), _load('plugin_hook'), _load('query_hook')
    N = _load('plugin_names')
    man = manifest()
    ucfg = man.get('userConfig') or {}
    sensitive = {k for k, v in ucfg.items() if isinstance(v, dict) and v.get('sensitive')}
    spath = settings_path(opts['settings'], env)
    saved, snote = saved_options(spath, N.INSTALL_ID, set(ucfg) - sensitive)
    cfg = Config(env, saved)
    tpath = T.toggle_path(env, opts['data_dir'])
    toggles, tstatus = T.read(tpath)
    eff = T.effective(cfg, toggles, tstatus)
    warn, lines = [], []
    w = lines.append

    w(f'davi status (plugin {N.INSTALL_ID})')
    w(f'toggle file: {tpath} [{tstatus}]')
    w(f'data dir id: {T.data_dir_id(os.path.dirname(tpath))} (the hooks log the same data_dir_id)')
    w(f'  scope: {SCOPE}')
    if tstatus in ('corrupt', 'unreadable'):
        warn.append(f'the toggle file is {tstatus}: the hooks ignore it and use the userConfig values; run '
                    '"reset" to remove it')
    w(f'userConfig values from: environment, then {spath}' + (f' ({snote})' if snote else '') + ', then defaults')
    w(f'global switch (userConfig enabled): {"on" if eff.enabled else "off"} [{cfg.source("enabled")}]')
    if T.flag_value(cfg('enabled')) == 'invalid':
        warn.append('userConfig enabled is not true/false: the hooks report config_error and change nothing')
    for feat in T.FEATURES:
        w(f'{feat:<7} {_onoff(getattr(eff, feat))}  [{eff.sources[feat]}]  {LABEL[feat]}')
    w(f'domain mode: {eff.domain_mode} [{eff.sources["domain_mode"]}]')
    for k, default, lo, hi in (('search_gate_threshold', 0.85, 0.5, 1.0), ('domain_min_confidence', 0.6, 0.0, 1.0)):
        try:
            w(f'{k}: {H._num(cfg(k), k, default, lo, hi)} [{cfg.source(k)}]')
        except Exception:  # noqa: BLE001
            w(f'{k}: invalid [{cfg.source(k)}]')
            warn.append(f'{k} is invalid: the query-stage hook reports config_error and changes nothing')
    w(f'filter preset: {cfg("preset") or H.DEFAULT_PRESET} [{cfg.source("preset")}]')
    try:
        pset = QH.parse_domain_presets(cfg('domain_presets'))
        names = [p.name for p in pset.presets]
        w(f'domain presets: {", ".join(names)} ({"built-in EXAMPLE presets" if pset.source == "examples" else "yours"})')
        if eff.domain and pset.source == 'examples':
            warn.append('domain presets are on with the built-in EXAMPLE presets (docs, academic); set domain_presets '
                        'to your own lists')
    except Exception as e:  # noqa: BLE001 - messages never include names, domains or paths
        w('domain presets: invalid')
        if eff.domain:
            warn.append(f'domain presets are on but domain_presets is invalid ({e}); the domain hook changes nothing')
    if eff.domain_mode not in ('ask', 'input_only'):
        warn.append('domain_apply_mode is invalid: the query-stage hook reports config_error')
    elif eff.domain and eff.domain_mode == 'ask':
        warn.append('domain mode "ask": every restricted search waits for your approval; in headless -p runs it can '
                    'block the search')

    visible = any(k.startswith(T.OPTION_PREFIXES) for k in env)
    for label, names in (('key (runpod_api_key)', H.KEY_ENVS), ('endpoint (choice_endpoint_id)', H.ENDPOINT_ENVS)):
        if H._present(env, names):
            w(f'{label}: set')
        elif visible:
            w(f'{label}: missing')
            warn.append(f'{label.split()[0]} missing: the hooks send nothing (fail-open)')
        else:
            w(f'{label}: not visible here (secure storage; passed only to hook processes; a hook log status '
              'missing_key / missing_endpoint means it is not set)')
    rate = cfg('usd_per_second')
    w(f'rate (usd_per_second): {"set" if rate else "missing"} [{cfg.source("usd_per_second")}]')
    any_on = eff.filter or eff.gate or eff.domain
    if any_on and not rate:
        warn.append('usd_per_second missing: the hooks send nothing (fail-open)')
    if any_on:
        import importlib.util
        venv = os.path.join(T.data_dir(env, opts['data_dir']), 'venv')
        if importlib.util.find_spec('httpx') is None and not os.path.isdir(venv):
            warn.append('httpx is not importable by this python3 and no venv exists in the data directory: the hooks '
                        'log dependency_missing and change nothing (see README, Install)')
    for n in eff.notes:
        w(f'note: {n}')
    for x in warn:
        w(f'warning: {x}')
    out.write('\n'.join(lines) + '\n')
    return 0


def change(opts, cmd, value, env, out):
    T = _load('toggles')
    path = T.toggle_path(env, opts['data_dir'])
    if cmd == 'reset':
        removed = T.reset(path)
        out.write(f'davi: {"removed" if removed else "no toggle file at"} {path}; the userConfig values apply.\n'
                  f'scope: {SCOPE}\n')
        return 0
    key = 'domain_mode' if cmd == 'domain-mode' else FEATURE_WORDS[cmd]
    new = T.update(path, {key: value})
    shown = value if key == 'domain_mode' else ('on' if value else 'off')
    out.write(f'davi: {key} set to {shown} in {path}\ndata dir id: {T.data_dir_id(os.path.dirname(path))}\n'
              f'scope: {SCOPE}\n')
    enabled = T.flag_value(T.option(env, 'enabled'))
    if enabled is None:
        man = manifest()
        allowed = {k for k, v in (man.get('userConfig') or {}).items() if isinstance(v, dict) and not v.get('sensitive')}
        saved, _ = saved_options(settings_path(opts['settings'], env), _load('plugin_names').INSTALL_ID, allowed)
        enabled = T.flag_value(saved.get('enabled'))
    if enabled == 'false' and key != 'domain_mode' and value:
        out.write('note: userConfig enabled=false turns every feature off; this toggle has no effect until enabled '
                  'is true\n')
    out.write('toggles now: ' + ', '.join(f'{k}={_shown(new[k])}' for k in ('filter', 'gate', 'domain', 'domain_mode')
                                         if k in new) + '\n')
    return 0


def main(argv=None, env=None, out=None, err=None):
    argv = sys.argv[1:] if argv is None else argv
    env = os.environ if env is None else env
    out, err = out or sys.stdout, err or sys.stderr
    try:
        opts, cmd, value = parse_args(argv)
    except UsageError as e:
        err.write(f'davi: {e}\n{USAGE}\n')
        return 2
    try:
        if cmd == 'status':
            return status(opts, env, out)
        return change(opts, cmd, value, env, out)
    except _load('toggles').ToggleError as e:
        err.write(f'davi: FAILED, nothing was changed: {e}\n')
        return 1
    except Exception as e:  # noqa: BLE001 - never report an unexpected failure as success
        err.write(f'davi: FAILED ({type(e).__name__}); nothing was reported as changed\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
