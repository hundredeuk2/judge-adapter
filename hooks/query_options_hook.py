#!/usr/bin/env python3
# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Plugin launcher for ``judge_adapter.query_hook`` (PreToolUse, optional query-stage options). Standard library only.

Fast path: unless the search gate or the domain presets are effectively on, it exits 0 at once, before reading stdin
or importing the package: no key read, no network, no state, no output, no log. To decide, it reads only the option
flags from the environment (``CLAUDE_PLUGIN_OPTION_<KEY>``, then ``JUDGE_ADAPTER_HOOK_<KEY>``) and the small toggle
file ``${CLAUDE_PLUGIN_DATA}/toggles.json`` (``judge_adapter/toggles.py``, loaded on its own without the package or
httpx): userConfig ``enabled=false`` -> off; else the toggle file's ``gate`` / ``domain`` when set; else
``search_gate_enabled`` / ``domain_presets_enabled``. Otherwise it behaves like ``context_filter_hook.py``: the plugin's own ``src`` first on
``sys.path``; httpx from ``python3`` or ``${CLAUDE_PLUGIN_DATA}/venv``; ``dependency_missing`` (no output) if neither.
It never installs anything.
"""
import os
import sys

_PREFIXES = ('CLAUDE_PLUGIN_OPTION_', 'JUDGE_ADAPTER_HOOK_')
_TRUE, _FALSE = ('1', 'true', 'yes', 'on'), ('0', 'false', 'no', 'off')
_SRC_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'judge_adapter')


def _opt(key):
    for p in _PREFIXES:
        v = os.environ.get(p + key)
        if v and v.strip():
            return v.strip().lower()
    return None


def _toggles():
    """``judge_adapter/toggles.py`` under a private stub package: no package ``__init__``, no httpx."""
    import importlib
    import types
    if '_davi_stub' not in sys.modules:
        stub = types.ModuleType('_davi_stub')
        stub.__path__ = [_SRC_PKG]
        sys.modules['_davi_stub'] = stub
    return importlib.import_module('_davi_stub.toggles')


def options_off():
    try:
        eff = _toggles().effective_from_env(os.environ)
        return not (eff.gate or eff.domain)
    except Exception:  # noqa: BLE001 - toggle module unusable: decide from the environment alone, as before
        if _opt('ENABLED') in _FALSE:
            return True
        return not (_opt('SEARCH_GATE_ENABLED') in _TRUE or _opt('DOMAIN_PRESETS_ENABLED') in _TRUE)


def main():
    import importlib.util
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'src')
    guard = '_JUDGE_ADAPTER_QUERY_HOOK_REEXEC'
    if importlib.util.find_spec('httpx') is None:
        data = os.environ.get('CLAUDE_PLUGIN_DATA')
        for rel in (('venv', 'bin', 'python'), ('venv', 'Scripts', 'python.exe')):
            py = os.path.join(data, *rel) if data else None
            if py and os.path.isfile(py) and not os.environ.get(guard):
                os.environ[guard] = '1'
                os.execv(py, [py, os.path.abspath(__file__)])
        try:
            spec = importlib.util.spec_from_file_location('_plugin_names', os.path.join(src, 'judge_adapter',
                                                                                        'plugin_names.py'))
            names = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(names)
            tag = names.PLUGIN_NAME + '-query-options-hook'
        except Exception:  # noqa: BLE001
            tag = 'query-options-hook'
        sys.stderr.write(json.dumps({'event': tag, 'status': 'dependency_missing',
                                     'detail': 'httpx is not importable (see README, Install)'}) + '\n')
        return 0
    sys.path.insert(0, src)
    from judge_adapter.query_hook import main as hook_main
    return hook_main()


if __name__ == '__main__':
    if options_off():
        sys.exit(0)
    try:
        main()
    except Exception:  # noqa: BLE001 - fail-open: the tool call goes through the normal permission flow
        pass
    sys.exit(0)
