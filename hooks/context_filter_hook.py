#!/usr/bin/env python3
# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Plugin launcher for ``judge_adapter.plugin_hook`` (PostToolUse). Standard library only.

hooks/hooks.json runs it in exec form: ``python3 ${CLAUDE_PLUGIN_ROOT}/hooks/context_filter_hook.py``. It puts the
plugin's own ``src`` first on ``sys.path`` (so the bundled package is used, not another installed copy) and runs the
hook. The only third-party dependency is httpx:

- if ``python3`` on PATH can import httpx, it is used;
- otherwise, if ``${CLAUDE_PLUGIN_DATA}/venv`` exists, the launcher re-runs itself with that venv's Python;
- otherwise it logs ``dependency_missing`` to stderr and exits 0 with no output, so the search result is unchanged.

Fast path: when the toggle file ``${CLAUDE_PLUGIN_DATA}/toggles.json`` sets ``"filter": false`` (``/davi:filter off``),
it exits 0 at once, before reading stdin or importing the package: no output, no log, no network, no state. Reading
that small file (``judge_adapter/toggles.py``, loaded on its own) is the only work. A missing or corrupt file changes
nothing; the hook itself checks the same toggle again.

The launcher never installs anything and never touches the network.
"""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
REEXEC_GUARD = '_JUDGE_ADAPTER_HOOK_REEXEC'


def _names():
    """Load plugin_names.py directly (importing the package would need httpx)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_plugin_names', os.path.join(SRC, 'judge_adapter',
                                                                                'plugin_names.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _venv_python():
    data = os.environ.get('CLAUDE_PLUGIN_DATA')
    if not data:
        return None
    for rel in (('venv', 'bin', 'python'), ('venv', 'Scripts', 'python.exe')):
        p = os.path.join(data, *rel)
        if os.path.isfile(p):
            return p
    return None


def filter_toggled_off():
    """True only if the toggle file says "filter": false. Any problem -> False (the normal path runs)."""
    try:
        import types
        if '_davi_stub' not in sys.modules:
            stub = types.ModuleType('_davi_stub')
            stub.__path__ = [os.path.join(SRC, 'judge_adapter')]
            sys.modules['_davi_stub'] = stub
        t = importlib.import_module('_davi_stub.toggles')
        return t.read(t.toggle_path(os.environ))[0].get('filter') is False
    except Exception:  # noqa: BLE001
        return False


def main():
    import importlib.util
    import json
    if importlib.util.find_spec('httpx') is None:
        py = _venv_python()
        if py and not os.environ.get(REEXEC_GUARD):
            os.environ[REEXEC_GUARD] = '1'
            os.execv(py, [py, os.path.abspath(__file__)])
        try:
            tag = _names().LOG_TAG
        except Exception:  # noqa: BLE001
            tag = 'context-filter-hook'
        sys.stderr.write(json.dumps({'event': tag, 'status': 'dependency_missing',
                                     'detail': 'httpx is not importable; run `python3 -m pip install "httpx>=0.24,<1"` '
                                               'or create ${CLAUDE_PLUGIN_DATA}/venv with httpx (see README)'}) + '\n')
        return 0
    sys.path.insert(0, SRC)
    from judge_adapter.plugin_hook import main as hook_main
    return hook_main()


if __name__ == '__main__':
    if filter_toggled_off():
        sys.exit(0)
    try:
        main()
    except Exception:  # noqa: BLE001 - fail-open: never break the tool result
        pass
    sys.exit(0)
