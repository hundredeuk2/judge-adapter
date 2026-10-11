#!/usr/bin/env python3
# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Plugin launcher for ``judge_adapter.prompt_signal`` (UserPromptSubmit; serves the optional search gate). Stdlib only.

hooks/hooks.json runs it in exec form on every user prompt (UserPromptSubmit has no matcher). It needs no httpx: the
modules it uses are loaded under a private stub package, without the package ``__init__``.

Fast path: unless the search gate is effectively on (userConfig ``enabled`` not false, and the toggle file's
``gate`` -- or, when unset, ``search_gate_enabled`` -- true), it exits 0 at once, before reading stdin: the prompt is
not read, no key is read, no state is written, no model is called, nothing is printed. The only work is reading the
option flags from the environment and the small toggle file ``${CLAUDE_PLUGIN_DATA}/toggles.json``.

When the gate is on, it reads the hook input, checks the prompt locally for an explicit search request and updates
the session's prompt state (see ``prompt_signal``). It NEVER writes to stdout: for UserPromptSubmit, "Plain-text
stdout is added as context for Claude" (https://code.claude.com/docs/en/hooks). It always exits 0, so it never blocks
a prompt.
"""
import importlib
import os
import sys
import types

SRC_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'judge_adapter')


def _load(name):
    if '_davi_stub' not in sys.modules:
        stub = types.ModuleType('_davi_stub')
        stub.__path__ = [SRC_PKG]
        sys.modules['_davi_stub'] = stub
    return importlib.import_module('_davi_stub.' + name)


def gate_on():
    try:
        return bool(_load('toggles').effective_from_env(os.environ).gate)
    except Exception:  # noqa: BLE001 - cannot tell: do nothing (the gate then finds no fresh state and keeps)
        return False


def main():
    return _load('prompt_signal').main()


if __name__ == '__main__':
    if not gate_on():
        sys.exit(0)
    try:
        main()
    except Exception:  # noqa: BLE001 - never block or disturb the prompt
        pass
    sys.exit(0)
