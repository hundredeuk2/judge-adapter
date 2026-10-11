# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Working names of the Claude Code plugin, kept in one place so a rename is a small, mechanical change.

A rename touches: this module, ``.claude-plugin/plugin.json`` (``name``), ``.claude-plugin/marketplace.json``
(``name`` and ``plugins[0].name``), the install commands in README.md, and ``tests/test_plugin_hook.py`` (which
checks that the manifests agree with this module). The hook code itself reads names only from here.

``CLAUDE_PLUGIN_OPTION_<KEY>`` comes from Claude Code and follows the ``userConfig`` keys, not the plugin name.
"""
PLUGIN_NAME = 'davi'                 # plugin.json "name"; install id is PLUGIN_NAME@MARKETPLACE_NAME
MARKETPLACE_NAME = 'judge-adapter'   # marketplace.json "name" (neutral working name: the repository name)
INSTALL_ID = f'{PLUGIN_NAME}@{MARKETPLACE_NAME}'

# Fallback environment variables for running the hook outside a plugin (e.g. from settings.json hooks).
ENV_PREFIX = 'JUDGE_ADAPTER_HOOK_'

# Text shown to the model in additionalContext and used as the log tag.
CONTEXT_LABEL = 'context filter'
LOG_TAG = f'{PLUGIN_NAME}-context-filter-hook'
