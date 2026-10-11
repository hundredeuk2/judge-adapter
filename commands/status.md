---
description: Show the davi plugin's effective settings (filter, search gate, domain presets); never shows secrets
allowed-tools: Bash(davi-toggle *)
disable-model-invocation: true
---
!`davi-toggle --data-dir "${CLAUDE_PLUGIN_DATA}" status`

Show the output above to the user exactly as printed, in one code block. Do not run any tool and add nothing else.
