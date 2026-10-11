---
description: Turn the davi search-result filter (PostToolUse, Tavily) off
allowed-tools: Bash(davi-toggle *)
disable-model-invocation: true
---
!`davi-toggle --data-dir "${CLAUDE_PLUGIN_DATA}" filter off`

Show the output above to the user exactly as printed, in one code block. Do not run any tool and add nothing else.
