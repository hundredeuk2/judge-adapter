# judge-adapter

**Fewer input tokens for search agents: score each retrieved document with a choice model and pass on only the ones
worth reading.** Also a pairwise LLM-judge adapter with fixed options, retries 0 and run-wide limits.

[Installation](#installation) · [Quickstart](#quickstart) · [Results](#results) ·
[Claude Code plugin](#claude-code-plugin) · [How it works](#how-it-works) · [한국어 요약](#한국어-요약) ·
[GitHub](https://github.com/hundredeuk2/judge-adapter) · [License](#license)

| | Name |
|---|---|
| Distribution (pip) | `judge-adapter` |
| Import | `judge_adapter` |
| JSONL CLI | `judge-adapter` (also `python -m judge_adapter.cli`) |
| FastChat wrapper CLI (optional) | `judge-adapter-fastchat` (needs the `fastchat` extra) |
| Claude Code plugin | `davi` (filters Tavily MCP search results), marketplace `judge-adapter` |
| License | Apache-2.0 (see `LICENSE` and `NOTICE`) |

## Installation

Python 3.9 or newer. Install from GitHub (this package is not on PyPI):

```bash
# core: depends only on httpx (no torch, no FastChat)
pip install "judge-adapter @ git+https://github.com/hundredeuk2/judge-adapter"

# optional FastChat binding: pulls FastChat at the pinned commit plus torch / transformers etc. (heavy)
pip install "judge-adapter[fastchat] @ git+https://github.com/hundredeuk2/judge-adapter"
```

The `fastchat` extra pins `fschat @ git+https://github.com/lm-sys/FastChat@587d5cfa1609a43d192cedb8441cac3c17db105d`
(fschat 0.2.36), the only host revision the binding was written and tested against, and adds the packages that
`fastchat.llm_judge` imports at load time (`torch`, `transformers`, `accelerate`, `anthropic`, `openai`, `pandas`,
`numpy`, `tqdm`). The core never imports any of them.

Install check of the extra (2026-10): `pip install "judge_adapter-0.1.0-py3-none-any.whl[fastchat]"` in a fresh
Python 3.13 venv on macOS arm64 resolved fschat 0.2.36 from the pinned commit and torch 2.14.1, and the installed
`judge-adapter-fastchat` ran a fake reference-category flow end to end with synthetic data files. Other platforms and
Python versions were not tried for the extra (the core was also tested on Python 3.9).

From a source checkout: `pip install .` (or `pip install -e ".[test]"` for development).

Claude Code users can install the search-result filter as a plugin instead; see [Claude Code plugin](#claude-code-plugin).

## About

### Context filter: less context, same agent

A search tool usually hands every result to the model. `judge_adapter.context_filter` asks a choice model one
question per document ("does this document give a direct answer or needed intermediate evidence?"), reads the
probability of the *keep* option, and keeps only documents at or above a threshold. The agent then reads fewer
documents, so its **input tokens go down** (see [Results](#results) for one measured run).

- **Model:** the filter talks to a Jev-compatible choice endpoint that you serve, for example on RunPod serverless.
  Davi is a Jev-style choice model that scores P(keep) per document in this way. This package ships no model and no
  worker; endpoint, model and price come only from your configuration.
- **Output:** document ids and decisions, in input order. The filter never rewrites, shortens or stores text.
- **Fail-open:** a document whose call fails is kept, and the reason is recorded.
- **Explicit policy:** there is no default policy; you pass `threshold(t)`, `top_k(k)`, `disabled()` or a named
  preset such as `v1-thr0.3-dev`.
- **Where it plugs in:** a Python API for your own tool executor, and a [Claude Code plugin](#claude-code-plugin)
  whose PostToolUse hook filters Tavily MCP search results before the model reads them.

### Pairwise judge adapter

Give it one question (one or two turns) and two responses, and it returns `A`, `B`, `tie` or an explicit `error`,
with a trace of every call. It has a JSONL command-line tool, a Python API and an optional binding for FastChat
`llm_judge` (MT-Bench style pairwise judging).

### What it keeps fixed on purpose

- **Fake by default, real by opt-in.** `--transport` has no default. A real (paid) CLI run needs `--allow-real`,
  `--max-submissions` and `--budget-cap-usd`.
- **Retries 0, fallback 0.** One model submission per call. A failure is reported, never turned into a tie or
  retried on another backend.
- **Run-wide limits.** `--max-submissions` and a client-side cost cap for the whole run (not reset per request).
- **Unknown cost stays unknown.** A submitted call without usage/timing has cost `null`, not `0`.
- **Failures before transmission send nothing.** A missing key, missing choice endpoint ID or rate, or a client setup
  error is `local_config` with 0 submissions, and the run stops.
- **Keys are never written.** Stored traces, CLI output files, error details and logs go through a scrubber (exact
  key value, `Bearer ...`, `sk-...`, `rpa_...`). Provider responses are parsed from the raw body, never from the
  scrubbed copy (see [Raw responses and stored traces](#raw-responses-and-stored-traces)).
- **No silent overwrite.** An output directory with results is refused unless `--overwrite`.
- **Rubric / reference are recorded.** Every result carries the prompt version and the sha256 of the rubric and
  reference that were sent.

## Quickstart

No key and no network: everything below runs against in-process fake endpoints.

### Context filter (Python)

```python
from judge_adapter import AdapterConfig, build_backend, fakes
from judge_adapter.context_filter import ContextFilter, Document, select_originals, threshold

cfg = AdapterConfig(transport='fake', backend='choice_endpoint', trace_dir='unused')
fake = fakes.FakeRelevanceEndpoint()                    # in-process; [[KEEP=p]] in the text sets the fake P(keep)
backend = build_backend(cfg, transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY)
flt = ContextFilter(cfg, backend, policy=threshold(0.5))   # no default policy: pass one, or preset=...
docs = [Document('d1', 'A title', 'https://example.invalid/1', 'synthetic text [[KEEP=0.8]]'),
        Document('d2', 'A title', 'https://example.invalid/2', 'synthetic text [[KEEP=0.1]]')]
res = flt.filter('a question', docs)
res.selected_ids                                        # ['d1'], input order
[(r.id, r.score, r.status, r.decision) for r in res.documents]
kept = select_originals(docs, res)                      # your own objects, unchanged
```

For a real endpoint: `AdapterConfig(transport='real', backend='choice_endpoint', choice_endpoint_id=...,
choice_usd_per_second=..., choice_expected_model=...)` (expected model optional), `build_backend(cfg)`, and
`ContextFilter(..., budget_cap_usd=..., max_submissions=...)`; both limits are required for a real transport unless
the policy is `disabled()`. Endpoint, model and rate come only from your configuration.

A runnable integration example (a function-call tool executor over a mock search with synthetic documents, no SDK):
`python examples/context_filter_tool_executor.py`.

### Pairwise judge (CLI)

```bash
judge-adapter examples                                   # paths of the bundled synthetic examples
EX=$(judge-adapter examples | grep pairs.example.jsonl)
judge-adapter validate "$EX" --mode two_direction        # no call, no key read
judge-adapter run "$EX" --out ./run-fake --transport fake --mode two_direction
judge-adapter run "$EX" --out ./run-fake-choice --transport fake --backend choice_endpoint
```

A fake run prints a `[MOCK]` summary and writes `results.jsonl`, `summary.json`, `gate.json` and `trace/`. The fake
judge ignores answer quality; it follows test markers only (see [Fake transport](#fake-transport)).

## Results

Single runs outside this package's test suite; not guarantees. Setup: HLE (Humanity's Last Exam), 90 English
text-only questions, a GPT-5.2 agent with web search, 5 documents per search, preset `v1-thr0.3-dev`, compared with
keeping the top 3 documents and keeping all 5. Answers were graded by Haiku against the reference answers.

| | Filter (`v1-thr0.3-dev`) | Keep top 3 | Keep all 5 |
|---|---|---|---|
| Agent input tokens, filter vs. this setting (HLE, 86 questions where all three finished) | | **−31.4%** | **−44.9%** |
| Correct answers, HLE (90 English questions) | 28 | 27 | 31 |
| Correct answers, earlier run (25 Korean questions) | 14 | 10 | 12 |

- **Input tokens went down.** On the 86 questions where all three settings finished, agent input tokens were 31.4%
  lower than keeping the top 3 documents and 44.9% lower than keeping all 5. (Correction: an earlier version of this
  README said "summed over all 90 questions, about 44% lower" than top-3. That sum counted 4 interrupted trajectories
  as 0 tokens, 3 of them in the filter setting, which inflated the reduction; it is replaced by the 86-question
  figures above.)
- **Accuracy was not shown to change.** In both runs the filter answered at least as many questions as top-3 (+1/90,
  +4/25), but neither difference is statistically established. Against all 5 documents the results were mixed (−3/90,
  +2/25). The preset was selected on development data (see [Preset](#preset-v1-thr03-dev-opt-in)).
- **Total cost was not shown to go down.** Filter decisions are input-only; at a Jev-style list price of $0.042 per 1M
  input tokens (an assumption; this package does not serve that model) they cost about $0.0003 per question. Total
  cost depended on the generator's output: on the 86 HLE questions the filter setting produced 6.7% more output tokens
  and cost 4.5% more in total than top-3. No total-cost saving is established.

In the same HLE run, all 5 documents were dropped in 25 of the 72 searches the filter setting made: a per-search
count from a single run, not a general drop rate or failure rate. Total cost also depends on the price you configure
for filter calls (`choice_usd_per_second`) and on the model you actually serve behind the endpoint; the package ties
neither to the other, and its estimate is not a bill.

**Intended use:** an opt-in input-reduction tool. It does not claim quality improvement or total-cost savings.

## Claude Code Plugin

The repository is also a Claude Code plugin marketplace. Its plugin `davi` adds a **PostToolUse hook on the Tavily
MCP search tool** (`tavily_search`). It runs each search result through `ContextFilter` (preset `v1-thr0.3-dev`) and
replaces the tool result with only the kept results before the model reads it.

**Requires a Jev-compatible choice endpoint** (for example a self-hosted model on RunPod serverless; see
[Choice endpoint contract](#choice-endpoint-contract)). Without an endpoint ID, key and per-second rate the hook sends
nothing and changes nothing. **Requires the Tavily MCP server** (`tavily-mcp` 0.2.22), whose results carry
per-result text.

**Built-in WebSearch is not filtered.** In one observed run, Claude Code's built-in WebSearch returned only a title and
a URL per result (`results: [{tool_use_id, content: [{title, url}, ...]}, "<model-written summary>", ...]`), with no
per-result text to score. The plugin therefore does not match WebSearch at all, so it adds no start-up latency there.
(The hook still parses that shape and passes it through unchanged if you route WebSearch to it yourself.)

### Supported search tool

Only one tool is supported and matched: **Tavily `tavily_search`** from the npm package `tavily-mcp` **0.2.22**. The
format was checked in the published 0.2.22 build and in the source at commit
[`1c1d54c`](https://github.com/tavily-ai/tavily-mcp/tree/1c1d54c2a619afe52544f775c8d82a56ac6d8bb5)
([format-results.ts](https://github.com/tavily-ai/tavily-mcp/blob/1c1d54c2a619afe52544f775c8d82a56ac6d8bb5/src/format-results.ts),
[index.ts](https://github.com/tavily-ai/tavily-mcp/blob/1c1d54c2a619afe52544f775c8d82a56ac6d8bb5/src/index.ts)) on
2026-10-11. In Claude Code the tool is `mcp__<server alias>__tavily_search`; the plugin's matcher is
`^mcp__[A-Za-z0-9_-]+__tavily_search$` (0.2.22 registers only `tavily_search`, so the hyphenated spelling in older
docs is not matched).

- Format: one text block with optional `Request ID:` / `Answer:` lines, `Detailed Results:`, then per result `Title:`,
  optional `ID:`, `URL:`, `Content:`, optional `Raw Content:` / `Favicon:`, then an optional `Images:` section.
- Only the `Content` snippet is scored (not `Raw Content`). The header, the `Answer:` summary and the image list are
  kept byte for byte and never scored; no text is generated or summarised.
- **Zero intervention** for every other tool: other Tavily tools (extract, crawl, map, research), other search MCP
  servers, read / write tools and the built-in WebSearch. Tests check this.
- A response with `isError: true` or with `structuredContent` is passed through unchanged (the hook could not keep
  `structuredContent` consistent with an edited text block). Other content blocks (text, image, resource) are kept.
- Not checked: Tavily's hosted remote server (`https://mcp.tavily.com/mcp/`) and other `tavily-mcp` versions.
- The MCP tool output shape that Claude Code passes to hooks is not documented. The hook accepts a list of content
  blocks, an object with a `content` list (other keys kept) or a plain string, and writes back the same shape.

### Install

```bash
claude plugin marketplace add hundredeuk2/judge-adapter
claude plugin install davi@judge-adapter
```

The install id is `<plugin>@<marketplace>`: plugin `davi`, marketplace `judge-adapter`. Inside a session,
`/plugin marketplace add hundredeuk2/judge-adapter` and `/plugin install davi@judge-adapter` do the same.

Add the Tavily MCP server as a local stdio server, **pinned to the verified version** (never an unpinned `npx`
latest). The key is your own Tavily API key from tavily.com; `--env` stores it in your Claude Code MCP configuration,
so prefer a shell variable over typing it:

```bash
claude mcp add --transport stdio --env TAVILY_API_KEY="$TAVILY_API_KEY" tavily -- npx -y tavily-mcp@0.2.22
```

The tool then appears as `mcp__tavily__tavily_search`. Tavily bills its own searches (credits); check its terms and
pricing. Without `TAVILY_API_KEY`, tavily-mcp 0.2.22 starts in a keyless mode in which search is still available.

The hook runs `python3` (exec form, no shell) and needs **httpx** in that interpreter. Pick one:

```bash
# A) into the python3 on your PATH
python3 -m pip install "httpx>=0.24,<1"

# B) a private venv in the plugin's data directory (used automatically when python3 lacks httpx).
#    The directory is ~/.claude/plugins/data/<id>/ with <id> = "davi@judge-adapter" with every character other than
#    a letter, digit, _ or - replaced by "-", i.e. ~/.claude/plugins/data/davi-judge-adapter/
D=~/.claude/plugins/data/davi-judge-adapter
python3 -m venv "$D/venv" && "$D/venv/bin/pip" install "httpx>=0.24,<1"
#    or with uv:  uv venv "$D/venv" && uv pip install --python "$D/venv/bin/python" "httpx>=0.24,<1"
```

B is the most robust choice: it does not depend on which `python3` comes first on PATH and does not touch your system
Python. The launcher (`hooks/context_filter_hook.py`) uses only the standard library, puts the plugin's bundled `src`
first on `sys.path`, re-runs itself with `${CLAUDE_PLUGIN_DATA}/venv` when `python3` cannot import httpx, and
otherwise logs `dependency_missing` and leaves the result unchanged. It never installs anything. On Windows, `python3`
must resolve to a real executable.

### Configure

Claude Code prompts for the plugin's options when it is enabled; later, use `/plugin` → the plugin → **Configure
options**, or `claude plugin configure davi@judge-adapter`. The key and endpoint ID are `sensitive`: Claude Code keeps
them in secure storage, not in `settings.json`. Prefer the dialog for them, so they do not land in shell history.

| Option (`userConfig`) | Default | Meaning |
|---|---|---|
| `runpod_api_key` (sensitive) | none | RunPod API key. Missing → nothing is sent |
| `choice_endpoint_id` (sensitive) | none | RunPod endpoint ID of your choice worker. Missing → nothing is sent |
| `usd_per_second` | none | Your endpoint's per-second price, for the client-side estimate and cap. Missing → nothing is sent |
| `enabled` | `true` | Turn the filter off without uninstalling |
| `preset` | `v1-thr0.3-dev` | Named prompt + threshold; currently the only preset (threshold 0.3) |
| `concurrency` | `1` | Filter calls in flight per search (1-8) |
| `max_docs` | `10` | Results scored per search (1-50); later results pass through unscored |
| `wall_budget_seconds` | `20` | Time budget per search (1-60); after it the original result is used |
| `on_all_dropped` | `keep_original` | `keep_original` or `drop_all` (see below) |
| `mcp_tools` | empty | Overrides: comma-separated `name` or `name=tavily\|off` |
| `log_file` | empty | Also append the per-search log line to this file |

`mcp_tools`: `mcp__tavily__tavily_search=off` excludes the default tool; `mcp__corp__web_lookup` (or `=tavily`) treats
another tool name as Tavily output. Claude Code starts the hook only for tools the
plugin's matcher names, so an extra tool name also needs a hook that reaches it, for example in your own
`settings.json` with `"command": "python3", "args": ["/path/to/judge-adapter/hooks/context_filter_hook.py"]` and the
options as `JUDGE_ADAPTER_HOOK_<KEY>` environment variables.

How the values reach the hook: Claude Code exports every option to hook processes as `CLAUDE_PLUGIN_OPTION_<KEY>`
(e.g. `CLAUDE_PLUGIN_OPTION_RUNPOD_API_KEY`). The hook reads those first and falls back to the package's own variables
`RUNPOD_API_KEY` and `JUDGE_ADAPTER_CHOICE_ENDPOINT_ID`, and to `JUDGE_ADAPTER_HOOK_<KEY>` for the other options, so
the same hook also works from a plain `settings.json` hook or a `--plugin-dir` session. The key is read only at the
first request and is never logged or passed on a command line.

### What the hook changes

```
Tavily MCP result (mcp__<alias>__tavily_search)
   │
   ▼
hook: split into per-result snippets ──(unknown format / isError / structuredContent)──▶ result unchanged
   │   (header, Answer:, images, other blocks kept aside, never scored)
   ▼
1 filter call per result: P(keep) ──(error)──▶ keep that result
   │
   ▼
keep if P(keep) >= 0.3, else drop ──▶ same text / blocks with dropped results removed ──▶ model
Any other tool, incl. built-in WebSearch: not matched (zero intervention)
```

- **Question:** the search query from `tool_input.query`.
- **Some dropped:** `updatedToolOutput` is the original result in the **same shape and format** with the dropped
  results removed (the result blocks are cut out of the text; nothing else in the text changes), plus `additionalContext: "context filter (v1-thr0.3-dev) kept k/n results"`.
- **None dropped:** no output; the result is unchanged.
- **All dropped:** by default (`keep_original`) the result is **left unchanged** and the hook adds
  `additionalContext: "context filter (v1-thr0.3-dev) kept 0/n results for this search; the original results are shown
  unchanged."` The filter can drop every document of a search (25 of 72 searches in the HLE run above); silently
  hiding every result would leave the model with nothing and no explanation. `drop_all` removes them all and says
  `kept 0/n results`.
- **Results without a snippet** are passed through unscored; if no result has one, nothing is done.
- **Limitation, the `Answer:` line:** when the Tavily response includes an `Answer:` summary, it is kept unchanged, so
  information from dropped results can still reach the model through it.
- **Hook field:** `updatedToolOutput`. The hooks reference lists `updatedMCPToolOutput` as "Replaces the output for MCP
  tools only. Prefer `updatedToolOutput`, which works for all tools", and warns that "MCP tool output is passed through
  without schema validation", which is why the hook rebuilds the exact input shape.

### Optional: route web searches to Tavily

The filter only sees Tavily results. If you want web searches in a session to go through Tavily instead of the
built-in WebSearch, you can deny the built-in tool yourself, e.g. in your own `settings.json`:

```json
{ "permissions": { "deny": ["WebSearch"] } }
```

What this does and does not do:

- It blocks only the built-in `WebSearch` tool. The hook then applies to **supported Tavily search responses in a setup
  where web search is limited to Tavily**.
- It does **not** route everything through the filter: `WebFetch`, `Bash` (e.g. `curl`), other MCP tools,
  unsupported or error responses, and every fail-open case below still reach the model unfiltered.
- The plugin never changes your permissions or settings; this is an opt-in step you take.

### Verification status

| What | How it was checked | Result |
|---|---|---|
| Parsing, keep/drop, same-shape rebuild, fail-open, zero intervention on other tools | Fake transport tests with fixtures rebuilt from the Tavily formatter (synthetic content) | Pass |
| Manifest and marketplace | `claude plugin validate --strict` (Claude Code 2.1.285) | Pass |
| Built-in WebSearch | One real headless session | Hook reached; no per-result text; 0 filter calls; result unchanged |
| Tavily `tavily_search` | One real headless session (Claude Code 2.1.285, tavily-mcp 0.2.22, one search, 5 results) | Hook ran; 5 filter calls, 5 valid answers; all 5 kept, so the result was left unchanged |
| Dropped results removed and the shorter result received by the model | Fake tests only | **Not verified in a real run** (no result was dropped in the real run) |

The real runs used a self-hosted choice endpoint; the exact filter-call cost of those runs was not recorded, and
client-side reservation caps are not bills.

### Fail-open

The hook always exits 0. Any of the following prints nothing, so Claude Code keeps the original result: the filter is
disabled; the tool is not enabled; no query; an unrecognised result format (including server error messages such as
`Tavily API error: ...`), `isError: true`, `structuredContent`; missing key, endpoint ID or rate; an invalid option; an unknown
preset; input the filter refuses; any exception; httpx missing; the wall budget running out (in-flight requests are
abandoned). Inside the filter, a failed call (HTTP error, per-request timeout, unusable answer) keeps that document,
and after an unsettled job the remaining documents are kept unscored. The hook's own `timeout` in `hooks/hooks.json`
is 75 s, above the largest wall budget; Claude Code discards the output of a hook it cancels, which also leaves the
result unchanged. A job abandoned at the wall budget may still run on the server until its own limits
(`executionTimeout` 60 s, `ttl` 600 s).

### Latency and cost

One filter call per scored result (Tavily returns 5 by default → 5 calls), sequential by default; raise
`concurrency` to overlap them. The search result reaches the model only after the hook finishes, so every search waits
for the filter, up to the wall budget. Cold starts on serverless endpoints can dominate. Per-search cap: the hook gives
each search its own client-side estimate cap of `usd_per_second × (scored results × 661 s + 60 s + wall budget)`, the
reservation the filter's ledger needs to admit every scored result; it is an estimate, not a bill. No latency
improvement is claimed.

### Logs

One JSON line per search on stderr (visible in Claude Code's debug log) and optionally in `log_file`: status, tool
name, format, result counts in / scored / out, filter status counts, submissions, text and response bytes in / out,
seconds, and the structural shape of the tool response (key names and types only). Never the query, document text,
titles, URLs, the key or the endpoint ID. The `shape` field shows a tool's result layout safely.

### Smoke test after installing

1. Configure the key, endpoint ID and rate, set `log_file` (e.g. `~/davi-hook.log`), and add a search MCP server.
2. Start a session and ask for something that needs a web search through that server.
3. Read the log line: `status` should be `filtered`, `all_kept` or `all_dropped_kept_original`. `unrecognized_shape`
   or `no_text_in_results` means the layout (`shape`) is not one the hook can filter; the result was left unchanged.
4. For a model-free check of the hook itself, pipe a sample hook input into it:

```bash
printf '%s' '{"hook_event_name":"PostToolUse","tool_name":"mcp__tavily__tavily_search","tool_input":{"query":"q"},
"tool_response":[{"type":"text","text":"Detailed Results:\n\nTitle: t\nURL: https://example.invalid\nContent: s"}]}' \
  | env -u RUNPOD_API_KEY python3 hooks/context_filter_hook.py     # -> stderr status "missing_key", no stdout
```

To try the plugin without installing it: `claude --plugin-dir /path/to/judge-adapter`.

## Directory Structure

```
judge-adapter/
├── .claude-plugin/
│   ├── plugin.json              # plugin manifest: name, version, userConfig
│   └── marketplace.json         # marketplace "judge-adapter" listing the plugin (source "./")
├── hooks/
│   ├── hooks.json               # PostToolUse matcher: Tavily tavily_search only
│   └── context_filter_hook.py   # stdlib launcher -> judge_adapter.plugin_hook
├── src/judge_adapter/
│   ├── context_filter.py        # ContextFilter, policies, preset, budget ledger
│   ├── plugin_hook.py           # PostToolUse hook: parse Tavily results, filter, rebuild, fail-open
│   ├── plugin_names.py          # plugin / marketplace working names in one place
│   ├── backends.py              # haiku_direct and choice_endpoint backends (httpx, retries 0)
│   ├── core.py                  # JudgeAdapter, build_backend, trace writer
│   ├── contract.py              # AdapterConfig, request / result types
│   ├── gate.py                  # RunGate: run-wide submission and cost limits
│   ├── keys.py                  # key loading and scrubbing
│   ├── prompt.py, user_prompt.py  # fixed judge prompts, rubric / reference prompt
│   ├── cli.py                   # judge-adapter JSONL CLI
│   ├── fakes.py                 # in-process fake endpoints
│   ├── fastchat_runner.py, hosts/fastchat.py   # optional FastChat binding
│   └── examples/                # bundled synthetic configs and pairs
├── examples/context_filter_tool_executor.py
├── tests/                       # model-free tests (fake transports only)
├── LICENSE, NOTICE
└── pyproject.toml
```

## How it works

### Context filter

```
 question ─┐
           ├─▶ one choice call per document ─▶ P(keep) ─▶ policy ─▶ keep / drop ─▶ selected_ids (input order)
 documents ┘   (question id "relevance",        (from the     (threshold,
               options opt_keep / opt_drop)      endpoint)     top_k, ...)
               a failed call ───────────────────────────────────────▶ keep (fail-open, status says why)
```

`judge_adapter.context_filter` scores retrieved documents against one question through the `choice_endpoint`
backend: one choice call per document, then a keep / drop decision by an explicit policy. Python API only (no CLI).
It returns ids and decisions; it never rewrites, shortens or stores document text.

#### Input and output

- Input: `question` (non-empty string) and `documents`: `Document(id, title, url, text)` or mappings with those keys
  (other keys are ignored and never sent). `id` is yours: a non-empty string, unique within the call, never sent to the
  model. `title` / `url` may be empty strings; `text` must be non-empty.
- `FilterResult.documents`: one `DocResult` per input document, **in input order**: `index`, `id`, `score`, `status`,
  `reason`, `decision`, `decision_basis`, `provenance`.
  - `score`: P(keep), the probability the served model returns for `opt_keep` (`ok` and exact-tie outcomes only).
    `None` when the document was not scored or the call failed. It is not a calibrated or validated probability of
    usefulness, and it is never imputed.
  - `status`: `ok`, `error` (`reason` = the adapter failure reason: `http`, `timeout`, `format_error`, `backend_error`,
    `contract`, `local_config`, ...) or `not_scored` (`reason` = `policy_disabled`, `over_limit` or `not_submitted`).
  - `provenance`: prompt version, question id, policy, preset, per-call outcome / detail, served model name (as
    reported), submissions, status polls, cancels, client seconds, server timing, server usage, the per-call cost
    estimate and sha256 of the semantic input and raw output. No text, no endpoint ID, no key.
- `selected_ids`: kept ids, in input order. If nothing is kept it is `[]`; no substitute document is added.
- `usage` (filter calls only), `cost_estimate` (`is_estimate: true`; `usd: null` if any submitted call has unknown
  cost; fake runs are labeled `MOCK`), `timing` (measured only), `budget` (run-wide snapshot), `reduction` (below).

**Default: an explicit policy is required.** `ContextFilter` has no default policy, the same convention as
`AdapterConfig.transport` and the CLI's `--transport`: a call that can cost money or change what an agent sees is never
started by a default. Omitting both `policy` and `preset`, or passing both, raises `ValueError` before anything is
built or sent. `policy=disabled()` is the explicit off switch: zero model calls, every document kept with status
`not_scored` / `policy_disabled`.

#### Policies

| Policy | Keeps |
|---|---|
| `threshold(t)` | scored documents with `score >= t` (t in [0, 1]); e.g. 0.2999 is dropped and 0.3 is kept at t = 0.3 |
| `top_k(k)` | the k scored documents with the highest score; exact ties go to the earlier input position |
| `disabled()` | everything, without calling the model |

Under every policy a document without a score (error, not submitted, over the limit when flagged) is **kept**
(fail-open; under `top_k` on top of the k) and its status shows why. Rows are never removed.

**Scope of fail-open (as implemented).** Fail-open is **per document**: an HTTP error, a timeout, an unusable
answer (format failure, handler error, truncation, invalid probabilities) or an unexpected client exception during
that document's call keeps that one document with status `error`; a run-wide stop keeps every later document as
`not_scored` / `not_submitted`. The other documents of the same `filter()` call are decided normally. An exact
0.5 / 0.5 answer is not a failure: it keeps its score and the policy decides. The whole call is refused, with an
exception and no result, only in these cases: invalid input (`FilterInputError`, raised before any call, 0
submissions), invalid constructor arguments (`ValueError`), or an exception raised by a caller-supplied `tokenizer`
or `logger`.

**Zero kept is a normal result.** With `threshold(t)` every scored document can fall below `t`; the result is then
`selected_ids == []` (an empty selection, not an error), and the rows show `status='ok'`, `decision='drop'`. The
package adds no fallback document, no minimum keep count and no re-search; what to do with an empty selection is the
caller's decision (for example, use the unfiltered documents, retry the search, or proceed without documents). In
the HLE run (see [Results](#results)) this happened in 25 of 72 searches with the preset `v1-thr0.3-dev`; that is a
per-search count from a single run, not a general drop rate or failure rate. `top_k(k)` with k >= 1 never returns an
empty selection when at least one document was scored. The Claude Code plugin's default for this case is to leave
the search result unchanged and say so (see [What the hook changes](#what-the-hook-changes)).

#### Preset `v1-thr0.3-dev` (opt-in)

`ContextFilter(cfg, backend, preset='v1-thr0.3-dev')` fixes the relevance wording `relevance/v1-dev` (English
instructions and option texts below), its document template (`제목: {title}\nURL: {url}\n\n{text}`; the title label
is Korean for "Title", as in the run it comes from), `threshold(0.3)` and the size range it was tried on (text <= 8,000,
title <= 300, URL <= 500 UTF-8 bytes). It is **a development candidate validated once on 25 questions; not a quality
guarantee**. It was chosen on development data and checked once in a private study; the evidence is not part of this
package. Results depend on the model you serve. It is not a default and not a recommendation for other tasks.
`limits=` may override its size range (recorded in the result); `prompt=` / `policy=` cannot be combined with it.

Instructions: "You are given a question and one web search document below. Decide whether this document provides a
direct answer to the question or intermediate evidence needed to find the answer. Being on the same topic as the
question is not enough on its own. The correct answer to the question is not given." Options: `opt_keep` "Provides:
this document contains a direct answer or needed intermediate evidence.", `opt_drop` "Does not provide: this document
contains neither a direct answer nor needed intermediate evidence."

Without a preset the default prompt `relevance/v1` uses the same wording with the English template
`Title: {title}\nURL: {url}\n\n{text}`, which was not the template that was tried. A custom
`RelevancePrompt(version, instructions, keep, drop, document_template)` may use only `{title}`, `{url}`, `{text}`.

#### Wire contract (relevance question)

Same endpoint, polling, cancel and strict restoration as the pairwise contract, with question id `relevance`:

```json
{"input": {"state": {"question": "...", "document": "<rendered document>"},
           "questions": {"relevance": {"type": "choice", "instructions": "...",
                                       "criteria": {"opt_keep": "...", "opt_drop": "..."}}}},
 "policy": {"executionTimeout": 60000, "ttl": 600000}}
```

A valid job has `output.answers.relevance = {"type": "choice", "choice": ..., "probabilities": {"opt_keep": p,
"opt_drop": q}}` (finite, in [0, 1], sum 1 +- 0.01, choice = unique argmax). Probabilities are read by option id,
never by position, and never renormalized. An exact 0.5 / 0.5 tie has no model label but keeps its score.

#### Who is responsible for what

- The package: validation before any call, one call per document through the existing choice-endpoint client, strict
  restoration, the policy decision, fail-open rows, run-wide stops, order restoration, the result record.
- The caller: storing the original documents and restoring them by id (`select_originals` helps; a tool can serve
  dropped ids on request, as in the example); choosing a policy; the endpoint, model, rate and caps; deciding whether
  filtering helps their task.
- Not verified: filter quality on your data, any model, task improvement, cost or latency savings. The tests use fake
  transports only; the single observed runs in [Results](#results) are outside the test suite and are not a
  verification.

### Pairwise judge adapter

```
 question + response A + response B ─▶ backend call (AB; BA in two_direction) ─▶ opt_1 / opt_2 / opt_tie ─▶ A / B / tie
                                                                       a failure ─▶ explicit error (never a tie)
```

#### JSONL input

One JSON object per line. The whole file is validated before any call; one bad line rejects the file (exit code 2)
with line numbers, and nothing is sent or written.

```json
{"id": "ex-1", "question_turns": ["Give me one tip for writing a clear bug report."], "response_a": {"id": "draft-v1", "turns": ["Write good reports."]}, "response_b": {"id": "draft-v2", "turns": ["State the exact steps to reproduce, what you expected, and what happened instead."]}}
{"id": "ex-2", "question_turns": ["Name a prime between 10 and 20.", "Now one between 20 and 30."], "response_a": {"id": "bot-x", "turns": ["13.", "23."]}, "response_b": {"id": "bot-y", "turns": ["11.", "25."]}, "turn": 2}
```

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | Request id, unique in the file |
| `question_turns` | yes | 1 or 2 user turns, non-empty strings |
| `response_a`, `response_b` | yes | `{"id": ..., "turns": [...]}`; one answer per question turn; ids must differ and must not be `"tie"` |
| `turn` | no | Judged turn; if given it must equal the number of question turns |
| `rubric` | no (v2) | Your evaluation standard, one string (max 8,000 UTF-8 bytes) |
| `reference` | no (v2) | One entry per question turn: a non-empty string or `null` (max 20,000 UTF-8 bytes per entry) |

- Schema v1 = the fields without `rubric` / `reference`; v1 files stay valid. The summary records
  `judge-adapter-pairs/v2`.
- The judged turn is always the last turn. With two turns, the first exchange is context.
- Response ids are never sent to the model; verdicts are restored to them.
- Rejected before any call: missing or unknown fields, duplicate ids, same A/B id, turn-count mismatch, more than 2
  turns, invalid JSON, empty lines, non-UTF-8 input, `criteria` / `instructions` / `judge_prompt` / `system_prompt`
  and near-miss names (`rubrics`, `references`, `reference_answer`, `ref_answer`), `rubric` / `reference` inside a
  response, and any size limit. The whole wire body of one call is limited to 200,000 UTF-8 bytes. Nothing is
  truncated.

#### Rubric and reference (prompt `user-criteria/v1`)

- Without `rubric` and `reference`, the fixed default prompt (`default-fixed/v1`) is sent byte for byte.
- `rubric` replaces the general quality standard ("helpfulness, relevance, accuracy, ...") instead of being appended
  to it. `reference` adds a reference-answer field that the instructions describe as a guide, not a candidate.
- Fixed on both paths: the pairwise framing, the bias sentence, options `opt_1` / `opt_2` / `opt_tie`, the strict
  parser and the generation settings. Custom option ids or score scales are not supported.
- The reference block is not tied to a slot, so it does not move when A/B are swapped.
- The field separation does **not** guarantee that a model follows the rubric or resists prompt injection. The
  recorded hashes show what was sent, not that it was followed.

#### CLI

```bash
judge-adapter validate PAIRS.jsonl [--backend choice_endpoint] [--mode two_direction] [--max-submissions N]
judge-adapter run PAIRS.jsonl --out DIR --transport fake [--backend choice_endpoint] [--mode two_direction]
judge-adapter run PAIRS.jsonl --out DIR --transport real --allow-real --max-submissions 4 --budget-cap-usd 0.05 \
    [--backend choice_endpoint --choice-endpoint-id ID --choice-usd-per-second RATE [--choice-expected-model NAME]] \
    [--mode two_direction] [--env-file PATH]
judge-adapter examples
```

| Option | Meaning |
|---|---|
| `--transport fake\|real` | Required, no default |
| `--backend haiku_direct\|choice_endpoint` | Default `haiku_direct` |
| `--mode single\|two_direction` | `two_direction` judges AB then BA; different labels -> `tie` with `tie_kind: inconsistent`; if AB fails, BA is not sent |
| `--max-submissions` | Run-wide submission limit (required for real) |
| `--budget-cap-usd` | Run-wide client-side cost cap (required for real) |
| `--allow-real` | Required for real |
| `--out`, `--overwrite` | Output dir; existing results are refused (exit 3) unless `--overwrite`, which replaces only this tool's 5 files |
| `--env-file` | Optional `KEY=VALUE` file, read only by a real run. No default path |
| `--choice-endpoint-id` | RunPod endpoint ID for `choice_endpoint` (else `JUDGE_ADAPTER_CHOICE_ENDPOINT_ID`) |
| `--choice-usd-per-second` | Per-second rate of your endpoint for the estimate and cap; required for a real `choice_endpoint` run (no default; missing -> refused, exit 3) |
| `--choice-expected-model` | Optional expected `output.model`; a different model is a contract failure. Unset: not checked, only recorded |

Output: `results.jsonl` (one row per input line: `outcome` ok / tie / error / unrun, `verdict`, `winner_id`,
`tie_kind`, `failure`, `calls[]`, `cost_estimate`, `criteria_info`), `summary.json`, `gate.json` (run-wide limit
log), `trace/judge_trace.jsonl`, `trace/judge_raw.jsonl` (HTTP bodies, keys removed). All output files are
scrubbed, so a stored body can differ from what the provider sent. No success rate is
reported; every request is counted, including errors and unrun requests.

#### Python API

```python
from judge_adapter import AdapterConfig, JudgeAdapter, JudgeRequest, Response, build_backend
from judge_adapter import fakes

cfg = AdapterConfig(transport='fake', backend='haiku_direct', mode='two_direction', trace_dir='trace')
fake = fakes.fake_for(cfg.backend)                      # in-process fake server; no key, no network
backend = build_backend(cfg, transport=fake.transport(), key_provider=lambda: fakes.DUMMY_KEY)
adapter = JudgeAdapter(cfg, backend)

req = JudgeRequest(request_id='q1', question_turns=('What is 17 * 23?',),
                   response_a=Response('calc-x', ('391',)), response_b=Response('calc-y', ('381',)),
                   rubric='Only correctness matters.', reference_turns=('391',))
res = adapter.judge(req)
print(res.status, res.verdict, res.winner_id, res.criteria_info['prompt_version'], res.submissions)
adapter.close()
```

For a real call use `AdapterConfig(transport='real', ...)` and `build_backend(cfg)` (no custom transport). To apply
run-wide limits in your own loop, wrap the backend in `judge_adapter.RunGate(backend, mode, cap_usd,
max_submissions)` and call `gate.begin(request)` before each `adapter.judge(request)`; `judge_adapter.cli.run_file`
does exactly this.

#### FastChat binding (optional)

`judge_adapter.hosts.fastchat` replaces `fastchat.llm_judge.common.run_judge_pair` at run time (host source
untouched); `judge-adapter-fastchat` runs FastChat's own `gen_judgment.py` / `show_result.py` through it.
Supported host revision: `587d5cfa1609a43d192cedb8441cac3c17db105d`. Other revisions are not checked.

```bash
# demo workdir: copies FastChat's own question / judge-prompt / reference files from YOUR FastChat checkout
judge-adapter-fastchat demo-workdir --workdir W --fastchat-root /path/to/FastChat
CFG=$(judge-adapter examples | grep config.fake.json)
judge-adapter-fastchat judge --config "$CFG" --workdir W --fastchat-root /path/to/FastChat --confirm -- \
    --mode pairwise-baseline --bench-name adapter_demo --judge-model judge-adapter \
    --baseline-model model-y --model-list model-x
judge-adapter-fastchat show --workdir W --fastchat-root /path/to/FastChat -- --mode pairwise-baseline \
    --bench-name adapter_demo --judge-model judge-adapter --baseline-model model-y
```

- No FastChat or MT-Bench data ships with this package. `--fastchat-root` points at a FastChat source checkout;
  without it the installed `fastchat` package is used. The pip-installed FastChat (`[fastchat]` extra) does **not**
  contain `fastchat/llm_judge/data`, so `demo-workdir` needs a source checkout; with only the extra installed, write
  your own `data/judge_prompts.jsonl` (entries named `pair-v2`, `pair-v2-multi-turn`, `pair-math-v1`,
  `pair-math-v1-multi-turn`; their text is replaced by the adapter) and `data/<bench>/question.jsonl` /
  `model_answer/*.jsonl` in the host's format.
- Host `--mode` must be `pairwise-baseline` or `pairwise-all`. Host `single` (its default) is refused before the host
  starts, because host single grading would call its own judge client outside the adapter.
- Adapter mode must be `single` (the host already judges both positions as g1 / g2).
- `transport: "real"` needs `--allow-real`. The wrapper has **no run-wide cap or submission limit** (only the JSONL
  CLI has `RunGate`).
- Host templates are **replaced, not reproduced**: `pair-v2`, `pair-v2-multi-turn` -> `default-fixed/v1`;
  `pair-math-v1`, `pair-math-v1-multi-turn` -> `user-criteria/v1`. The host's scores, explanations and
  `[[A]]`/`[[B]]`/`[[C]]` format are not used; the host's result rows still show the host prompt names.

##### Reference categories (explicit mapping)

Host categories `math`, `reasoning`, `coding`, `arena-hard-200` need a reference answer. The mapping is explicit:

- `--reference-file PATH`: a FastChat answer-format JSONL you provide (`question_id`, `choices[0].turns`), or
- `--reference-model NAME`: `W/data/<bench>/reference_answer/NAME.jsonl` (e.g. `gpt-4` for FastChat's MT-Bench
  reference in your own workdir).
- Python API: `judge_adapter.fastchat_runner.judge(..., reference_file=..., reference_model=...)`.

The reference is **never inferred from the judge name** and **never auto-filled**. The host looks references up by
`--judge-model`; the wrapper runs the host in a temporary staged workdir (built with **symlinks**) whose
`reference_answer/` holds only a byte copy of the mapped file, named `<judge-model>.jsonl`. A file under the judge
name in your workdir is ignored. The mapping is recorded in `<trace_dir>/fastchat_reference_mapping.jsonl` (source
path, sha256, staged name, mapped questions, `inferred_from_judge_name: false`, `auto_fill: false`, host revision,
template replacement). Reference-category requests send the user reference turns 1..turn as `reference_turns`;
other categories are built exactly as without a mapping (byte-identical wire bodies).

Rejected before any call (exit code 2; the host does not run, no trace dir is created): a missing row for a
reference-category question that will run (after `--first-n`), a reference turn count different from the
question's, an empty or non-text turn, a turn over 20,000 UTF-8 bytes, a duplicate `question_id`, a malformed row,
a missing file, both options at once, a name containing a path, a relative `--judge-file` outside the workdir, and a
host whose `NEED_REF_CATS` differs from the supported revision. Without a mapping, reference-category requests are
`unsupported_input` (0 calls).

Limits of the reference path: host single-mode reference grading (`single-math-v1*`) stays blocked; environments
that cannot create symlinks (some Windows setups) are untested; whether a model follows the reference, judgment
quality and preservation of the host's `pair-math-v1` policy are not verified.

## Configuration and limits

### Backends and configuration

| Backend | Endpoint | Key | Output |
|---|---|---|---|
| `haiku_direct` (default) | Anthropic Messages API, model `claude-haiku-4-5-20251001`, `max_tokens` 8, temperature 0, no system prompt | `ANTHROPIC_API_KEY` | Text that must be exactly `opt_1`, `opt_2` or `opt_tie` |
| `choice_endpoint` | `https://api.runpod.ai/v2/<endpoint-id>/runsync`: a choice-judge worker that you serve on RunPod serverless | `RUNPOD_API_KEY` | Probability per option (see contract below) |

Keys: from the environment variable, or from an env file only if you pass one (`--env-file` /
`AdapterConfig.env_file`). There is no default file path. A key is read only at the first real request, never at
import, validation or in a fake run. (The Claude Code plugin reads `CLAUDE_PLUGIN_OPTION_RUNPOD_API_KEY` first; see
[Configure](#configure).)

Choice endpoint settings (nothing endpoint-specific is built in):

- endpoint ID: `AdapterConfig.choice_endpoint_id`, `--choice-endpoint-id` or `JUDGE_ADAPTER_CHOICE_ENDPOINT_ID`;
- per-second rate: `AdapterConfig.choice_usd_per_second` / `--choice-usd-per-second` (required for a real run);
- expected model name: `AdapterConfig.choice_expected_model` / `--choice-expected-model` (optional).

Without an endpoint ID or a rate, every real `choice_endpoint` call fails as `local_config` before anything is sent
(no key read, 0 submissions); the CLI refuses a real run without a rate before it starts (exit 3). The fake transport
needs no endpoint and uses a placeholder rate that is labeled `FAKE PLACEHOLDER` in every cost basis.

`AdapterConfig` fields: `transport` (required), `backend`, `mode`, `trace_dir`, `timeout_seconds`,
`choice_endpoint_id`, `choice_poll_interval_seconds`, `choice_max_status_polls`, `choice_usd_per_second`,
`choice_expected_model`, `env_file`.
`AdapterConfig.load(path)` reads a JSON file (keys starting with `_` are comments); see the bundled
`config.*.json` examples.

### Choice endpoint contract

`choice_endpoint` talks to a RunPod serverless worker that you deploy and that answers a multiple-choice judging
request with one probability per option. This package ships no model and no worker. The adapter sends one
`POST /runsync?wait=110000`:

```json
{"input": {"state": {"question": "...", "assistant_1_conversation": "...", "assistant_2_conversation": "..."},
           "questions": {"preference": {"type": "choice", "instructions": "...",
                                        "criteria": {"opt_1": "...", "opt_2": "...", "opt_tie": "..."}}}},
 "policy": {"executionTimeout": 60000, "ttl": 600000}}
```

With a rubric / reference, `state` also has `evaluation_criteria` / `reference_answers`. A valid completed job is
`status: COMPLETED` with `output.answers.preference = {"type": "choice", "choice": <option id>, "probabilities":
{"opt_1": p1, "opt_2": p2, "opt_tie": p3}}`. Probabilities must be finite, in [0, 1] and sum to 1 +- 0.01, and
`choice` must be the unique argmax. `output.model` is optional: it is recorded as `calls[].response_model`, and it is
checked only if you set an expected model. `output.usage`, `executionTime` and `delayTime` are read when present;
`output.error` / `output.truncated` flags are failures. If `/runsync` returns `IN_QUEUE` / `IN_PROGRESS`, the
adapter polls `GET /status/<job>` of the same job (bounded by `choice_max_status_polls` and the 600 s ttl), then
sends one `POST /cancel/<job>`. Anything else is a contract / format failure. The context filter uses the same
contract with the `relevance` question shown [above](#wire-contract-relevance-question).

### Context filter limits, stops and budget

Everything is validated before the first call; one problem refuses the whole input with `FilterInputError`
(0 submissions). Nothing is truncated.

| `FilterLimits` field | Default | Over the limit |
|---|---|---|
| `max_docs` (per `filter()` call) | 50 | refused |
| `max_question_bytes` | 8,000 | refused |
| `max_title_bytes` / `max_url_bytes` / `max_text_bytes` | 1,000 / 2,048 / 20,000 | refused (`oversize='reject'`, default) or kept unscored as `over_limit` (`oversize='flag'`) |
| `max_body_bytes` (one wire body) | 200,000 | as the per-document limits |

Missing, empty or duplicate ids, non-string fields and empty text are refused the same way. (The plugin hook uses the
preset's size range with `oversize='flag'`, so an oversized search result is kept unscored instead of refusing the
whole search.)

One `ContextFilter` is one run: `budget_cap_usd` and `max_submissions` cover all its `filter()` calls. The estimate
is the RunGate time estimate for the choice endpoint (wall clock since the first submission + 60 s idle allowance +
661 s reservation per open job, times your rate); it is client-side, not the bill. Submissions stop, and every later
document is `not_scored` / `not_submitted` and kept, after: the cap or the submission limit would be exceeded; a job
not observed final (HTTP error, timeout, unresolved queue; its reservation stays held); a final job without server
execution time (cost unknown); a contract failure; a local configuration failure (nothing sent, nothing held).
Format failures and handler errors of completed jobs are settled; that document is kept as `error` and the run goes
on. Retries 0.

### Concurrency

Default `concurrency=1`: one request at a time, in input order. `concurrency=N` (opt-in) runs at most N requests in
a thread pool. Each submission is admitted by one atomic check-and-reserve against the cap and the submission limit;
after a stop nothing new is admitted; requests already in flight are waited for and settled; results are restored to
input order; a partial failure affects only its document plus the stop rule above. This is tested with fake barriers
only. No latency improvement is claimed: parallel behaviour and numeric stability of a real server were not tested.

### Logging (local estimate, not billing)

Each `filter()` emits one INFO record on the logger `judge_adapter.context_filter` (also in `FilterResult.reduction`):
documents in / out, UTF-8 bytes of document text in / out, status counts, stop code, mock flag. With
`tokenizer=` (an object with `name`, `version`, `count(text) -> int`) it adds text token counts in / out and the
tokenizer name and version; without one, tokens are reported as unsupported. These are **local estimates of document
text reduction**, not billed usage, not all-turn cached / uncached / output usage of your agent, and they are never
converted into money. The record holds no question, no document text, no ids and no keys. Note that `httpx` logs request
URLs (which include your endpoint ID) at INFO; keep the `httpx` logger at WARNING if your logs are shared (the plugin
hook does this itself).

### Cost estimates (read before a real run)

- All cost numbers are **client-side estimates, not the bill**. The provider console is the source of truth. The
  cap is not an account billing limit.
- Haiku: usage x list price ($1 / $5 per MTok, as checked 2026-09; check the current price). The pre-call
  reservation is (body bytes + 200) x 2 x input price + 8 output tokens.
- choice_endpoint: server `executionTime` x the per-second rate **you configure** (no built-in rate; the package
  does not know your hardware or price). Excludes cold start, idle timeout and queue billing. The cap reserves
  661 s (ttl 600 s + idle 60 s + 1 s) per open job plus 60 s idle once, so the first job needs a cap of at least
  `(661 + 60) x rate` USD, and the first two_direction request `(2 x 661 + 60) x rate` USD.
- Failed requests are included. A submitted call with unknown usage keeps its full reservation and stops the run.

### Raw responses and stored traces

Provider responses are parsed from the body **as received**. Scrubbing (your exact key, `Bearer ...`, and key-like
substrings `sk-...` / `rpa_...`) is applied only where data leaves the process or is written down:

| Where | Content |
|---|---|
| In-memory results: `JudgeResult` / `CallRecord` fields parsed from the response (`label`, `usage`, `response_model`, `job_id`, `job_status`, `server_timing`), `FilterResult` scores and provenance, a backend's `call()` result (`raw_text`, `exchanges[].text`, `exchanges[].json`) | the provider's data, unmodified |
| `detail` / `failure.detail` and `exchanges[].error` (error messages, including text taken from provider error bodies) | scrubbed |
| `trace/judge_raw.jsonl`, `trace/judge_trace.jsonl`, CLI `results.jsonl` and `gate.json` | scrubbed (the whole record) |
| The context filter's INFO log | counts and sizes only |
| The plugin hook's log line | counts, bytes, status and the response's key names / types only |

Why: a legitimate payload can contain a key-like substring by chance. A URL-safe base64 value (for example an
opaque `encrypted_content` blob that a caller replays to the provider) may contain `-sk-` followed by 8+ characters;
scrubbing before parsing rewrote it to `[redacted]` and corrupted the value. Since this fix, values inside any JSON
string reach the caller byte-for-byte.

Compatibility: no signature changed. Before the fix, a response value that matched the scrubber came back with
`[redacted]` in it; it now comes back as sent. The stored files are unchanged in kind (scrubbed) and now scrub the
whole record, not only the HTTP body. `raw_output_sha256` keeps its meaning: the sha256 of the **stored (scrubbed)**
body in `judge_raw.jsonl`, so it can be checked against that file; it equals the hash of the received body whenever
nothing was scrubbed. If you log or save in-memory results yourself, scrub them first (`judge_adapter.keys.Scrubber`;
a backend's `scrub` attribute knows the key after the first request).

### Fake transport

`--transport fake` uses an in-process `httpx.MockTransport` (`judge_adapter.fakes`). The fake verdict reads only
markers in the judged answers:

| Marker | Fake result |
|---|---|
| `[[GOOD]]` | that answer wins (follows the response across swaps) |
| none / both | `tie` (explicit) |
| `FAKE:POSITION` | always slot 1 (two_direction -> `tie` inconsistent) |
| `FAKE:FORMAT` | `format_error` |
| `FAKE:TRUNC` / `FAKE:REFUSE` | `truncated` / `refused` (choice_endpoint: `backend_error`) |
| `FAKE:HTTP500` / `FAKE:TIMEOUT` | `http` / `timeout`; usage unknown, later requests unrun |
| `FAKE:QUEUE` | choice_endpoint: one status poll of the same job |
| `FAKE:LEAK` | the fake server echoes the auth header (scrubbing check) |

The context filter's fake (`fakes.FakeRelevanceEndpoint`) reads only the rendered document: `[[KEEP=p]]` sets P(keep) = p, `[[RELEVANT]]` 0.9, otherwise 0.1; `FAKE:HTTP500`, `FAKE:TIMEOUT`, `FAKE:FORMAT`, `FAKE:REFUSE`, `FAKE:QUEUE` as above; `FAKE:NOTIMING` omits the server timing (cost unknown).

Usage numbers in fake runs are invented. Identical fake results in both directions show that ids are restored
across the swap; they are **not evidence that a real model is order-invariant**.

### Tests

```bash
pip install -e ".[test]"
pytest                                   # model-free; FastChat tests are skipped
JUDGE_ADAPTER_FASTCHAT_ROOT=/path/to/FastChat pytest   # also runs the FastChat host tests (needs the extra's deps)
```

All tests use fake transports; network sockets and key reads are blocked, and the key / endpoint environment
variables are removed for each test. The plugin hook tests (`tests/test_plugin_hook.py`) drive the hook entry point
with sample hook inputs and a fake choice endpoint, and run the launcher as a subprocess.

### Limits

- **No quality claim.** Nothing here shows that a backend judges well or better than another, or that a model
  follows a rubric or reference.
- **No guaranteed savings.** Cost numbers are estimates; the cap is client-side.
- **No claim of general host-policy preservation.** The FastChat binding replaces the host's judge prompts with this
  package's fixed prompts; it does not reproduce them.
- **Tests are mock tests.** This package's test suite runs only against fake transports. An earlier experiment
  version of this code made a 16-call real smoke run (8 Haiku, 8 choice-endpoint calls) through FastChat; that result applies to that
  experiment version, **not to this package**. The context filter at commit `331081c` was run once against a real
  choice endpoint (30 calls on synthetic documents, checking the wire contract, budget, ordering and concurrency; no
  quality claim). The pairwise judge in this package has not been run against a real model or API.
- **Plugin: drop → replace → model not verified in a real run.** See [Verification status](#verification-status).
  The replacement path is covered by fake tests only; the shape in which Claude Code hands MCP output to hooks is not
  documented and was observed once.
- Pairwise requests run sequentially; no parallelism, no resume. Two turns at most. Custom option ids and score
  scales are not supported. The context filter has opt-in bounded concurrency (default 1), tested with fakes only.

## 한국어 요약

- **무엇인가:** 두 기능을 담은 패키지다. (1) **컨텍스트 필터**: 검색 문서마다 선택지 모델에 한 번 물어 P(keep)을
  받고 기준 이상만 넘겨 agent의 **입력 토큰을 줄인다**. (2) **pairwise judge 어댑터**: 질문 하나(1~2 turn)와 응답
  두 개를 넣으면 `A`/`B`/`tie`/명시 `error`와 호출 기록을 남긴다. JSONL CLI(`judge-adapter`), Python API(`import
  judge_adapter`), 선택 FastChat binding(`judge-adapter-fastchat`, extra `fastchat`), Claude Code plugin(`davi`)을
  제공한다. 배포 이름 `judge-adapter`, 라이선스 Apache-2.0.
- **설치:** `pip install "judge-adapter @ git+https://github.com/hundredeuk2/judge-adapter"`. 코어 의존성은 `httpx`
  하나다. FastChat은 `[fastchat]` extra이며 commit `587d5cfa`로 고정하고 torch 등 무거운 의존성을 함께 설치한다.
- **결과(단일 실행, 보장 아님):** HLE 영어 텍스트 90문항, 웹 검색을 쓰는 GPT-5.2 agent, 검색당 5문서, preset
  `v1-thr0.3-dev`. 세 설정이 모두 끝난 86문항에서 agent 입력 토큰은 top-3 대비 −31.4%, 5문서 전부 대비 −44.9%였다.
  이전 README의 "90문항 전체 합 기준 약 44% 감소(top-3 대비)"는 중단된 궤적 4개(그중 3개가 필터 설정)를 0토큰으로 센
  과대 수치였으므로 이 86문항 수치로 바로잡는다. 정답 수(Haiku가 참고답 기준 채점): HLE 90문항 필터 28 / top-3 27 /
  5문서 31, 이전 한국어 25문항 실행 필터 14 / top-3 10 / 전체 12. 두 실행 모두 필터가 top-3 이상(+1/90, +4/25)이었지만
  통계적으로 확립된 차이는 아니며, 5문서 전부와는 엇갈렸다(−3/90, +2/25). 필터 판단은 입력 전용이고, Jev 방식 정가
  입력 1M 토큰당 $0.042(가정; 이 패키지가 그 모델을 제공하지 않음) 기준 문항당 약 $0.0003이다. 총비용은 생성 출력에
  따라 달랐다: HLE 86문항에서 필터 설정은 top-3 대비 출력 토큰 +6.7%, 총비용 +4.5%였다. 총비용 절감은 확립되지
  않았다. 같은 HLE 실행에서 72회 검색 중 25회는 5문서가 모두 drop되었다(그 실행의 검색 단위 관찰이며 일반 drop률·실패율이
  아님). 용도는 opt-in 입력 축소 도구이며 품질 향상·총비용 절감을 주장하지 않는다.
- **Claude Code plugin `davi`:** `claude plugin marketplace add hundredeuk2/judge-adapter` 후 `claude plugin install
  davi@judge-adapter`. 대상은 **Tavily MCP의 `tavily_search` 하나**다(`tavily-mcp` npm 0.2.22, 소스 commit `1c1d54c`에서
  출력 형식 확인). 서버 별칭과 상관없이 `^mcp__[A-Za-z0-9_-]+__tavily_search$`로만 매칭하고, 다른 도구(다른 Tavily 도구·
  다른 검색 MCP·읽기/쓰기 도구·내장 WebSearch)에는 전혀 개입하지 않는다. **Claude Code 내장 WebSearch는 필터하지
  않는다**: 한 번의 실제 실행에서 결과마다 제목·URL만 있고 본문이 없었다(점수 매길 텍스트 없음). Tavily 서버 추가는 버전을
  고정한다: `claude mcp add --transport stdio --env TAVILY_API_KEY="$TAVILY_API_KEY" tavily -- npx -y tavily-mcp@0.2.22`
  (Tavily API 키는 본인 것, 검색 비용은 Tavily 크레딧 과금, 키가 없으면 keyless 모드로 검색은 동작). hook은 검색어
  (`tool_input.query`)를 질문으로 결과의 `Content` 스니펫마다 필터를 한 번씩 호출하고, drop된 결과 블록만 텍스트에서 잘라 낸
  **같은 모양·같은 형식**으로 `updatedToolOutput`을 돌려주며(`updatedMCPToolOutput`은 쓰지 않음) `context filter (...)
  kept k/n results`를 덧붙인다. 헤더·`Answer:` 요약·이미지 목록·다른 블록은 그대로 두고 점수에 넣지 않으며, 새 텍스트를
  만들지 않는다. 한계: `Answer:` 요약은 그대로 남으므로 drop된 문서의 정보가 거기 남을 수 있다. `isError: true`이거나
  `structuredContent`가 있으면 그대로 통과시킨다. 하나도 drop되지 않으면 아무것도 출력하지 않는다. 모두 drop되면 기본값
  (`keep_original`)은 원래 결과를 그대로 두고 0개를 남겼다는 한 줄을 덧붙인다(`drop_all`로 바꿀 수 있음). 알 수 없는
  형식(서버 오류 메시지 포함)·키/endpoint/단가 누락·설정 오류·예외·httpx 없음·호출당 시간 예산(기본 20초) 초과에서는
  아무것도 출력하지 않아 원래 결과가 그대로 쓰인다(fail-open). 설정은 `userConfig`(RunPod 키와 endpoint ID는 sensitive,
  단가, enable, preset, 동시성, 최대 문서 수, 시간 예산, 전부 drop 시 동작, `mcp_tools` 재정의(`name=tavily|off`), 로그
  파일)이며 hook에는 `CLAUDE_PLUGIN_OPTION_<KEY>`로 전달되고, 패키지 변수 `RUNPOD_API_KEY`·
  `JUDGE_ADAPTER_CHOICE_ENDPOINT_ID`와 `JUDGE_ADAPTER_HOOK_<KEY>`를 대체값으로 읽는다. 결과 하나당 필터 호출 1회라서
  검색마다 지연이 생긴다. Jev 호환 선택지 endpoint(예: RunPod serverless에 직접 띄운 모델)가 필요하다. 로그는 개수·
  bytes·상태·형식·응답 구조(키 이름과 형식)만 남기고 검색어·본문·제목·URL·키·endpoint ID는 남기지 않는다. 실제
  Tavily 검색 1회(결과 5개)에서 hook 실행과 필터 호출 5회·정상 응답 5회를 확인했으나 5개 모두 keep되어 결과는 바뀌지
  않았다. drop된 결과를 빼고 줄어든 결과가 모델에 전달되는 경로는 fake 테스트로만 확인했고 실제 실행으로는 미검증이다.
  Claude Code가 MCP 출력을 hook에 넘기는 모양은 문서화되어 있지 않다. 선택 사항으로, 본인 `settings.json`에서 내장
  WebSearch를 deny하면 웹 검색을 Tavily로 제한한 구성에서 지원 응답에 hook이 적용된다. 이는 WebSearch만 막을 뿐
  WebFetch·Bash·다른 MCP·미지원/오류 응답·fail-open 경우는 필터를 거치지 않으며, plugin이 권한을 자동으로 바꾸지 않는다.
- **유지하는 동작:** 기본 fake·real은 명시 opt-in(`--allow-real` + `--max-submissions` + `--budget-cap-usd`),
  재시도 0, 실행 전체 제출·비용 상한, 비용 불명은 `null`, 전송 전 오류는 제출 0(`local_config`), 저장 trace·CLI
  출력·오류 메시지·log 키 제거(응답 파싱은 원문 body로 하고 메모리 결과의 provider 값은 바꾸지 않음),
  결과 덮어쓰기 거부, rubric/reference 전달과 버전·sha256 기록.
- **키·endpoint:** 키는 환경변수(`ANTHROPIC_API_KEY`, `RUNPOD_API_KEY`) 또는 명시한 env 파일에서만 읽는다. 기본
  파일 경로는 없다. `choice_endpoint` backend(사용자가 RunPod serverless에 직접 띄운 선택지 judge worker)의
  endpoint ID는 설정·`--choice-endpoint-id`·`JUDGE_ADAPTER_CHOICE_ENDPOINT_ID`로만, 초당 단가는
  `--choice-usd-per-second`로만 받는다(기본값 없음). 둘 중 하나라도 없으면 real 호출은 전송 0인 `local_config`이고,
  CLI는 단가 없는 real 실행을 시작 전에 거부한다. 기대 모델 이름은 선택 설정이며, 주면 불일치가 contract 실패다.
  선택지 endpoint 계약(요청·응답 형식)만 문서화했고 모델·단가·하드웨어 정보는 넣지 않았다.
- **FastChat 참고답:** `--reference-file` 또는 `--reference-model`로 명시 매핑한다. judge 이름으로 추정하지 않고
  자동으로 채우지 않는다. 누락·turn 불일치 등은 호출 전에 거부한다. 임시 작업 폴더는 symlink를 쓰며, symlink가
  없는 환경(일부 Windows)은 확인하지 않았다. host single 모드 참고답 채점은 계속 차단한다. host 템플릿은 재현이
  아니라 교체한다. FastChat·MT-Bench 데이터는 패키지에 포함하지 않는다.
- **컨텍스트 필터:** `judge_adapter.context_filter`는 질문 하나와 문서들(id·title·url·text)을 받아 문서마다
  `choice_endpoint`에 한 번 묻고, 입력 순서대로 문서별 P(keep)·상태(ok / error 사유 / not_scored)·근거와
  `selected_ids`를 돌려준다. 정책은 명시해야 한다(`threshold(t)`, `top_k(k)`, `disabled()` 또는 preset; 기본 정책 없음,
  `transport`와 같은 관례). 실패·미전송 문서는 지우지 않고 keep(fail-open)하며 상태로 이유를 보인다. 선택이 없으면 빈
  목록이다. 입력 오류·크기 초과는 호출 전에 거부한다(자르지 않음; 문서 크기 초과는 `oversize='flag'`로 미채점 keep
  가능). fail-open은 문서 단위다: 그 문서의 호출 오류·쓸 수 없는 답은 그 문서만 keep하고 나머지는 정상 판정한다.
  입력 오류(`FilterInputError`, 호출 전)·생성자 인자 오류·호출자가 넘긴 tokenizer/logger 예외만 호출 전체를
  실패시킨다. threshold 정책에서 모든 문서가 drop되면 `selected_ids == []`(오류 아님)이며, 대체 문서·최소 keep·재검색은
  없고 처리 방법은 호출자가 정한다. 비용 상한·제출 상한은 인스턴스 전체에 걸치며, 비용 불명·미정산 job·계약/설정 실패·상한에서 추가 제출을
  멈춘다. 동시성 기본 1, 상한 있는 동시성은 opt-in이며 fake barrier로만 검사했다(지연 개선 주장 없음). 원문 보관·복구는
  호출자 책임이다(`select_originals`). 로그는 문서 수·텍스트 bytes(토크나이저를 주면 토큰 수와 이름·버전)만 남기는
  로컬 추정이며 과금 usage나 절감액이 아니다. 질문·원문·id·키는 기본 로그에 없다. 필터 호출 단가는 사용자가 설정한
  가정이며 실제로 띄운 모델과 별개다.
- **preset `v1-thr0.3-dev`:** 영어 V1 지시문·선택지, 실험 때의 문서 템플릿, `threshold(0.3)`, 시험한 크기 범위를 묶은
  opt-in 이름이다. 25문항에서 한 번 검증한 개발 후보이며 품질 보장이 아니다. 근거 자료는 비공개 연구 문서에 있고
  패키지에 넣지 않았다. 기본값이 아니며 결과는 사용자가 띄운 모델에 따라 달라진다.
- **한계:** 품질 우위·절감 보장·범용 host 정책 보존을 주장하지 않는다. 이 패키지의 검사는 fake transport만
  쓴다. 과거 16회 실제 smoke는 실험 버전에 대한 결과이며 이 패키지의 실제 실행 결과가 아니다. fake에서 두 방향
  결과가 같다는 것은 실제 모델의 순서 불변성 증거가 아니다. 비용은 클라이언트 추정이며 청구액이 아니다. plugin은
  `claude plugin validate --strict`와 서버 formatter로 재구성한 fixture 검사를 통과했다. 실제 실행은 내장 WebSearch 1회
  (통과만 확인)와 Tavily 1회(필터 호출 5회, 전부 keep)다. context filter는 commit `331081c`에서 실제 endpoint로 30회
  smoke(합성 문서, 계약·예산·순서·동시성 확인, 품질 주장 없음)를 했고, pairwise judge는 이 패키지로 실제 실행한 적이 없다.

## Citation

If you use this software, please cite the repository:

```bibtex
@software{judge_adapter_2026,
  author  = {hundredeuk2},
  title   = {judge-adapter: context filtering and pairwise LLM-judge calls},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/hundredeuk2/judge-adapter}
}
```

## License

Apache-2.0. See `LICENSE` and `NOTICE` (the default pairwise judge wording is adapted from FastChat's MT-Bench judge
prompts).
