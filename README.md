# judge-adapter

A small adapter for **pairwise LLM-judge calls**: give it one question (one or two turns) and two responses, and it
returns `A`, `B`, `tie` or an explicit `error`, with a trace of every call. It has a JSONL command-line tool, a
Python API and an optional binding for FastChat `llm_judge` (MT-Bench style pairwise judging).

| | Name |
|---|---|
| Distribution (pip) | `judge-adapter` |
| Import | `judge_adapter` |
| JSONL CLI | `judge-adapter` (also `python -m judge_adapter.cli`) |
| FastChat wrapper CLI (optional) | `judge-adapter-fastchat` (needs the `fastchat` extra) |
| License | Apache-2.0 (see `LICENSE` and `NOTICE`) |

What it keeps fixed on purpose:

- **Fake by default, real by opt-in.** `--transport` has no default. A real (paid) CLI run needs `--allow-real`,
  `--max-submissions` and `--budget-cap-usd`.
- **Retries 0, fallback 0.** One model submission per call. A failure is reported, never turned into a tie or
  retried on another backend.
- **Run-wide limits.** `--max-submissions` and a client-side cost cap for the whole run (not reset per request).
- **Unknown cost stays unknown.** A submitted call without usage/timing has cost `null`, not `0`.
- **Failures before transmission send nothing.** A missing key, missing choice endpoint ID or rate, or a client setup error is
  `local_config` with 0 submissions, and the run stops.
- **Keys are never written.** Raw HTTP bodies and logs go through a scrubber (exact key value, `Bearer ...`,
  `sk-...`, `rpa_...`).
- **No silent overwrite.** An output directory with results is refused unless `--overwrite`.
- **Rubric / reference are recorded.** Every result carries the prompt version and the sha256 of the rubric and
  reference that were sent.

## Install

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

## Quick start (no key, no network)

```bash
judge-adapter examples                                   # paths of the bundled synthetic examples
EX=$(judge-adapter examples | grep pairs.example.jsonl)
judge-adapter validate "$EX" --mode two_direction        # no call, no key read
judge-adapter run "$EX" --out ./run-fake --transport fake --mode two_direction
judge-adapter run "$EX" --out ./run-fake-choice --transport fake --backend choice_endpoint
```

A fake run prints a `[MOCK]` summary and writes `results.jsonl`, `summary.json`, `gate.json` and `trace/`. The fake
judge ignores answer quality; it follows test markers only (see [Fake transport](#fake-transport)).

## JSONL input

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

### Rubric and reference (prompt `user-criteria/v1`)

- Without `rubric` and `reference`, the fixed default prompt (`default-fixed/v1`) is sent byte for byte.
- `rubric` replaces the general quality standard ("helpfulness, relevance, accuracy, ...") instead of being appended
  to it. `reference` adds a reference-answer field that the instructions describe as a guide, not a candidate.
- Fixed on both paths: the pairwise framing, the bias sentence, options `opt_1` / `opt_2` / `opt_tie`, the strict
  parser and the generation settings. Custom option ids or score scales are not supported.
- The reference block is not tied to a slot, so it does not move when A/B are swapped.
- The field separation does **not** guarantee that a model follows the rubric or resists prompt injection. The
  recorded hashes show what was sent, not that it was followed.

## CLI

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
log), `trace/judge_trace.jsonl`, `trace/judge_raw.jsonl` (raw HTTP bodies, keys removed). No success rate is
reported; every request is counted, including errors and unrun requests.

## Python API

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

## Backends and configuration

| Backend | Endpoint | Key | Output |
|---|---|---|---|
| `haiku_direct` (default) | Anthropic Messages API, model `claude-haiku-4-5-20251001`, `max_tokens` 8, temperature 0, no system prompt | `ANTHROPIC_API_KEY` | Text that must be exactly `opt_1`, `opt_2` or `opt_tie` |
| `choice_endpoint` | `https://api.runpod.ai/v2/<endpoint-id>/runsync`: a choice-judge worker that you serve on RunPod serverless | `RUNPOD_API_KEY` | Probability per option (see contract below) |

Keys: from the environment variable, or from an env file only if you pass one (`--env-file` /
`AdapterConfig.env_file`). There is no default file path. A key is read only at the first real request, never at
import, validation or in a fake run.

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
sends one `POST /cancel/<job>`. Anything else is a contract / format failure.

## Cost estimates (read before a real run)

- All cost numbers are **client-side estimates, not the bill**. The provider console is the source of truth. The
  cap is not an account billing limit.
- Haiku: usage x list price ($1 / $5 per MTok, as checked 2026-09; check the current price). The pre-call
  reservation is (body bytes + 200) x 2 x input price + 8 output tokens.
- choice_endpoint: server `executionTime` x the per-second rate **you configure** (no built-in rate; the package
  does not know your hardware or price). Excludes cold start, idle timeout and queue billing. The cap reserves
  661 s (ttl 600 s + idle 60 s + 1 s) per open job plus 60 s idle once, so the first job needs a cap of at least
  `(661 + 60) x rate` USD, and the first two_direction request `(2 x 661 + 60) x rate` USD.
- Failed requests are included. A submitted call with unknown usage keeps its full reservation and stops the run.

## FastChat binding (optional)

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

### Reference categories (explicit mapping)

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

## Fake transport

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

Usage numbers in fake runs are invented. Identical fake results in both directions show that ids are restored
across the swap; they are **not evidence that a real model is order-invariant**.

## Tests

```bash
pip install -e ".[test]"
pytest                                   # model-free; FastChat tests are skipped
JUDGE_ADAPTER_FASTCHAT_ROOT=/path/to/FastChat pytest   # also runs the FastChat host tests (needs the extra's deps)
```

All tests use fake transports; network sockets and key reads are blocked, and the key / endpoint environment
variables are removed for each test.

## Limits

- **No quality claim.** Nothing here shows that a backend judges well or better than another, or that a model
  follows a rubric or reference.
- **No guaranteed savings.** Cost numbers are estimates; the cap is client-side.
- **No claim of general host-policy preservation.** The FastChat binding replaces the host's judge prompts with this
  package's fixed prompts; it does not reproduce them.
- **Tests are mock tests.** This package's test suite runs only against fake transports. An earlier experiment
  version of this code made a 16-call real smoke run (8 Haiku, 8 choice-endpoint calls) through FastChat; that result applies to that
  experiment version, **not to this package**. This package itself has not been run against a real model or API.
- Requests run sequentially; no parallelism, no resume. Two turns at most. Custom option ids and score scales are
  not supported.

## 한국어 요약

- **무엇인가:** 질문 하나(1~2 turn)와 응답 두 개를 넣으면 pairwise judge 결과(`A`/`B`/`tie`/명시 `error`)와 호출
  기록을 남기는 어댑터다. JSONL CLI(`judge-adapter`), Python API(`import judge_adapter`), 선택 FastChat binding
  (`judge-adapter-fastchat`, extra `fastchat`)을 제공한다. 배포 이름 `judge-adapter`, 라이선스 Apache-2.0.
- **설치:** `pip install "judge-adapter @ git+https://github.com/hundredeuk2/judge-adapter"`. 코어 의존성은 `httpx`
  하나다. FastChat은 `[fastchat]` extra이며 commit `587d5cfa`로 고정하고 torch 등 무거운 의존성을 함께 설치한다.
- **유지하는 동작:** 기본 fake·real은 명시 opt-in(`--allow-real` + `--max-submissions` + `--budget-cap-usd`),
  재시도 0, 실행 전체 제출·비용 상한, 비용 불명은 `null`, 전송 전 오류는 제출 0(`local_config`), raw·log 키 제거,
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
- **한계:** 품질 우위·절감 보장·범용 host 정책 보존을 주장하지 않는다. 이 패키지의 검사는 fake transport만
  쓴다. 과거 16회 실제 smoke는 실험 버전에 대한 결과이며 이 패키지의 실제 실행 결과가 아니다. fake에서 두 방향
  결과가 같다는 것은 실제 모델의 순서 불변성 증거가 아니다. 비용은 클라이언트 추정이며 청구액이 아니다.
