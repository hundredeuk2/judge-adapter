# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Adapter contract: request, result, config. Stdlib only.

A judge call compares two responses with stable host-owned IDs. The model never sees the IDs; they are restored after
the call. ``tie`` is a valid verdict. ``error`` is a failure with a reason; it is never turned into a tie or a score.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ('haiku_direct', 'choice_endpoint')
DEFAULT_BACKEND = 'haiku_direct'
MODES = ('single', 'two_direction')
TRANSPORTS = ('real', 'fake')

STATUS_OK, STATUS_TIE, STATUS_ERROR = 'ok', 'tie', 'error'
# failure reasons (JudgeResult.failure['reason'])
HTTP, TIMEOUT, FORMAT_ERROR, TRUNCATED, REFUSED = 'http', 'timeout', 'format_error', 'truncated', 'refused'
NO_DECISION, BACKEND_ERROR, CONTRACT, UNSUPPORTED_INPUT = 'no_decision', 'backend_error', 'contract', 'unsupported_input'
NOT_SUBMITTED = 'not_submitted'      # a pre-submission gate refused the call (no request sent)
LOCAL_CONFIG = 'local_config'        # failed before any HTTP transmission (key provider, missing key, client init);
                                     # submissions 0; the backend refuses every later call of the run
FAILURE_REASONS = (HTTP, TIMEOUT, FORMAT_ERROR, TRUNCATED, REFUSED, NO_DECISION, BACKEND_ERROR, CONTRACT,
                   UNSUPPORTED_INPUT, NOT_SUBMITTED, LOCAL_CONFIG)
TIE = 'tie'


@dataclass(frozen=True)
class Response:
    response_id: str                 # host-owned, stable; never sent to the model
    turns: tuple                     # this response's answers, turn 1 [, turn 2]


@dataclass(frozen=True)
class JudgeRequest:
    request_id: str
    question_turns: tuple            # user turns, turn 1 [, turn 2]
    response_a: Response             # host position A
    response_b: Response             # host position B
    turn: int = 1                    # judged turn (1 or 2); turn 2 keeps the turn-1 exchange as context
    instructions: str | None = None  # None -> the fixed default wording for the turn
    criteria: dict | None = None     # None -> fixed; keys must be exactly opt_1 / opt_2 / opt_tie
    reference: object = None         # host-format reference object: NOT supported -> explicit unsupported_input
    host: dict = field(default_factory=dict)   # host metadata, recorded in the trace only
    rubric: str | None = None        # user rubric: replaces the general quality standard (adapter/user_prompt.py)
    reference_turns: tuple | None = None   # user reference, one entry per question turn (str or None)


@dataclass
class CallRecord:
    call_id: str
    direction: str                   # 'AB' (slot 1 = A) or 'BA' (slot 1 = B)
    slot_ids: dict                   # {'1': response_id, '2': response_id}
    outcome: str                     # 'ok' or a failure reason
    label: str | None                # restored: response_id or 'tie'; None on failure
    detail: str | None
    submissions: int                 # model submissions (POST runsync / POST messages); retries are 0
    status_polls: int = 0            # choice_endpoint only: GET /status of the SAME job (not a resubmission)
    cancels: int = 0
    http_status: int | None = None
    client_seconds: float | None = None
    server_timing: dict | None = None    # choice_endpoint delay/execution/latency ms; never replaces client time
    usage: dict | None = None
    raw_output_sha256: str | None = None
    raw_output_pointer: dict | None = None
    response_model: str | None = None
    job_id: str | None = None
    job_status: str | None = None        # choice_endpoint: last OBSERVED job status
    job_final: bool | None = None        # choice_endpoint: True only if a terminal state was observed (not a cancel)
    gate: dict | None = None             # pre-submission gate record, when a gate wraps the backend


@dataclass
class JudgeResult:
    request_id: str
    status: str                      # ok | tie | error
    verdict: str | None              # 'A' | 'B' | 'tie' in HOST positions; None on error
    winner_id: str | None            # restored response id; None for tie / error
    tie_kind: str | None             # 'explicit' | 'inconsistent' | None
    failure: dict | None             # {'reason', 'detail', 'direction'} on error
    backend: str
    model: str
    mode: str
    transport: str
    calls: list
    submissions: int
    retries: int                     # always 0
    usage: dict | None
    cost_estimate: dict
    total_seconds: float
    semantic_sha256: str | None
    host: dict
    criteria_info: dict | None = None    # prompt version, effective rubric default/user + sha256, reference sha256

    def to_dict(self):
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


@dataclass(frozen=True)
class AdapterConfig:
    transport: str                   # 'real' or 'fake'; REQUIRED, no default (no accidental paid call)
    backend: str = DEFAULT_BACKEND
    mode: str = 'single'
    trace_dir: str = 'judge_adapter_trace'
    timeout_seconds: float = 120.0
    choice_endpoint_id: str | None = None    # RunPod endpoint ID; None -> env JUDGE_ADAPTER_CHOICE_ENDPOINT_ID; unset
                                             # -> every real choice_endpoint call is local_config, 0 transmissions
    choice_poll_interval_seconds: float = 5.0
    choice_max_status_polls: int = 120       # bounded also by the per-request ttl (600 s)
    choice_usd_per_second: float | None = None   # REQUIRED for a real choice_endpoint run (no built-in rate); unset ->
                                                 # local_config, 0 transmissions. Fake runs use a labeled placeholder
    choice_expected_model: str | None = None     # optional: if set, a different output.model is a contract failure
    env_file: str | None = None          # optional KEY=VALUE file, used only if given; None -> environment variables.
                                         # Either source is read ONLY by a real call, at its first request

    def __post_init__(self):
        if self.transport not in TRANSPORTS:
            raise ValueError(f'transport must be one of {TRANSPORTS} (explicit; no default)')
        if self.backend not in BACKENDS:
            raise ValueError(f'backend must be one of {BACKENDS}')
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f'unknown config keys {sorted(unknown)}')
        return cls(**data)
