"""Model-free test suite. Every test runs with the key and endpoint environment variables removed, so no test can
read a real key or address a real endpoint even if the developer's shell has them set."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from judge_adapter import keys as K

SECRET_ENV = ('ANTHROPIC_API_KEY', 'RUNPOD_API_KEY', 'JUDGE_ADAPTER_CHOICE_ENDPOINT_ID')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path_factory):
    for name in SECRET_ENV:
        monkeypatch.delenv(name, raising=False)
    # The davi toggle file and prompt state default to the user's real plugin data / temp directory: point them at a
    # fresh per-test directory so no test reads or writes the developer's own files.
    from judge_adapter import prompt_signal as PS
    from judge_adapter import toggles as T
    data = tmp_path_factory.mktemp('plugin-data')
    monkeypatch.setattr(T, 'default_data_dir', lambda home=None: str(data))
    real_state_dir = PS.default_state_dir
    monkeypatch.setattr(PS, 'default_state_dir', lambda env: real_state_dir(env) if env.get('CLAUDE_PLUGIN_DATA')
                        else str(data / PS.STATE_DIRNAME))


def block_key_reads(monkeypatch):
    """Any key read (explicit env file or environment variable) fails the test."""
    def boom(*a, **k):
        raise AssertionError('key read attempted')
    monkeypatch.setattr(K, 'dotenv_key', boom)
    monkeypatch.setattr(K, 'env_key', boom)


def fastchat_root():
    """FastChat for the optional host tests, which need the llm_judge data files: env JUDGE_ADAPTER_FASTCHAT_ROOT (a
    source checkout), else the installed fastchat package (None) if it contains llm_judge/data. False otherwise (the
    pip-installed FastChat has no data files; tests/test_reference_mapping.py covers the mapping without them)."""
    root = os.environ.get('JUDGE_ADAPTER_FASTCHAT_ROOT')
    if root:
        return root if (Path(root) / 'fastchat' / 'llm_judge' / 'data' / 'judge_prompts.jsonl').is_file() else False
    spec = importlib.util.find_spec('fastchat')
    if spec is None or not spec.origin:
        return False
    return None if (Path(spec.origin).parent / 'llm_judge' / 'data' / 'judge_prompts.jsonl').is_file() else False
