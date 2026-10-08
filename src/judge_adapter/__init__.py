# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""judge-adapter: a pairwise LLM-judge call adapter with fixed options, retries 0 and run-wide client-side limits.

Backends: ``haiku_direct`` (default; Anthropic Messages API) and ``choice_endpoint`` (a user-served choice-judge
worker on RunPod serverless, configured explicitly).
Transport ``fake`` runs fully in-process (no key, no network). See README.md.
"""
__version__ = '0.1.0'

from .contract import AdapterConfig, JudgeRequest, JudgeResult, Response  # noqa: E402,F401
from .core import JudgeAdapter, build_backend  # noqa: E402,F401
from .gate import RunGate  # noqa: E402,F401

__all__ = ['AdapterConfig', 'JudgeAdapter', 'JudgeRequest', 'JudgeResult', 'Response', 'RunGate', 'build_backend',
           '__version__']
