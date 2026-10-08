# Copyright 2026 hundredeuk2
# SPDX-License-Identifier: Apache-2.0
"""Credential loading and scrubbing.

Key sources, in this order of precedence:
- an env file passed EXPLICITLY (``AdapterConfig.env_file`` / ``--env-file``): only the ``NAME=`` line is parsed;
- otherwise the process environment variable ``NAME`` (``ANTHROPIC_API_KEY`` or ``RUNPOD_API_KEY``).
There is no default file path. The key is read only when a real backend sends its first request (never at import,
construction, validation or in a fake run). It is passed only in a request header and is never printed, logged,
stored or put in an exception.
"""
from __future__ import annotations

import os
import re


class CredentialMissing(RuntimeError):
    pass


def dotenv_key(path, name):
    """Parse ONLY the `name=` line of the file. Raises CredentialMissing without any value."""
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        raise CredentialMissing('env file not readable') from None
    for line in lines:
        s = line.strip()
        if s.startswith('export '):
            s = s[len('export '):].lstrip()
        if s.startswith(name + '='):
            v = s[len(name) + 1:].strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
                v = v[1:-1]
            if v:
                return v
    raise CredentialMissing(f'{name} line absent or empty in the env file')


def env_key(name):
    """Read the environment variable ``name``. Raises CredentialMissing without any value."""
    v = os.environ.get(name, '').strip()
    if not v:
        raise CredentialMissing(f'environment variable {name} is not set')
    return v


def dotenv_provider(path, name):
    """A lazy key provider for an explicit env file: nothing is read until the provider is called."""
    return lambda: dotenv_key(path, name)


def env_provider(name):
    """A lazy key provider for an environment variable: nothing is read until the provider is called."""
    return lambda: env_key(name)


def default_provider(env_file, name):
    """Explicit env file if given, else the environment variable. Lazy in both cases."""
    return dotenv_provider(env_file, name) if env_file else env_provider(name)


_BEARER_RE = re.compile(r'(?i)(bearer\s+)[A-Za-z0-9._\-]+')
_KEYLIKE_RE = re.compile(r'\b(rpa_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9_\-]{8,})\b')


class Scrubber:
    """Removes the exact key value plus bearer-token and key-like substrings from every stored string."""

    def __init__(self, secret=None):
        self._secret = secret

    def set_secret(self, secret):
        self._secret = secret

    def text(self, s):
        s = str(s)
        if self._secret:
            s = s.replace(self._secret, '[redacted]')
        s = _BEARER_RE.sub(r'\1[redacted]', s)
        return _KEYLIKE_RE.sub('[redacted]', s)

    def obj(self, o):
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, dict):
            return {self.text(k): self.obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.obj(v) for v in o]
        return o
