"""Secret detection and redaction.

Used before anything is persisted (memory, session transcripts, logs) and before
command output or environment values are shown. Pattern-based detection cannot
catch every secret. It is a safety net, not a guarantee.
"""

from __future__ import annotations

import re
from typing import List, Tuple

REDACTED = "[REDACTED]"

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("private key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)", re.S)),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("Slack token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("API key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("URL credentials", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b([A-Z0-9_.-]*(?:api[_-]?key|secret|token|passwd|password|auth|credential)[A-Z0-9_.-]*)"
            r"(\s*[:=]\s*)(['\"]?)([^\s'\"]{8,})\3"
        ),
    ),
]


def find_secrets(text: str) -> List[str]:
    """Return the kinds of secrets detected in ``text``."""
    if not text:
        return []
    return [kind for kind, pattern in _PATTERNS if pattern.search(text)]


def redact(text: str) -> str:
    if not text:
        return text
    for kind, pattern in _PATTERNS:
        if kind == "assigned secret":
            text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


_SECRET_ENV_NAME = re.compile(r"(?i)(key|token|secret|passw|credential|auth|cookie|session)")


def is_secret_env_name(name: str) -> bool:
    return bool(_SECRET_ENV_NAME.search(name))
