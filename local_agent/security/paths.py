"""Workspace boundary enforcement and sensitive-file detection.

Every path a tool touches goes through ``Workspace.resolve``. The path is joined
to the workspace root, fully resolved (symlinks included) with ``realpath``, and
then checked for containment using path components, not string prefixes. A
symlink inside the workspace that points outside it is therefore rejected.

This is an application-level check. It is not an OS sandbox: shell commands the
user approves run with the user's own privileges (see docs/SECURITY.md).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath
from typing import Optional

from local_agent.errors import PathViolation

_SENSITIVE_NAMES = {
    ".npmrc", ".pypirc", ".netrc", "_netrc", ".git-credentials", ".htpasswd", ".pgpass",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "credentials.json",
    "secrets.json", "secrets.yaml", "secrets.yml", "secrets.toml", ".secrets",
    "terraform.tfstate", "terraform.tfstate.backup", "keystore.jks",
}
_SENSITIVE_GLOBS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.kdbx", "*.tfstate",
    "*.ovpn", "service-account*.json", "*_rsa", "*_ed25519", "*.asc",
)
_SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"}
_ENV_SAFE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".defaults")


def sensitive_reason(rel_path: str) -> Optional[str]:
    """Return why a workspace-relative path is sensitive, or None."""
    parts = PurePosixPath(rel_path.replace("\\", "/")).parts
    if not parts:
        return None
    name = parts[-1].lower()
    for part in parts[:-1]:
        if part.lower() in _SENSITIVE_DIRS:
            return f"inside credential directory '{part}'"
    if name == ".env" or (name.startswith(".env.") and not name.endswith(_ENV_SAFE_SUFFIXES)):
        return "environment file that typically contains secrets"
    if name.endswith(".env") and not name.endswith(_ENV_SAFE_SUFFIXES):
        return "environment file that typically contains secrets"
    if name in _SENSITIVE_NAMES:
        return "credential or key file"
    for pattern in _SENSITIVE_GLOBS:
        if fnmatch.fnmatch(name, pattern):
            return f"matches sensitive pattern '{pattern}'"
    return None


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_norm(str(path)), _norm(str(root))]) == _norm(str(root))
    except ValueError:  # different drives on Windows
        return False


class Workspace:
    """The directory the agent is authorized to operate in."""

    def __init__(self, root: Path) -> None:
        resolved = Path(os.path.realpath(str(root)))
        if not resolved.is_dir():
            raise PathViolation(f"Workspace {root} is not a directory")
        self.root = resolved

    def resolve(self, path_str: str, *, must_exist: bool = False) -> Path:
        if not isinstance(path_str, str) or not path_str.strip():
            raise PathViolation("Path must be a non-empty string")
        if "\x00" in path_str:
            raise PathViolation("Path contains a NUL byte")
        raw = path_str.strip()
        if raw.startswith("~"):
            raise PathViolation(
                f"'{raw}' refers to the home directory, which is outside the workspace. Use a workspace-relative path."
            )
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        real = Path(os.path.realpath(str(candidate)))
        if not is_within(real, self.root):
            raise PathViolation(f"'{raw}' resolves to {real}, which is outside the workspace {self.root}")
        if must_exist and not real.exists():
            raise PathViolation(f"'{raw}' does not exist")
        return real

    def relative(self, path: Path) -> str:
        rel = os.path.relpath(str(path), str(self.root))
        return "." if rel == "." else PurePosixPath(Path(rel).as_posix()).as_posix()

    def is_git_internal(self, path: Path) -> bool:
        rel = self.relative(path)
        return rel == ".git" or rel.startswith(".git/")

    def sensitive_reason(self, path: Path) -> Optional[str]:
        return sensitive_reason(self.relative(path))
