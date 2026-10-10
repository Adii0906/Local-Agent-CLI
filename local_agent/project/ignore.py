"""`.gitignore`-compatible ignore rules and a workspace walker.

Supported syntax: comments, negation (``!``), directory-only patterns (trailing
``/``), anchored patterns (leading or embedded ``/``), ``*``, ``?``, ``[...]``
and ``**``. Nested ``.gitignore`` files are honoured, with deeper files taking
precedence. ``.local-agentignore`` at the root adds agent-only exclusions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_IGNORES = (
    ".git/", "node_modules/", "__pycache__/", ".venv/", "venv/", ".tox/", ".mypy_cache/",
    ".pytest_cache/", ".ruff_cache/", ".idea/", ".vscode/", "dist/", "build/", "target/",
    ".next/", ".nuxt/", "coverage/", ".gradle/", "*.pyc", "*.pyo", ".DS_Store", "*.egg-info/",
)


@dataclass
class _Rule:
    regex: "re.Pattern[str]"
    negate: bool
    dir_only: bool


def _translate(pattern: str) -> str:
    out: List[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pattern[i]))
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def parse_rules(lines: Iterable[str]) -> List[_Rule]:
    rules: List[_Rule] = []
    for raw in lines:
        line = raw.rstrip("\n\r")
        if not line.strip() or line.startswith("#"):
            continue
        line = line.rstrip()
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if not line:
            continue
        anchored = "/" in line
        line = line.lstrip("/")
        body = _translate(line)
        regex = "^" + body + "$" if anchored else "^(?:.*/)?" + body + "$"
        rules.append(_Rule(re.compile(regex), negate, dir_only))
    return rules


class IgnoreRules:
    def __init__(self, root: Path, extra: Sequence[str] = ()) -> None:
        self.root = root
        base = list(DEFAULT_IGNORES) + list(extra)
        agent_ignore = root / ".local-agentignore"
        if agent_ignore.is_file():
            try:
                base += agent_ignore.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                pass
        self._base = parse_rules(base)
        self._dir_rules: Dict[str, List[_Rule]] = {}

    def _rules_for(self, rel_dir: str) -> List[_Rule]:
        if rel_dir not in self._dir_rules:
            path = self.root / rel_dir / ".gitignore" if rel_dir else self.root / ".gitignore"
            rules: List[_Rule] = []
            if path.is_file():
                try:
                    rules = parse_rules(path.read_text(encoding="utf-8", errors="replace").splitlines())
                except OSError:
                    rules = []
            self._dir_rules[rel_dir] = rules
        return self._dir_rules[rel_dir]

    def is_ignored(self, rel_path: str, is_dir: bool) -> bool:
        rel_path = rel_path.replace("\\", "/").strip("/")
        if not rel_path or rel_path == ".":
            return False
        ignored = False
        layers: List[Tuple[str, List[_Rule]]] = [("", self._base), ("", self._rules_for(""))]
        parts = rel_path.split("/")
        for depth in range(1, len(parts)):
            prefix = "/".join(parts[:depth])
            layers.append((prefix, self._rules_for(prefix)))
        for prefix, rules in layers:
            sub = rel_path[len(prefix) + 1 :] if prefix else rel_path
            for rule in rules:
                if rule.dir_only and not is_dir:
                    continue
                if rule.regex.match(sub):
                    ignored = not rule.negate
        return ignored

    def is_path_ignored(self, rel_path: str) -> bool:
        """True if the file or any of its ancestor directories is ignored."""
        parts = rel_path.replace("\\", "/").strip("/").split("/")
        for depth in range(1, len(parts)):
            if self.is_ignored("/".join(parts[:depth]), True):
                return True
        return self.is_ignored(rel_path, False)


def walk(
    root: Path,
    rules: IgnoreRules,
    start: Optional[Path] = None,
    *,
    max_depth: Optional[int] = None,
    include_dirs: bool = False,
    limit: Optional[int] = None,
) -> Iterator[Tuple[str, bool]]:
    """Yield ``(relative_posix_path, is_dir)`` for non-ignored entries.

    Symlinked directories are not followed, which avoids cycles and escapes.
    """
    start = start or root
    count = 0
    stack: List[Tuple[Path, int]] = [(start, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            continue
        subdirs: List[Tuple[Path, int]] = []
        for entry in entries:
            rel = Path(os.path.relpath(entry.path, root)).as_posix()
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if rules.is_ignored(rel, is_dir):
                continue
            if is_dir:
                if include_dirs:
                    yield rel, True
                    count += 1
                if max_depth is None or depth + 1 < max_depth:
                    subdirs.append((Path(entry.path), depth + 1))
            else:
                yield rel, False
                count += 1
            if limit is not None and count >= limit:
                return
        stack.extend(reversed(subdirs))
