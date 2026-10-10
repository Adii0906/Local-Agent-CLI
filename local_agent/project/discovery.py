"""Detect a project's languages, frameworks, package managers and commands.

Detection only reads marker files. The commands it suggests (test, lint,
build) are hints for the model and the user. They are never run automatically
and never pre-approved: running one still goes through the permission engine.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from local_agent.project.ignore import IgnoreRules, walk

_EXT_LANG = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#", ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++",
    ".swift": "Swift", ".scala": "Scala", ".sh": "Shell", ".html": "HTML", ".css": "CSS", ".scss": "CSS",
    ".vue": "Vue", ".svelte": "Svelte", ".sql": "SQL", ".dart": "Dart", ".ex": "Elixir", ".lua": "Lua",
}

INSTRUCTION_FILES = ("AGENTS.md", "LOCAL-AGENT.md")


@dataclass
class ProjectInfo:
    root: Path
    languages: List[str] = field(default_factory=list)
    file_counts: Dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    frameworks: List[str] = field(default_factory=list)
    package_managers: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    test_commands: List[str] = field(default_factory=list)
    lint_commands: List[str] = field(default_factory=list)
    build_commands: List[str] = field(default_factory=list)
    git_branch: Optional[str] = None
    is_git: bool = False
    instruction_file: Optional[str] = None

    def to_text(self) -> str:
        lines = [f"Project root: {self.root}"]
        if self.languages:
            counts = ", ".join(f"{lang} ({self.file_counts.get(lang, 0)})" for lang in self.languages[:6])
            lines.append(f"Languages (files): {counts}; {self.total_files} files scanned")
        for label, items in (
            ("Frameworks/tools", self.frameworks),
            ("Package managers", self.package_managers),
            ("Config files", self.config_files),
            ("Likely entry points", self.entry_points),
            ("Suggested test commands", self.test_commands),
            ("Suggested lint commands", self.lint_commands),
            ("Suggested build commands", self.build_commands),
        ):
            if items:
                lines.append(f"{label}: {', '.join(items)}")
        if self.is_git:
            lines.append(f"Git: yes (branch {self.git_branch or 'unknown'})")
        else:
            lines.append("Git: no (undo still works for files the agent changes)")
        if self.instruction_file:
            lines.append(f"Project instruction file: {self.instruction_file}")
        return "\n".join(lines)


def _read(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def _add(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _git_branch(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def discover(root: Path, rules: Optional[IgnoreRules] = None, scan_limit: int = 5000) -> ProjectInfo:
    rules = rules or IgnoreRules(root)
    info = ProjectInfo(root=root)
    exists = lambda name: (root / name).exists()  # noqa: E731

    counts: Counter = Counter()
    for rel, _ in walk(root, rules, limit=scan_limit):
        info.total_files += 1
        lang = _EXT_LANG.get(Path(rel).suffix.lower())
        if lang:
            counts[lang] += 1
    info.file_counts = dict(counts)
    info.languages = [lang for lang, _ in counts.most_common()]

    markers = [
        "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "package.json", "tsconfig.json",
        "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "Gemfile", "composer.json",
        "Makefile", "Dockerfile", "docker-compose.yml", "pytest.ini", "tox.ini", ".eslintrc.json",
        "vite.config.ts", "vite.config.js", "next.config.js", "next.config.mjs", "jest.config.js",
    ]
    info.config_files = [m for m in markers if exists(m)]

    # Python
    if any(exists(m) for m in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile")):
        if exists("poetry.lock"):
            _add(info.package_managers, "poetry")
        elif exists("uv.lock"):
            _add(info.package_managers, "uv")
        elif exists("Pipfile"):
            _add(info.package_managers, "pipenv")
        else:
            _add(info.package_managers, "pip")
        py_text = _read(root / "pyproject.toml") + _read(root / "requirements.txt") + _read(root / "setup.py")
        lowered = py_text.lower()
        for name in ("django", "flask", "fastapi", "pytest", "ruff", "mypy", "black"):
            if name in lowered:
                _add(info.frameworks, name)
        if exists("manage.py"):
            _add(info.frameworks, "django")
        if (
            "pytest" in lowered
            or exists("pytest.ini")
            or exists("conftest.py")
            or (root / "tests").is_dir()
        ):
            _add(info.test_commands, "python -m pytest")
        elif (root / "tests").is_dir() or (root / "test").is_dir():
            _add(info.test_commands, "python -m unittest discover")
        if "ruff" in lowered:
            _add(info.lint_commands, "ruff check .")
        if "mypy" in lowered:
            _add(info.lint_commands, "mypy .")
    # Node
    if exists("package.json"):
        if exists("pnpm-lock.yaml"):
            pm = "pnpm"
        elif exists("yarn.lock"):
            pm = "yarn"
        elif exists("bun.lockb") or exists("bun.lock"):
            pm = "bun"
        else:
            pm = "npm"
        _add(info.package_managers, pm)
        try:
            pkg = json.loads(_read(root / "package.json"))
        except json.JSONDecodeError:
            pkg = {}
        if isinstance(pkg, dict):
            deps = {}
            for key in ("dependencies", "devDependencies"):
                if isinstance(pkg.get(key), dict):
                    deps.update(pkg[key])
            for name in ("next", "react", "vue", "svelte", "express", "vite", "jest", "vitest", "typescript", "eslint"):
                if name in deps:
                    _add(info.frameworks, name)
            scripts = pkg.get("scripts") if isinstance(pkg.get("scripts"), dict) else {}
            run = "npm run" if pm == "npm" else pm
            if "test" in scripts:
                _add(info.test_commands, f"{pm} test")
            if "lint" in scripts:
                _add(info.lint_commands, f"{run} lint")
            if "typecheck" in scripts:
                _add(info.lint_commands, f"{run} typecheck")
            if "build" in scripts:
                _add(info.build_commands, f"{run} build")
            if isinstance(pkg.get("main"), str):
                _add(info.entry_points, pkg["main"])
    # Others
    if exists("Cargo.toml"):
        _add(info.package_managers, "cargo")
        _add(info.test_commands, "cargo test")
        _add(info.build_commands, "cargo build")
        _add(info.lint_commands, "cargo clippy")
    if exists("go.mod"):
        _add(info.package_managers, "go modules")
        _add(info.test_commands, "go test ./...")
        _add(info.build_commands, "go build ./...")
        _add(info.lint_commands, "go vet ./...")
    if exists("pom.xml"):
        _add(info.package_managers, "maven")
        _add(info.test_commands, "mvn test")
    if exists("build.gradle") or exists("build.gradle.kts"):
        _add(info.package_managers, "gradle")
        _add(info.test_commands, "./gradlew test" if exists("gradlew") else "gradle test")
    if exists("Gemfile"):
        _add(info.package_managers, "bundler")
    if exists("composer.json"):
        _add(info.package_managers, "composer")
    if exists("Makefile"):
        targets = {line.split(":")[0].strip() for line in _read(root / "Makefile").splitlines() if ":" in line}
        if "test" in targets:
            _add(info.test_commands, "make test")
        if "lint" in targets:
            _add(info.lint_commands, "make lint")

    for candidate in (
        "main.py", "app.py", "manage.py", "cli.py", "__main__.py", "src/main.py", "src/index.ts", "src/index.js",
        "src/main.ts", "src/main.rs", "src/lib.rs", "index.js", "server.js", "main.go", "cmd",
    ):
        if exists(candidate):
            _add(info.entry_points, candidate)

    info.is_git = (root / ".git").exists()
    if info.is_git:
        info.git_branch = _git_branch(root)
    for name in INSTRUCTION_FILES:
        if (root / name).is_file():
            info.instruction_file = name
            break
    return info


def read_instruction_file(info: ProjectInfo, max_chars: int = 4000) -> str:
    if not info.instruction_file:
        return ""
    return _read(info.root / info.instruction_file, max_chars)
