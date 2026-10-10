"""Read-only git tools plus helpers used for checkpoints and the final report."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent.security.secrets import redact
from local_agent.tools.base import Assessment, Risk, Tool, ToolContext, ToolResult


def _git(root: Path, *args: str, timeout: int = 20) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def is_git_repo(root: Path) -> bool:
    proc = _git(root, "rev-parse", "--is-inside-work-tree")
    return bool(proc and proc.returncode == 0 and proc.stdout.strip() == "true")


def porcelain_status(root: Path) -> Optional[Dict[str, str]]:
    """Map of path -> two-letter status, or None if not a git repo."""
    proc = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if not proc or proc.returncode != 0:
        return None
    result: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        result[path.strip('"')] = line[:2]
    return result


def head_commit(root: Path) -> Optional[str]:
    proc = _git(root, "rev-parse", "HEAD")
    return proc.stdout.strip() if proc and proc.returncode == 0 else None


class GitStatus(Tool):
    name = "git_status"
    description = "Show the git branch and working-tree status (modified, staged and untracked files)."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(Risk.READ, "git status")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        proc = _git(ctx.workspace.root, "status", "--short", "--branch")
        if proc is None:
            return ToolResult.error("git is not installed or not runnable")
        if proc.returncode != 0:
            return ToolResult.error(proc.stderr.strip() or "not a git repository")
        return ToolResult(True, proc.stdout.strip() or "(clean)", summary="git status")


class GitDiff(Tool):
    name = "git_diff"
    description = "Show the git diff of the working tree (or staged changes), optionally for one path."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}},
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        paths: List[str] = []
        if args.get("path"):
            paths = [ctx.workspace.relative(ctx.workspace.resolve(args["path"]))]
        return Assessment(Risk.READ, "git diff" + (f" -- {paths[0]}" if paths else ""), paths)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        cmd = ["diff", "--no-color", "--no-ext-diff"]
        if args.get("staged"):
            cmd.append("--staged")
        if args.get("path"):
            cmd += ["--", ctx.workspace.relative(ctx.workspace.resolve(args["path"]))]
        proc = _git(ctx.workspace.root, *cmd)
        if proc is None:
            return ToolResult.error("git is not installed or not runnable")
        if proc.returncode != 0:
            return ToolResult.error(proc.stderr.strip() or "git diff failed")
        out = redact(proc.stdout)
        limit = ctx.settings.max_output_chars
        if len(out) > limit:
            out = out[:limit] + f"\n... [diff truncated, {len(out) - limit} more characters; pass a path]"
        return ToolResult(True, out or "(no differences)", summary="git diff")
