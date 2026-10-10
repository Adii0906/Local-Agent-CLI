"""Code search (ripgrep with a pure-Python fallback) and file finding.

Results never include the contents of sensitive files such as ``.env`` or
private keys. A match in one is reported only as a redacted path.
"""

from __future__ import annotations

import fnmatch
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from local_agent.project.ignore import walk
from local_agent.security.paths import sensitive_reason
from local_agent.tools.base import Assessment, Risk, Tool, ToolContext, ToolResult

MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024


def _ripgrep() -> Optional[str]:
    return shutil.which("rg")


class SearchCode(Tool):
    name = "search_code"
    description = (
        "Search file contents in the workspace (ripgrep when available). Returns path:line: text. "
        "Use this to locate definitions, usages and error messages before reading files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Text or regular expression to search for."},
            "path": {"type": "string", "description": "Directory or file to search (default '.')."},
            "regex": {"type": "boolean", "description": "Treat pattern as a regex (default false = literal)."},
            "case_sensitive": {"type": "boolean"},
            "glob": {"type": "string", "description": "Only search files matching this glob, e.g. '*.py'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        return Assessment(Risk.READ, f"Search for {args['pattern']!r} in {ctx.workspace.relative(path)}")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        if not pattern:
            return ToolResult.error("pattern must not be empty")
        if args.get("regex"):
            try:
                re.compile(pattern)
            except re.error as exc:
                return ToolResult.error(f"Invalid regular expression: {exc}")
        target = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        max_results = args.get("max_results", 100)
        rg = _ripgrep()
        try:
            matches, engine = (self._rg(rg, args, ctx, target), "ripgrep") if rg else (self._python(args, ctx, target), "python")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Search timed out after 30s; narrow the path or pattern")

        hidden_sensitive = 0
        visible: List[Tuple[str, int, str]] = []
        for rel, line_no, text in matches:
            if sensitive_reason(rel):
                hidden_sensitive += 1
                continue
            if ctx.ignore.is_path_ignored(rel):
                continue
            visible.append((rel, line_no, text))
        total = len(visible)
        lines = [f"{rel}:{n}: {text[:300]}" for rel, n, text in visible[:max_results]]
        note = []
        if total > max_results:
            note.append(f"[{total - max_results} more matches not shown; narrow the search]")
        if hidden_sensitive:
            note.append(f"[{hidden_sensitive} match(es) in sensitive files hidden]")
        body = "\n".join(lines + note) if lines or note else "No matches."
        return ToolResult(True, body, summary=f"Searched {pattern!r}: {total} match(es) ({engine})", data={"matches": total})

    def _rg(self, rg: str, args: Dict[str, Any], ctx: ToolContext, target) -> List[Tuple[str, int, str]]:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--with-filename", "--no-require-git",
               "--hidden", "--max-columns", "400", "--max-filesize", "2M", "-g", "!.git", "--max-count", "200"]
        if not args.get("case_sensitive"):
            cmd.append("-i")
        if not args.get("regex"):
            cmd.append("-F")
        if args.get("glob"):
            cmd += ["-g", args["glob"]]
        cmd += ["-e", args["pattern"], "--", str(target)]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=str(ctx.workspace.root))
        results: List[Tuple[str, int, str]] = []
        root = str(ctx.workspace.root)
        for line in proc.stdout.splitlines():
            # path:line:text  (path may contain ':' on Windows drive letters; split from the left after root)
            if line.startswith(root):
                line = line[len(root):].lstrip("/\\")
            parts = line.split(":", 2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            results.append((parts[0].replace("\\", "/"), int(parts[1]), parts[2].strip()))
            if len(results) >= 5000:
                break
        return results

    def _python(self, args: Dict[str, Any], ctx: ToolContext, target) -> List[Tuple[str, int, str]]:
        flags = 0 if args.get("case_sensitive") else re.IGNORECASE
        regex = re.compile(args["pattern"] if args.get("regex") else re.escape(args["pattern"]), flags)
        glob = args.get("glob")
        results: List[Tuple[str, int, str]] = []
        if target.is_file():
            files = [ctx.workspace.relative(target)]
        else:
            files = [rel for rel, _ in walk(ctx.workspace.root, ctx.ignore, target)]
        for rel in files:
            if ctx.cancel_event.is_set():
                break
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], glob)):
                continue
            path = ctx.workspace.root / rel
            try:
                if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                with open(path, "rb") as fh:
                    raw = fh.read()
            except OSError:
                continue
            if b"\x00" in raw[:8192]:
                continue
            for n, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append((rel, n, line.strip()))
                    if len(results) >= 5000:
                        return results
        return results


class FindFiles(Tool):
    name = "find_files"
    description = "Find files by name or glob pattern, e.g. '*.py', 'test_*', 'src/**/auth*'."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Directory to search (default '.')."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        return Assessment(Risk.READ, f"Find files matching {args['pattern']!r} in {ctx.workspace.relative(path)}")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        pattern = args["pattern"]
        limit = args.get("max_results", 200)
        found: List[str] = []
        for rel, _ in walk(ctx.workspace.root, ctx.ignore, target):
            name = rel.rsplit("/", 1)[-1]
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern) or (
                not any(ch in pattern for ch in "*?[") and pattern.lower() in rel.lower()
            ):
                found.append(rel)
                if len(found) > limit:
                    break
        lines = found[:limit]
        if len(found) > limit:
            lines.append(f"[more than {limit} results; refine the pattern]")
        return ToolResult(True, "\n".join(lines) if lines else "No files found.", summary=f"Found {min(len(found), limit)} file(s) for {pattern!r}")
