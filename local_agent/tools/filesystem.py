"""Filesystem tools. All paths are resolved through ``Workspace``.

Writes are atomic (temp file + ``os.replace``) and snapshotted by the
``ChangeTracker`` first, so ``/undo`` can restore them.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from local_agent.errors import ToolError
from local_agent.project.ignore import walk
from local_agent.tools.base import Assessment, Risk, Tool, ToolContext, ToolResult

MAX_READ_LINES = 2000
DEFAULT_READ_LINES = 400
MAX_LINE_CHARS = 2000
MAX_READ_CHARS = 30000
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TRACKED_FILES = 1000


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return False


def _read_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"{path.name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB; refusing to load it whole")
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.local-agent-tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    if path.exists():
        try:
            os.chmod(tmp, path.stat().st_mode)
        except OSError:
            pass
    os.replace(tmp, path)


def _diff(before: str, after: str, rel: str, max_lines: int = 120) -> str:
    lines = list(
        difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=2)
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more diff lines)"]
    return "\n".join(lines)


def _write_risk(ctx: ToolContext, path: Path, exists: bool) -> Tuple[Risk, List[str], str]:
    warnings: List[str] = []
    if ctx.workspace.is_git_internal(path):
        return Risk.BLOCKED, warnings, "Writing inside .git is not permitted; use git commands instead."
    reason = ctx.workspace.sensitive_reason(path)
    if reason:
        warnings.append(f"Sensitive file: {reason}")
        return Risk.WRITE_SENSITIVE, warnings, ""
    return (Risk.EDIT if exists else Risk.CREATE), warnings, ""


def _file_signature(path: Path) -> Tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


class ListDirectory(Tool):
    name = "list_directory"
    description = (
        "List files and folders under a workspace directory (respects .gitignore). "
        "Use depth to control recursion. Prefer this over shell `ls`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory (default '.')."},
            "depth": {"type": "integer", "minimum": 1, "maximum": 6, "description": "Recursion depth (default 2)."},
        },
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        return Assessment(Risk.READ, f"List {ctx.workspace.relative(path)}", [ctx.workspace.relative(path)])

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args.get("path", "."), must_exist=True)
        if not path.is_dir():
            return ToolResult.error(f"{args.get('path')} is not a directory")
        limit = 400
        entries = list(walk(ctx.workspace.root, ctx.ignore, path, max_depth=args.get("depth", 2), include_dirs=True, limit=limit + 1))
        lines = []
        for rel, is_dir in entries[:limit]:
            if is_dir:
                lines.append(rel + "/")
            else:
                try:
                    size = (ctx.workspace.root / rel).stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{rel}  ({size} B)")
        if len(entries) > limit:
            lines.append(f"... truncated at {limit} entries; list a subdirectory for more")
        rel_root = ctx.workspace.relative(path)
        body = "\n".join(lines) if lines else "(empty, or everything is ignored)"
        return ToolResult(True, f"Contents of {rel_root}:\n{body}", summary=f"Listed {rel_root} ({min(len(entries), limit)} entries)")


class ReadFile(Tool):
    name = "read_file"
    description = (
        "Read a text file with line numbers. Large files are paged: pass start_line and max_lines to read "
        "further. Always read a file before editing it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1, "description": "1-based first line (default 1)."},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LINES},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args["path"], must_exist=True)
        rel = ctx.workspace.relative(path)
        reason = ctx.workspace.sensitive_reason(path)
        if reason:
            return Assessment(
                Risk.READ_SENSITIVE, f"Read {rel}", [rel], warnings=[f"Sensitive file: {reason}. Its contents would be sent to the model."]
            )
        return Assessment(Risk.READ, f"Read {rel}", [rel])

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"], must_exist=True)
        rel = ctx.workspace.relative(path)
        if path.is_dir():
            return ToolResult.error(f"{rel} is a directory; use list_directory")
        if _is_binary(path):
            return ToolResult.error(f"{rel} appears to be a binary file ({path.stat().st_size} bytes)")
        start = args.get("start_line", 1)
        count = args.get("max_lines", DEFAULT_READ_LINES)
        signature = _file_signature(path)
        previous = ctx.reads.get(rel)
        if (
            previous
            and previous.get("in_context")
            and previous.get("sig") == signature
            and previous.get("range") == (start, count)
        ):
            return ToolResult(
                True,
                f"{rel} is unchanged since you last read lines {start}-{start + count - 1}; that content is already "
                "in the conversation above.",
                summary=f"{rel} unchanged (not re-sent)",
            )
        out: List[str] = []
        total = 0
        budget = MAX_READ_CHARS
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for total, line in enumerate(fh, 1):
                if start <= total < start + count:
                    line = line.rstrip("\n").rstrip("\r")
                    if len(line) > MAX_LINE_CHARS:
                        line = line[:MAX_LINE_CHARS] + " ...[line truncated]"
                    budget -= len(line) + 8
                    if budget < 0:
                        count = total - start  # stop here; report the real range below
                        continue
                    out.append(f"{total:>6}\t{line}")
        ctx.reads[rel] = {"sig": signature, "range": (start, count), "in_context": True}
        if total == 0:
            return ToolResult(True, f"{rel} is empty.", summary=f"Read {rel} (empty)")
        if start > total:
            return ToolResult.error(f"start_line {start} is past the end of {rel} ({total} lines)")
        end = min(total, start + count - 1)
        header = f"{rel} (lines {start}-{end} of {total})"
        footer = f"\n[{total - end} more lines; call read_file with start_line={end + 1}]" if end < total else ""
        return ToolResult(True, header + "\n" + "\n".join(out) + footer, summary=f"Read {rel} lines {start}-{end} of {total}")


class WriteFile(Tool):
    name = "write_file"
    description = (
        "Create a new file or completely replace an existing one. For small changes to existing files use "
        "edit_file instead. Parent directories are created automatically."
    )
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args["path"])
        rel = ctx.workspace.relative(path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory")
        exists = path.exists()
        risk, warnings, block = _write_risk(ctx, path, exists)
        if exists:
            details = _diff(_read_text(path), args["content"], rel)
            summary = f"Overwrite {rel}"
        else:
            preview = args["content"].splitlines()
            details = "\n".join(f"+{line}" for line in preview[:60])
            if len(preview) > 60:
                details += f"\n... ({len(preview) - 60} more lines)"
            summary = f"Create {rel} ({len(preview)} lines)"
        return Assessment(risk, summary, [rel], details, warnings, scope_key=f"write:{rel}", block_reason=block)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"])
        rel = ctx.workspace.relative(path)
        existed = path.exists()
        ctx.tracker.before_change(path, "overwrite" if existed else "create")
        missing_parents = [p for p in reversed(path.parents) if not p.exists()]
        _atomic_write(path, args["content"])
        for parent in missing_parents:
            ctx.tracker.record_dir(parent)
        ctx.tracker.after_change(path)
        ctx.reads[rel] = {"sig": _file_signature(path), "range": None, "in_context": False}
        lines = args["content"].count("\n") + (0 if args["content"].endswith("\n") or not args["content"] else 1)
        verb = "Overwrote" if existed else "Created"
        return ToolResult(True, f"{verb} {rel} ({lines} lines, {len(args['content'].encode('utf-8'))} bytes).", summary=f"{verb} {rel}", changed_paths=[rel])


class EditFile(Tool):
    name = "edit_file"
    description = (
        "Replace an exact snippet in an existing file. old_string must match the file exactly (including "
        "indentation) and be unique unless replace_all is true. Read the file first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def _apply(self, args: Dict[str, Any], ctx: ToolContext) -> Tuple[Path, str, str, int]:
        path = ctx.workspace.resolve(args["path"], must_exist=True)
        rel = ctx.workspace.relative(path)
        if path.is_dir():
            raise ToolError(f"{rel} is a directory")
        if rel not in ctx.reads:
            raise ToolError(f"Read {rel} with read_file before editing it, so the edit is based on its real content.")
        old, new = args["old_string"], args["new_string"]
        if not old:
            raise ToolError("old_string must not be empty; use write_file to create or replace whole files")
        if old == new:
            raise ToolError("old_string and new_string are identical; nothing to change")
        text = _read_text(path)
        count = text.count(old)
        if count == 0:
            hint = ""
            first = old.strip().splitlines()[0].strip() if old.strip() else ""
            if first:
                candidates = difflib.get_close_matches(first, [l.strip() for l in text.splitlines()], n=1, cutoff=0.6)
                if candidates:
                    hint = f" Closest line in the file: {candidates[0]!r}. Check whitespace and re-read the file."
            raise ToolError(f"old_string was not found in {rel}.{hint}")
        if count > 1 and not args.get("replace_all"):
            raise ToolError(f"old_string occurs {count} times in {rel}; include more surrounding context or set replace_all")
        updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        return path, text, updated, count if args.get("replace_all") else 1

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path, before, after, n = self._apply(args, ctx)
        rel = ctx.workspace.relative(path)
        risk, warnings, block = _write_risk(ctx, path, True)
        return Assessment(risk, f"Edit {rel} ({n} replacement{'s' if n > 1 else ''})", [rel], _diff(before, after, rel), warnings, f"write:{rel}", block)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path, before, after, n = self._apply(args, ctx)
        rel = ctx.workspace.relative(path)
        ctx.tracker.before_change(path, "edit")
        _atomic_write(path, after)
        ctx.tracker.after_change(path)
        ctx.reads[rel] = {"sig": _file_signature(path), "range": None, "in_context": False}
        return ToolResult(True, f"Edited {rel}: {n} replacement(s).\n{_diff(before, after, rel, 60)}", summary=f"Edited {rel}", changed_paths=[rel])


class CreateDirectory(Tool):
    name = "create_directory"
    description = "Create a directory (and parents) inside the workspace."
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args["path"])
        rel = ctx.workspace.relative(path)
        risk, warnings, block = _write_risk(ctx, path, False)
        if risk is Risk.WRITE_SENSITIVE:
            risk = Risk.CREATE
        return Assessment(risk, f"Create directory {rel}", [rel], warnings=warnings, scope_key=f"mkdir:{rel}", block_reason=block)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"])
        rel = ctx.workspace.relative(path)
        if path.exists():
            return ToolResult(path.is_dir(), f"{rel} already exists" + ("" if path.is_dir() else " and is a file"), summary=f"{rel} exists")
        missing = [p for p in reversed(path.parents) if not p.exists()] + [path]
        path.mkdir(parents=True)
        for p in missing:
            ctx.tracker.record_dir(p)
        return ToolResult(True, f"Created directory {rel}", summary=f"Created {rel}/", changed_paths=[rel])


def _files_under(path: Path) -> List[Path]:
    if path.is_file() or path.is_symlink():
        return [path]
    files: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            files.append(Path(dirpath) / name)
            if len(files) > MAX_TRACKED_FILES:
                raise ToolError(f"{path.name} contains more than {MAX_TRACKED_FILES} files; refusing (too large to snapshot for undo)")
    return files


class MovePath(Tool):
    name = "move_path"
    description = "Rename or move a file or directory within the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean", "description": "Replace an existing destination file (destructive)."},
        },
        "required": ["source", "destination"],
        "additionalProperties": False,
    }

    def _paths(self, args: Dict[str, Any], ctx: ToolContext) -> Tuple[Path, Path]:
        src = ctx.workspace.resolve(args["source"], must_exist=True)
        dst = ctx.workspace.resolve(args["destination"])
        if src == ctx.workspace.root:
            raise ToolError("Cannot move the workspace root")
        if dst.exists() and dst.is_dir() and src.is_dir():
            raise ToolError(f"Destination directory {ctx.workspace.relative(dst)} already exists")
        if dst.exists() and not args.get("overwrite"):
            raise ToolError(f"Destination {ctx.workspace.relative(dst)} exists; set overwrite=true to replace it")
        return src, dst

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        src, dst = self._paths(args, ctx)
        rels = [ctx.workspace.relative(src), ctx.workspace.relative(dst)]
        risk = Risk.DESTRUCTIVE if dst.exists() else Risk.EDIT
        warnings: List[str] = []
        for p in (src, dst):
            if ctx.workspace.is_git_internal(p):
                return Assessment(Risk.BLOCKED, f"Move {rels[0]} -> {rels[1]}", rels, block_reason="Cannot move files into or out of .git")
            reason = ctx.workspace.sensitive_reason(p)
            if reason:
                warnings.append(f"Sensitive path {ctx.workspace.relative(p)}: {reason}")
                risk = Risk.WRITE_SENSITIVE if risk is not Risk.DESTRUCTIVE else risk
        if dst.exists():
            warnings.append(f"{rels[1]} will be overwritten")
        return Assessment(risk, f"Move {rels[0]} -> {rels[1]}", rels, warnings=warnings)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        src, dst = self._paths(args, ctx)
        moved = _files_under(src)
        targets = [dst / f.relative_to(src) if src.is_dir() else dst for f in moved]
        for path in moved + targets:
            ctx.tracker.before_change(path, "move")
        missing = [p for p in reversed(dst.parents) if not p.exists()]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        for p in missing:
            ctx.tracker.record_dir(p)
        for path in moved + targets:
            ctx.tracker.after_change(path)
        rels = [ctx.workspace.relative(src), ctx.workspace.relative(dst)]
        return ToolResult(True, f"Moved {rels[0]} -> {rels[1]}", summary=f"Moved {rels[0]} -> {rels[1]}", changed_paths=rels)


class DeletePath(Tool):
    name = "delete_path"
    description = "Delete a file, or a directory when recursive=true. Always requires the user's approval."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        "required": ["path"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        path = ctx.workspace.resolve(args["path"], must_exist=True)
        rel = ctx.workspace.relative(path)
        if path == ctx.workspace.root:
            return Assessment(Risk.BLOCKED, f"Delete {rel}", [rel], block_reason="Refusing to delete the workspace root")
        if ctx.workspace.is_git_internal(path):
            return Assessment(Risk.BLOCKED, f"Delete {rel}", [rel], block_reason="Refusing to delete git internals")
        if path.is_dir() and not args.get("recursive"):
            raise ToolError(f"{rel} is a directory; set recursive=true to delete it and its contents")
        files = _files_under(path)
        warnings = [f"{len(files)} file(s) will be deleted (snapshotted for /undo)"] if path.is_dir() else []
        reason = ctx.workspace.sensitive_reason(path)
        if reason:
            warnings.append(f"Sensitive file: {reason}")
        details = "\n".join(ctx.workspace.relative(f) for f in files[:40])
        return Assessment(Risk.DESTRUCTIVE, f"Delete {rel}{'/ (recursive)' if path.is_dir() else ''}", [rel], details, warnings)

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.workspace.resolve(args["path"], must_exist=True)
        rel = ctx.workspace.relative(path)
        files = _files_under(path)
        for f in files:
            ctx.tracker.before_change(f, "delete")
        if path.is_dir() and not path.is_symlink():
            import shutil

            shutil.rmtree(path)
        else:
            path.unlink()
        for f in files:
            ctx.tracker.after_change(f)
            ctx.reads.pop(ctx.workspace.relative(f), None)
        return ToolResult(True, f"Deleted {rel} ({len(files)} file(s))", summary=f"Deleted {rel}", changed_paths=[rel])
