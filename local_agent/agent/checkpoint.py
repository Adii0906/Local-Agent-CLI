"""Checkpoints: record original file contents before the agent changes them.

Each file the agent's tools modify is snapshotted once, before the first
change. ``undo`` restores only files whose current content still matches what
the agent last wrote. A file someone else changed since then is reported as a
conflict and left alone unless ``force=True``.

Changes made by shell commands are not captured here. They show up in the git
status delta, which the final report lists separately.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent.security.paths import Workspace


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


@dataclass
class FileChange:
    path: str
    original_exists: bool
    original_sha: Optional[str]
    last_sha: Optional[str] = None  # None => agent deleted it / it does not exist
    operations: List[str] = field(default_factory=list)


@dataclass
class UndoResult:
    path: str
    action: str
    ok: bool
    reason: str = ""


class ChangeTracker:
    def __init__(self, workspace: Workspace, store_dir: Optional[Path] = None) -> None:
        self.workspace = workspace
        self.store_dir = store_dir
        self.changes: Dict[str, FileChange] = {}
        self.created_dirs: List[str] = []
        self._blobs: Dict[str, bytes] = {}

    # -- recording -------------------------------------------------------
    def _store_blob(self, data: bytes) -> str:
        digest = _sha(data)
        if self.store_dir is not None:
            blob = self.store_dir / "blobs" / digest
            if not blob.exists():
                blob.parent.mkdir(parents=True, exist_ok=True)
                blob.write_bytes(data)
        else:
            self._blobs[digest] = data
        return digest

    def _load_blob(self, digest: str) -> Optional[bytes]:
        if digest in self._blobs:
            return self._blobs[digest]
        if self.store_dir is not None:
            return _read_bytes(self.store_dir / "blobs" / digest)
        return None

    def before_change(self, path: Path, operation: str) -> None:
        rel = self.workspace.relative(path)
        if rel not in self.changes:
            data = _read_bytes(path)
            self.changes[rel] = FileChange(
                path=rel,
                original_exists=data is not None,
                original_sha=self._store_blob(data) if data is not None else None,
            )
        self.changes[rel].operations.append(operation)

    def after_change(self, path: Path) -> None:
        rel = self.workspace.relative(path)
        change = self.changes.get(rel)
        if change is None:
            return
        data = _read_bytes(path)
        change.last_sha = _sha(data) if data is not None else None

    def record_dir(self, path: Path) -> None:
        rel = self.workspace.relative(path)
        if rel not in self.created_dirs:
            self.created_dirs.append(rel)

    # -- inspection ------------------------------------------------------
    def changed_files(self) -> List[str]:
        return [rel for rel, ch in self.changes.items() if self._differs_from_original(ch)]

    def _differs_from_original(self, change: FileChange) -> bool:
        current = _read_bytes(self.workspace.root / change.path)
        current_sha = _sha(current) if current is not None else None
        return current_sha != change.original_sha

    def describe(self) -> List[Dict[str, str]]:
        out = []
        for rel in self.changed_files():
            ch = self.changes[rel]
            exists_now = (self.workspace.root / rel).exists()
            if not ch.original_exists:
                kind = "created"
            elif not exists_now:
                kind = "deleted"
            else:
                kind = "modified"
            out.append({"path": rel, "change": kind})
        return out

    def diff(self, only: Optional[str] = None, context: int = 3) -> str:
        chunks: List[str] = []
        for rel in self.changed_files():
            if only and rel != only:
                continue
            ch = self.changes[rel]
            before = self._load_blob(ch.original_sha) if ch.original_sha else b""
            after = _read_bytes(self.workspace.root / rel) or b""
            if before is None:
                chunks.append(f"--- {rel}: original snapshot missing\n")
                continue
            if b"\x00" in before[:8192] or b"\x00" in after[:8192]:
                chunks.append(f"Binary file {rel} changed\n")
                continue
            a = before.decode("utf-8", errors="replace").splitlines(keepends=True)
            b = after.decode("utf-8", errors="replace").splitlines(keepends=True)
            from_name = f"a/{rel}" if ch.original_exists else "/dev/null"
            to_name = f"b/{rel}" if (self.workspace.root / rel).exists() else "/dev/null"
            lines = list(difflib.unified_diff(a, b, from_name, to_name, n=context))
            chunks.append("".join(line if line.endswith("\n") else line + "\n" for line in lines))
        return "".join(chunks)

    # -- undo ------------------------------------------------------------
    def undo(self, force: bool = False) -> List[UndoResult]:
        results: List[UndoResult] = []
        for rel, ch in list(self.changes.items()):
            path = self.workspace.root / rel
            current = _read_bytes(path)
            current_sha = _sha(current) if current is not None else None
            if current_sha == ch.original_sha:
                results.append(UndoResult(rel, "unchanged", True))
                del self.changes[rel]
                continue
            if current_sha != ch.last_sha and not force:
                results.append(
                    UndoResult(rel, "skipped", False, "modified after the agent's last write; use --force to overwrite")
                )
                continue
            try:
                if ch.original_exists:
                    data = self._load_blob(ch.original_sha or "")
                    if data is None:
                        results.append(UndoResult(rel, "skipped", False, "original snapshot missing"))
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.name + ".local-agent-undo")
                    tmp.write_bytes(data)
                    os.replace(tmp, path)
                    results.append(UndoResult(rel, "restored", True))
                else:
                    if path.exists():
                        path.unlink()
                    results.append(UndoResult(rel, "removed", True))
                del self.changes[rel]
            except OSError as exc:
                results.append(UndoResult(rel, "failed", False, str(exc)))
        for rel in sorted(self.created_dirs, key=len, reverse=True):
            path = self.workspace.root / rel
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    results.append(UndoResult(rel + "/", "removed", True))
                    self.created_dirs.remove(rel)
            except OSError:
                pass
        return results

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"changes": [asdict(c) for c in self.changes.values()], "created_dirs": list(self.created_dirs)}

    def load_dict(self, data: Dict[str, Any]) -> None:
        self.changes = {c["path"]: FileChange(**c) for c in data.get("changes", [])}
        self.created_dirs = list(data.get("created_dirs", []))

    def discard_store(self) -> None:
        if self.store_dir is not None and self.store_dir.exists():
            shutil.rmtree(self.store_dir, ignore_errors=True)
