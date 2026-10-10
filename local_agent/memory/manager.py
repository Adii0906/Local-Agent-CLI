"""Persistent memory: project facts, user preferences, task history, sessions.

Layout under the data directory::

    preferences.json                 global, user-approved preferences
    projects/<project-id>/
        project.json                 facts about this project (verified flag)
        tasks.json                   task history (last 200)
        sessions/<session-id>.json   resumable session state
        checkpoints/<session-id>/    original file snapshots for /undo

``project-id`` is a hash of the workspace's real path, so memories from
different projects never mix. Everything is redacted for secrets before it is
written. Retrieved memory is injected into the prompt as *reference data*,
clearly labelled as untrusted: it cannot grant permissions.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets as _rand
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent.memory.store import JsonDocument
from local_agent.security.secrets import redact

MAX_TASKS = 200
MAX_FACTS = 300
_WORD = re.compile(r"[a-z0-9_]{2,}")
_STOP = {"the", "and", "for", "with", "this", "that", "from", "into", "are", "was", "use", "can", "you", "fix", "add", "make", "file", "files"}


def project_id(root: Path) -> str:
    return hashlib.sha256(os.path.normcase(str(root)).encode("utf-8")).hexdigest()[:16]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{_rand.token_hex(4)}"


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> List[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


class MemoryManager:
    def __init__(self, data_dir: Path, project_root: Path) -> None:
        self.data_dir = data_dir
        self.project_root = project_root
        self.project_dir = data_dir / "projects" / project_id(project_root)
        self._project = JsonDocument(self.project_dir / "project.json", lambda: {"project_root": str(project_root), "facts": []})
        self._tasks = JsonDocument(self.project_dir / "tasks.json", lambda: {"tasks": []})
        self._prefs = JsonDocument(data_dir / "preferences.json", lambda: {"preferences": []})

    def warnings(self) -> List[str]:
        return [d.warning for d in (self._project, self._tasks, self._prefs) if d.warning]

    # -- project facts ---------------------------------------------------
    def facts(self) -> List[Dict[str, Any]]:
        return list(self._project.load()["facts"])

    def add_fact(self, content: str, category: str = "other", verified: bool = False, evidence: str = "", source: str = "user") -> Dict[str, Any]:
        content = redact(content.strip())
        data = self._project.load()
        data["project_root"] = str(self.project_root)
        for fact in data["facts"]:
            if _norm(fact["content"]) == _norm(content):
                fact["updated_at"] = _now()
                fact["verified"] = fact.get("verified", False) or verified
                if evidence:
                    fact["evidence"] = redact(evidence)
                self._project.save(data)
                return fact
        fact = {
            "id": _new_id("f"), "content": content, "category": category, "verified": verified,
            "evidence": redact(evidence), "source": source, "created_at": _now(), "updated_at": _now(),
        }
        data["facts"].append(fact)
        data["facts"] = data["facts"][-MAX_FACTS:]
        self._project.save(data)
        return fact

    # -- preferences -----------------------------------------------------
    def preferences(self) -> List[Dict[str, Any]]:
        return list(self._prefs.load()["preferences"])

    def add_preference(self, content: str, source: str = "user") -> Dict[str, Any]:
        content = redact(content.strip())
        data = self._prefs.load()
        for pref in data["preferences"]:
            if _norm(pref["content"]) == _norm(content):
                return pref
        pref = {"id": _new_id("p"), "content": content, "source": source, "created_at": _now()}
        data["preferences"].append(pref)
        self._prefs.save(data)
        return pref

    # -- task history ----------------------------------------------------
    def tasks(self) -> List[Dict[str, Any]]:
        return list(self._tasks.load()["tasks"])

    def record_task(self, record: Dict[str, Any]) -> Dict[str, Any]:
        data = self._tasks.load()
        record = {k: (redact(v) if isinstance(v, str) else v) for k, v in record.items()}
        record.setdefault("id", _new_id("t"))
        record.setdefault("finished_at", _now())
        data["tasks"].append(record)
        data["tasks"] = data["tasks"][-MAX_TASKS:]
        self._tasks.save(data)
        return record

    # -- search / delete / clear -----------------------------------------
    def _all_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for f in self.facts():
            items.append({"kind": "fact", **f})
        for p in self.preferences():
            items.append({"kind": "preference", **p})
        for t in self.tasks():
            text = f"{t.get('task', '')} | {t.get('summary', '')} | files: {', '.join(t.get('files_changed', []))}"
            items.append({"kind": "task", "id": t["id"], "content": text, "created_at": t.get("finished_at", ""), "status": t.get("status")})
        return items

    def search(self, query: str, limit: int = 10, kinds: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        q_terms = set(_terms(query))
        if not q_terms:
            return []
        items = [i for i in self._all_items() if not kinds or i["kind"] in kinds]
        doc_terms = [set(_terms(i["content"] + " " + i.get("category", ""))) for i in items]
        n = len(items) or 1
        df = {t: sum(1 for d in doc_terms if t in d) for t in q_terms}
        scored = []
        for item, terms in zip(items, doc_terms):
            score = sum((1.0 + (n / (1 + df[t]))) for t in q_terms if t in terms)
            if query.lower() in item["content"].lower():
                score += 3
            if item.get("verified"):
                score *= 1.2
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda x: -x[0])
        return [dict(item, score=round(score, 2)) for score, item in scored[:limit]]

    def delete(self, item_id: str) -> bool:
        for doc, key in ((self._project, "facts"), (self._prefs, "preferences"), (self._tasks, "tasks")):
            data = doc.load()
            kept = [i for i in data[key] if i.get("id") != item_id]
            if len(kept) != len(data[key]):
                data[key] = kept
                doc.save(data)
                return True
        return False

    def clear(self, scope: str) -> List[str]:
        cleared = []
        if scope in ("project", "all"):
            self._project.delete()
            cleared.append("project facts")
        if scope in ("tasks", "all"):
            self._tasks.delete()
            cleared.append("task history")
        if scope in ("preferences", "prefs", "all"):
            self._prefs.delete()
            cleared.append("preferences")
        if scope in ("sessions", "all"):
            sessions = self.project_dir / "sessions"
            for path in sessions.glob("*.json") if sessions.exists() else []:
                path.unlink()
            cleared.append("saved sessions")
        return cleared

    # -- prompt context --------------------------------------------------
    def context_block(self, task: str, budget_chars: int = 3000) -> str:
        """Relevant memory formatted for the system prompt (bounded)."""
        lines: List[str] = []
        prefs = self.preferences()
        if prefs:
            lines.append("User preferences (approved by the user):")
            lines += [f"- {p['content']}" for p in prefs[-10:]]
        facts = self.facts()
        if facts:
            relevant_ids = {r["id"] for r in self.search(task, limit=15, kinds=["fact"])}
            ordered = [f for f in facts if f["id"] in relevant_ids] + [f for f in reversed(facts) if f["id"] not in relevant_ids]
            lines.append("Project memory (from earlier sessions; may be outdated, verify before relying on it):")
            for f in ordered[:25]:
                tag = "verified" if f.get("verified") else "unverified"
                lines.append(f"- [{tag}, {f.get('category', 'other')}] {f['content']}")
        related = self.search(task, limit=3, kinds=["task"])
        if related:
            lines.append("Related past tasks:")
            lines += [f"- ({r.get('status')}) {r['content'][:300]}" for r in related]
        text = "\n".join(lines)
        if len(text) > budget_chars:
            text = text[:budget_chars] + "\n- [memory truncated]"
        return text

    # -- sessions (recovery) --------------------------------------------
    def session_doc(self, session_id: str) -> JsonDocument:
        return JsonDocument(self.project_dir / "sessions" / f"{session_id}.json", lambda: {})

    def checkpoint_dir(self, session_id: str) -> Path:
        return self.project_dir / "checkpoints" / session_id

    def list_sessions(self) -> List[Dict[str, Any]]:
        sessions_dir = self.project_dir / "sessions"
        out = []
        if sessions_dir.exists():
            for path in sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                data = JsonDocument(path, lambda: {}).load()
                if data.get("session_id"):
                    out.append(data)
        return out

    def prune_sessions(self, keep: int = 20) -> int:
        """Delete all but the ``keep`` most recent sessions and their checkpoints."""
        sessions_dir = self.project_dir / "sessions"
        if not sessions_dir.exists():
            return 0
        paths = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        removed = 0
        for path in paths[keep:]:
            shutil.rmtree(self.checkpoint_dir(path.stem), ignore_errors=True)
            path.unlink()
            removed += 1
        return removed

    def latest_interrupted_session(self) -> Optional[Dict[str, Any]]:
        for data in self.list_sessions():
            if data.get("status") in ("running", "interrupted"):
                return data
        return None
