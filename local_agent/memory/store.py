"""Versioned JSON documents with atomic writes.

Every file has ``{"schema_version": N, ...}``. Older versions are upgraded by
registered migrations. A file from a *newer* version, or a corrupt one, is
moved aside to ``<name>.bak-<timestamp>`` rather than overwritten, so data is
never silently lost.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

SCHEMA_VERSION = 1

Migration = Callable[[Dict[str, Any]], Dict[str, Any]]
MIGRATIONS: Dict[int, Migration] = {}  # from_version -> function producing from_version + 1


class JsonDocument:
    def __init__(self, path: Path, default: Callable[[], Dict[str, Any]]) -> None:
        self.path = path
        self._default = default
        self.warning: Optional[str] = None

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._fresh()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            self._quarantine(f"unreadable ({exc})")
            return self._fresh()
        version = data.get("schema_version", 0)
        if not isinstance(version, int) or version > SCHEMA_VERSION:
            self._quarantine(f"schema version {version!r} is newer than supported {SCHEMA_VERSION}")
            return self._fresh()
        while version < SCHEMA_VERSION:
            migrate = MIGRATIONS.get(version)
            if migrate is None:
                if version == 0:
                    data["schema_version"] = SCHEMA_VERSION
                    break
                self._quarantine(f"no migration from schema version {version}")
                return self._fresh()
            data = migrate(data)
            version = data.get("schema_version", version + 1)
        base = self._default()
        base.update(data)
        return base

    def save(self, data: Dict[str, Any]) -> None:
        data["schema_version"] = SCHEMA_VERSION
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _fresh(self) -> Dict[str, Any]:
        data = self._default()
        data["schema_version"] = SCHEMA_VERSION
        return data

    def _quarantine(self, why: str) -> None:
        backup = self.path.with_name(f"{self.path.name}.bak-{int(time.time())}")
        try:
            os.replace(self.path, backup)
            self.warning = f"{self.path.name} was {why}; moved to {backup.name} and started fresh."
        except OSError:
            self.warning = f"{self.path.name} was {why}; ignoring it."
