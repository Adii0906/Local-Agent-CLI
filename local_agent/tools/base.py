"""Tool contracts: schemas, validation, risk assessment and results.

Each tool declares a JSON Schema for its arguments. The agent loop validates
model-supplied arguments before anything else happens. A tool then *assesses*
the call, which yields a risk category and the affected paths for the
permission engine, and only after approval is the tool *run*.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from local_agent.agent.checkpoint import ChangeTracker
    from local_agent.config import Settings
    from local_agent.memory.manager import MemoryManager
    from local_agent.project.discovery import ProjectInfo
    from local_agent.project.ignore import IgnoreRules
    from local_agent.security.paths import Workspace


class Risk(str, Enum):
    READ = "read"
    READ_SENSITIVE = "read-sensitive"
    CREATE = "create"
    EDIT = "edit"
    WRITE_SENSITIVE = "write-sensitive"
    DESTRUCTIVE = "destructive"
    SHELL_READONLY = "shell-readonly"
    SHELL = "shell"
    NETWORK = "network"
    PRIVILEGED = "privileged"
    MEMORY = "memory"
    PREFERENCE = "preference"
    BLOCKED = "blocked"


RISK_DESCRIPTIONS = {
    Risk.READ: "read-only inspection",
    Risk.READ_SENSITIVE: "reads a file that may contain secrets",
    Risk.CREATE: "creates new files or directories",
    Risk.EDIT: "modifies existing files (undoable with /undo)",
    Risk.WRITE_SENSITIVE: "writes to a sensitive file",
    Risk.DESTRUCTIVE: "deletes or overwrites data",
    Risk.SHELL_READONLY: "runs a read-only command",
    Risk.SHELL: "runs a command whose effects cannot be verified in advance",
    Risk.NETWORK: "accesses the network or has external side effects",
    Risk.PRIVILEGED: "privileged or system-level operation",
    Risk.MEMORY: "saves a note to project memory",
    Risk.PREFERENCE: "saves a long-term user preference",
    Risk.BLOCKED: "blocked by safety policy",
}


@dataclass
class Assessment:
    risk: Risk
    summary: str  # one line: the exact action
    paths: List[str] = field(default_factory=list)
    details: str = ""  # e.g. a diff preview or the full command
    warnings: List[str] = field(default_factory=list)
    scope_key: Optional[str] = None  # what "always allow" would cover; None = not offered
    block_reason: str = ""


@dataclass
class ToolResult:
    ok: bool
    output: str  # text returned to the model
    summary: str = ""  # short line for the execution trace
    data: Dict[str, Any] = field(default_factory=dict)
    changed_paths: List[str] = field(default_factory=list)

    @classmethod
    def error(cls, message: str, **data: Any) -> "ToolResult":
        return cls(ok=False, output=f"ERROR: {message}", summary=message, data=data)


@dataclass
class ToolContext:
    workspace: "Workspace"
    ignore: "IgnoreRules"
    settings: "Settings"
    tracker: "ChangeTracker"
    memory: Optional["MemoryManager"] = None
    project: Optional["ProjectInfo"] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    plan: List[Dict[str, str]] = field(default_factory=list)
    # path -> (mtime_ns, size, start_line, max_lines) of reads still present in context
    reads: Dict[str, Any] = field(default_factory=dict)


class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

    def spec(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


_JSON_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_args(schema: Dict[str, Any], args: Any, path: str = "arguments") -> List[str]:
    """Validate ``args`` against the subset of JSON Schema the tools use."""
    errors: List[str] = []
    expected = schema.get("type")
    if expected:
        py_type = _JSON_TYPES[expected]
        is_bool = isinstance(args, bool)
        if not isinstance(args, py_type) or (is_bool and expected in ("integer", "number")):
            return [f"{path} must be of type {expected}, got {type(args).__name__}"]
    if "enum" in schema and args not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}, got {args!r}")
    if expected == "integer" or expected == "number":
        if "minimum" in schema and args < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and args > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
    if expected == "string" and "maxLength" in schema and len(args) > schema["maxLength"]:
        errors.append(f"{path} is longer than {schema['maxLength']} characters")
    if expected == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in args:
                errors.append(f"{path}.{req} is required")
        for key, value in args.items():
            if key in props:
                errors.extend(validate_args(props[key], value, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{key} is not a recognized parameter (allowed: {', '.join(props) or 'none'})")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(args):
            errors.extend(validate_args(schema["items"], item, f"{path}[{i}]"))
    return errors
