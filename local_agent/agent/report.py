"""Evidence-based task report.

Built only from recorded tool results, never from the model's own claims.
The model's final message is shown separately, and this report lets the user
check it against what actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CommandRecord:
    command: str
    exit_code: Optional[int]
    duration: float
    timed_out: bool
    cancelled: bool
    is_test: bool
    after_change_index: int  # number of file changes made before this command ran
    tail: str = ""


@dataclass
class TaskReport:
    task: str
    status: str  # completed | stopped | cancelled | failed
    stop_reason: str
    final_message: str = ""
    files: List[Dict[str, str]] = field(default_factory=list)  # path, change
    changed_by_commands: List[str] = field(default_factory=list)
    commands: List[CommandRecord] = field(default_factory=list)
    denied: List[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    file_change_count: int = 0

    def verification(self) -> List[str]:
        notes: List[str] = []
        tests = [c for c in self.commands if c.is_test]
        if self.files and not tests:
            notes.append("Files were changed but no test command was run, so the changes are UNVERIFIED.")
        elif tests:
            last = tests[-1]
            if last.after_change_index < self.file_change_count:
                notes.append("Files changed after the last test run; the final state was NOT re-tested.")
            if last.timed_out:
                notes.append(f"The last test run TIMED OUT: `{last.command}`.")
            elif last.exit_code == 0:
                notes.append(f"The last test run passed: `{last.command}` (exit 0).")
            else:
                notes.append(f"The last test run FAILED: `{last.command}` (exit {last.exit_code}).")
        if self.changed_by_commands:
            notes.append(
                "Commands changed files outside the agent's file tools (not covered by /undo): "
                + ", ".join(self.changed_by_commands[:10])
            )
        if self.denied:
            notes.append(f"{len(self.denied)} action(s) were denied and not performed.")
        if self.status != "completed":
            notes.append(f"Task did not complete normally: {self.stop_reason}.")
        return notes

    def to_record(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "summary": self.final_message[:1500],
            "files_changed": [f["path"] for f in self.files],
            "commands": [
                {"command": c.command, "exit_code": c.exit_code, "is_test": c.is_test} for c in self.commands[-15:]
            ],
            "verification": self.verification(),
        }
