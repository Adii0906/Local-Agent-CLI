"""The interface between the agent loop and whatever displays it.

The loop reports only real events: a tool started, a tool finished with its
actual result, an approval was needed. The terminal UI and the tests both
implement this interface.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from local_agent.permissions.engine import ApprovalRequest, ApprovalResponse, deny_all_approver
from local_agent.tools.base import Assessment, ToolResult


class AgentUI:
    def assistant_start(self) -> None: ...

    def assistant_text(self, chunk: str) -> None: ...

    def assistant_end(self, full_text: str, streamed: bool) -> None: ...

    def thinking(self, active: bool) -> None: ...

    def tool_started(self, tool: str, assessment: Assessment, auto_reason: str = "") -> None: ...

    def tool_finished(self, tool: str, assessment: Assessment, result: ToolResult) -> None: ...

    def tool_denied(self, tool: str, assessment: Assessment, reason: str) -> None: ...

    def tool_invalid(self, tool: str, message: str) -> None: ...

    def plan_updated(self, plan: List[Dict[str, str]]) -> None: ...

    def notice(self, message: str, level: str = "info") -> None: ...

    def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        return deny_all_approver(request)


class RecordingUI(AgentUI):
    """Collects events. Approvals come from a scripted list (for tests)."""

    def __init__(self, approvals: Optional[List[ApprovalResponse]] = None) -> None:
        self.events: List[tuple] = []
        self.approvals = list(approvals or [])
        self.requests: List[ApprovalRequest] = []
        self.text = ""

    def assistant_text(self, chunk: str) -> None:
        self.text += chunk

    def tool_started(self, tool, assessment, auto_reason=""):
        self.events.append(("start", tool, assessment.summary))

    def tool_finished(self, tool, assessment, result):
        self.events.append(("finish", tool, result.ok, result.summary))

    def tool_denied(self, tool, assessment, reason):
        self.events.append(("denied", tool, reason))

    def tool_invalid(self, tool, message):
        self.events.append(("invalid", tool, message))

    def notice(self, message, level="info"):
        self.events.append(("notice", level, message))

    def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        self.requests.append(request)
        if self.approvals:
            return self.approvals.pop(0)
        return deny_all_approver(request)
