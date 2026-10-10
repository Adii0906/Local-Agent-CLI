"""The permission engine: maps (mode, risk) to allow / ask / deny.

Invariants:
* Only the human, through the approval callback, can turn ASK into an
  approval. No tool lets the model approve its own actions.
* "Always allow" rules are session-scoped, narrowly keyed (one file path or one
  command prefix), and can only turn ASK into ALLOW. They never override a
  DENY, and they are never offered for destructive, sensitive, network or
  privileged actions.
* Project files cannot change the mode or add rules. Only the user can, via a
  CLI flag, the user config file or the /permissions command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Set

from local_agent.tools.base import Assessment, Risk

A, Q, D = "allow", "ask", "deny"

#                     plan  ask  auto full
_MATRIX: Dict[Risk, Dict[str, str]] = {
    Risk.READ:            {"plan": A, "ask": A, "auto": A, "full": A},
    Risk.SHELL_READONLY:  {"plan": A, "ask": A, "auto": A, "full": A},
    Risk.MEMORY:          {"plan": A, "ask": A, "auto": A, "full": A},
    Risk.READ_SENSITIVE:  {"plan": Q, "ask": Q, "auto": Q, "full": Q},
    Risk.PREFERENCE:      {"plan": Q, "ask": Q, "auto": Q, "full": Q},
    Risk.CREATE:          {"plan": D, "ask": Q, "auto": A, "full": A},
    Risk.EDIT:            {"plan": D, "ask": Q, "auto": A, "full": A},
    Risk.WRITE_SENSITIVE: {"plan": D, "ask": Q, "auto": Q, "full": Q},
    Risk.DESTRUCTIVE:     {"plan": D, "ask": Q, "auto": Q, "full": Q},
    Risk.SHELL:           {"plan": D, "ask": Q, "auto": Q, "full": A},
    Risk.NETWORK:         {"plan": D, "ask": Q, "auto": Q, "full": Q},
    Risk.PRIVILEGED:      {"plan": D, "ask": D, "auto": D, "full": Q},
    Risk.BLOCKED:         {"plan": D, "ask": D, "auto": D, "full": D},
}

ALWAYS_ALLOWABLE = {Risk.CREATE, Risk.EDIT, Risk.SHELL}

MODE_DESCRIPTIONS = {
    "plan": "Read-only. Inspect and propose a plan; no file changes or state-changing commands.",
    "ask": "Read freely; ask before every file change and every non-read-only command.",
    "auto": "Edit/create files in the workspace without asking; ask before commands, deletes, network and sensitive files.",
    "full": "Also run ordinary commands without asking. Still asks for deletes, network, privileged and sensitive actions. Higher risk.",
}


class Choice(str, Enum):
    ONCE = "once"
    ALWAYS = "always"
    DENY = "deny"


@dataclass
class ApprovalRequest:
    tool: str
    assessment: Assessment
    mode: str
    reason: str  # the model's stated intent, if any
    cwd: str
    can_always: bool


@dataclass
class ApprovalResponse:
    choice: Choice
    feedback: str = ""


Approver = Callable[[ApprovalRequest], ApprovalResponse]


def deny_all_approver(_: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(Choice.DENY, "No interactive user is available to approve this action.")


@dataclass
class Verdict:
    decision: str  # allow | ask | deny
    reason: str = ""
    via_rule: bool = False


@dataclass
class PermissionEngine:
    mode: str = "ask"
    session_rules: Set[str] = field(default_factory=set)
    log: List[Dict[str, str]] = field(default_factory=list)

    def evaluate(self, assessment: Assessment) -> Verdict:
        if assessment.risk is Risk.BLOCKED:
            return Verdict(D, assessment.block_reason or "Blocked by safety policy")
        decision = _MATRIX[assessment.risk][self.mode]
        if decision == D:
            if self.mode == "plan":
                return Verdict(D, "Plan mode is read-only. Describe this change in your plan instead of performing it.")
            return Verdict(D, f"'{assessment.risk.value}' actions are not permitted in {self.mode} mode.")
        if decision == Q and assessment.scope_key and assessment.risk in ALWAYS_ALLOWABLE and assessment.scope_key in self.session_rules:
            return Verdict(A, f"allowed by session rule '{assessment.scope_key}'", via_rule=True)
        return Verdict(decision)

    def can_always(self, assessment: Assessment) -> bool:
        return bool(assessment.scope_key) and assessment.risk in ALWAYS_ALLOWABLE

    def remember(self, assessment: Assessment) -> None:
        if self.can_always(assessment):
            self.session_rules.add(assessment.scope_key or "")

    def record(self, tool: str, assessment: Assessment, outcome: str) -> None:
        self.log.append({"tool": tool, "action": assessment.summary, "risk": assessment.risk.value, "outcome": outcome})
