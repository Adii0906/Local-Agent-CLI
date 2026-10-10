"""Validate a model with a small real request before setup completes.

The test asks the model to call a harmless ``report_status`` tool. Three
outcomes are distinguished:
* native   - a structured tool call came back (best)
* text     - the call came back as JSON text (works through the agent's
             text-call recovery, less reliably)
* none     - the model answered but never called the tool, or the server
             rejected tools; the agent would fall back to prompt-based calls
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from local_agent.config import Settings
from local_agent.errors import AgentError, ProviderError
from local_agent.llm.base import Message
from local_agent.llm.factory import create_base_provider
from local_agent.llm.prompted import extract_tool_calls

_TOOL = {
    "name": "report_status",
    "description": "Report that the coding agent is ready.",
    "parameters": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]},
}


@dataclass
class ValidationResult:
    responded: bool
    tool_calling: str  # native | text | none
    seconds: float
    error: str = ""
    sample: str = ""

    @property
    def ok(self) -> bool:
        return self.responded

    def recommended_tool_mode(self) -> str:
        return "auto" if self.tool_calling in ("native", "text") else "prompted"


def validate_model(settings: Settings) -> ValidationResult:
    provider = create_base_provider(settings)
    started = time.monotonic()
    messages = [
        Message("system", "You are a connectivity test. Call the report_status tool with status \"ready\". Do not write anything else."),
        Message("user", "Call report_status now."),
    ]
    try:
        resp = provider.chat(messages, [_TOOL])
    except ProviderError as exc:
        if "tool" not in exc.message.lower():
            return ValidationResult(False, "none", time.monotonic() - started, exc.message)
        # The server refuses tools for this model: check plain chat still works.
        try:
            plain = provider.chat([Message("user", "Reply with the single word: ready")], None)
        except AgentError as exc2:
            return ValidationResult(False, "none", time.monotonic() - started, exc2.message)
        return ValidationResult(True, "none", time.monotonic() - started, sample=plain.content.strip()[:80])
    except AgentError as exc:
        return ValidationResult(False, "none", time.monotonic() - started, exc.message)
    elapsed = time.monotonic() - started
    if any(c.name == "report_status" for c in resp.tool_calls):
        return ValidationResult(True, "native", elapsed)
    _, salvaged = extract_tool_calls(resp.content, {"report_status"})
    if salvaged:
        return ValidationResult(True, "text", elapsed, sample=resp.content.strip()[:80])
    responded = bool(resp.content.strip())
    return ValidationResult(responded, "none", elapsed, "" if responded else "the model returned an empty reply", resp.content.strip()[:80])


def describe(result: ValidationResult) -> Optional[str]:
    """A user-facing limitation note, or None when everything is fine."""
    if not result.responded:
        return None
    if result.tool_calling == "text":
        return ("The model answers tool requests as plain JSON text instead of real tool calls. The agent can still "
                "use it, but it will be less reliable at editing files and running commands.")
    if result.tool_calling == "none":
        return ("The model did not make a tool call. The agent will use prompt-based tool calling, which often makes "
                "small models reply with code instead of editing files. A tool-capable coding model is recommended.")
    return None
