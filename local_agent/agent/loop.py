"""The agent loop: model -> validated tool call -> permission -> execution -> model.

Termination is guaranteed by bounded iterations, wall-clock time, an optional
token budget, repeated-call detection and a consecutive-failure limit. Each
stop records a concrete reason in the TaskReport.
"""

from __future__ import annotations

import json
import secrets
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from local_agent.agent.context import build_system_prompt, compact, estimate_tokens
from local_agent.agent.events import AgentUI
from local_agent.agent.report import CommandRecord, TaskReport
from local_agent.config import Settings
from local_agent.errors import ProviderError, ToolError
from local_agent.llm.base import ChatResponse, Message, Provider, ToolCall
from local_agent.llm.prompted import extract_tool_calls
from local_agent.memory.manager import MemoryManager
from local_agent.permissions.engine import ApprovalRequest, Choice, PermissionEngine
from local_agent.security.secrets import redact
from local_agent.tools.base import Risk, ToolContext, ToolResult, validate_args
from local_agent.tools.git import porcelain_status
from local_agent.tools.registry import ToolRegistry

MAX_TOOL_OUTPUT_CHARS = 40000
REPEAT_LIMIT = 3
FAILURE_HINT_AT = 5
FAILURE_STOP_AT = 10
STATE_CHANGING = {Risk.CREATE, Risk.EDIT, Risk.WRITE_SENSITIVE, Risk.DESTRUCTIVE, Risk.SHELL, Risk.NETWORK, Risk.PRIVILEGED}


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_obj(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_obj(v) for k, v in value.items()}
    return value


class _Stop(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        ctx: ToolContext,
        engine: PermissionEngine,
        ui: AgentUI,
        settings: Settings,
        memory: Optional[MemoryManager] = None,
        session_id: Optional[str] = None,
        context_window: int = 8192,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.ctx = ctx
        self.engine = engine
        self.ui = ui
        self.settings = settings
        self.memory = memory
        self.session_id = session_id or new_session_id()
        self.context_window = context_window
        self.messages: List[Message] = []
        self.total_tokens = 0
        self.file_changes = 0
        self.state_version = 0
        self.last_report: Optional[TaskReport] = None
        self._read_calls: Dict[str, str] = {}  # tool_call_id -> path, for compaction bookkeeping
        self._current_task = ""

    # -- public API --------------------------------------------------------
    def run_task(self, task: str, *, resumed: bool = False) -> TaskReport:
        self._current_task = task
        system = Message("system", self._system_prompt(task))
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = system
        else:
            self.messages.insert(0, system)
        if resumed:
            self.messages.append(Message(
                "user",
                "The previous run of this task was interrupted. Continue from where you left off. Files may have "
                "changed since: re-check their current state before editing.",
            ))
        else:
            self.messages.append(Message("user", task))
        report = self._loop(task)
        self.last_report = report
        if self.memory is not None:
            try:
                self.memory.record_task(report.to_record())
            except OSError as exc:
                self.ui.notice(f"Could not save task history: {exc}", "warning")
        return report

    def compact_now(self) -> Dict[str, object]:
        result = compact(self.messages, self._budget(), self._summarize, force=True)
        self._after_elision(result.get("elided", []))  # type: ignore[arg-type]
        self._save_session("idle", "")
        return result

    def load_session(self, data: Dict[str, Any]) -> str:
        self.session_id = data["session_id"]
        self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        self.ctx.plan = data.get("plan", [])
        self.ctx.tracker.load_dict(data.get("tracker", {}))
        return data.get("task", "")

    # -- internals ---------------------------------------------------------
    def _system_prompt(self, task: str) -> str:
        memory_block = ""
        if self.memory is not None:
            try:
                memory_block = self.memory.context_block(task)
            except OSError:
                memory_block = ""
        return build_system_prompt(str(self.ctx.workspace.root), self.engine.mode, self.ctx.project, memory_block)

    def _budget(self) -> int:
        tools_tokens = estimate_tokens([Message("system", json.dumps(self.registry.specs()))])
        return max(1024, int(self.context_window * 0.75) - tools_tokens)

    def _summarize(self, msgs: List[Message]) -> str:
        return self.provider.chat(msgs, None).content

    def _after_elision(self, elided_ids: List[str]) -> None:
        for call_id in elided_ids:
            path = self._read_calls.pop(call_id, None)
            if path and path in self.ctx.reads:
                self.ctx.reads[path]["in_context"] = False

    def _maybe_compact(self) -> None:
        budget = self._budget()
        if estimate_tokens(self.messages) <= budget:
            return
        result = compact(self.messages, budget, self._summarize)
        self._after_elision(result.get("elided", []))  # type: ignore[arg-type]
        self.ui.notice(
            f"Context compacted: ~{result['before']} -> ~{result['after']} tokens"
            + (" (older history summarized)" if result.get("summarized") else ""),
            "info",
        )

    def _call_model(self) -> ChatResponse:
        delays = (1.0, 3.0)
        attempt = 0
        while True:
            streamed = {"any": False}

            def on_text(chunk: str) -> None:
                if not streamed["any"]:
                    self.ui.thinking(False)
                    self.ui.assistant_start()
                    streamed["any"] = True
                self.ui.assistant_text(chunk)

            self.ui.thinking(True)
            try:
                resp = self.provider.chat(self.messages, self.registry.specs(), on_text=on_text, cancel_event=self.ctx.cancel_event)
            except ProviderError as exc:
                self.ui.thinking(False)
                if streamed["any"]:
                    self.ui.assistant_end("", True)
                if not exc.retryable or attempt >= len(delays):
                    raise
                self.ui.notice(f"Model request failed ({exc.message}); retrying in {delays[attempt]:.0f}s", "warning")
                time.sleep(delays[attempt])
                attempt += 1
                continue
            finally:
                self.ui.thinking(False)
            if streamed["any"]:
                self.ui.assistant_end(resp.content, True)
            return resp

    def _loop(self, task: str) -> TaskReport:
        settings = self.settings
        started = time.monotonic()
        baseline = porcelain_status(self.ctx.workspace.root)
        report = TaskReport(task=task, status="completed", stop_reason="the model reported it was done")
        changed: "OrderedDict[str, None]" = OrderedDict()
        call_counts: Dict[str, Tuple[int, int]] = {}
        failures = 0
        empty_replies = 0
        known = set(self.registry.names())
        self._save_session("running", "")
        try:
            while True:
                if report.iterations >= settings.max_iterations:
                    raise _Stop("stopped", f"iteration limit reached ({settings.max_iterations} model calls); raise max_iterations to allow more")
                if time.monotonic() - started > settings.max_seconds:
                    raise _Stop("stopped", f"time limit reached ({settings.max_seconds}s)")
                if settings.max_total_tokens and self.total_tokens >= settings.max_total_tokens:
                    raise _Stop("stopped", f"token budget exhausted ({self.total_tokens}/{settings.max_total_tokens})")
                self._maybe_compact()
                report.iterations += 1
                resp = self._call_model()
                if resp.finish_reason == "cancelled":
                    raise KeyboardInterrupt
                if resp.usage and (resp.usage.prompt_tokens or resp.usage.completion_tokens):
                    self.total_tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
                else:
                    self.total_tokens += estimate_tokens(self.messages[-2:]) + estimate_tokens([Message("assistant", resp.content)])
                text, calls = resp.content, list(resp.tool_calls)
                if not calls and text.strip():
                    remaining, salvaged = extract_tool_calls(text, known)
                    if salvaged:
                        self.ui.notice("Model wrote a tool call as text; interpreting it as a tool call.", "info")
                        text, calls = remaining, salvaged
                self.messages.append(Message("assistant", text, calls))
                if not calls:
                    if not text.strip():
                        empty_replies += 1
                        if empty_replies >= 2:
                            raise _Stop("stopped", "the model returned empty responses")
                        self.messages.append(Message("user", "Your last reply was empty. Continue the task with tools, or give your final summary."))
                        continue
                    report.final_message = text
                    break
                empty_replies = 0
                stuck = ""
                for call in calls:
                    report.tool_calls += 1
                    result, stuck_reason = self._handle_call(call, text, call_counts, report, changed)
                    output = result.output
                    if len(output) > MAX_TOOL_OUTPUT_CHARS:
                        output = output[:MAX_TOOL_OUTPUT_CHARS] + f"\n[output truncated at {MAX_TOOL_OUTPUT_CHARS} characters]"
                    failures = 0 if result.ok else failures + 1
                    if failures == FAILURE_HINT_AT:
                        output += "\n\nNOTE: several tool calls in a row have failed. Step back, re-read the relevant files or errors, and change approach."
                    self.messages.append(Message("tool", output, tool_call_id=call.id, name=call.name))
                    stuck = stuck or stuck_reason
                self._save_session("running", "")
                if stuck:
                    raise _Stop("stopped", stuck)
                if failures >= FAILURE_STOP_AT:
                    raise _Stop("stopped", f"{failures} consecutive tool calls failed")
        except _Stop as stop:
            report.status, report.stop_reason = stop.status, stop.reason
        except KeyboardInterrupt:
            report.status, report.stop_reason = "cancelled", "cancelled by user (Ctrl+C)"
            self._close_dangling_calls("Cancelled by the user; this tool call did not complete.")
        except ProviderError as exc:
            report.status, report.stop_reason = "failed", f"model error: {exc}"
            self._close_dangling_calls("Not executed: the model request failed.")
        report.seconds = round(time.monotonic() - started, 1)
        report.tokens = self.total_tokens
        report.file_change_count = self.file_changes
        current = {d["path"]: d for d in self.ctx.tracker.describe()}
        report.files = [current[p] for p in changed if p in current]
        after = porcelain_status(self.ctx.workspace.root)
        if baseline is not None and after is not None:
            agent_paths = set(current)
            report.changed_by_commands = sorted(
                p for p, st in after.items() if baseline.get(p) != st and p not in agent_paths
            ) + sorted(p for p in baseline if p not in after and p not in agent_paths)
        self._save_session(
            {"completed": "completed", "cancelled": "interrupted"}.get(report.status, "stopped"), report.stop_reason
        )
        return report

    def _close_dangling_calls(self, note: str) -> None:
        """Keep the transcript valid: every tool call must have a result."""
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if msg.role == "assistant":
                answered = {m.tool_call_id for m in self.messages[i + 1:] if m.role == "tool"}
                for call in msg.tool_calls:
                    if call.id not in answered:
                        self.messages.append(Message("tool", note, tool_call_id=call.id, name=call.name))
                break

    def _handle_call(
        self,
        call: ToolCall,
        intent: str,
        call_counts: Dict[str, Tuple[int, int]],
        report: TaskReport,
        changed: "OrderedDict[str, None]",
    ) -> Tuple[ToolResult, str]:
        tool = self.registry.get(call.name)
        if tool is None:
            msg = f"Unknown tool '{call.name}'. Available tools: {', '.join(self.registry.names())}"
            self.ui.tool_invalid(call.name, msg)
            return ToolResult.error(msg), ""
        if call.parse_error or call.arguments is None:
            msg = f"Invalid arguments for {call.name}: {call.parse_error or 'missing'}. Send a JSON object matching the schema."
            self.ui.tool_invalid(call.name, msg)
            return ToolResult.error(msg), ""
        args = call.arguments
        errors = validate_args(tool.parameters, args)
        if errors:
            msg = f"Invalid arguments for {call.name}: " + "; ".join(errors)
            self.ui.tool_invalid(call.name, msg)
            return ToolResult.error(msg), ""

        key = call.name + json.dumps(args, sort_keys=True)
        count, version = call_counts.get(key, (0, -1))
        count = count + 1 if version == self.state_version else 1
        call_counts[key] = (count, self.state_version)
        if count >= REPEAT_LIMIT:
            msg = f"the model repeated the same {call.name} call {count} times with nothing changing in between"
            self.ui.tool_invalid(call.name, "Repeated identical call; stopping to avoid a loop.")
            return ToolResult.error("Refused: identical call repeated without any change in between. Stopping."), msg

        try:
            assessment = tool.assess(args, self.ctx)
        except ToolError as exc:
            self.ui.tool_invalid(call.name, exc.message)
            return ToolResult.error(exc.message), ""
        except OSError as exc:
            self.ui.tool_invalid(call.name, str(exc))
            return ToolResult.error(f"{type(exc).__name__}: {exc}"), ""

        verdict = self.engine.evaluate(assessment)
        if verdict.decision == "deny":
            self.ui.tool_denied(call.name, assessment, verdict.reason)
            self.engine.record(call.name, assessment, "denied by policy")
            report.denied.append(assessment.summary)
            return ToolResult(False, f"DENIED by permission policy: {verdict.reason}", summary="denied"), ""
        approval_note = verdict.reason
        if verdict.decision == "ask":
            reason = args.get("reason") if isinstance(args.get("reason"), str) else ""
            request = ApprovalRequest(
                tool=call.name,
                assessment=assessment,
                mode=self.engine.mode,
                reason=(reason or intent or "").strip()[:600],
                cwd=str(self.ctx.workspace.root),
                can_always=self.engine.can_always(assessment),
            )
            response = self.ui.approve(request)
            if response.choice is Choice.DENY:
                self.ui.tool_denied(call.name, assessment, "denied by user")
                self.engine.record(call.name, assessment, "denied by user")
                report.denied.append(assessment.summary)
                feedback = f" User feedback: {response.feedback}" if response.feedback else ""
                return ToolResult(False, f"DENIED by the user; the action was not performed.{feedback} Do not retry it unchanged.", summary="denied by user"), ""
            if response.choice is Choice.ALWAYS:
                self.engine.remember(assessment)
            approval_note = "approved by user"
            self.engine.record(call.name, assessment, approval_note)

        self.ui.tool_started(call.name, assessment, approval_note)
        try:
            result = tool.run(args, self.ctx)
        except KeyboardInterrupt:
            raise
        except ToolError as exc:
            result = ToolResult.error(exc.message)
        except OSError as exc:
            result = ToolResult.error(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill the session; it is reported, not hidden
            if self.settings.verbose:
                self.ui.notice(traceback.format_exc(), "error")
            result = ToolResult.error(f"Internal error in {call.name}: {type(exc).__name__}: {exc}")
        self.ui.tool_finished(call.name, assessment, result)

        if assessment.risk in STATE_CHANGING:
            self.state_version += 1
        for path in result.changed_paths:
            changed[path] = None
        self.file_changes += len(result.changed_paths)
        if call.name == "run_command" and "exit_code" in result.data:
            d = result.data
            report.commands.append(CommandRecord(
                command=d.get("command", ""), exit_code=d.get("exit_code"), duration=d.get("duration", 0.0),
                timed_out=d.get("timed_out", False), cancelled=d.get("cancelled", False), is_test=d.get("is_test", False),
                after_change_index=self.file_changes, tail=d.get("tail", ""),
            ))
            if d.get("cancelled"):
                raise KeyboardInterrupt
        if call.name == "update_plan" and result.ok:
            self.ui.plan_updated(self.ctx.plan)
        if call.name == "read_file" and result.ok:
            self._read_calls[call.id] = assessment.paths[0] if assessment.paths else ""
        if count == 2:
            result.output += "\n(Note: this exact call was already made and nothing has changed since. Avoid repeating it.)"
        return result, ""

    def _save_session(self, status: str, stop_reason: str) -> None:
        if self.memory is None:
            return
        data = {
            "session_id": self.session_id,
            "task": self._current_task,
            "status": status,
            "stop_reason": stop_reason,
            "mode": self.engine.mode,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "messages": _redact_obj([m.to_dict() for m in self.messages if m.role != "system"]),
            "plan": self.ctx.plan,
            "tracker": self.ctx.tracker.to_dict(),
        }
        try:
            self.memory.session_doc(self.session_id).save(data)
        except OSError as exc:
            self.ui.notice(f"Could not save session state: {exc}", "warning")
