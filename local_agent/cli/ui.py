"""Terminal rendering: execution trace, approval prompts, diffs and reports.

Plain ANSI output with no required dependencies. Colors are disabled when stdout
is not a TTY or ``NO_COLOR`` is set. Every line shown corresponds to a real
event from the agent loop. There are no simulated progress indicators.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import Dict, List, Optional, TextIO

from local_agent.agent.events import AgentUI
from local_agent.agent.report import TaskReport
from local_agent.permissions.engine import ApprovalRequest, ApprovalResponse, Choice
from local_agent.tools.base import RISK_DESCRIPTIONS, Assessment, Risk, ToolResult


def _enable_windows_terminal() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass
        os.system("")  # enables ANSI escape processing in classic consoles  # noqa: S605,S607


class Style:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str: return self._wrap("1", t)
    def dim(self, t: str) -> str: return self._wrap("2", t)
    def red(self, t: str) -> str: return self._wrap("31", t)
    def green(self, t: str) -> str: return self._wrap("32", t)
    def yellow(self, t: str) -> str: return self._wrap("33", t)
    def blue(self, t: str) -> str: return self._wrap("34", t)
    def magenta(self, t: str) -> str: return self._wrap("35", t)
    def cyan(self, t: str) -> str: return self._wrap("36", t)


FANCY_SYMBOLS = {"ok": "✓", "fail": "✗", "run": "→", "warn": "!", "dot": "•", "on": "●", "off": "○",
                 "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"}
ASCII_SYMBOLS = {"ok": "+", "fail": "x", "run": ">", "warn": "!", "dot": "*", "on": "(*)", "off": "( )",
                 "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"}


def _symbols(stream: TextIO) -> Dict[str, str]:
    try:
        "".join(FANCY_SYMBOLS.values()).encode(getattr(stream, "encoding", None) or "ascii")
        return dict(FANCY_SYMBOLS)
    except (UnicodeEncodeError, LookupError):
        return dict(ASCII_SYMBOLS)


def visible_len(text: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;?]*[A-Za-z]", "", text))


class TerminalUI(AgentUI):
    def __init__(self, verbose: bool = False, out: Optional[TextIO] = None, interactive: Optional[bool] = None) -> None:
        _enable_windows_terminal()
        out = out if out is not None else sys.stdout
        self.out = out
        self.verbose = verbose
        self.is_tty = hasattr(out, "isatty") and out.isatty()
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.s = Style(self.is_tty and not os.environ.get("NO_COLOR"))
        self.sym = _symbols(out)
        self._thinking_shown = False
        self._streaming = False

    # -- primitives --------------------------------------------------------
    def print(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def width(self) -> int:
        return max(60, min(shutil.get_terminal_size((100, 20)).columns, 120))

    def box(self, lines: List[str], width: int = 48) -> None:
        """A rounded box (ASCII on terminals that can't encode box characters)."""
        y = self.sym
        inner = max([width] + [visible_len(l) + 4 for l in lines])
        self.print(self.s.cyan(y["tl"] + y["h"] * inner + y["tr"]))
        for line in [""] + lines + [""]:
            pad = inner - visible_len(line) - 3
            self.print(self.s.cyan(y["v"]) + "   " + line + " " * max(pad, 0) + self.s.cyan(y["v"]))
        self.print(self.s.cyan(y["bl"] + y["h"] * inner + y["br"]))

    def check(self, ok: Optional[bool], label: str, detail: str = "", hint: str = "") -> None:
        """One result line. Only call after the thing being reported has actually happened."""
        if ok is True:
            mark = self.s.green(self.sym["ok"])
        elif ok is False:
            mark = self.s.red(self.sym["fail"])
        else:
            mark = self.s.yellow(self.sym["warn"])
        text = f"{label}: {detail}" if detail else label
        self.print(f"  {mark} {text}")
        if hint:
            self.print(self.s.dim(f"      {hint}"))

    def step(self, number: int, total: int, title: str) -> None:
        self.print()
        self.print(self.s.bold(f"[{number}/{total}] {title}"))

    def diff(self, text: str, max_lines: int = 60) -> None:
        lines = text.splitlines()
        for line in lines[:max_lines]:
            if line.startswith("+") and not line.startswith("+++"):
                self.print("    " + self.s.green(line))
            elif line.startswith("-") and not line.startswith("---"):
                self.print("    " + self.s.red(line))
            elif line.startswith("@@"):
                self.print("    " + self.s.cyan(line))
            else:
                self.print("    " + self.s.dim(line))
        if len(lines) > max_lines:
            self.print(self.s.dim(f"    ... {len(lines) - max_lines} more lines"))

    # -- AgentUI -----------------------------------------------------------
    def thinking(self, active: bool) -> None:
        if not self.is_tty:
            return
        if active and not self._thinking_shown:
            self.out.write(self.s.dim("  waiting for model..."))
            self.out.flush()
            self._thinking_shown = True
        elif not active and self._thinking_shown:
            self.out.write("\r\033[K")
            self.out.flush()
            self._thinking_shown = False

    def assistant_start(self) -> None:
        self._streaming = True
        self.out.write("\n")

    def assistant_text(self, chunk: str) -> None:
        self.out.write(chunk)
        self.out.flush()

    def assistant_end(self, full_text: str, streamed: bool) -> None:
        if self._streaming:
            self.out.write("\n")
            self.out.flush()
        self._streaming = False

    def tool_started(self, tool: str, assessment: Assessment, auto_reason: str = "") -> None:
        note = ""
        if self.verbose and auto_reason:
            note = self.s.dim(f"  ({auto_reason})")
        self.print(f"  {self.s.cyan(self.sym['run'])} {assessment.summary}{note}")

    def tool_finished(self, tool: str, assessment: Assessment, result: ToolResult) -> None:
        mark = self.s.green(self.sym["ok"]) if result.ok else self.s.red(self.sym["fail"])
        self.print(f"  {mark} {result.summary or ('done' if result.ok else 'failed')}")
        if tool == "edit_file" and result.ok and "\n" in result.output:
            self.diff(result.output.split("\n", 1)[1], max_lines=30 if not self.verbose else 200)
        elif tool == "run_command" and (not result.ok or self.verbose):
            body = result.output.split("\n", 2)[2] if result.output.count("\n") >= 2 else ""
            lines = body.splitlines()
            shown = lines[-15:] if not self.verbose else lines[-80:]
            if len(lines) > len(shown):
                self.print(self.s.dim(f"    ... ({len(lines) - len(shown)} earlier lines)"))
            for line in shown:
                self.print(self.s.dim("    " + line))
        elif self.verbose and result.output:
            for line in result.output.splitlines()[:25]:
                self.print(self.s.dim("    " + line))

    def tool_denied(self, tool: str, assessment: Assessment, reason: str) -> None:
        self.print(f"  {self.s.yellow(self.sym['warn'])} Not performed: {assessment.summary} {self.s.dim('(' + reason + ')')}")

    def tool_invalid(self, tool: str, message: str) -> None:
        self.print(f"  {self.s.red(self.sym['fail'])} {tool}: {message}")

    def plan_updated(self, plan: List[Dict[str, str]]) -> None:
        marks = {"pending": " ", "in_progress": self.sym["run"], "done": self.sym["ok"], "blocked": self.sym["warn"], "skipped": "-"}
        self.print(self.s.bold("  Plan:"))
        for step in plan:
            mark = marks.get(step["status"], " ")
            text = step["step"]
            if step["status"] == "done":
                text = self.s.dim(text)
            elif step["status"] == "in_progress":
                text = self.s.bold(text)
            self.print(f"    [{mark}] {text}")

    def notice(self, message: str, level: str = "info") -> None:
        color = {"warning": self.s.yellow, "error": self.s.red}.get(level, self.s.dim)
        prefix = {"warning": self.sym["warn"] + " ", "error": self.sym["fail"] + " "}.get(level, "")
        self.print(color(f"  {prefix}{message}"))

    def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        a = request.assessment
        s = self.s
        if not self.interactive:
            self.print(s.yellow(f"  {self.sym['warn']} Needs approval but no interactive terminal: denied: {a.summary}"))
            return ApprovalResponse(Choice.DENY, "Running non-interactively; actions needing approval are denied.")
        if self._thinking_shown:
            self.thinking(False)
        self.print()
        self.print(s.yellow(s.bold(f"{self.sym['warn']} Permission needed")) + s.dim(f"  [{request.mode} mode, {a.risk.value}: {RISK_DESCRIPTIONS[a.risk]}]"))
        self.print(f"  {s.bold('Action:')}  {a.summary}")
        self.print(f"  {s.bold('Workdir:')} {request.cwd}")
        if a.paths:
            self.print(f"  {s.bold('Paths:')}   {', '.join(a.paths)}")
        if request.reason:
            wrapped = textwrap.fill(request.reason, width=self.width() - 12, subsequent_indent=" " * 11)
            self.print(f"  {s.bold('Why:')}     {wrapped}")
        for w in a.warnings:
            self.print(f"  {s.red('Risk:')}    {w}")
        if a.risk in (Risk.SHELL, Risk.NETWORK, Risk.PRIVILEGED):
            self.print(s.dim("           Commands run with your user's privileges; this is not a sandbox."))
        if a.details:
            if a.details.lstrip().startswith(("---", "+", "@@")):
                self.diff(a.details, max_lines=80)
            else:
                for line in a.details.splitlines()[:40]:
                    self.print(s.dim("    " + line))
        options = ["[y] approve once", "[n] deny", "[f] deny with feedback"]
        if request.can_always and a.scope_key:
            options.insert(1, f"[a] always allow '{a.scope_key.split(':', 1)[1]}' this session")
        self.print("  " + "   ".join(options))
        while True:
            try:
                answer = input("  > ").strip().lower()
            except EOFError:
                return ApprovalResponse(Choice.DENY, "No input available.")
            if answer in ("y", "yes"):
                return ApprovalResponse(Choice.ONCE)
            if answer in ("n", "no", ""):
                return ApprovalResponse(Choice.DENY)
            if answer == "a" and request.can_always and a.scope_key:
                return ApprovalResponse(Choice.ALWAYS)
            if answer == "f":
                try:
                    fb = input("  feedback for the agent: ").strip()
                except EOFError:
                    fb = ""
                return ApprovalResponse(Choice.DENY, fb)
            self.print(s.dim("  Please answer y, n, f" + (", or a" if request.can_always and a.scope_key else "")))

    # -- reports -------------------------------------------------------------
    def report(self, report: TaskReport, show_message: bool = False) -> None:
        s = self.s
        if show_message and report.final_message:
            self.print()
            self.print(report.final_message)
        self.print()
        status_color = {"completed": s.green, "cancelled": s.yellow}.get(report.status, s.red)
        self.print(s.bold("── Task report ") + s.dim("(from recorded tool results)"))
        self.print(f"  Status:   {status_color(report.status)}: {report.stop_reason}")
        if report.files:
            self.print("  Files:    " + ", ".join(f"{f['path']} ({f['change']})" for f in report.files))
        else:
            self.print("  Files:    no files changed by the agent's tools")
        if report.commands:
            self.print("  Commands:")
            for c in report.commands[-10:]:
                if c.timed_out:
                    st = s.red("timed out")
                elif c.cancelled:
                    st = s.yellow("cancelled")
                else:
                    st = s.green("exit 0") if c.exit_code == 0 else s.red(f"exit {c.exit_code}")
                self.print(f"    {'[test] ' if c.is_test else ''}{c.command}  -> {st} ({c.duration}s)")
        for note in report.verification():
            self.print(f"  {s.yellow(self.sym['warn'])} {note}")
        self.print(s.dim(f"  {report.iterations} model calls, {report.tool_calls} tool calls, ~{report.tokens} tokens, {report.seconds}s"))
        if report.files:
            self.print(s.dim("  Review with /diff, revert with /undo."))
