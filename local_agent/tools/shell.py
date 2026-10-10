"""Command execution with timeouts, cancellation, output limits and a scrubbed env.

* Commands without shell syntax run as an argv list (no shell is involved).
  Commands with pipes, redirects or globs run under ``/bin/sh -c`` on POSIX
  or ``cmd.exe`` on Windows, and only after the permission engine allows them.
* The child gets ``stdin=DEVNULL``, so interactive prompts fail fast instead of
  hanging. Pagers and git credential prompts are disabled.
* Environment variables whose names look like secrets (``*_KEY``, ``*TOKEN*``,
  ...) are removed unless listed in ``shell_env_passthrough``.
* On timeout or Ctrl+C the whole process group is terminated.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Any, Dict, List, Optional

from local_agent.permissions.commands import classify_command
from local_agent.security.secrets import is_secret_env_name, redact
from local_agent.tools.base import Assessment, Tool, ToolContext, ToolResult

_IS_WINDOWS = sys.platform == "win32"


class _CappedBuffer:
    """Keeps the head and the tail of a stream once it exceeds ``limit`` chars."""

    def __init__(self, limit: int) -> None:
        self.head_limit = max(limit * 3 // 10, 1)
        self.tail_limit = max(limit - self.head_limit, 1)
        self.head: List[str] = []
        self.head_len = 0
        self.tail: List[str] = []
        self.tail_len = 0
        self.dropped = 0

    def write(self, chunk: str) -> None:
        if self.head_len < self.head_limit:
            take = chunk[: self.head_limit - self.head_len]
            self.head.append(take)
            self.head_len += len(take)
            chunk = chunk[len(take):]
        if not chunk:
            return
        self.tail.append(chunk)
        self.tail_len += len(chunk)
        while self.tail_len > self.tail_limit and self.tail:
            excess = self.tail_len - self.tail_limit
            first = self.tail[0]
            if len(first) <= excess:
                self.tail.pop(0)
                self.tail_len -= len(first)
                self.dropped += len(first)
            else:
                self.tail[0] = first[excess:]
                self.tail_len -= excess
                self.dropped += excess

    def text(self) -> str:
        head = "".join(self.head)
        tail = "".join(self.tail)
        if self.dropped:
            return f"{head}\n... [{self.dropped} characters omitted] ...\n{tail}"
        return head + tail


def _pump(stream: IO[bytes], buf: _CappedBuffer) -> None:
    try:
        for raw in iter(lambda: stream.read(4096), b""):
            buf.write(raw.decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        pass


def child_env(passthrough: List[str]) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not is_secret_env_name(k) or k in passthrough}
    env.update({"PAGER": "cat", "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", "PYTHONUNBUFFERED": "1", "LOCAL_AGENT": "1"})
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


def _limit_resources() -> None:  # pragma: no cover - runs in the child
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


def execute(
    command: str,
    cwd: str,
    timeout: float,
    *,
    use_shell: bool,
    env: Dict[str, str],
    max_chars: int,
    cancel_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    """Run a command and return a structured, already-redacted result."""
    started = time.monotonic()
    kwargs: Dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if _IS_WINDOWS:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        argv: Any = command
        kwargs["shell"] = True
    else:
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = _limit_resources
        try:
            argv = ["/bin/sh", "-c", command] if use_shell else shlex.split(command)
        except ValueError as exc:
            return {"exit_code": 2, "stdout": "", "stderr": f"could not parse command: {exc}", "duration": 0.0, "timed_out": False, "cancelled": False}
    try:
        proc = subprocess.Popen(argv, **kwargs)
    except FileNotFoundError as exc:
        return {"exit_code": 127, "stdout": "", "stderr": f"command not found: {exc.filename or command}", "duration": 0.0, "timed_out": False, "cancelled": False}
    except OSError as exc:
        return {"exit_code": 126, "stdout": "", "stderr": f"could not start command: {exc}", "duration": 0.0, "timed_out": False, "cancelled": False}

    out_buf, err_buf = _CappedBuffer(max_chars), _CappedBuffer(max_chars // 2)
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, out_buf), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err_buf), daemon=True),
    ]
    for t in threads:
        t.start()
    timed_out = cancelled = False
    deadline = started + timeout
    try:
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                _kill_tree(proc)
                break
            if time.monotonic() > deadline:
                timed_out = True
                _kill_tree(proc)
                break
    except KeyboardInterrupt:
        _kill_tree(proc)
        for t in threads:
            t.join(timeout=1)
        raise
    for t in threads:
        t.join(timeout=2)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
    return {
        "exit_code": proc.returncode,
        "stdout": redact(out_buf.text()),
        "stderr": redact(err_buf.text()),
        "duration": round(time.monotonic() - started, 2),
        "timed_out": timed_out,
        "cancelled": cancelled,
        "truncated": bool(out_buf.dropped or err_buf.dropped),
    }


TEST_HINTS = ("pytest", "unittest", " test", "jest", "vitest", "mocha", "cargo test", "go test", "rspec", "phpunit", "tox", "nox")


def is_test_command(command: str) -> bool:
    lowered = f" {command.lower()}"
    return any(h in lowered for h in TEST_HINTS)


class RunCommand(Tool):
    name = "run_command"
    description = (
        "Run a shell command in the workspace and get its exit code, stdout and stderr. Use for tests, "
        "linters, builds and git. Commands run non-interactively (no stdin). Prefer the file tools for "
        "reading, searching and editing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command line to run.", "maxLength": 8000},
            "cwd": {"type": "string", "description": "Workspace-relative working directory (default '.')."},
            "timeout": {"type": "integer", "minimum": 1, "description": "Seconds before the command is killed."},
            "reason": {"type": "string", "description": "Why this command is needed (shown to the user)."},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        cwd = ctx.workspace.resolve(args.get("cwd", "."), must_exist=True)
        info = classify_command(args["command"], ctx.workspace)
        timeout = min(args.get("timeout", ctx.settings.command_timeout), ctx.settings.max_command_timeout)
        details = f"cwd: {ctx.workspace.relative(cwd)}\ntimeout: {timeout}s"
        if info.needs_shell:
            details += "\nruns via: " + ("cmd.exe" if _IS_WINDOWS else "/bin/sh -c")
        return Assessment(
            risk=info.risk,
            summary=f"$ {args['command']}",
            paths=[ctx.workspace.relative(cwd)],
            details=details,
            warnings=info.warnings,
            scope_key=info.scope_key,
            block_reason=info.block_reason,
        )

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        cwd = ctx.workspace.resolve(args.get("cwd", "."), must_exist=True)
        timeout = min(args.get("timeout", ctx.settings.command_timeout), ctx.settings.max_command_timeout)
        info = classify_command(command, ctx.workspace)
        res = execute(
            command,
            str(cwd),
            timeout,
            use_shell=info.needs_shell,
            env=child_env(ctx.settings.shell_env_passthrough),
            max_chars=ctx.settings.max_output_chars,
            cancel_event=ctx.cancel_event,
        )
        status = f"exit code {res['exit_code']}"
        if res["timed_out"]:
            status = f"TIMED OUT after {timeout}s (process killed)"
        elif res["cancelled"]:
            status = "CANCELLED (process killed)"
        parts = [f"$ {command}", f"{status} in {res['duration']}s"]
        if res["stdout"].strip():
            parts.append("--- stdout ---\n" + res["stdout"].rstrip())
        if res["stderr"].strip():
            parts.append("--- stderr ---\n" + res["stderr"].rstrip())
        if not res["stdout"].strip() and not res["stderr"].strip():
            parts.append("(no output)")
        ok = res["exit_code"] == 0 and not res["timed_out"] and not res["cancelled"]
        data = dict(res, command=command, is_test=is_test_command(command))
        data.pop("stdout")
        data.pop("stderr")
        data["tail"] = (res["stdout"] + "\n" + res["stderr"]).strip()[-600:]
        return ToolResult(ok, "\n".join(parts), summary=f"`{command}` -> {status}", data=data)
