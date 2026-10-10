"""Command-line entry point: ``local-agent [task] [options]``."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent import __version__
from local_agent.cli.banner import BANNER, SUBTITLE
from local_agent.cli.repl import Repl
from local_agent.cli.session import Session, build_session, resume_into
from local_agent.cli.ui import TerminalUI
from local_agent.config import MODES, PROVIDERS, TOOL_MODES, Settings, default_config_path, load_settings, save_user_config
from local_agent.errors import AgentError, ConfigError, ProviderUnavailable
from local_agent.llm.catalog import RECOMMENDED_MODELS
from local_agent.llm.factory import create_base_provider, create_provider
from local_agent.permissions.engine import MODE_DESCRIPTIONS


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="local-agent",
        description="A local-first coding agent. Run it in a project directory and describe a task.",
        epilog="Examples:\n  local-agent\n  local-agent \"Explain this codebase\"\n  local-agent --plan \"Add input validation to the signup form\"\n  local-agent --mode auto --model qwen2.5-coder:7b \"Fix the failing tests\"",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="*", help="Task to run once, then exit. Omit for an interactive session.")
    p.add_argument("--plan", action="store_true", help="Plan mode: inspect and propose, never modify (same as --mode plan).")
    p.add_argument("--mode", choices=MODES, help="Permission mode (default: ask).")
    p.add_argument("--model", help="Model name, e.g. qwen2.5-coder:7b")
    p.add_argument("--provider", choices=PROVIDERS, help="Model backend (default: ollama).")
    p.add_argument("--base-url", help="Model server URL (default: http://localhost:11434 for Ollama).")
    p.add_argument("--tool-mode", choices=TOOL_MODES, help="Native tool calling, prompt-based, or auto-detect.")
    p.add_argument("--workspace", "-C", help="Authorized workspace directory (default: current directory).")
    p.add_argument("--max-iterations", type=int, help="Max model calls per task.")
    p.add_argument("--max-seconds", type=int, help="Max wall-clock seconds per task.")
    p.add_argument("--max-tokens", type=int, dest="max_total_tokens", help="Token budget per session (0 = unlimited).")
    p.add_argument("--context-window", type=int, help="Override the model context size (tokens).")
    p.add_argument("--no-memory", action="store_true", help="Do not read or write persistent memory.")
    p.add_argument("--resume", action="store_true", help="Resume the last interrupted task in this project.")
    p.add_argument("--setup", action="store_true", help="Interactive first-time configuration.")
    p.add_argument("--doctor", action="store_true", help="Check model server, model capabilities and tools.")
    p.add_argument("--verbose", "-v", action="store_true", help="Show tool output and debugging details.")
    p.add_argument("--version", action="version", version=f"local-agent {__version__}")
    return p.parse_args(argv)


def overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    o: Dict[str, Any] = {
        "model": args.model, "provider": args.provider, "base_url": args.base_url, "tool_mode": args.tool_mode,
        "max_iterations": args.max_iterations, "max_seconds": args.max_seconds,
        "max_total_tokens": args.max_total_tokens, "context_window": args.context_window,
        "mode": "plan" if args.plan else args.mode,
    }
    if args.no_memory:
        o["memory_enabled"] = False
    if args.verbose:
        o["verbose"] = True
    return o


def run_setup(ui: TerminalUI, settings: Settings) -> int:
    s = ui.s
    ui.print(s.bold("Local Agent setup") + s.dim(f"  (writes {default_config_path()})"))
    try:
        provider = input(f"  Provider [ollama/openai] ({settings.provider}): ").strip() or settings.provider
        if provider not in PROVIDERS:
            ui.print(s.red("  Unknown provider."))
            return 2
        settings.provider = provider
        default_url = settings.resolved_base_url() if provider == settings.provider else ""
        base_url = input(f"  Server URL ({default_url}): ").strip() or None
        settings.base_url = base_url
        values: Dict[str, Any] = {"provider": provider}
        if base_url:
            values["base_url"] = base_url
        if provider == "openai":
            env_name = input(f"  Name of the env var holding the API key ({settings.api_key_env}, blank if none): ").strip()
            if env_name:
                values["api_key_env"] = env_name
                settings.api_key_env = env_name
            ui.print(s.dim("  The key itself is never written to the config file."))
        installed: List[str] = []
        settings.model = settings.model or "placeholder"
        try:
            installed = create_base_provider(settings).list_models()
        except AgentError as exc:
            ui.print(s.yellow(f"  Could not reach the server: {exc}"))
        choices = installed or [m["name"] for m in RECOMMENDED_MODELS]
        ui.print("  Models " + ("installed on the server:" if installed else "(suggested; not yet installed):"))
        for i, name in enumerate(choices, 1):
            desc = next((m["description"] for m in RECOMMENDED_MODELS if m["name"] == name), "")
            ui.print(f"    {i:2}. {name:28} {s.dim(desc)}")
        pick = input("  Choose a number or type a model name: ").strip()
        model = choices[int(pick) - 1] if pick.isdigit() and 0 < int(pick) <= len(choices) else pick
        if not model:
            ui.print(s.red("  No model chosen."))
            return 2
        values["model"] = model
        if provider == "ollama" and installed and model not in installed and shutil.which("ollama"):
            if input(f"  Download {model} now with `ollama pull`? [y/N] ").strip().lower() == "y":
                import subprocess

                subprocess.run(["ollama", "pull", model])
        ui.print("  Default permission mode:")
        for m in MODES:
            ui.print(f"    {m:5} {MODE_DESCRIPTIONS[m]}")
        mode = input("  Mode (ask): ").strip() or "ask"
        if mode not in MODES:
            ui.print(s.red("  Unknown mode."))
            return 2
        values["mode"] = mode
    except (EOFError, KeyboardInterrupt):
        ui.print("\n  Setup cancelled.")
        return 1
    path = save_user_config(values)
    ui.print(s.green(f"  Saved {path}. Run `local-agent --doctor` to verify."))
    return 0


def run_doctor(ui: TerminalUI, settings: Settings, workspace: Path) -> int:
    s = ui.s
    ok = True

    def line(good: Optional[bool], label: str, detail: str) -> None:
        mark = s.green(ui.sym["ok"]) if good else (s.yellow(ui.sym["warn"]) if good is None else s.red(ui.sym["fail"]))
        ui.print(f"  {mark} {label:16} {detail}")

    ui.print(s.bold("Local Agent doctor"))
    line(True, "Python", sys.version.split()[0])
    line(True, "Config file", str(default_config_path()) + ("" if default_config_path().exists() else " (not created; using defaults)"))
    line(True, "Workspace", str(workspace))
    line(bool(shutil.which("git")), "git", shutil.which("git") or "not found (diff/undo still work for agent edits)")
    line(True if shutil.which("rg") else None, "ripgrep", shutil.which("rg") or "not found; using slower Python search")
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        line(True, "Data dir", str(settings.data_dir))
    except OSError as exc:
        ok = False
        line(False, "Data dir", f"{settings.data_dir}: {exc}")
    if not settings.model:
        line(False, "Model", "none configured: run `local-agent --setup` or pass --model")
        return 1
    try:
        provider, info, why = create_provider(settings)
        line(True, "Server", f"{settings.provider} at {settings.resolved_base_url()}")
        line(True, "Model", f"{info.name} {info.details}".strip())
        line(True if info.context_length else None, "Context", f"{info.context_length or 'unknown'} tokens (model)")
        native = provider.tool_mode == "native"
        line(True if native else None, "Tool calling", f"{provider.tool_mode}: {why}")
    except AgentError as exc:
        ok = False
        line(False, "Model server", str(exc))
    return 0 if ok else 1


def _make_session(settings: Settings, workspace: Path, ui: TerminalUI) -> Session:
    provider, info, why = create_provider(settings)
    return build_session(settings, workspace, ui, provider, info, why)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    ui = TerminalUI(verbose=args.verbose)
    try:
        settings = load_settings(overrides_from_args(args))
        workspace = Path(args.workspace or os.getcwd()).resolve()
        if not workspace.is_dir():
            raise ConfigError(f"Workspace {workspace} is not a directory")
        if args.setup:
            return run_setup(ui, settings)
        if args.doctor:
            return run_doctor(ui, settings, workspace)
        if not settings.model:
            if sys.stdin.isatty() and not args.task:
                ui.print("No model configured yet. Starting setup.\n")
                rc = run_setup(ui, settings)
                if rc != 0:
                    return rc
                settings = load_settings(overrides_from_args(args))
            else:
                raise ConfigError("No model configured.", hint="Run `local-agent --setup` or pass --model NAME.")
        if settings.mode == "full" and sys.stdin.isatty():
            ui.print(ui.s.red("Full-control mode: ordinary commands run without asking and are not sandboxed."))

        session = _make_session(settings, workspace, ui)
        if session.memory:
            for w in session.memory.warnings():
                ui.notice(w, "warning")
        task = " ".join(args.task).strip()

        if args.resume:
            data = session.memory.latest_interrupted_session() if session.memory else None
            if not data:
                ui.print("No interrupted task to resume in this project.")
                return 1
            task = resume_into(session, data)
            ui.print(f"Resuming: {task[:100]!r}")
            report = session.agent.run_task(task, resumed=True)
            ui.report(report, show_message=not ui.is_tty)
            if not sys.stdin.isatty():
                return 0 if report.status == "completed" else 1
        elif task:
            ui.print(ui.s.dim(f"Workspace {session.workspace.root} | model {settings.model} | mode {session.engine.mode}"))
            report = session.agent.run_task(task)
            ui.report(report, show_message=not ui.is_tty)
            return 0 if report.status == "completed" else 1

        if ui.is_tty:
            ui.print(ui.s.cyan(BANNER))
            ui.print(ui.s.dim(SUBTITLE))
        ui.print()
        ui.print(f"  Workspace: {session.workspace.root}")
        ui.print(f"  Model:     {settings.model} ({session.provider.tool_mode} tool calling)")
        ui.print(f"  Mode:      {session.engine.mode}: {MODE_DESCRIPTIONS[session.engine.mode]}")
        if session.memory and session.memory.facts():
            ui.print(f"  Memory:    {len(session.memory.facts())} project fact(s) loaded (/memory to inspect)")
        ui.print(ui.s.dim("  Type a task, or /help. Ctrl+C cancels a running task."))
        if session.provider.tool_mode != "native":
            ui.notice(f"{session.tool_mode_reason}. Prompt-based tool calls are less reliable; a tool-capable model is recommended.", "warning")

        def switch(model: str) -> Optional[Session]:
            new_settings = load_settings(dict(overrides_from_args(args), model=model, mode=session.engine.mode))
            try:
                return _make_session(new_settings, workspace, ui)
            except AgentError as exc:
                ui.print(ui.s.red(f"  {exc}"))
                return None

        return Repl(session, ui, switch_model=switch).run()
    except ProviderUnavailable as exc:
        ui.print(ui.s.red(f"Model unavailable: {exc.message}"))
        hint = exc.hint or (
            "Is Ollama running? Start it with `ollama serve`, or check --base-url." if "ollama" in str(exc).lower() or "11434" in str(exc) else "Check the server URL and model name."
        )
        ui.print(f"  hint: {hint}")
        return 2
    except AgentError as exc:
        ui.print(ui.s.red(f"Error: {exc.message}"))
        if exc.hint:
            ui.print(f"  hint: {exc.hint}")
        return 2
    except KeyboardInterrupt:
        ui.print("\nInterrupted.")
        return 130
    except Exception:  # noqa: BLE001 - last-resort handler for genuine bugs
        if args.verbose:
            traceback.print_exc()
        else:
            ui.print(ui.s.red(f"Unexpected error: {sys.exc_info()[1]!r}. Re-run with --verbose for the traceback."))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
