"""Command-line entry point.

    local-agent                      interactive session (first run: setup wizard)
    local-agent "task"               run one task and exit
    local-agent setup                (re)run the setup wizard
    local-agent models [list]        installed models
    local-agent models use <name>    test and switch the default model
    local-agent models pull <name>   download a model with progress
    local-agent config [show|path|set <key> <value>]
    local-agent doctor [--test]      diagnose Ollama, config and model
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent import __version__
from local_agent.cli.prompts import Option, Prompter
from local_agent.cli.repl import Repl
from local_agent.cli.session import Session, build_session, resume_into
from local_agent.cli.startup import StartupAborted, start_session
from local_agent.cli.ui import TerminalUI
from local_agent.config import MODES, PROVIDERS, TOOL_MODES, Settings, load_settings
from local_agent.errors import AgentError, ConfigError, ProviderUnavailable
from local_agent.llm.factory import create_provider

SUBCOMMANDS = ("setup", "models", "config", "doctor")


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="local-agent",
        description="A local-first coding agent. Run it in a project directory and describe a task.",
        epilog=(
            "Commands:\n  local-agent setup | models [list|use <name>|pull <name>] | config [show|path|set <k> <v>] | doctor [--test]\n\n"
            "Examples:\n  local-agent\n  local-agent \"Explain this codebase\"\n  local-agent --plan \"Add input validation to the signup form\"\n"
            "  local-agent --mode auto --model qwen2.5-coder:7b \"Fix the failing tests\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("task", nargs="*", help="Task to run once, then exit. Omit for an interactive session.")
    p.add_argument("--plan", action="store_true", help="Plan mode: inspect and propose, never modify (same as --mode plan).")
    p.add_argument("--mode", choices=MODES, help="Permission mode for this run.")
    p.add_argument("--model", help="Model name for this run, e.g. qwen2.5-coder:7b")
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
    p.add_argument("--setup", action="store_true", help="Same as `local-agent setup`.")
    p.add_argument("--doctor", action="store_true", help="Same as `local-agent doctor`.")
    p.add_argument("--test", action="store_true", help="With doctor: also send a real test request to the model.")
    p.add_argument("--quiet", "-q", action="store_true", help="Skip the startup checklist.")
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


def _split_subcommand(argv: List[str]) -> tuple:
    """`setup`/`doctor` only count as commands when not followed by task words,
    so `local-agent setup the database` still runs a task."""
    if argv and argv[0] in SUBCOMMANDS:
        rest = argv[1:]
        if argv[0] in ("models", "config"):
            return argv[0], rest
        try:
            if not parse_args(rest).task:  # only options follow, e.g. `doctor -C path --test`
                return argv[0], rest
        except SystemExit:
            pass
    return None, argv


def _header(ui: TerminalUI, workspace: Path) -> None:
    s = ui.s
    ui.print(f"{s.bold(s.cyan('>_ Local-Agent-CLI'))} {s.dim('v' + __version__)}  {s.dim(str(workspace))}")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub, rest = _split_subcommand(argv)
    args = parse_args(rest)
    if args.setup:
        sub = "setup"
    elif args.doctor:
        sub = "doctor"
    ui = TerminalUI(verbose=args.verbose)
    prompter = Prompter(ui.s, ui.out, symbols=ui.sym)
    interactive = ui.interactive and not args.task and sub is None

    try:
        workspace = Path(args.workspace or os.getcwd()).resolve()
        if not workspace.is_dir():
            raise ConfigError(f"Workspace {workspace} is not a directory")

        if sub == "setup":
            from local_agent.cli.commands import cmd_setup

            return cmd_setup(ui, prompter, load_settings(overrides_from_args(args), lenient=True))
        if sub == "doctor":
            from local_agent.cli.commands import cmd_doctor

            return cmd_doctor(ui, load_settings(overrides_from_args(args), lenient=True), workspace, test=args.test)
        if sub == "models":
            from local_agent.cli.commands import cmd_models

            return cmd_models(ui, prompter, load_settings(overrides_from_args(args), lenient=True), args.task)
        if sub == "config":
            from local_agent.cli.commands import cmd_config

            try:
                current: Optional[Settings] = load_settings(overrides_from_args(args))
            except ConfigError:
                current = None
            return cmd_config(ui, prompter, current, args.task)

        settings = _load_or_repair(args, ui, prompter, interactive)
        if settings is None:
            return 1

        def run_wizard() -> Optional[Settings]:
            from local_agent.onboarding.wizard import Wizard

            base = load_settings(overrides_from_args(args), lenient=True)
            if Wizard(ui, prompter, base).run(first_run=not base.onboarding_completed) is None:
                return None
            return load_settings(overrides_from_args(args))

        if not settings.onboarding_completed and not settings.model:
            if not interactive:
                raise ConfigError("No model configured.", hint="Run `local-agent setup` once, or pass --model NAME.")
            settings = run_wizard()
            if settings is None:
                return 1
            ui.print()

        if interactive or args.task:
            _header(ui, workspace)
        session = start_session(
            settings, workspace, ui, prompter,
            interactive=interactive or (ui.interactive and args.resume),
            run_setup=run_wizard if interactive else None,
            show_checks=not args.quiet and (interactive or args.verbose),
        )
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
            ui.print(ui.s.dim(f"model {settings.model} | mode {session.engine.mode}"))
            report = session.agent.run_task(task)
            ui.report(report, show_message=not ui.is_tty)
            return 0 if report.status == "completed" else 1

        ui.print()
        ui.print(ui.s.bold("Local-Agent-CLI is ready.") + ui.s.dim("  /help for commands, Ctrl+C cancels a running task."))
        if settings.mode == "full":
            ui.print(ui.s.red("Full Control: ordinary commands run without asking and are not sandboxed."))
        ui.print("What would you like to work on?")

        def switch(model: str) -> Optional[Session]:
            from local_agent.config import load_settings as _ls

            new_settings = _ls(dict(overrides_from_args(args), model=model, mode=session.engine.mode))
            try:
                provider, info, why = create_provider(new_settings)
                return build_session(new_settings, workspace, ui, provider, info, why)
            except AgentError as exc:
                ui.print(ui.s.red(f"  {exc}"))
                return None

        return Repl(session, ui, switch_model=switch).run()
    except StartupAborted as exc:
        ui.print(ui.s.red(str(exc)))
        return exc.code
    except ProviderUnavailable as exc:
        ui.print(ui.s.red(f"Model unavailable: {exc.message}"))
        ui.print(f"  hint: {exc.hint or 'Is Ollama running? Start it with `ollama serve`, or run `local-agent doctor`.'}")
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


def _load_or_repair(args: argparse.Namespace, ui: TerminalUI, prompter: Prompter, interactive: bool) -> Optional[Settings]:
    """Load settings; on an invalid config offer to repair it via setup (interactive only)."""
    try:
        return load_settings(overrides_from_args(args))
    except ConfigError as exc:
        if not interactive:
            raise
        ui.check(False, "Configuration problem", exc.message)
        choice = prompter.select("How would you like to continue?", [
            Option("Repair it with setup", "setup", "the old file is backed up; valid settings are kept"),
            Option("Quit", "quit"),
        ])
        if choice == "quit":
            return None
        from local_agent.onboarding.wizard import Wizard

        base = load_settings(overrides_from_args(args), lenient=True)
        if Wizard(ui, prompter, base).run(first_run=False) is None:
            return None
        return load_settings(overrides_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
