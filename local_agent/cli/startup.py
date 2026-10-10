"""Launch sequence for returning users: verify each piece, then start the session.

Every success line is printed only after that step actually succeeded. When a
step fails in an interactive terminal, the user gets a recovery menu (retry,
start Ollama, pick another installed model, download, run setup) instead of a
traceback.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from local_agent.cli.prompts import Option, Prompter
from local_agent.cli.session import Session, build_session
from local_agent.cli.ui import TerminalUI
from local_agent.config import Settings, default_config_path, save_user_config
from local_agent.errors import AgentError, ModelNotFound, ProviderUnavailable
from local_agent.llm.factory import create_provider
from local_agent.llm.ollama import OllamaProvider
from local_agent.onboarding import system
from local_agent.onboarding.models import human_size

MODE_LABELS = {"plan": "Plan Mode", "ask": "Ask Before Changes", "auto": "Auto Edit", "full": "Full Control"}


class StartupAborted(Exception):
    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def start_session(
    settings: Settings,
    workspace: Path,
    ui: TerminalUI,
    prompter: Prompter,
    *,
    interactive: bool,
    run_setup: Optional[Callable[[], Optional[Settings]]] = None,
    show_checks: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> Session:
    """Verify config, server and model, then build the session. Raises StartupAborted."""
    show = show_checks and settings.show_startup_checks
    while True:
        try:
            return _attempt(settings, workspace, ui, prompter, interactive, show, sleep)
        except _Restart as restart:
            if restart.settings is not None:
                settings = restart.settings
            elif restart.setup:
                if run_setup is None:
                    raise StartupAborted("Run `local-agent setup` to configure a model.", 2) from None
                new = run_setup()
                if new is None:
                    raise StartupAborted("Setup was cancelled.", 1) from None
                settings = new
            ui.print()


class _Restart(Exception):
    """Start over: with new settings, by running setup, or as a plain retry."""

    def __init__(self, settings: Optional[Settings] = None, setup: bool = False) -> None:
        super().__init__("restart")
        self.settings = settings
        self.setup = setup


def _attempt(settings: Settings, workspace: Path, ui: TerminalUI, p: Prompter, interactive: bool, show: bool, sleep) -> Session:
    check = ui.check if show else (lambda *a, **k: None)
    path = default_config_path()
    check(True, "Configuration loaded", str(path) if path.exists() else "defaults (no config file)")
    for w in settings.load_warnings:
        ui.check(None, "Config", w)

    # Server
    url = settings.resolved_base_url()
    if settings.provider == "ollama":
        while True:
            try:
                version = OllamaProvider(url, settings.model or "").version()
                check(True, "Ollama connected", f"v{version} at {url}")
                break
            except AgentError as exc:
                ui.check(False, "Ollama not reachable", exc.message)
                binary = system.ollama_binary()
                hint = system.start_instructions() if binary else "Ollama is not installed. " + system.install_instructions()
                if not interactive:
                    raise StartupAborted(f"Cannot reach Ollama at {url}. {hint}") from None
                ui.print(ui.s.dim("  " + hint))
                options = [Option("Retry", "retry")]
                if binary:
                    options.insert(0, Option("Start Ollama for me", "start", "runs `ollama serve` in the background"))
                options += [Option("Run setup", "setup"), Option("Quit", "quit")]
                choice = p.select("Ollama is not responding. What would you like to do?", options)
                if choice == "quit":
                    raise StartupAborted("Ollama is not running.", 2) from None
                if choice == "setup":
                    raise _Restart(setup=True) from None
                if choice == "start" and binary:
                    if system.start_ollama_background(binary):
                        ui.print(ui.s.dim("  Started `ollama serve`; waiting for it..."))
                        for _ in range(10):
                            sleep(1)
                            try:
                                OllamaProvider(url, "").version()
                                break
                            except AgentError:
                                continue

    # Model
    if not settings.model:
        if interactive:
            ui.check(False, "No model configured")
            raise _Restart(setup=True)
        raise StartupAborted("No model configured. Run `local-agent setup` or pass --model NAME.")
    try:
        provider, info, why = create_provider(settings)
    except ModelNotFound as exc:
        ui.check(False, "Model unavailable", exc.message)
        if not interactive:
            raise StartupAborted(f"{exc.message} Run `ollama pull {settings.model}` or `local-agent models use <name>`.") from None
        recovered = _recover_model(settings, ui, p)
        raise _Restart(recovered, setup=recovered is None) from None
    except ProviderUnavailable as exc:
        ui.check(False, "Model server unavailable", exc.message)
        if not interactive:
            raise StartupAborted(exc.message + (f" ({exc.hint})" if exc.hint else "")) from None
        choice = p.select("What would you like to do?", [Option("Retry", "retry"), Option("Run setup", "setup"), Option("Quit", "quit")])
        if choice == "quit":
            raise StartupAborted("Model server unavailable.") from None
        raise _Restart(setup=choice == "setup") from None
    tool_note = "native tool calling" if provider.tool_mode == "native" else "prompt-based tool calling"
    check(True, "Model", f"{settings.model} ({tool_note})")
    if provider.tool_mode != "native":
        ui.check(None, "Tool calling", f"{why}. A tool-capable coding model edits files more reliably.")
    check(True, "Permission mode", f"{MODE_LABELS.get(settings.mode, settings.mode)} ({settings.mode})")

    session = build_session(settings, workspace, ui, provider, info, why)
    langs = ", ".join(session.project.languages[:3]) or "no source files detected"
    check(True, "Project detected", f"{session.workspace.root.name} ({langs})")
    if session.memory is not None:
        facts = len(session.memory.facts())
        check(True, "Memory initialized", f"{facts} project fact(s), {len(session.memory.tasks())} past task(s)")
        for w in session.memory.warnings():
            ui.check(None, "Memory", w)
    else:
        check(None, "Memory", "disabled (--no-memory)")
    return session


def _recover_model(settings: Settings, ui: TerminalUI, p: Prompter) -> Optional[Settings]:
    """The configured model is gone. Offer installed alternatives, download, or setup."""
    from local_agent.onboarding.environment import probe_ollama
    from local_agent.onboarding.wizard import _ProgressRenderer

    url = settings.resolved_base_url()
    state = probe_ollama(url, with_capabilities=False) if settings.provider == "ollama" else None
    options = []
    if state is not None:
        for m in state.chat_models[:12]:
            options.append(Option(f"Use {m.name}", ("use", m.name), human_size(m.size)))
        options.append(Option(f"Download {settings.model}", ("pull", settings.model)))
    options += [Option("Run setup", ("setup", None)), Option("Quit", ("quit", None))]
    action, name = p.select(f"'{settings.model}' is not available. What would you like to do?", options)  # type: ignore[misc]
    if action == "quit":
        raise StartupAborted("No usable model.", 2)
    if action == "setup":
        return None
    if action == "pull":
        renderer = _ProgressRenderer(ui)
        try:
            OllamaProvider(url, str(name)).pull(str(name), renderer.update)
            renderer.finish()
            ui.check(True, "Downloaded", str(name))
        except KeyboardInterrupt:
            renderer.finish()
            ui.check(None, "Download cancelled", "the partial download resumes next time")
        except AgentError as exc:
            renderer.finish()
            ui.check(False, "Download failed", exc.message)
        return settings
    settings.model = str(name)
    if p.confirm(f"Make {name} your default model?", True):
        try:
            save_user_config({"model": str(name)})
            ui.check(True, "Default model saved", str(name))
        except (AgentError, OSError) as exc:
            ui.check(False, "Could not save", str(exc))
    return settings
