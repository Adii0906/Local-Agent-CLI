"""Subcommands: `setup`, `models [list|use|pull]`, `config [show|path|set]`, `doctor`."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

from local_agent.cli.prompts import Prompter
from local_agent.cli.ui import TerminalUI
from local_agent.config import (
    CONFIG_SCHEMA_VERSION, Settings, apply_overrides, default_config_path, read_config_file, save_user_config,
)
from local_agent.errors import AgentError, ConfigError, ModelNotFound
from local_agent.llm.ollama import OllamaProvider
from local_agent.onboarding import environment as env
from local_agent.onboarding import system
from local_agent.onboarding.models import human_size, recommend_installed
from local_agent.onboarding.validate import describe, validate_model
from local_agent.onboarding.wizard import Wizard, _ProgressRenderer

MODELS_USAGE = "Usage: local-agent models [list] | models use <name> | models pull <name>"
CONFIG_USAGE = "Usage: local-agent config [show] | config path | config set <key> <value>"


def cmd_setup(ui: TerminalUI, prompter: Prompter, settings: Settings) -> int:
    if not prompter.arrow_keys and not ui.interactive:
        ui.print(ui.s.red("Setup needs an interactive terminal."))
        ui.print("  Configure non-interactively instead: local-agent config set model <name>")
        return 2
    first = not settings.onboarding_completed
    return 0 if Wizard(ui, prompter, settings).run(first_run=first) is not None else 1


def cmd_models(ui: TerminalUI, prompter: Prompter, settings: Settings, args: List[str]) -> int:
    sub = args[0] if args else "list"
    if sub == "list" and len(args) <= 1:
        return _models_list(ui, settings)
    if sub == "use" and len(args) == 2:
        return _models_use(ui, prompter, settings, args[1])
    if sub == "pull" and len(args) == 2:
        return _models_pull(ui, settings, args[1])
    ui.print(MODELS_USAGE)
    return 2


def _models_list(ui: TerminalUI, settings: Settings) -> int:
    s = ui.s
    url = settings.resolved_base_url()
    if settings.provider != "ollama":
        from local_agent.llm.openai_compat import OpenAICompatProvider

        try:
            names = OpenAICompatProvider(url, "", api_key=settings.api_key()).list_models()
        except AgentError as exc:
            ui.check(False, "Server", exc.message)
            return 1
        for n in names:
            ui.print(f"  {'*' if n == settings.model else ' '} {n}")
        return 0
    state = env.probe_ollama(url)
    if not state.reachable:
        ui.check(False, "Ollama not reachable", state.error, system.start_instructions() if state.binary else system.install_instructions())
        return 1
    chat = state.chat_models
    if not chat:
        ui.print("No chat models installed. Download one, e.g.:  local-agent models pull qwen2.5-coder:3b")
        return 0
    rec = recommend_installed(chat, system.total_memory_bytes())
    width = max(len(m.name) for m in chat) + 2
    ui.print(s.bold(f"  {'MODEL':<{width}} {'SIZE':>9}  {'PARAMS':>7}  CAPABILITIES"))
    for m in sorted(chat, key=lambda x: x.name):
        current = s.green("*") if m.name == settings.model else " "
        notes = ", ".join(m.tags())
        if rec and m.name == rec.name:
            notes += s.green("  (recommended)")
        ui.print(f"{current} {m.name:<{width}} {human_size(m.size):>9}  {m.parameter_size or '?':>7}  {notes}")
    ui.print(s.dim("  * = configured model.  Switch: local-agent models use <name>"))
    return 0


def _models_use(ui: TerminalUI, prompter: Prompter, settings: Settings, name: str) -> int:
    if settings.provider == "ollama":
        state = env.probe_ollama(settings.resolved_base_url(), with_capabilities=False)
        if not state.reachable:
            ui.check(False, "Ollama not reachable", state.error)
            return 1
        found = state.find(name)
        if found is None:
            ui.check(False, "Not installed", name)
            if not (ui.interactive and prompter.confirm(f"Download {name} now?", False)):
                ui.print(f"  Download it with: local-agent models pull {name}")
                return 1
            if _models_pull(ui, settings, name) != 0:
                return 1
        else:
            name = found.name
    probe = Settings(**{**settings.to_public_dict(), "model": name})
    ui.print(ui.s.dim(f"  Testing {name} (the first load can take a minute)..."))
    result = validate_model(probe)
    if not result.responded:
        ui.check(False, "Model test failed", result.error)
        if not (ui.interactive and prompter.confirm(f"Save {name} anyway?", False)):
            return 1
    else:
        ui.check(True if result.tool_calling == "native" else None, "Model responded", f"{result.seconds:.1f}s, tool calling: {result.tool_calling}")
        note = describe(result)
        if note:
            ui.print(ui.s.yellow("  " + note))
    try:
        path = save_user_config({"model": name, "tool_mode": result.recommended_tool_mode() if result.responded else settings.tool_mode})
    except (AgentError, OSError) as exc:
        ui.check(False, "Could not save", str(exc))
        return 1
    ui.check(True, "Default model", f"{name} (saved to {path})")
    return 0


def _models_pull(ui: TerminalUI, settings: Settings, name: str) -> int:
    if settings.provider != "ollama":
        ui.print("Downloading models is only supported for Ollama.")
        return 2
    renderer = _ProgressRenderer(ui)
    try:
        OllamaProvider(settings.resolved_base_url(), name).pull(name, renderer.update, threading.Event())
    except KeyboardInterrupt:
        renderer.finish()
        ui.check(None, "Download cancelled", "the partial download is kept and resumes next time")
        return 130
    except ModelNotFound as exc:
        renderer.finish()
        ui.check(False, "Download failed", exc.message, exc.hint)
        return 1
    except AgentError as exc:
        renderer.finish()
        ui.check(False, "Download failed", exc.message)
        return 1
    renderer.finish()
    ui.check(True, "Downloaded", name)
    return 0


def cmd_config(ui: TerminalUI, prompter: Prompter, settings: Optional[Settings], args: List[str]) -> int:
    path = default_config_path()
    sub = args[0] if args else "show"
    if sub == "path" and len(args) == 1:
        ui.print(str(path))
        return 0
    if sub == "show" and len(args) <= 1:
        cf = read_config_file(path)
        ui.print(ui.s.bold("Config file: ") + str(path) + ("" if cf.exists else ui.s.dim("  (not created yet)")))
        if cf.error:
            ui.check(False, "Problem", cf.error, "Run `local-agent setup` to repair it.")
        elif cf.exists:
            ui.print(ui.s.dim(f"  schema {CONFIG_SCHEMA_VERSION}" + (f" (migrated from {cf.migrated_from}; saved on next change)" if cf.migrated_from else "")))
        if settings is not None:
            ui.print(ui.s.bold("Effective settings") + ui.s.dim(" (defaults < file < environment < flags):"))
            for k, v in settings.to_public_dict().items():
                mark = "" if k not in cf.data else ui.s.dim("  (from file)")
                ui.print(f"  {k:22} {json.dumps(v)}{mark}")
            ui.print(ui.s.dim(f"  data_dir               {settings.data_dir}"))
        return 0
    if sub == "set" and len(args) >= 3:
        key, raw = args[1], " ".join(args[2:])
        probe = Settings()
        try:
            value = json.loads(raw) if raw[:1] in "[{" or raw in ("true", "false") else raw
            apply_overrides(probe, {key: value}, "new config values")
            if key == "mode" and probe.mode == "full" and not (ui.interactive and prompter.text(
                    "Full Control runs commands without asking and is not sandboxed. Type 'yes' to confirm", "").lower() == "yes"):
                ui.print("Unchanged.")
                return 1
            save_user_config({key: getattr(probe, key)}, path)
        except (ConfigError, ValueError) as exc:
            ui.check(False, "Not saved", str(exc))
            return 2
        ui.check(True, "Saved", f"{key} = {json.dumps(getattr(probe, key))}  ({path})")
        return 0
    ui.print(CONFIG_USAGE)
    return 2


def cmd_doctor(ui: TerminalUI, settings: Settings, workspace: Path, test: bool = False) -> int:
    s = ui.s
    ui.print(s.bold("Local-Agent-CLI doctor"))
    checks = [env.config_check(read_config_file())]
    checks += [env.Check("Config value", None, w) for w in settings.load_warnings]
    checks += env.local_checks(settings)
    checks.append(env.Check("Workspace", workspace.is_dir(), str(workspace)))
    if settings.provider == "ollama":
        state = env.probe_ollama(settings.resolved_base_url())
        checks += env.ollama_checks(state, settings.model)
        if not settings.model:
            checks.append(env.Check("Configured model", False, "none", "Run `local-agent setup` or `local-agent models use <name>`."))
    else:
        from local_agent.llm.openai_compat import OpenAICompatProvider

        try:
            names = OpenAICompatProvider(settings.resolved_base_url(), "", api_key=settings.api_key()).list_models()
            checks.append(env.Check("Server", True, f"{settings.resolved_base_url()} ({len(names)} models)"))
            checks.append(env.Check("Configured model", settings.model in names, settings.model or "none"))
        except AgentError as exc:
            checks.append(env.Check("Server", False, exc.message))
    for c in checks:
        ui.check(c.ok, c.name, c.detail, c.hint if c.ok is not True else "")
    problems = [c for c in checks if c.ok is False]
    if test and settings.model and not problems:
        ui.print(s.dim(f"  Sending a test request to {settings.model}..."))
        result = validate_model(settings)
        ui.check(True if result.tool_calling == "native" else (None if result.responded else False), "Model test",
                 f"{result.seconds:.1f}s, tool calling: {result.tool_calling}" if result.responded else result.error)
        if not result.responded:
            problems.append(env.Check("Model test", False, result.error))
    elif not test:
        ui.print(s.dim("  Run `local-agent doctor --test` to also send a real test request to the model."))
    ui.print(s.green("No problems found.") if not problems else s.red(f"{len(problems)} problem(s) found."))
    return 0 if not problems else 1
