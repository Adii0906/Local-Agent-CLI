"""The interactive first-run setup (`local-agent setup`).

Steps: welcome -> [1/4] environment -> [2/4] model -> [3/4] permissions -> [4/4] save.

Nothing is downloaded, started or written without an explicit choice by the
user. The config file is only written at the end, after a summary, and
unrelated settings in it are preserved.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from local_agent import __version__
from local_agent.cli.prompts import Option, Prompter
from local_agent.cli.ui import TerminalUI
from local_agent.config import (
    DEFAULT_BASE_URLS, Settings, config_diff, default_config_path, read_config_file, save_user_config,
)
from local_agent.errors import AgentError, ModelNotFound
from local_agent.llm.ollama import OllamaProvider, PullProgress
from local_agent.onboarding import environment as env
from local_agent.onboarding import system
from local_agent.onboarding.models import ModelEntry, human_size, recommend_download, recommend_installed
from local_agent.onboarding.validate import ValidationResult, describe, validate_model
from local_agent.permissions.engine import MODE_DESCRIPTIONS

TAGLINE = "Your local coding agent. Your rules."

MODE_OPTIONS = [
    ("plan", "Plan Mode", "Inspect and propose changes without modifying files."),
    ("ask", "Ask Before Changes", "Request approval for file modifications and risky commands."),
    ("auto", "Auto Edit", "Edit files in this project without asking; approval needed for commands, deletes and sensitive actions."),
    ("full", "Full Control", "Also run ordinary commands without asking. Deletes, network, privileged and secret-file actions still ask."),
]


class SetupCancelled(Exception):
    pass


def welcome(ui: TerminalUI) -> None:
    s = ui.s
    ui.print()
    ui.box([
        s.bold(s.cyan(">_  Local-Agent-CLI")) + s.dim(f"  v{__version__}"),
        "",
        TAGLINE,
        "",
        "Read files. Write code. Run commands.",
        "Remember your projects.",
        "",
        s.dim("Runs on local LLMs via Ollama."),
        s.dim("Private by default. Permission first."),
    ])


class Wizard:
    def __init__(
        self,
        ui: TerminalUI,
        prompter: Prompter,
        settings: Settings,
        config_path: Optional[Path] = None,
        *,
        probe: Callable[[str], env.OllamaState] = env.probe_ollama,
        validator: Callable[[Settings], ValidationResult] = validate_model,
        memory_bytes: Optional[int] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ui = ui
        self.p = prompter
        self.settings = settings
        self.config_path = config_path or default_config_path()
        self.probe = probe
        self.validator = validator
        self.memory = memory_bytes if memory_bytes is not None else system.total_memory_bytes()
        self.sleep = sleep
        self.chosen: Dict[str, object] = {}

    # -- public --------------------------------------------------------------
    def run(self, first_run: bool = True) -> Optional[Settings]:
        """Run the whole flow. Returns the saved settings, or None if cancelled."""
        s = self.ui.s
        welcome(self.ui)
        self.ui.print()
        if first_run:
            self.ui.print("Welcome! Let's get your agent ready.")
        else:
            self.ui.print("Let's review your setup. Current values are pre-selected; nothing changes until you confirm.")
        for i, name in enumerate(("Check your environment", "Choose your model", "Configure permissions", "Start coding"), 1):
            self.ui.print(s.dim(f"  [{i}/4] {name}"))
        try:
            state = self.step_environment()
            if state is None:
                return None
            model = self.step_model(state)
            if model is None:
                return None
            mode = self.step_permissions()
            return self.step_save(model, mode)
        except (KeyboardInterrupt, SetupCancelled):
            self.ui.print()
            self.ui.print(s.yellow("Setup cancelled. Nothing was saved.") + s.dim("  Run `local-agent setup` any time."))
            return None

    # -- [1/4] environment ------------------------------------------------------
    def step_environment(self) -> Optional[env.OllamaState]:
        self.ui.step(1, 4, "Check your environment")
        cf = read_config_file(self.config_path)
        c = env.config_check(cf)
        self.ui.check(c.ok, c.name, c.detail)
        for c in env.local_checks(self.settings):
            self.ui.check(c.ok, c.name, c.detail, c.hint if c.ok is False else "")
        if self.settings.provider != "ollama":
            return self._openai_server()
        base_url = self.settings.resolved_base_url()
        while True:
            state = self.probe(base_url)
            for c in env.ollama_checks(state, None):
                self.ui.check(c.ok, c.name, c.detail, c.hint)
            if state.reachable:
                if base_url != self.settings.resolved_base_url():  # only persist a URL the user typed here
                    self.chosen["base_url"] = base_url
                return state
            options = [Option("Retry", "retry", "after installing or starting Ollama")]
            if state.binary:
                options.insert(0, Option("Start Ollama for me", "start", f"runs `{Path(state.binary).name} serve` in the background"))
            options += [
                Option("Use a different Ollama URL", "url", f"currently {base_url}"),
                Option("Use an OpenAI-compatible server", "openai", "LM Studio, llama.cpp, vLLM, ..."),
                Option("Quit setup", "quit"),
            ]
            self.ui.print()
            if not state.binary:
                self.ui.print("Ollama runs the models locally. Install it, then choose Retry:")
                self.ui.print("  " + self.ui.s.cyan(system.install_instructions()))
            else:
                self.ui.print("Ollama is installed but its service is not responding.")
            choice = self.p.select("What would you like to do?", options)
            if choice == "quit":
                raise SetupCancelled
            if choice == "start" and state.binary:
                if system.start_ollama_background(state.binary):
                    self.ui.print(self.ui.s.dim("  Started `ollama serve`; waiting for it to respond..."))
                    for _ in range(10):
                        self.sleep(1)
                        if self.probe(base_url).reachable:
                            break
                else:
                    self.ui.check(False, "Could not start Ollama", system.start_instructions())
            elif choice == "url":
                base_url = self.p.text("Ollama URL", base_url).rstrip("/")
            elif choice == "openai":
                self.settings.provider = "openai"
                return self._openai_server()

    def _openai_server(self) -> Optional[env.OllamaState]:
        """OpenAI-compatible server: ask URL and key variable, list its models."""
        from local_agent.llm.openai_compat import OpenAICompatProvider

        default_url = self.settings.base_url if self.settings.provider == "openai" and self.settings.base_url else DEFAULT_BASE_URLS["openai"]
        while True:
            url = self.p.text("Server URL", default_url).rstrip("/")
            key_env = self.p.text("Environment variable holding the API key (blank if none)", self.settings.api_key_env)
            self.ui.print(self.ui.s.dim("  The key itself is never written to the config file."))
            client = OpenAICompatProvider(url, "", api_key=Settings(api_key_env=key_env).api_key() if key_env else None)
            try:
                names = client.list_models()
            except AgentError as exc:
                self.ui.check(False, "Server", exc.message)
                if not self.p.confirm("Try again?", True):
                    raise SetupCancelled
                default_url = url
                continue
            self.ui.check(True, "Server", f"{url} ({len(names)} model(s))")
            self.chosen.update(provider="openai", base_url=url, api_key_env=key_env)
            state = env.OllamaState(base_url=url, reachable=True, version="openai-compatible")
            state.models = [ModelEntry(n) for n in names]
            return state

    # -- [2/4] model ------------------------------------------------------------
    def step_model(self, state: env.OllamaState) -> Optional[str]:
        self.ui.step(2, 4, "Choose your model")
        is_ollama = self.chosen.get("provider", self.settings.provider) == "ollama"
        while True:
            chat = state.chat_models
            suggestion = recommend_download(self.memory)
            if not chat and is_ollama:
                self.ui.print("No language models are installed yet. Local-Agent-CLI needs one to run.")
                options = [
                    Option(f"Download {suggestion['name']}", ("pull", suggestion["name"]), f"{suggestion['size']}, {suggestion['note']}", "recommended"),
                    Option("Enter a model name to download", ("custom", None)),
                    Option("Retry", ("retry", None), "after running `ollama pull <model>` yourself"),
                    Option("Quit setup", ("quit", None)),
                ]
                choice = self.p.select("How would you like to get a model?", options)
            else:
                choice = self._choose_installed(chat, suggestion, is_ollama)
            action, name = choice  # type: ignore[misc]
            if action == "quit":
                raise SetupCancelled
            if action == "retry":
                state = self.probe(state.base_url) if is_ollama else state
                continue
            if action == "custom":
                name = self.p.text("Model name (e.g. qwen2.5-coder:7b)").strip()
                if not name:
                    continue
            if is_ollama and state.find(str(name)) is None:
                if not self._download(state.base_url, str(name)):
                    continue
                state = self.probe(state.base_url)
                if state.find(str(name)) is None:
                    self.ui.check(False, "Model", f"{name} still isn't listed by Ollama after the download")
                    continue
            if self._validate(str(name)) == "retry":
                continue
            return str(name)

    def _choose_installed(self, chat: List[ModelEntry], suggestion: dict, is_ollama: bool):
        rec = recommend_installed(chat, self.memory)
        current = self.settings.model
        options: List[Option] = []
        ordered = sorted(chat, key=lambda m: (m is not rec, not m.coding, m.name))
        for m in ordered:
            hint = "  ".join(x for x in (human_size(m.size) if m.size > 0 else "", ", ".join(m.tags()) if is_ollama else "") if x)
            badge = "recommended" if rec is not None and m.name == rec.name else ("current" if m.name == current else "")
            options.append(Option(m.name, ("use", m.name), hint, badge))
        if is_ollama and not any(m.name == suggestion["name"] for m in chat):
            options.append(Option(f"Download {suggestion['name']}", ("pull", suggestion["name"]), f"{suggestion['size']}, {suggestion['note']}"))
        options.append(Option("Enter a custom model name", ("custom", None), "any model from ollama.com/library" if is_ollama else ""))
        options.append(Option("Quit setup", ("quit", None)))
        default = next((i for i, o in enumerate(options) if o.label == current), 0)
        footer_lines = []
        if is_ollama:
            footer_lines.append(self.ui.s.dim("Coding models (e.g. qwen2.5-coder) work best for editing and debugging; 'tools' = native tool calling."))
        if rec is not None:
            footer_lines.append(f"Recommended for your system: {self.ui.s.green(rec.name)}"
                                + (self.ui.s.dim(f"  ({human_size(self.memory)} RAM)") if self.memory else ""))
        return self.p.select("Choose your language model", options, default, "\n".join(footer_lines))

    def _download(self, base_url: str, name: str) -> bool:
        s = self.ui.s
        self.ui.print(f"{s.bold(name)} is not installed. It will be downloaded from the Ollama library (this can be several GB).")
        if not self.p.confirm(f"Download {name} now?", False):
            return False
        client = OllamaProvider(base_url, name)
        while True:
            renderer = _ProgressRenderer(self.ui)
            try:
                client.pull(name, renderer.update, threading.Event())
                renderer.finish()
                self.ui.check(True, "Downloaded", name)
                return True
            except KeyboardInterrupt:
                renderer.finish()
                self.ui.check(None, "Download cancelled", "the partial download is kept by Ollama and resumes next time")
            except ModelNotFound as exc:
                renderer.finish()
                self.ui.check(False, "Download failed", exc.message, exc.hint)
                return False
            except AgentError as exc:
                renderer.finish()
                self.ui.check(False, "Download failed", exc.message)
            choice = self.p.select("What next?", [
                Option("Retry the download", "retry"),
                Option("Choose a different model", "other"),
                Option("Quit setup", "quit"),
            ])
            if choice == "quit":
                raise SetupCancelled
            if choice == "other":
                return False

    def _validate(self, name: str) -> str:
        """Test the model. Returns "ok" or "retry" (the user wants to pick again)."""
        s = self.ui.s
        probe_settings = Settings(**{**self.settings.to_public_dict(), **{k: v for k, v in self.chosen.items() if k in ("provider", "base_url", "api_key_env")}, "model": name})
        probe_settings.data_dir = self.settings.data_dir
        self.ui.print(s.dim(f"  Sending a test request to {name} (the first load can take a minute)..."))
        try:
            result = self.validator(probe_settings)
        except KeyboardInterrupt:
            self.ui.check(None, "Test skipped", "cancelled; the model was not verified")
            return "ok" if self.p.confirm(f"Use {name} without testing it?", False) else "retry"
        if not result.responded:
            self.ui.check(False, "Model test failed", result.error)
            choice = self.p.select("What next?", [Option("Choose a different model", "retry"), Option(f"Use {name} anyway", "keep"), Option("Quit setup", "quit")])
            if choice == "quit":
                raise SetupCancelled
            if choice == "retry":
                return "retry"
            self.chosen["model"] = name
            return "ok"
        label = {"native": "native tool calling works", "text": "tool calls arrive as text", "none": "no tool calling"}[result.tool_calling]
        self.ui.check(True if result.tool_calling == "native" else None, "Model responded", f"{name} in {result.seconds:.1f}s, {label}")
        note = describe(result)
        self.chosen["tool_mode"] = result.recommended_tool_mode()
        self.chosen["model"] = name
        if note:
            self.ui.print(s.yellow("  " + note))
            choice = self.p.select("How would you like to continue?", [
                Option("Choose a different model", "retry", "a coding model with tool support is recommended"),
                Option(f"Continue with {name}", "keep"),
            ], default=0 if result.tool_calling == "none" else 1)
            if choice == "retry":
                return "retry"
        return "ok"

    # -- [3/4] permissions ------------------------------------------------------
    def step_permissions(self) -> str:
        s = self.ui.s
        self.ui.step(3, 4, "Configure permissions")
        self.ui.print("Local-Agent-CLI can read your files, modify code and run commands in the project you start it in.")
        self.ui.print("You decide what it may do without asking. This is enforced for every action, not just shown here.")
        self.ui.print(s.dim("Always protected: paths outside the project, secret files (.env, keys), destructive and network actions."))
        self.ui.print()
        current = self.settings.mode if self.settings.onboarding_completed else "ask"
        options = [Option(label, value, desc, "recommended" if value == "ask" else "") for value, label, desc in MODE_OPTIONS]
        default = [v for v, _, _ in MODE_OPTIONS].index(current)
        while True:
            mode = str(self.p.select("How should Local-Agent-CLI handle your actions?", options, default))
            if mode != "full":
                return mode
            self.ui.print(s.red("  Full Control runs ordinary shell commands (tests, builds, scripts) without asking."))
            self.ui.print("  They run with your user's privileges and are NOT sandboxed. Deletes, network, privileged and")
            self.ui.print("  secret-file actions still ask, and blocked commands stay blocked.")
            if self.p.text("  Type 'yes' to enable Full Control", "").lower() == "yes":
                return mode
            default = 1

    # -- [4/4] save -------------------------------------------------------------
    def step_save(self, model: str, mode: str) -> Optional[Settings]:
        s = self.ui.s
        self.ui.step(4, 4, "Start coding")
        values: Dict[str, object] = {
            "provider": self.chosen.get("provider", self.settings.provider),
            "model": model,
            "mode": mode,
            "tool_mode": self.chosen.get("tool_mode", self.settings.tool_mode),
            "onboarding_completed": True,
            "setup_completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        base_url = self.chosen.get("base_url")
        if base_url and base_url != DEFAULT_BASE_URLS[str(values["provider"])]:
            values["base_url"] = base_url
        if "api_key_env" in self.chosen:
            values["api_key_env"] = self.chosen["api_key_env"]
        changes = [(k, o, n) for k, o, n in config_diff(self.settings, values) if k not in ("setup_completed_at",)]
        self.ui.print(f"  Model:       {model}")
        self.ui.print(f"  Permissions: {dict((v, l) for v, l, _ in MODE_OPTIONS)[mode]} ({mode}) - {MODE_DESCRIPTIONS[mode]}")
        self.ui.print(f"  Config file: {self.config_path}")
        cf = read_config_file(self.config_path)
        if cf.exists and cf.error:
            self.ui.print(s.yellow("  The existing config file is unreadable. It will be renamed to config.json.bak-<time>"))
            self.ui.print(s.yellow("  and a new one written with these settings."))
            if not self.p.confirm("Continue?", True):
                raise SetupCancelled
        elif cf.exists and changes:
            self.ui.print(s.dim("  Changes: " + ", ".join(f"{k}: {o!r} -> {n!r}" for k, o, n in changes)))
            self.ui.print(s.dim("  Other settings in the file are kept."))
            if not self.p.confirm("Save these settings?", True):
                raise SetupCancelled
        try:
            save_user_config(values, self.config_path)  # type: ignore[arg-type]
        except (AgentError, OSError) as exc:
            self.ui.check(False, "Could not save configuration", str(exc))
            return None
        self.ui.check(True, "Configuration saved", str(self.config_path))
        for k, v in values.items():
            setattr(self.settings, k, v)
        return self.settings


class _ProgressRenderer:
    """Renders real progress events from Ollama's pull API."""

    def __init__(self, ui: TerminalUI) -> None:
        self.ui = ui
        self.last_status = ""
        self.last_draw = 0.0
        self.inline = False

    def update(self, p: PullProgress) -> None:
        now = time.monotonic()
        if p.total and p.completed is not None:
            if now - self.last_draw < 0.2 and p.completed < p.total:
                return
            self.last_draw = now
            frac = min(p.completed / p.total, 1.0)
            bar = "#" * int(frac * 30)
            line = f"  {p.status[:28]:28} [{bar:<30}] {frac * 100:5.1f}%  {human_size(p.completed)}/{human_size(p.total)}"
            if self.ui.is_tty:
                self.ui.out.write("\r\033[K" + line)
                self.ui.out.flush()
                self.inline = True
            elif frac >= 1.0 or p.status != self.last_status:
                self.ui.print(line)
        elif p.status and p.status != self.last_status:
            self.finish()
            self.ui.print(self.ui.s.dim(f"  {p.status}"))
        self.last_status = p.status

    def finish(self) -> None:
        if self.inline:
            self.ui.out.write("\n")
            self.ui.out.flush()
            self.inline = False
