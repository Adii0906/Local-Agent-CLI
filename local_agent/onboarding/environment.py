"""Environment and model-server checks shared by onboarding, startup and `doctor`.

A check reports what was actually observed. "Ollama is installed" (binary on
PATH) and "Ollama is running" (its HTTP API answered) are separate checks, so
an installed-but-stopped Ollama is never reported as running.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from local_agent.config import CONFIG_SCHEMA_VERSION, ConfigFile, Settings
from local_agent.errors import AgentError
from local_agent.llm.ollama import OllamaProvider
from local_agent.onboarding.models import ModelEntry
from local_agent.onboarding.system import ollama_binary

MAX_CAPABILITY_PROBES = 30


@dataclass
class Check:
    name: str
    ok: Optional[bool]  # True ok, False problem, None optional/warning
    detail: str
    hint: str = ""


@dataclass
class OllamaState:
    base_url: str
    binary: Optional[str] = None
    reachable: bool = False
    version: str = ""
    error: str = ""
    models: List[ModelEntry] = field(default_factory=list)

    @property
    def chat_models(self) -> List[ModelEntry]:
        return [m for m in self.models if m.is_chat]

    def find(self, name: str) -> Optional[ModelEntry]:
        for m in self.models:
            if m.name == name or (":" not in name and m.name == f"{name}:latest"):
                return m
        return None


def probe_ollama(base_url: str, with_capabilities: bool = True) -> OllamaState:
    state = OllamaState(base_url=base_url, binary=ollama_binary())
    client = OllamaProvider(base_url, "")
    try:
        state.version = client.version()
        state.reachable = True
        raw = client.list_models_detailed()
    except AgentError as exc:
        state.error = exc.message
        return state
    for item in raw:
        entry = ModelEntry(item["name"], item["size"], item["family"], item["parameter_size"], item["quantization"])
        if with_capabilities and len(raw) <= MAX_CAPABILITY_PROBES:
            try:
                caps = client.capabilities(entry.name)
                entry.tools, entry.embedding = caps["tools"], caps["embedding"]
            except AgentError:
                pass
        state.models.append(entry)
    return state


def config_check(cf: ConfigFile) -> Check:
    if not cf.exists:
        return Check("Configuration", None, f"not created yet ({cf.path})", "Run `local-agent setup`.")
    if cf.error:
        return Check("Configuration", False, cf.error, "Run `local-agent setup` to repair it (a backup is kept).")
    note = f" (migrates from schema {cf.migrated_from})" if cf.migrated_from else ""
    return Check("Configuration", True, f"{cf.path} (schema {CONFIG_SCHEMA_VERSION}){note}")


def local_checks(settings: Settings) -> List[Check]:
    checks = [
        Check("Python", sys.version_info >= (3, 9), sys.version.split()[0], "Python 3.9 or newer is required."),
        Check("git", True if shutil.which("git") else None, shutil.which("git") or "not found (git status/diff in reports unavailable)"),
        Check("ripgrep", True if shutil.which("rg") else None, shutil.which("rg") or "not found; using the slower built-in search"),
    ]
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        probe = Path(settings.data_dir) / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        checks.append(Check("Data directory", True, str(settings.data_dir)))
    except OSError as exc:
        checks.append(Check("Data directory", False, f"{settings.data_dir}: {exc}", "Set LOCAL_AGENT_HOME to a writable directory."))
    return checks


def ollama_checks(state: OllamaState, model: Optional[str]) -> List[Check]:
    from local_agent.onboarding.system import install_instructions, start_instructions

    checks = [
        Check("Ollama installed", True if state.binary else (None if state.reachable else False),
              state.binary or ("not on PATH (but a server is reachable)" if state.reachable else "not found"),
              "" if state.binary or state.reachable else install_instructions()),
        Check("Ollama running", state.reachable,
              f"version {state.version} at {state.base_url}" if state.reachable else f"no response from {state.base_url}",
              "" if state.reachable else (start_instructions() if state.binary else install_instructions())),
    ]
    if state.reachable:
        chat = state.chat_models
        checks.append(Check("Models installed", bool(chat), f"{len(chat)} chat model(s)" if chat else "none",
                            "" if chat else "Download one: ollama pull qwen2.5-coder:3b   (or run `local-agent setup`)"))
        if model:
            entry = state.find(model)
            if entry is None:
                checks.append(Check("Configured model", False, f"{model} is not installed",
                                    f"ollama pull {model}   or choose another: local-agent models use <name>"))
            else:
                tool_note = {True: "native tool calling", False: "no native tool calling (prompt-based fallback)", None: "tool support unknown"}[entry.tools]
                checks.append(Check("Configured model", True if entry.tools is not False else None, f"{entry.name} ({tool_note})",
                                    "" if entry.tools is not False else "A tool-capable model (e.g. qwen2.5-coder) is much more reliable."))
    return checks
