"""Centralized configuration.

Precedence (lowest to highest): built-in defaults, user config file, environment
variables, command-line flags.

Only the *user* config file is read. A project directory cannot ship a config
that changes permissions or the model endpoint: anything inside the workspace
is treated as untrusted.

API keys are never read from or written to the config file. ``api_key_env``
names the environment variable that holds the key.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent.errors import ConfigError

MODES = ("plan", "ask", "auto", "full")
PROVIDERS = ("ollama", "openai")
TOOL_MODES = ("auto", "native", "prompted")

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:1234/v1",  # LM Studio / llama.cpp server default style
}


def _platform_dir(kind: str) -> Path:
    """Return the per-user config or data directory for this OS."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "local-agent"
    if kind == "config":
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "local-agent"


def default_config_path() -> Path:
    override = os.environ.get("LOCAL_AGENT_CONFIG")
    return Path(override) if override else _platform_dir("config") / "config.json"


def default_data_dir() -> Path:
    override = os.environ.get("LOCAL_AGENT_HOME")
    return Path(override) if override else _platform_dir("data")


@dataclass
class Settings:
    provider: str = "ollama"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    mode: str = "ask"
    tool_mode: str = "auto"
    temperature: float = 0.2
    # Limits
    max_iterations: int = 40
    max_seconds: int = 1800
    max_total_tokens: int = 0  # 0 = unlimited
    context_window: int = 0  # 0 = ask the provider, else fall back to 8192
    request_timeout: int = 300
    command_timeout: int = 120
    max_command_timeout: int = 1800
    max_output_chars: int = 12000
    # Workspace
    extra_ignore: List[str] = field(default_factory=list)
    shell_env_passthrough: List[str] = field(default_factory=list)
    # Memory
    memory_enabled: bool = True
    verbose: bool = False
    # Not loaded from file; resolved at runtime.
    data_dir: Path = field(default_factory=default_data_dir)

    def resolved_base_url(self) -> str:
        return (self.base_url or DEFAULT_BASE_URLS[self.provider]).rstrip("/")

    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def to_public_dict(self) -> Dict[str, Any]:
        """Settings safe to print or save (never includes key material)."""
        data = asdict(self)
        data.pop("data_dir", None)
        return data


_FILE_KEYS = {f.name for f in fields(Settings)} - {"data_dir"}


def _coerce(name: str, value: Any, current: Any) -> Any:
    expected = type(current) if current is not None else str
    if name in ("model", "base_url"):
        expected = str
    if value is None and name in ("model", "base_url"):
        return None
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            return value.lower() in ("1", "true", "yes", "on")
    elif expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
    elif expected is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    elif expected is list:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
    elif expected is str:
        if isinstance(value, str):
            return value
    raise ConfigError(f"Invalid value for '{name}': {value!r} (expected {expected.__name__})")


def apply_overrides(settings: Settings, overrides: Dict[str, Any], source: str) -> None:
    for key, value in overrides.items():
        if key in ("api_key", "apikey", "token", "password"):
            raise ConfigError(
                f"Refusing to read a credential from {source}.",
                hint="Put the key in an environment variable and set 'api_key_env' to its name.",
            )
        if key not in _FILE_KEYS:
            raise ConfigError(f"Unknown setting '{key}' in {source}.")
        setattr(settings, key, _coerce(key, value, getattr(settings, key)))


def validate(settings: Settings) -> None:
    if settings.mode not in MODES:
        raise ConfigError(f"Unknown mode '{settings.mode}'. Choose one of: {', '.join(MODES)}")
    if settings.provider not in PROVIDERS:
        raise ConfigError(f"Unknown provider '{settings.provider}'. Choose one of: {', '.join(PROVIDERS)}")
    if settings.tool_mode not in TOOL_MODES:
        raise ConfigError(f"Unknown tool_mode '{settings.tool_mode}'. Choose one of: {', '.join(TOOL_MODES)}")
    for name in ("max_iterations", "max_seconds", "command_timeout", "max_command_timeout", "request_timeout"):
        if getattr(settings, name) <= 0:
            raise ConfigError(f"'{name}' must be positive")
    if settings.max_total_tokens < 0 or settings.context_window < 0:
        raise ConfigError("Token limits must be >= 0")


_ENV_MAP = {
    "LOCAL_AGENT_PROVIDER": "provider",
    "LOCAL_AGENT_MODEL": "model",
    "LOCAL_AGENT_BASE_URL": "base_url",
    "LOCAL_AGENT_MODE": "mode",
    "LOCAL_AGENT_TOOL_MODE": "tool_mode",
}


def load_settings(cli_overrides: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None) -> Settings:
    settings = Settings()
    path = config_path or default_config_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Could not read config file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Config file {path} must contain a JSON object")
        apply_overrides(settings, data, str(path))

    env: Dict[str, Any] = {}
    if os.environ.get("OLLAMA_HOST") and settings.provider == "ollama" and not settings.base_url:
        host = os.environ["OLLAMA_HOST"]
        env["base_url"] = host if host.startswith("http") else f"http://{host}"
    for var, key in _ENV_MAP.items():
        if os.environ.get(var):
            env[key] = os.environ[var]
    apply_overrides(settings, env, "environment")

    if cli_overrides:
        apply_overrides(settings, {k: v for k, v in cli_overrides.items() if v is not None}, "command line")
    validate(settings)
    return settings


def save_user_config(values: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
    """Persist selected settings to the user config file (merging with existing)."""
    path = config_path or default_config_path()
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    probe = Settings()
    apply_overrides(probe, values, "new config values")  # validates keys and types
    existing.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
