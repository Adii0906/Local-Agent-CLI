"""Centralized configuration.

Precedence (lowest to highest): built-in defaults, user config file, environment
variables, command-line flags.

Only the *user* config file is read. A project directory cannot ship a config
that changes permissions or the model endpoint: anything inside the workspace
is treated as untrusted.

API keys are never read from or written to the config file. ``api_key_env``
names the environment variable that holds the key.

The file carries a ``schema_version``. Older files are migrated in memory when
loaded and written in the new format the next time settings are saved. A file
from a newer version is never overwritten.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from local_agent.errors import ConfigError

MODES = ("plan", "ask", "auto", "full")
PROVIDERS = ("ollama", "openai")
TOOL_MODES = ("auto", "native", "prompted")
CONFIG_SCHEMA_VERSION = 2

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openai": "http://localhost:1234/v1",  # LM Studio / llama.cpp server default style
}

_SECRET_KEYS = ("api_key", "apikey", "token", "password", "secret")


def _platform_dir(kind: str) -> Path:
    """Return the per-user config or data directory for this OS.

    * Windows: ``%APPDATA%\\local-agent``
    * macOS:   ``~/Library/Application Support/local-agent`` (an existing
      ``~/.config/local-agent`` from earlier versions keeps being used)
    * Linux:   ``$XDG_CONFIG_HOME`` / ``$XDG_DATA_HOME`` (``~/.config``, ``~/.local/share``)
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "local-agent"
    env_var = "XDG_CONFIG_HOME" if kind == "config" else "XDG_DATA_HOME"
    xdg_default = Path.home() / (".config" if kind == "config" else ".local/share")
    if os.environ.get(env_var):
        return Path(os.environ[env_var]) / "local-agent"
    if sys.platform == "darwin":
        legacy = xdg_default / "local-agent"
        if legacy.exists():
            return legacy
        return Path.home() / "Library" / "Application Support" / "local-agent"
    return xdg_default / "local-agent"


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
    # Onboarding / preferences
    onboarding_completed: bool = False
    setup_completed_at: str = ""
    show_startup_checks: bool = True
    # Not loaded from file; resolved at runtime.
    data_dir: Path = field(default_factory=default_data_dir)
    load_warnings: List[str] = field(default_factory=list)

    def resolved_base_url(self) -> str:
        return (self.base_url or DEFAULT_BASE_URLS[self.provider]).rstrip("/")

    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def to_public_dict(self) -> Dict[str, Any]:
        """Settings safe to print or save (never includes key material)."""
        data = asdict(self)
        data.pop("data_dir", None)
        data.pop("load_warnings", None)
        return data


_RUNTIME_ONLY = {"data_dir", "load_warnings"}
_FILE_KEYS = {f.name for f in fields(Settings)} - _RUNTIME_ONLY
_META_KEYS = {"schema_version"}


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


def _check_value(key: str, value: Any) -> None:
    allowed = {"mode": MODES, "provider": PROVIDERS, "tool_mode": TOOL_MODES}.get(key)
    if allowed and value not in allowed:
        raise ConfigError(f"Invalid value for '{key}': {value!r} (choose one of: {', '.join(allowed)})")


def apply_overrides(settings: Settings, overrides: Dict[str, Any], source: str, *, lenient: bool = False) -> List[str]:
    """Apply values to ``settings``. Returns warnings (unknown/ignored keys).

    With ``lenient=True`` invalid values are skipped with a warning instead of
    raising, which lets ``local-agent setup`` repair a broken config.
    """
    warnings: List[str] = []
    for key, value in overrides.items():
        if key in _META_KEYS:
            continue
        if key.lower() in _SECRET_KEYS:
            raise ConfigError(
                f"Refusing to read a credential ('{key}') from {source}.",
                hint="Remove it, put the key in an environment variable and set 'api_key_env' to its name.",
            )
        if key not in _FILE_KEYS:
            msg = f"Unknown setting '{key}' in {source} (ignored)."
            if source in ("command line", "environment", "new config values"):
                raise ConfigError(msg.replace(" (ignored)", ""))
            warnings.append(msg)
            continue
        try:
            coerced = _coerce(key, value, getattr(settings, key))
            _check_value(key, coerced)
        except ConfigError as exc:
            if not lenient:
                raise ConfigError(f"{exc.message} in {source}", hint="Run `local-agent setup` to repair it, or edit the file.") from None
            warnings.append(f"{exc.message} in {source} (ignored).")
            continue
        setattr(settings, key, coerced)
    return warnings


def validate(settings: Settings) -> None:
    _check_value("mode", settings.mode)
    _check_value("provider", settings.provider)
    _check_value("tool_mode", settings.tool_mode)
    for name in ("max_iterations", "max_seconds", "command_timeout", "max_command_timeout", "request_timeout"):
        if getattr(settings, name) <= 0:
            raise ConfigError(f"'{name}' must be positive")
    if settings.max_total_tokens < 0 or settings.context_window < 0:
        raise ConfigError("Token limits must be >= 0")


# -- file handling ---------------------------------------------------------

def _migrate_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """v1 (no schema_version, written by `--setup` in 2.0): users who configured
    a model are treated as already onboarded."""
    data = dict(data)
    data.setdefault("onboarding_completed", bool(data.get("model")))
    data["schema_version"] = 2
    return data


MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {1: _migrate_v1}


@dataclass
class ConfigFile:
    path: Path
    exists: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = CONFIG_SCHEMA_VERSION
    migrated_from: Optional[int] = None
    error: str = ""  # unreadable / newer version

    @property
    def usable(self) -> bool:
        return not self.error


def read_config_file(path: Optional[Path] = None) -> ConfigFile:
    """Read and migrate the user config. Never raises; problems go in ``error``."""
    path = path or default_config_path()
    cf = ConfigFile(path=path)
    if not path.is_file():
        return cf
    cf.exists = True
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        cf.error = f"{path} is not valid JSON ({exc})"
        return cf
    if not isinstance(raw, dict):
        cf.error = f"{path} must contain a JSON object"
        return cf
    version = raw.get("schema_version", 1)
    if not isinstance(version, int) or version < 1:
        cf.error = f"{path} has an invalid schema_version {version!r}"
        return cf
    if version > CONFIG_SCHEMA_VERSION:
        cf.error = (
            f"{path} was written by a newer Local-Agent-CLI (config schema {version}, this version supports "
            f"{CONFIG_SCHEMA_VERSION}). Upgrade local-agent-cli; the file was left unchanged."
        )
        cf.version = version
        return cf
    data = raw
    if version < CONFIG_SCHEMA_VERSION:
        cf.migrated_from = version
        while version < CONFIG_SCHEMA_VERSION:
            data = MIGRATIONS[version](data)
            version = data["schema_version"]
    cf.data, cf.version = data, version
    return cf


_ENV_MAP = {
    "LOCAL_AGENT_PROVIDER": "provider",
    "LOCAL_AGENT_MODEL": "model",
    "LOCAL_AGENT_BASE_URL": "base_url",
    "LOCAL_AGENT_MODE": "mode",
    "LOCAL_AGENT_TOOL_MODE": "tool_mode",
}


def load_settings(
    cli_overrides: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    *,
    lenient: bool = False,
) -> Settings:
    settings = Settings()
    cf = read_config_file(config_path)
    if cf.error:
        if not lenient or cf.version > CONFIG_SCHEMA_VERSION:
            raise ConfigError(cf.error, hint="Run `local-agent setup` to repair it (a backup is kept), or fix the file by hand.")
        settings.load_warnings.append(cf.error + " Ignoring it.")
    else:
        settings.load_warnings += apply_overrides(settings, cf.data, str(cf.path), lenient=lenient)
        if cf.migrated_from is not None:
            settings.load_warnings.append(
                f"Config migrated from schema {cf.migrated_from} to {CONFIG_SCHEMA_VERSION} (saved on the next change)."
            )

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


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(path, backup)
    return backup


def save_user_config(values: Dict[str, Any], config_path: Optional[Path] = None) -> Path:
    """Merge ``values`` into the user config file and write it atomically.

    Unrelated settings (including keys this version does not know) are kept.
    An unreadable file is moved to ``config.json.bak-<time>`` first. A file
    from a newer version is never touched.
    """
    path = config_path or default_config_path()
    probe = Settings()
    apply_overrides(probe, values, "new config values")  # validates keys and types
    cf = read_config_file(path)
    existing: Dict[str, Any] = {}
    if cf.error:
        if cf.version > CONFIG_SCHEMA_VERSION:
            raise ConfigError(cf.error)
        _backup(path)
    else:
        existing = dict(cf.data)
    existing.update(values)
    existing["schema_version"] = CONFIG_SCHEMA_VERSION
    ordered = {"schema_version": CONFIG_SCHEMA_VERSION, **{k: v for k, v in existing.items() if k != "schema_version"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def config_diff(old: Settings, values: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """(key, old, new) for values that would change."""
    return [(k, getattr(old, k, None), v) for k, v in values.items() if getattr(old, k, None) != v]
