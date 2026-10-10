"""Build the configured provider and choose native vs prompted tool calling."""

from __future__ import annotations

from typing import Tuple

from local_agent.config import Settings
from local_agent.errors import ConfigError
from local_agent.llm.base import ModelInfo, Provider
from local_agent.llm.ollama import OllamaProvider
from local_agent.llm.openai_compat import OpenAICompatProvider
from local_agent.llm.prompted import PromptedToolsProvider


def create_base_provider(settings: Settings) -> Provider:
    if not settings.model:
        raise ConfigError("No model configured.", hint="Pass --model NAME, run `local-agent --setup`, or set LOCAL_AGENT_MODEL.")
    if settings.provider == "ollama":
        return OllamaProvider(settings.resolved_base_url(), settings.model, temperature=settings.temperature,
                              timeout=settings.request_timeout, context_window=settings.context_window)
    return OpenAICompatProvider(settings.resolved_base_url(), settings.model, api_key=settings.api_key(),
                                temperature=settings.temperature, timeout=settings.request_timeout,
                                context_window=settings.context_window)


def resolve_tool_mode(settings: Settings, info: ModelInfo) -> Tuple[str, str]:
    """Return (mode, explanation)."""
    if settings.tool_mode == "native":
        if info.supports_tools is False:
            return "native", "native tool calling forced, but the model reports no tool support; calls may fail"
        return "native", "native tool calling (forced)"
    if settings.tool_mode == "prompted":
        return "prompted", "prompt-based tool calling (forced)"
    if info.supports_tools:
        return "native", "model supports native tool calling"
    if info.supports_tools is False:
        return "prompted", "model lacks native tool calling; using prompt-based tool calls"
    if settings.provider == "openai":
        return "native", "tool support unknown; trying native tool calling (use --tool-mode prompted if it fails)"
    return "prompted", "tool support unknown; using prompt-based tool calls"


def create_provider(settings: Settings) -> Tuple[Provider, ModelInfo, str]:
    """Create the provider, probe the model and apply the tool-calling mode.

    Raises ProviderUnavailable with an actionable hint if the backend or model
    cannot be reached.
    """
    base = create_base_provider(settings)
    info = base.model_info()
    mode, why = resolve_tool_mode(settings, info)
    provider: Provider = PromptedToolsProvider(base) if mode == "prompted" else base
    provider.tool_mode = mode
    return provider, info, why
