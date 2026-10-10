"""Error hierarchy.

Ordinary failures (model unreachable, bad config, path outside workspace) are
raised as ``AgentError`` subclasses carrying a user-facing message, so the CLI can
print them without a stack trace. Anything else is a bug and is shown with a
traceback only in ``--verbose`` mode.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for expected, user-presentable errors."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}\n  hint: {self.hint}" if self.hint else self.message


class ConfigError(AgentError):
    pass


class ProviderError(AgentError):
    """The model backend failed in a way that may be transient."""

    def __init__(self, message: str, hint: str = "", retryable: bool = False) -> None:
        super().__init__(message, hint)
        self.retryable = retryable


class ProviderUnavailable(ProviderError):
    """The backend cannot be reached or the model is not installed."""


class ModelNotFound(ProviderUnavailable):
    """The server is reachable but does not have the requested model."""

    def __init__(self, model: str, message: str = "", hint: str = "") -> None:
        super().__init__(message or f"Model '{model}' is not installed.", hint)
        self.model = model


class ToolError(AgentError):
    """A tool could not complete. The message is returned to the model."""


class PathViolation(ToolError):
    """A path resolved outside the authorized workspace or is otherwise invalid."""


class Cancelled(AgentError):
    def __init__(self, message: str = "Cancelled by user") -> None:
        super().__init__(message)
