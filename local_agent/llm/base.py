"""Provider-neutral chat types and the Provider interface."""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from local_agent.errors import ProviderError, ProviderUnavailable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Optional[Dict[str, Any]]
    raw_arguments: str = ""
    parse_error: Optional[str] = None


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # tool name for role=tool

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments, "raw_arguments": c.raw_arguments, "parse_error": c.parse_error}
                for c in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=[ToolCall(**c) for c in d.get("tool_calls", [])],
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ChatResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Usage] = None
    finish_reason: str = ""


@dataclass
class ModelInfo:
    name: str
    context_length: Optional[int] = None
    supports_tools: Optional[bool] = None  # None = unknown
    details: str = ""


TextCallback = Callable[[str], None]


class Provider(ABC):
    name = "provider"
    model = ""
    tool_mode = "native"

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        on_text: Optional[TextCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ChatResponse: ...

    @abstractmethod
    def model_info(self) -> ModelInfo: ...

    def list_models(self) -> List[str]:
        return []


# -- HTTP helpers (stdlib only) -------------------------------------------

def _decode_error_body(err: urllib.error.HTTPError) -> str:
    try:
        body = err.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - best effort only
        return ""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            e = data.get("error")
            if isinstance(e, dict):
                return str(e.get("message", e))
            if e:
                return str(e)
    except ValueError:
        pass
    return body[:500]


def open_request(
    url: str,
    payload: Optional[Dict[str, Any]],
    *,
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
    method: Optional[str] = None,
    service: str = "model server",
):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 - URL comes from user config
    except urllib.error.HTTPError as err:
        detail = _decode_error_body(err)
        if err.code in (401, 403):
            raise ProviderError(f"{service} rejected the credentials (HTTP {err.code}): {detail}", hint="Check the API key environment variable.") from None
        if err.code == 404:
            raise ProviderUnavailable(f"{service} returned 404 for {url}: {detail}") from None
        retryable = err.code in (408, 429, 500, 502, 503, 504)
        raise ProviderError(f"{service} error HTTP {err.code}: {detail}", retryable=retryable) from None
    except urllib.error.URLError as err:
        reason = err.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ProviderError(f"{service} timed out after {timeout}s", retryable=True) from None
        raise ProviderUnavailable(f"Cannot reach {service} at {url} ({reason})") from None
    except (socket.timeout, TimeoutError):
        raise ProviderError(f"{service} timed out after {timeout}s", retryable=True) from None
    except ConnectionError as err:
        raise ProviderUnavailable(f"Cannot reach {service} at {url} ({err})") from None


def iter_lines(resp, cancel_event: Optional[threading.Event], service: str) -> Iterator[str]:
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return
            try:
                line = resp.readline()
            except (socket.timeout, TimeoutError):
                raise ProviderError(f"{service} stopped responding (read timeout)", retryable=True) from None
            except (ConnectionError, OSError) as err:
                raise ProviderError(f"Connection to {service} dropped: {err}", retryable=True) from None
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                yield text
    finally:
        resp.close()


def parse_arguments(raw: Any) -> tuple:
    """Return (arguments dict or None, raw string, error or None)."""
    if isinstance(raw, dict):
        return raw, json.dumps(raw), None
    if raw is None or raw == "":
        return {}, "", None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError as exc:
            return None, raw, f"arguments are not valid JSON: {exc}"
        if not isinstance(value, dict):
            return None, raw, "arguments must be a JSON object"
        return value, raw, None
    return None, str(raw), f"unexpected arguments type {type(raw).__name__}"
