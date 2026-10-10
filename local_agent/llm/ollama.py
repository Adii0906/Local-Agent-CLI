"""Ollama adapter (``/api/chat`` with native tool calling and streaming)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from local_agent.errors import ModelNotFound, ProviderError, ProviderUnavailable
from local_agent.llm.base import (
    ChatResponse, Message, ModelInfo, Provider, TextCallback, ToolCall, Usage, iter_lines, open_request, parse_arguments,
)

DEFAULT_NUM_CTX = 8192
MAX_DEFAULT_NUM_CTX = 32768


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, *, temperature: float = 0.2, timeout: float = 300, context_window: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self._context_override = context_window
        self._info: Optional[ModelInfo] = None

    def _service(self) -> str:
        return f"Ollama ({self.base_url})"

    def version(self) -> str:
        """Server version; proves the service is actually running (not just installed)."""
        resp = open_request(f"{self.base_url}/api/version", None, timeout=5, service=self._service())
        with resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        return str(data.get("version", "unknown"))

    def list_models_detailed(self) -> List[Dict[str, Any]]:
        """Installed models with size and details, as reported by ``/api/tags``."""
        resp = open_request(f"{self.base_url}/api/tags", None, timeout=10, service=self._service())
        with resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
        out = []
        for m in data.get("models", []) or []:
            if not m.get("name"):
                continue
            details = m.get("details") or {}
            out.append({
                "name": m["name"],
                "size": int(m.get("size") or 0),
                "family": details.get("family", ""),
                "parameter_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
            })
        return out

    def list_models(self) -> List[str]:
        return [m["name"] for m in self.list_models_detailed()]

    def _show(self, name: str) -> Dict[str, Any]:
        try:
            resp = open_request(f"{self.base_url}/api/show", {"model": name}, timeout=30, service=self._service())
        except ProviderUnavailable as exc:
            if "404" in str(exc):
                raise ModelNotFound(name, f"Model '{name}' is not installed in Ollama.", hint=f"Run: ollama pull {name}") from None
            raise ProviderUnavailable(str(exc), hint="Start Ollama with `ollama serve` (or open the Ollama app), or set base_url.") from None
        with resp:
            return json.loads(resp.read().decode("utf-8") or "{}")

    @staticmethod
    def _supports_tools(data: Dict[str, Any]) -> Optional[bool]:
        caps = data.get("capabilities")
        if isinstance(caps, list):
            return "tools" in caps
        template = data.get("template") or ""  # older Ollama: infer from the chat template
        return True if ".Tools" in template or "tools" in template.lower() else None

    def capabilities(self, name: str) -> Dict[str, Any]:
        """{'tools': bool|None, 'embedding': bool, 'context_length': int|None} for any installed model."""
        data = self._show(name)
        caps = data.get("capabilities") if isinstance(data.get("capabilities"), list) else []
        ctx = None
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                ctx = value
        return {"tools": self._supports_tools(data), "embedding": "embedding" in caps and "completion" not in caps, "context_length": ctx}

    def model_info(self) -> ModelInfo:
        if self._info is not None:
            return self._info
        data = self._show(self.model)
        ctx = None
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                ctx = value
        details = data.get("details") or {}
        self._info = ModelInfo(
            name=self.model,
            context_length=ctx,
            supports_tools=self._supports_tools(data),
            details=f"{details.get('family', '?')} {details.get('parameter_size', '')} {details.get('quantization_level', '')}".strip(),
        )
        return self._info

    def pull(
        self,
        name: str,
        on_progress: Optional[Callable[["PullProgress"], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Download a model through ``/api/pull``, reporting real progress.

        Raises ProviderError on failure. Ctrl+C propagates as KeyboardInterrupt;
        Ollama keeps the partial download and resumes it on the next pull.
        """
        resp = open_request(f"{self.base_url}/api/pull", {"model": name, "stream": True}, timeout=120, service=self._service())
        success = False
        for line in iter_lines(resp, cancel_event, self._service()):
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                err = str(chunk["error"])
                if "file does not exist" in err or "not found" in err:
                    raise ModelNotFound(name, f"'{name}' was not found in the Ollama library.", hint="Check the exact name at https://ollama.com/library")
                raise ProviderError(f"Download of {name} failed: {err}", retryable=True)
            progress = PullProgress(str(chunk.get("status", "")), str(chunk.get("digest", "")), int(chunk.get("total") or 0), int(chunk.get("completed") or 0))
            if on_progress:
                on_progress(progress)
            if progress.status == "success":
                success = True
        if not success:
            if cancel_event is not None and cancel_event.is_set():
                raise KeyboardInterrupt
            raise ProviderError(f"Download of {name} ended before completing.", retryable=True)

    def num_ctx(self) -> int:
        if self._context_override:
            return self._context_override
        ctx = self._info.context_length if self._info else None
        return min(ctx, MAX_DEFAULT_NUM_CTX) if ctx else DEFAULT_NUM_CTX

    @staticmethod
    def _convert(messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                d: Dict[str, Any] = {"role": "assistant", "content": m.content}
                if m.tool_calls:
                    d["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments or {}}} for c in m.tool_calls]
                out.append(d)
            elif m.role == "tool":
                out.append({"role": "tool", "content": m.content, "tool_name": m.name or ""})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        on_text: Optional[TextCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ChatResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert(messages),
            "stream": True,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx()},
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = open_request(f"{self.base_url}/api/chat", payload, timeout=self.timeout, service=self._service())
        content: List[str] = []
        calls: List[ToolCall] = []
        usage = Usage()
        finish = ""
        for line in iter_lines(resp, cancel_event, self._service()):
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                err = str(chunk["error"])
                if "does not support tools" in err:
                    raise ProviderError(f"Model '{self.model}' does not support native tool calling.", hint="Use --tool-mode prompted.")
                raise ProviderError(f"Ollama error: {err}")
            msg = chunk.get("message") or {}
            if msg.get("content"):
                content.append(msg["content"])
                if on_text:
                    on_text(msg["content"])
            for raw in msg.get("tool_calls") or []:
                fn = raw.get("function") or {}
                args, raw_args, error = parse_arguments(fn.get("arguments"))
                calls.append(ToolCall(id=raw.get("id") or f"call_{len(calls) + 1}", name=str(fn.get("name", "")), arguments=args, raw_arguments=raw_args, parse_error=error))
            if chunk.get("done"):
                usage = Usage(chunk.get("prompt_eval_count", 0) or 0, chunk.get("eval_count", 0) or 0)
                finish = chunk.get("done_reason", "stop")
        if cancel_event is not None and cancel_event.is_set():
            finish = "cancelled"
        return ChatResponse("".join(content), calls, usage, finish)


@dataclass
class PullProgress:
    status: str
    digest: str = ""
    total: int = 0
    completed: int = 0
