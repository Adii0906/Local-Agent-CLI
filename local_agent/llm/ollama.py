"""Ollama adapter (``/api/chat`` with native tool calling and streaming)."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

from local_agent.errors import ProviderError, ProviderUnavailable
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

    def list_models(self) -> List[str]:
        resp = open_request(f"{self.base_url}/api/tags", None, timeout=10, service=self._service())
        with resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]

    def model_info(self) -> ModelInfo:
        if self._info is not None:
            return self._info
        try:
            resp = open_request(f"{self.base_url}/api/show", {"model": self.model}, timeout=30, service=self._service())
        except ProviderUnavailable as exc:
            if "404" in str(exc):
                raise ProviderUnavailable(
                    f"Model '{self.model}' is not installed in Ollama.", hint=f"Run: ollama pull {self.model}"
                ) from None
            raise ProviderUnavailable(str(exc), hint="Start Ollama with `ollama serve` or set base_url.") from None
        with resp:
            data = json.loads(resp.read().decode("utf-8"))
        ctx = None
        for key, value in (data.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                ctx = value
        caps = data.get("capabilities")
        if isinstance(caps, list):
            supports = "tools" in caps
        else:  # older Ollama: infer from the chat template
            template = data.get("template") or ""
            supports = True if ".Tools" in template or "tools" in template.lower() else None
        details = data.get("details") or {}
        self._info = ModelInfo(
            name=self.model,
            context_length=ctx,
            supports_tools=supports,
            details=f"{details.get('family', '?')} {details.get('parameter_size', '')} {details.get('quantization_level', '')}".strip(),
        )
        return self._info

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
