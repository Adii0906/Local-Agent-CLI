"""OpenAI-compatible Chat Completions adapter.

Works with LM Studio, llama.cpp ``llama-server``, vLLM, and hosted
OpenAI-compatible APIs. The API key is read from the environment variable named
by ``api_key_env`` and is never logged.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

from local_agent.errors import ProviderUnavailable
from local_agent.llm.base import (
    ChatResponse, Message, ModelInfo, Provider, TextCallback, ToolCall, Usage, iter_lines, open_request, parse_arguments,
)


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(self, base_url: str, model: str, *, api_key: Optional[str] = None, temperature: float = 0.2,
                 timeout: float = 300, context_window: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self._context = context_window or None
        self._info: Optional[ModelInfo] = None

    def __repr__(self) -> str:  # never include the key
        return f"OpenAICompatProvider(base_url={self.base_url!r}, model={self.model!r})"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _service(self) -> str:
        return f"OpenAI-compatible server ({self.base_url})"

    def list_models(self) -> List[str]:
        resp = open_request(f"{self.base_url}/models", None, timeout=15, headers=self._headers(), service=self._service())
        with resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]

    def model_info(self) -> ModelInfo:
        if self._info is None:
            models = self.list_models()
            if models and self.model not in models:
                raise ProviderUnavailable(
                    f"Model '{self.model}' is not served by {self.base_url}.",
                    hint=f"Available: {', '.join(models[:10])}",
                )
            self._info = ModelInfo(name=self.model, context_length=self._context, supports_tools=None)
        return self._info

    @staticmethod
    def _convert(messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant":
                d: Dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    d["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments) if c.arguments is not None else c.raw_arguments}}
                        for c in m.tool_calls
                    ]
                out.append(d)
            elif m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content})
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
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = open_request(f"{self.base_url}/chat/completions", payload, timeout=self.timeout, headers=self._headers(), service=self._service())
        content: List[str] = []
        partial: Dict[int, Dict[str, str]] = {}
        usage: Optional[Usage] = None
        finish = ""
        for line in iter_lines(resp, cancel_event, self._service()):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("usage"):
                u = chunk["usage"]
                usage = Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                    if on_text:
                        on_text(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = partial.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        calls = []
        for idx in sorted(partial):
            slot = partial[idx]
            args, raw, error = parse_arguments(slot["arguments"])
            calls.append(ToolCall(id=slot["id"] or f"call_{idx}", name=slot["name"], arguments=args, raw_arguments=raw, parse_error=error))
        if cancel_event is not None and cancel_event.is_set():
            finish = "cancelled"
        return ChatResponse("".join(content), calls, usage, finish)
