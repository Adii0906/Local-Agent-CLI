"""Text-based tool calling for models without native tool support.

The model is told to emit tool calls as fenced JSON blocks::

    ```tool
    {"name": "read_file", "arguments": {"path": "src/app.py"}}
    ```

``extract_tool_calls`` also recognizes ``<tool_call>...</tool_call>`` (used by
Qwen-family templates) and a bare JSON object that is the whole reply. Native
mode uses it as well, because small local models often answer with JSON text
instead of a structured tool call.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from local_agent.llm.base import ChatResponse, Message, ModelInfo, Provider, TextCallback, ToolCall

_BLOCK = re.compile(r"```(?:tool|tool_call|json)?\s*\n?(\{.*?\})\s*```", re.S)
_TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

PROTOCOL = """\
## How to call tools
You can call tools. To call one, reply with a fenced block exactly like this (one block per call):

```tool
{{"name": "<tool name>", "arguments": {{ ...JSON arguments... }}}}
```

After the block, STOP and wait: the result will come back in the next message, starting with "TOOL RESULT".
Never invent tool results. When the task is complete, reply with plain text and no tool block.

Available tools (JSON Schema for arguments):
{tools}
"""


def _parse_obj(text: str, known: Optional[set]) -> Optional[Tuple[str, Dict[str, Any], Optional[str]]]:
    try:
        obj = json.loads(text)
    except ValueError as exc:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
        if name_match and (known is None or name_match.group(1) in known):
            return name_match.group(1), {}, f"arguments are not valid JSON: {exc}"
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or (obj.get("function") or {}).get("name")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", (obj.get("function") or {}).get("arguments", {}))))
    if not isinstance(name, str) or (known is not None and name not in known):
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return name, {}, "arguments are not valid JSON"
    if not isinstance(args, dict):
        return name, {}, "arguments must be a JSON object"
    return name, args, None


def extract_tool_calls(text: str, known: Optional[set] = None) -> Tuple[str, List[ToolCall]]:
    """Return (text without tool blocks, parsed calls)."""
    calls: List[ToolCall] = []
    spans: List[Tuple[int, int]] = []
    for regex in (_TAGGED, _BLOCK):
        for m in regex.finditer(text):
            if any(s <= m.start() < e for s, e in spans):
                continue
            parsed = _parse_obj(m.group(1), known)
            if parsed:
                name, args, err = parsed
                calls.append(ToolCall(id=f"call_{len(calls) + 1}", name=name, arguments=None if err else args, raw_arguments=m.group(1), parse_error=err))
                spans.append((m.start(), m.end()))
    if not calls:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = _parse_obj(stripped, known)
            if parsed:
                name, args, err = parsed
                calls.append(ToolCall(id="call_1", name=name, arguments=None if err else args, raw_arguments=stripped, parse_error=err))
                spans.append((text.find(stripped), text.find(stripped) + len(stripped)))
    remaining = text
    for s, e in sorted(spans, reverse=True):
        remaining = remaining[:s] + remaining[e:]
    return remaining.strip(), calls


class _StreamFilter:
    """Forwards streamed text but hides tool-call blocks from the user."""

    MARKERS = ("```tool", "<tool_call>")

    def __init__(self, sink: Optional[TextCallback]) -> None:
        self.sink = sink
        self.pending = ""
        self.suppressed = False

    def feed(self, chunk: str) -> None:
        if not self.sink or self.suppressed:
            return
        self.pending += chunk
        for marker in self.MARKERS:
            idx = self.pending.find(marker)
            if idx != -1:
                if idx:
                    self.sink(self.pending[:idx])
                self.pending = ""
                self.suppressed = True
                return
        keep = max(len(m) for m in self.MARKERS)
        if len(self.pending) > keep:
            self.sink(self.pending[:-keep])
            self.pending = self.pending[-keep:]

    def flush(self) -> None:
        if self.sink and not self.suppressed and self.pending:
            self.sink(self.pending)
        self.pending = ""


class PromptedToolsProvider(Provider):
    """Wraps a provider and implements tool calling through the prompt."""

    tool_mode = "prompted"

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model

    def model_info(self) -> ModelInfo:
        return self.inner.model_info()

    def list_models(self) -> List[str]:
        return self.inner.list_models()

    @staticmethod
    def _convert(messages: List[Message], tools: Optional[List[Dict[str, Any]]]) -> List[Message]:
        out: List[Message] = []
        for m in messages:
            if m.role == "system" and tools:
                spec = json.dumps(tools, indent=1)
                out.append(Message("system", m.content + "\n\n" + PROTOCOL.format(tools=spec)))
            elif m.role == "assistant" and m.tool_calls:
                blocks = "\n".join(
                    "```tool\n" + json.dumps({"name": c.name, "arguments": c.arguments if c.arguments is not None else c.raw_arguments}) + "\n```"
                    for c in m.tool_calls
                )
                out.append(Message("assistant", (m.content + "\n" if m.content else "") + blocks))
            elif m.role == "tool":
                text = f"TOOL RESULT ({m.name}):\n{m.content}"
                if out and out[-1].role == "user" and out[-1].content.startswith("TOOL RESULT"):
                    out[-1] = Message("user", out[-1].content + "\n\n" + text)
                else:
                    out.append(Message("user", text))
            else:
                out.append(Message(m.role, m.content))
        return out

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        on_text: Optional[TextCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ChatResponse:
        filt = _StreamFilter(on_text)
        resp = self.inner.chat(self._convert(messages, tools), None, on_text=filt.feed, cancel_event=cancel_event)
        filt.flush()
        known = {t["name"] for t in tools} if tools else set()
        text, calls = extract_tool_calls(resp.content, known)
        return ChatResponse(text, calls, resp.usage, resp.finish_reason)
