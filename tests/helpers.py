"""Deterministic test doubles (no network, no paid APIs)."""

from typing import Callable, List, Optional, Union

from local_agent.llm.base import ChatResponse, Message, ModelInfo, Provider, ToolCall

Step = Union[ChatResponse, Callable[[List[Message]], ChatResponse], Exception]


class ScriptedProvider(Provider):
    """Deterministic fake model: returns pre-scripted responses in order."""

    name = "scripted"
    model = "scripted-model"
    tool_mode = "native"

    def __init__(self, steps: List[Step], supports_tools: bool = True) -> None:
        self.steps = list(steps)
        self.seen: List[List[Message]] = []
        self._supports = supports_tools

    def chat(self, messages, tools=None, *, on_text=None, cancel_event=None):
        self.seen.append([Message.from_dict(m.to_dict()) for m in messages])
        if not self.steps:
            return ChatResponse("All done.")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            step = step(messages)
        if on_text and step.content:
            on_text(step.content)
        return step

    def model_info(self):
        return ModelInfo(self.model, 8192, self._supports)


def call(name: str, _id: Optional[str] = None, **arguments) -> ToolCall:
    return ToolCall(id=_id or f"c-{name}-{len(arguments)}-{id(arguments)}", name=name, arguments=arguments)


def respond(*calls: ToolCall, text: str = "") -> ChatResponse:
    return ChatResponse(text, list(calls))
