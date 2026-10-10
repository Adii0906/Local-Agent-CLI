"""The set of tools exposed to the model."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from local_agent.tools.agent_tools import ProjectInfoTool, Remember, UpdatePlan
from local_agent.tools.base import Tool
from local_agent.tools.filesystem import CreateDirectory, DeletePath, EditFile, ListDirectory, MovePath, ReadFile, WriteFile
from local_agent.tools.git import GitDiff, GitStatus
from local_agent.tools.search import FindFiles, SearchCode
from local_agent.tools.shell import RunCommand


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def specs(self) -> List[dict]:
        return [t.spec() for t in self._tools.values()]


def default_registry(memory_enabled: bool = True) -> ToolRegistry:
    tools: List[Tool] = [
        ProjectInfoTool(), ListDirectory(), FindFiles(), SearchCode(), ReadFile(),
        EditFile(), WriteFile(), CreateDirectory(), MovePath(), DeletePath(),
        RunCommand(), GitStatus(), GitDiff(), UpdatePlan(),
    ]
    if memory_enabled:
        tools.append(Remember())
    return ToolRegistry(tools)
