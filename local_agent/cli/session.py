"""Assemble a working agent session from settings (shared by CLI and tests)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from local_agent.agent.checkpoint import ChangeTracker
from local_agent.agent.events import AgentUI
from local_agent.agent.loop import Agent, new_session_id
from local_agent.config import Settings
from local_agent.llm.base import ModelInfo, Provider
from local_agent.llm.ollama import DEFAULT_NUM_CTX, OllamaProvider
from local_agent.memory.manager import MemoryManager
from local_agent.permissions.engine import PermissionEngine
from local_agent.project.discovery import ProjectInfo, discover
from local_agent.project.ignore import IgnoreRules
from local_agent.security.paths import Workspace
from local_agent.tools.base import ToolContext
from local_agent.tools.registry import default_registry


@dataclass
class Session:
    settings: Settings
    workspace: Workspace
    provider: Provider
    model_info: ModelInfo
    tool_mode_reason: str
    agent: Agent
    memory: Optional[MemoryManager]
    engine: PermissionEngine
    ctx: ToolContext
    project: ProjectInfo


def context_window_for(provider: Provider, info: ModelInfo, settings: Settings) -> int:
    if settings.context_window:
        return settings.context_window
    inner = getattr(provider, "inner", provider)
    if isinstance(inner, OllamaProvider):
        return inner.num_ctx()
    return info.context_length or DEFAULT_NUM_CTX


def build_session(
    settings: Settings,
    workspace_root: Path,
    ui: AgentUI,
    provider: Provider,
    model_info: ModelInfo,
    tool_mode_reason: str = "",
    session_id: Optional[str] = None,
) -> Session:
    workspace = Workspace(workspace_root)
    ignore = IgnoreRules(workspace.root, settings.extra_ignore)
    memory = MemoryManager(settings.data_dir, workspace.root) if settings.memory_enabled else None
    if memory is not None:
        try:
            memory.prune_sessions(keep=20)
        except OSError:
            pass
    session_id = session_id or new_session_id()
    tracker = ChangeTracker(workspace, memory.checkpoint_dir(session_id) if memory else None)
    project = discover(workspace.root, ignore)
    ctx = ToolContext(workspace=workspace, ignore=ignore, settings=settings, tracker=tracker, memory=memory, project=project)
    engine = PermissionEngine(mode=settings.mode)
    agent = Agent(
        provider, default_registry(settings.memory_enabled), ctx, engine, ui, settings, memory,
        session_id=session_id, context_window=context_window_for(provider, model_info, settings),
    )
    return Session(settings, workspace, provider, model_info, tool_mode_reason, agent, memory, engine, ctx, project)


def resume_into(session: Session, data: dict) -> str:
    """Load a saved session (messages, plan, checkpoints). Returns its task."""
    session.ctx.tracker.store_dir = session.memory.checkpoint_dir(data["session_id"]) if session.memory else None
    return session.agent.load_session(data)
