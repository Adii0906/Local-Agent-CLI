"""Tools that act on the agent's own state: plan tracking, project info, memory."""

from __future__ import annotations

from typing import Any, Dict

from local_agent.project.discovery import discover
from local_agent.security.secrets import find_secrets
from local_agent.tools.base import Assessment, Risk, Tool, ToolContext, ToolResult

PLAN_STATUSES = ("pending", "in_progress", "done", "blocked", "skipped")


class UpdatePlan(Tool):
    name = "update_plan"
    description = (
        "Record or update your step-by-step plan for a non-trivial task. Send the full list each time with "
        "each step's status. The user sees it in the execution trace."
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string", "maxLength": 300},
                        "status": {"type": "string", "enum": list(PLAN_STATUSES)},
                    },
                    "required": ["step", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(Risk.READ, "Update plan")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not args["steps"]:
            return ToolResult.error("steps must not be empty")
        ctx.plan = [dict(s) for s in args["steps"]]
        marks = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "blocked": "[!]", "skipped": "[-]"}
        text = "\n".join(f"{marks[s['status']]} {s['step']}" for s in ctx.plan)
        return ToolResult(True, "Plan recorded:\n" + text, summary="Plan updated", data={"plan": ctx.plan})


class ProjectInfoTool(Tool):
    name = "project_info"
    description = "Detect languages, frameworks, package managers, entry points and suggested test/lint/build commands."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        return Assessment(Risk.READ, "Inspect project structure")

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.project = discover(ctx.workspace.root, ctx.ignore)
        return ToolResult(True, ctx.project.to_text(), summary="Detected project structure")


class Remember(Tool):
    name = "remember"
    description = (
        "Save a durable, useful fact about THIS project (build/test commands, architecture, conventions, "
        "verified root causes) so future sessions can use it. Only save things you verified with tools; "
        "set verified=false for assumptions. Never save secrets. scope='preference' saves a user "
        "preference and requires the user's approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {"type": "string", "maxLength": 1000},
            "category": {"type": "string", "enum": ["architecture", "commands", "conventions", "discovery", "issue", "other"]},
            "verified": {"type": "boolean", "description": "True only if confirmed by tool output."},
            "evidence": {"type": "string", "maxLength": 300, "description": "How it was verified (file, command)."},
            "scope": {"type": "string", "enum": ["project", "preference"]},
        },
        "required": ["content", "verified"],
        "additionalProperties": False,
    }

    def assess(self, args: Dict[str, Any], ctx: ToolContext) -> Assessment:
        scope = args.get("scope", "project")
        risk = Risk.PREFERENCE if scope == "preference" else Risk.MEMORY
        return Assessment(risk, f"Remember ({scope}): {args['content'][:120]}", details=args.get("evidence", ""))

    def run(self, args: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.memory is None:
            return ToolResult.error("Memory is disabled")
        secrets = find_secrets(args["content"]) + find_secrets(args.get("evidence", ""))
        if secrets:
            return ToolResult.error(f"Refusing to store content that looks like a secret ({', '.join(sorted(set(secrets)))})")
        if args.get("scope") == "preference":
            item = ctx.memory.add_preference(args["content"], source="agent (user-approved)")
        else:
            item = ctx.memory.add_fact(
                args["content"],
                category=args.get("category", "other"),
                verified=bool(args.get("verified")),
                evidence=args.get("evidence", ""),
                source="agent",
            )
        return ToolResult(True, f"Saved memory {item['id']}.", summary=f"Remembered: {args['content'][:60]}")
