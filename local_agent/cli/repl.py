"""Interactive session and slash commands."""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, Dict, List, Optional

from local_agent.cli.session import Session, resume_into
from local_agent.cli.ui import TerminalUI
from local_agent.config import MODES
from local_agent.errors import AgentError
from local_agent.llm.base import Message
from local_agent.llm.catalog import RECOMMENDED_MODELS
from local_agent.permissions.engine import MODE_DESCRIPTIONS

HELP = """\
Describe a task in plain language, e.g. "find why login fails and fix it, then run the auth tests".

Commands:
  /help                     this help
  /status                   model, mode, workspace, context and token usage
  /plan [task]              switch to plan mode (read-only), or run one task in plan mode
  /mode <plan|ask|auto|full>  change permission mode (alias: /permissions <mode>)
  /permissions              show mode, session rules and recent permission decisions
  /memory                   show saved project facts, preferences and recent tasks
  /memory search <query>    search memory
  /memory add <fact>        save a verified project fact
  /memory pref <text>       save a long-term preference (applies to all projects)
  /memory forget <id>       delete one memory item
  /memory clear [project|tasks|preferences|sessions|all]
  /diff [path]              diff of files the agent changed this session
  /undo [--force]           revert the agent's file changes from this session
  /compact                  summarize history to free context
  /analyze                  project structure and language statistics
  /model [name]             list models or switch model
  /resume                   continue the last interrupted task
  /history                  recent tasks in this project
  /clear                    start a fresh conversation (keeps memory and checkpoints)
  /exit                     quit (Ctrl+D also works; Ctrl+C cancels the running task)"""


class Repl:
    def __init__(self, session: Session, ui: TerminalUI, switch_model: Optional[Callable[[str], Optional[Session]]] = None) -> None:
        self.session = session
        self.ui = ui
        self.switch_model = switch_model
        self.commands: Dict[str, Callable[[str], Optional[bool]]] = {
            "help": self.cmd_help, "status": self.cmd_status, "plan": self.cmd_plan, "mode": self.cmd_mode,
            "permissions": self.cmd_permissions, "memory": self.cmd_memory, "diff": self.cmd_diff, "undo": self.cmd_undo,
            "compact": self.cmd_compact, "analyze": self.cmd_analyze, "model": self.cmd_model, "models": self.cmd_model,
            "resume": self.cmd_resume, "history": self.cmd_history, "clear": self.cmd_clear,
            "exit": self.cmd_exit, "quit": self.cmd_exit,
        }

    # -- main loop -----------------------------------------------------------
    def run(self) -> int:
        s = self.ui.s
        interrupted = self.session.memory.latest_interrupted_session() if self.session.memory else None
        if interrupted and interrupted.get("session_id") != self.session.agent.session_id:
            self.ui.print(s.yellow(f"  An earlier task was interrupted: {interrupted.get('task', '')[:80]!r}. Type /resume to continue it."))
        while True:
            try:
                line = input(self._prompt())
            except EOFError:
                self.ui.print()
                return 0
            except KeyboardInterrupt:
                self.ui.print(s.dim("\n  (Ctrl+C) Type /exit or press Ctrl+D to quit."))
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                name, _, rest = line[1:].partition(" ")
                handler = self.commands.get(name.lower())
                if handler is None:
                    self.ui.print(s.red(f"  Unknown command /{name}. Type /help."))
                    continue
                try:
                    if handler(rest.strip()) is True:
                        return 0
                except AgentError as exc:
                    self.ui.print(s.red(f"  {exc}"))
                except KeyboardInterrupt:
                    self.ui.print(s.yellow("  Cancelled."))
                continue
            self.run_task(line)

    def _prompt(self) -> str:
        mode = self.session.engine.mode
        color = {"plan": self.ui.s.blue, "ask": self.ui.s.green, "auto": self.ui.s.yellow, "full": self.ui.s.red}[mode]
        return f"\n{color('[' + mode + ']')} {self.ui.s.bold('>')} "

    def run_task(self, task: str, resumed: bool = False) -> None:
        try:
            report = self.session.agent.run_task(task, resumed=resumed)
        except AgentError as exc:
            self.ui.print(self.ui.s.red(f"  {exc}"))
            return
        streamed = self.ui.is_tty
        self.ui.report(report, show_message=not streamed)

    # -- commands --------------------------------------------------------------
    def cmd_help(self, _: str) -> None:
        self.ui.print(HELP)

    def cmd_exit(self, _: str) -> bool:
        return True

    def cmd_status(self, _: str) -> None:
        sess, agent = self.session, self.session.agent
        from local_agent.agent.context import estimate_tokens

        self.ui.print(f"  Workspace:  {sess.workspace.root}")
        self.ui.print(f"  Model:      {sess.settings.model} via {sess.settings.provider} ({sess.settings.resolved_base_url()})")
        self.ui.print(f"  Tool calls: {sess.provider.tool_mode} ({sess.tool_mode_reason})")
        self.ui.print(f"  Mode:       {sess.engine.mode} - {MODE_DESCRIPTIONS[sess.engine.mode]}")
        self.ui.print(f"  Context:    ~{estimate_tokens(agent.messages)} / {agent.context_window} tokens in history")
        self.ui.print(f"  Tokens:     ~{agent.total_tokens} used this session")
        self.ui.print(f"  Limits:     {sess.settings.max_iterations} model calls, {sess.settings.max_seconds}s per task" + (f", {sess.settings.max_total_tokens} tokens" if sess.settings.max_total_tokens else ""))
        self.ui.print(f"  Memory:     {'on (' + str(sess.memory.project_dir) + ')' if sess.memory else 'off'}")
        changed = sess.ctx.tracker.describe()
        self.ui.print(f"  Changes:    {len(changed)} file(s) changed by the agent this session")
        if sess.ctx.plan:
            self.ui.plan_updated(sess.ctx.plan)

    def _set_mode(self, mode: str) -> None:
        if mode not in MODES:
            self.ui.print(self.ui.s.red(f"  Unknown mode '{mode}'. Choose: {', '.join(MODES)}"))
            return
        if mode == "full":
            self.ui.print(self.ui.s.red("  Full-control mode runs ordinary shell commands without asking."))
            self.ui.print("  Commands run with your user's privileges and are NOT sandboxed. Deletes, network, privileged and")
            self.ui.print("  sensitive-file actions still ask. Only use this in a project you trust, ideally under version control.")
            try:
                if input("  Type 'yes' to enable full mode: ").strip().lower() != "yes":
                    self.ui.print("  Mode unchanged.")
                    return
            except EOFError:
                return
        self.session.engine.mode = mode
        self.ui.print(f"  Mode: {mode} - {MODE_DESCRIPTIONS[mode]}")

    def cmd_mode(self, arg: str) -> None:
        if not arg:
            self.ui.print(f"  Current mode: {self.session.engine.mode}")
            for m in MODES:
                self.ui.print(f"    {m:5} {MODE_DESCRIPTIONS[m]}")
            return
        self._set_mode(arg.lower())

    def cmd_plan(self, arg: str) -> None:
        if not arg:
            self._set_mode("plan")
            return
        previous = self.session.engine.mode
        self.session.engine.mode = "plan"
        try:
            self.run_task(arg)
        finally:
            self.session.engine.mode = previous
            self.ui.print(self.ui.s.dim(f"  (back in {previous} mode; say 'go ahead' to implement the plan)"))

    def cmd_permissions(self, arg: str) -> None:
        if arg.lower() == "reset":
            self.session.engine.session_rules.clear()
            self.ui.print("  Session rules cleared.")
            return
        if arg:
            self._set_mode(arg.lower())
            return
        engine = self.session.engine
        self.ui.print(f"  Mode: {engine.mode} - {MODE_DESCRIPTIONS[engine.mode]}")
        if engine.session_rules:
            self.ui.print("  Session 'always allow' rules:")
            for rule in sorted(engine.session_rules):
                self.ui.print(f"    {rule}")
        else:
            self.ui.print("  No session rules.")
        if engine.log:
            self.ui.print("  Recent decisions:")
            for entry in engine.log[-10:]:
                self.ui.print(f"    [{entry['outcome']}] {entry['action']} ({entry['risk']})")
        self.ui.print(self.ui.s.dim("  Rules last for this session only. 'reset' clears them: /permissions reset"))

    def cmd_memory(self, arg: str) -> None:
        mem = self.session.memory
        if mem is None:
            self.ui.print("  Memory is disabled (--no-memory).")
            return
        sub, _, rest = arg.partition(" ")
        rest = rest.strip()
        s = self.ui.s
        if not sub:
            facts, prefs, tasks = mem.facts(), mem.preferences(), mem.tasks()
            self.ui.print(s.bold(f"  Project memory ({len(facts)})") + s.dim(f"  {mem.project_dir}"))
            for f in facts[-30:]:
                tag = s.green("verified") if f.get("verified") else s.yellow("unverified")
                self.ui.print(f"    {f['id']}  [{tag}] {f['content']}" + s.dim(f"  ({f.get('category')}, {f.get('source')})"))
            self.ui.print(s.bold(f"  Preferences ({len(prefs)})"))
            for p in prefs:
                self.ui.print(f"    {p['id']}  {p['content']}")
            self.ui.print(s.bold(f"  Recent tasks ({len(tasks)} stored)"))
            for t in tasks[-5:]:
                self.ui.print(f"    {t['id']}  ({t.get('status')}) {t.get('task', '')[:90]}")
            for w in mem.warnings():
                self.ui.print(s.yellow(f"  {w}"))
        elif sub == "search":
            if not rest:
                self.ui.print("  Usage: /memory search <query>")
                return
            results = mem.search(rest)
            if not results:
                self.ui.print("  No matches.")
            for r in results:
                self.ui.print(f"    {r['id']}  [{r['kind']}] {r['content'][:150]}  " + s.dim(f"score {r['score']}"))
        elif sub == "add":
            if not rest:
                self.ui.print("  Usage: /memory add <fact>")
                return
            item = mem.add_fact(rest, category="other", verified=True, source="user")
            self.ui.print(f"  Saved {item['id']}.")
        elif sub in ("pref", "preference"):
            if not rest:
                self.ui.print("  Usage: /memory pref <preference>")
                return
            item = mem.add_preference(rest, source="user")
            self.ui.print(f"  Saved preference {item['id']}.")
        elif sub in ("forget", "delete", "rm"):
            self.ui.print(f"  {'Deleted' if mem.delete(rest) else 'No item with id'} {rest}.")
        elif sub == "clear":
            scope = rest or "project"
            if scope not in ("project", "tasks", "preferences", "prefs", "sessions", "all"):
                self.ui.print("  Usage: /memory clear [project|tasks|preferences|sessions|all]")
                return
            try:
                ok = input(f"  Permanently delete {scope} memory? [y/N] ").strip().lower() == "y"
            except EOFError:
                ok = False
            if ok:
                self.ui.print("  Cleared: " + ", ".join(mem.clear(scope)))
            else:
                self.ui.print("  Nothing deleted.")
        else:
            self.ui.print("  Usage: /memory [search <q> | add <fact> | pref <text> | forget <id> | clear [scope]]")

    def cmd_diff(self, arg: str) -> None:
        tracker = self.session.ctx.tracker
        diff = tracker.diff(only=arg or None)
        if not diff.strip():
            self.ui.print("  No changes by the agent in this session." if not arg else f"  No agent changes to {arg}.")
            return
        for item in tracker.describe():
            self.ui.print(f"  {item['change']:9} {item['path']}")
        self.ui.diff(diff, max_lines=400)

    def cmd_undo(self, arg: str) -> None:
        tracker = self.session.ctx.tracker
        files = tracker.describe()
        if not files:
            self.ui.print("  Nothing to undo.")
            return
        force = arg.strip() == "--force"
        self.ui.print("  This will revert:")
        for item in files:
            self.ui.print(f"    {item['change']:9} {item['path']}")
        if force:
            self.ui.print(self.ui.s.red("  --force: files modified by someone else since the agent's last write will be overwritten."))
        try:
            if input("  Proceed? [y/N] ").strip().lower() != "y":
                self.ui.print("  Nothing reverted.")
                return
        except EOFError:
            return
        for r in tracker.undo(force=force):
            if r.action == "unchanged":
                continue
            mark = self.ui.s.green("ok") if r.ok else self.ui.s.yellow("skipped")
            self.ui.print(f"    [{mark}] {r.action} {r.path}" + (f": {r.reason}" if r.reason else ""))
        self.session.agent.messages.append(
            Message("user", "[Note: the user reverted the agent's file changes with /undo. Re-read files before relying on earlier content.]")
        )
        self.session.ctx.reads.clear()

    def cmd_compact(self, _: str) -> None:
        result = self.session.agent.compact_now()
        self.ui.print(f"  Context: ~{result['before']} -> ~{result['after']} tokens" + (" (history summarized)" if result.get("summarized") else ""))

    def cmd_analyze(self, _: str) -> None:
        from local_agent.project.discovery import discover

        info = discover(self.session.workspace.root, self.session.ctx.ignore)
        self.session.ctx.project = info
        for line in info.to_text().splitlines():
            self.ui.print("  " + line)

    def cmd_model(self, arg: str) -> None:
        provider = self.session.provider
        if not arg:
            try:
                installed = provider.list_models()
            except AgentError as exc:
                installed = []
                self.ui.print(self.ui.s.yellow(f"  Could not list models: {exc}"))
            self.ui.print(f"  Current: {self.session.settings.model}")
            if installed:
                self.ui.print("  Available on the server:")
                for name in installed:
                    self.ui.print(f"    {name}")
            if self.session.settings.provider == "ollama":
                self.ui.print("  Suggested (install with `ollama pull <name>`):")
                for m in RECOMMENDED_MODELS:
                    if m["name"] not in installed:
                        self.ui.print(f"    {m['name']:24} {m['size']:>7}  {m['description']}")
            self.ui.print(self.ui.s.dim("  Switch with /model <name>"))
            return
        if self.session.settings.provider == "ollama":
            try:
                installed = provider.list_models()
            except AgentError:
                installed = []
            if installed and arg not in installed and shutil.which("ollama"):
                try:
                    ok = input(f"  '{arg}' is not installed. Download it now with `ollama pull {arg}`? [y/N] ").strip().lower() == "y"
                except EOFError:
                    ok = False
                if not ok:
                    return
                if subprocess.run(["ollama", "pull", arg]).returncode != 0:
                    self.ui.print(self.ui.s.red("  Download failed."))
                    return
        if self.switch_model is None:
            self.ui.print("  Model switching is not available here.")
            return
        new_session = self.switch_model(arg)
        if new_session is not None:
            self.session = new_session
            self.ui.print(f"  Now using {arg} ({new_session.provider.tool_mode} tool calling: {new_session.tool_mode_reason}).")

    def cmd_resume(self, _: str) -> None:
        mem = self.session.memory
        data = mem.latest_interrupted_session() if mem else None
        if not data:
            self.ui.print("  No interrupted task found for this project.")
            return
        task = resume_into(self.session, data)
        self.ui.print(f"  Resuming: {task[:100]!r} ({len(self.session.agent.messages)} messages restored)")
        changed = self.session.ctx.tracker.describe()
        if changed:
            self.ui.print(f"  {len(changed)} file(s) changed by that session; /diff to review.")
        self.run_task(task, resumed=True)

    def cmd_history(self, _: str) -> None:
        mem = self.session.memory
        if mem is None:
            self.ui.print("  Memory is disabled.")
            return
        tasks: List[dict] = mem.tasks()[-15:]
        if not tasks:
            self.ui.print("  No tasks recorded yet.")
        for t in tasks:
            files = ", ".join(t.get("files_changed", [])[:4])
            self.ui.print(f"  {t.get('finished_at', '')}  ({t.get('status')}) {t.get('task', '')[:70]}" + (f"  [{files}]" if files else ""))

    def cmd_clear(self, _: str) -> None:
        self.session.agent.messages = []
        self.session.ctx.reads.clear()
        self.session.ctx.plan = []
        self.ui.print("  Conversation cleared. Memory and /undo checkpoints are kept.")
