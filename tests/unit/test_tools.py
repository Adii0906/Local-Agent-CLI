import sys
import threading
import time

import pytest

from local_agent.agent.checkpoint import ChangeTracker
from local_agent.agent.context import compact, estimate_tokens, safe_cut_index
from local_agent.config import Settings
from local_agent.errors import PathViolation, ToolError
from local_agent.llm.base import Message, ToolCall
from local_agent.llm.prompted import PromptedToolsProvider, _StreamFilter, extract_tool_calls
from local_agent.project.ignore import IgnoreRules
from local_agent.security.paths import Workspace
from local_agent.tools import search as search_mod
from local_agent.tools.base import Risk, ToolContext
from local_agent.tools.filesystem import CreateDirectory, DeletePath, EditFile, ListDirectory, MovePath, ReadFile, WriteFile
from local_agent.tools.search import FindFiles, SearchCode
from local_agent.tools.shell import RunCommand, _CappedBuffer, child_env, execute


@pytest.fixture
def ctx(workspace, tmp_path):
    ws = Workspace(workspace)
    return ToolContext(workspace=ws, ignore=IgnoreRules(ws.root), settings=Settings(model="m", data_dir=tmp_path / "d"),
                       tracker=ChangeTracker(ws, tmp_path / "ckpt"))


PY = f'"{sys.executable}"' if " " in sys.executable else sys.executable


# ---------------------------------------------------------------- filesystem
class TestFilesystem:
    def test_read_paging_and_dedupe(self, ctx, workspace):
        (workspace / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 1001)))
        out = ReadFile().run({"path": "big.txt", "max_lines": 10}, ctx)
        assert out.ok and "lines 1-10 of 1000" in out.output and "start_line=11" in out.output
        again = ReadFile().run({"path": "big.txt", "max_lines": 10}, ctx)
        assert "unchanged since you last read" in again.output
        page = ReadFile().run({"path": "big.txt", "start_line": 995}, ctx)
        assert "lines 995-1000 of 1000" in page.output and "more lines" not in page.output

    def test_read_rejects_binary_and_dirs(self, ctx, workspace):
        (workspace / "b.bin").write_bytes(b"\x00\x01\x02")
        assert not ReadFile().run({"path": "b.bin"}, ctx).ok
        assert not ReadFile().run({"path": "src"}, ctx).ok

    def test_read_sensitive_assessed(self, ctx):
        assert ReadFile().assess({"path": ".env"}, ctx).risk is Risk.READ_SENSITIVE
        assert ReadFile().assess({"path": "src/auth.py"}, ctx).risk is Risk.READ

    def test_edit_requires_prior_read(self, ctx):
        with pytest.raises(ToolError, match="Read src/auth.py"):
            EditFile().assess({"path": "src/auth.py", "old_string": "None", "new_string": "x"}, ctx)

    def test_edit_exact_unique_and_diff(self, ctx, workspace):
        ReadFile().run({"path": "src/auth.py"}, ctx)
        tool = EditFile()
        with pytest.raises(ToolError, match="not found"):
            tool.assess({"path": "src/auth.py", "old_string": "if token is None", "new_string": "x"}, ctx)
        a = tool.assess({"path": "src/auth.py", "old_string": "return True", "new_string": "return False"}, ctx)
        assert a.risk is Risk.EDIT and "+        return False" in a.details
        r = tool.run({"path": "src/auth.py", "old_string": "return True", "new_string": "return False"}, ctx)
        assert r.ok and r.changed_paths == ["src/auth.py"]
        assert "return False" in (workspace / "src/auth.py").read_text()
        (workspace / "dup.txt").write_text("a\na\n")
        ReadFile().run({"path": "dup.txt"}, ctx)
        with pytest.raises(ToolError, match="occurs 2 times"):
            tool.assess({"path": "dup.txt", "old_string": "a", "new_string": "b"}, ctx)
        assert tool.run({"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True}, ctx).ok
        assert (workspace / "dup.txt").read_text() == "b\nb\n"

    def test_edit_preserves_crlf(self, ctx, workspace):
        (workspace / "w.txt").write_bytes(b"one\r\ntwo\r\n")
        ReadFile().run({"path": "w.txt"}, ctx)
        EditFile().run({"path": "w.txt", "old_string": "two", "new_string": "three"}, ctx)
        assert (workspace / "w.txt").read_bytes() == b"one\r\nthree\r\n"

    def test_write_create_vs_overwrite_and_git_block(self, ctx, workspace):
        assert WriteFile().assess({"path": "new/mod.py", "content": "x=1\n"}, ctx).risk is Risk.CREATE
        assert WriteFile().assess({"path": "src/auth.py", "content": "x"}, ctx).risk is Risk.EDIT
        assert WriteFile().assess({"path": ".env", "content": "x"}, ctx).risk is Risk.WRITE_SENSITIVE
        (workspace / ".git").mkdir()
        assert WriteFile().assess({"path": ".git/config", "content": "x"}, ctx).risk is Risk.BLOCKED
        r = WriteFile().run({"path": "new/mod.py", "content": "x=1\n"}, ctx)
        assert r.ok and (workspace / "new/mod.py").read_text() == "x=1\n"
        with pytest.raises(PathViolation):
            WriteFile().assess({"path": "../evil.py", "content": "x"}, ctx)

    def test_move_and_delete(self, ctx, workspace):
        (workspace / "a.txt").write_text("A")
        (workspace / "b.txt").write_text("B")
        assert MovePath().assess({"source": "a.txt", "destination": "c.txt"}, ctx).risk is Risk.EDIT
        with pytest.raises(ToolError, match="exists"):
            MovePath().assess({"source": "a.txt", "destination": "b.txt"}, ctx)
        assert MovePath().assess({"source": "a.txt", "destination": "b.txt", "overwrite": True}, ctx).risk is Risk.DESTRUCTIVE
        assert MovePath().run({"source": "a.txt", "destination": "moved/c.txt"}, ctx).ok
        assert (workspace / "moved/c.txt").read_text() == "A" and not (workspace / "a.txt").exists()
        assert DeletePath().assess({"path": "b.txt"}, ctx).risk is Risk.DESTRUCTIVE
        with pytest.raises(ToolError, match="recursive"):
            DeletePath().assess({"path": "src"}, ctx)
        assert DeletePath().assess({"path": "."}, ctx).risk is Risk.BLOCKED
        assert DeletePath().run({"path": "src", "recursive": True}, ctx).ok
        assert not (workspace / "src").exists()
        # Everything is recoverable
        results = ctx.tracker.undo()
        assert all(r.ok for r in results), results
        assert (workspace / "a.txt").read_text() == "A" and (workspace / "src/auth.py").exists()
        assert not (workspace / "moved").exists()

    def test_list_directory_respects_ignore(self, ctx, workspace):
        (workspace / "node_modules" / "x").mkdir(parents=True)
        (workspace / ".gitignore").write_text("secret_build/\n")
        (workspace / "secret_build").mkdir()
        out = ListDirectory().run({"path": ".", "depth": 3}, ctx).output
        assert "src/auth.py" in out and "node_modules" not in out and "secret_build" not in out

    def test_create_directory(self, ctx, workspace):
        assert CreateDirectory().run({"path": "a/b"}, ctx).ok
        assert (workspace / "a/b").is_dir()


# ---------------------------------------------------------------- search
@pytest.mark.parametrize("engine", ["rg", "python"])
def test_search_hides_sensitive_and_ignored(ctx, workspace, monkeypatch, engine):
    if engine == "python":
        monkeypatch.setattr(search_mod, "_ripgrep", lambda: None)
    elif search_mod._ripgrep() is None:
        pytest.skip("ripgrep not installed")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "lib.js").write_text("API_KEY usage\n")
    (workspace / "src" / "config.py").write_text("API_KEY = os.environ['API_KEY']\n")
    out = SearchCode().run({"pattern": "api_key"}, ctx)
    assert out.ok
    assert "src/config.py:1" in out.output
    assert "sk-live" not in out.output and "node_modules" not in out.output
    assert "sensitive files hidden" in out.output
    regex = SearchCode().run({"pattern": r"def \w+_token", "regex": True}, ctx)
    assert "src/auth.py:1" in regex.output
    assert not SearchCode().run({"pattern": "(unclosed", "regex": True}, ctx).ok


def test_find_files(ctx):
    out = FindFiles().run({"pattern": "test_*.py"}, ctx).output
    assert out.strip() == "tests/test_auth.py"
    assert "src/auth.py" in FindFiles().run({"pattern": "auth"}, ctx).output


# ---------------------------------------------------------------- shell
class TestShell:
    def test_exit_code_and_streams(self, ctx):
        r = RunCommand().run({"command": f"{PY} -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\""}, ctx)
        assert not r.ok and r.data["exit_code"] == 3
        assert "out" in r.output and "err" in r.output and "exit code 3" in r.output

    def test_runs_in_workspace_cwd(self, ctx, workspace):
        r = RunCommand().run({"command": f"{PY} -c \"import os; print(os.getcwd())\"", "cwd": "src"}, ctx)
        assert r.ok and str((workspace / "src").resolve()) in r.output

    def test_timeout_kills(self, ctx):
        start = time.monotonic()
        r = RunCommand().run({"command": f"{PY} -c \"import time; time.sleep(30)\"", "timeout": 1}, ctx)
        assert r.data["timed_out"] and not r.ok and time.monotonic() - start < 10
        assert "TIMED OUT" in r.output

    def test_cancel_event(self, ctx):
        ctx.cancel_event = threading.Event()
        threading.Timer(0.5, ctx.cancel_event.set).start()
        r = RunCommand().run({"command": f"{PY} -c \"import time; time.sleep(30)\""}, ctx)
        assert r.data["cancelled"]

    def test_no_stdin_hang(self, ctx):
        r = RunCommand().run({"command": f"{PY} -c \"input()\"", "timeout": 10}, ctx)
        assert not r.data["timed_out"] and r.data["exit_code"] != 0

    def test_output_truncated_keeps_tail(self, ctx):
        ctx.settings.max_output_chars = 2000
        r = RunCommand().run({"command": f"{PY} -c \"[print('x'*50) for _ in range(2000)]; print('THE_END')\""}, ctx)
        assert r.data["truncated"] and "THE_END" in r.output and "characters omitted" in r.output

    def test_secret_env_scrubbed_and_output_redacted(self, ctx, monkeypatch):
        monkeypatch.setenv("MY_SERVICE_TOKEN", "zzz-very-secret-zzz")
        env = child_env([])
        assert "MY_SERVICE_TOKEN" not in env and env["GIT_TERMINAL_PROMPT"] == "0"
        assert "MY_SERVICE_TOKEN" in child_env(["MY_SERVICE_TOKEN"])
        r = RunCommand().run({"command": f"{PY} -c \"print('sk-abcdefghijklmnopqrstuvwxyz123')\""}, ctx)
        stdout = r.output.split("--- stdout ---", 1)[1]
        assert "sk-abcdef" not in stdout and "[REDACTED]" in stdout

    def test_missing_program(self, ctx):
        res = execute("definitely-not-a-real-program-xyz", str(ctx.workspace.root), 5, use_shell=False, env=child_env([]), max_chars=1000)
        assert res["exit_code"] == 127

    def test_capped_buffer(self):
        buf = _CappedBuffer(100)
        for i in range(100):
            buf.write(f"{i:04d}\n")
        text = buf.text()
        assert text.startswith("0000") and text.rstrip().endswith("0099") and "omitted" in text


# ---------------------------------------------------------------- checkpoints
def test_undo_skips_files_changed_by_others(ctx, workspace):
    ReadFile().run({"path": "src/auth.py"}, ctx)
    original = (workspace / "src/auth.py").read_text()
    EditFile().run({"path": "src/auth.py", "old_string": "return True", "new_string": "return False"}, ctx)
    WriteFile().run({"path": "notes.md", "content": "hi"}, ctx)
    assert "+        return False" in ctx.tracker.diff()
    assert {d["path"]: d["change"] for d in ctx.tracker.describe()} == {"src/auth.py": "modified", "notes.md": "created"}
    (workspace / "notes.md").write_text("user edited this")  # someone else changed it
    results = {r.path: r for r in ctx.tracker.undo()}
    assert results["src/auth.py"].ok and (workspace / "src/auth.py").read_text() == original
    assert not results["notes.md"].ok and (workspace / "notes.md").read_text() == "user edited this"
    forced = {r.path: r for r in ctx.tracker.undo(force=True)}
    assert forced["notes.md"].ok and not (workspace / "notes.md").exists()


def test_tracker_persistence_roundtrip(ctx, workspace, tmp_path):
    ReadFile().run({"path": "src/auth.py"}, ctx)
    EditFile().run({"path": "src/auth.py", "old_string": "return True", "new_string": "return 1"}, ctx)
    restored = ChangeTracker(ctx.workspace, tmp_path / "ckpt")
    restored.load_dict(ctx.tracker.to_dict())
    assert restored.undo()[0].ok
    assert "return True" in (workspace / "src/auth.py").read_text()


# ---------------------------------------------------------------- prompted tool calls
def test_extract_tool_calls_variants():
    known = {"read_file", "run_command"}
    text, calls = extract_tool_calls('Let me look.\n```tool\n{"name": "read_file", "arguments": {"path": "a.py"}}\n```', known)
    assert text == "Let me look." and calls[0].name == "read_file" and calls[0].arguments == {"path": "a.py"}
    _, calls = extract_tool_calls('<tool_call>{"name": "run_command", "arguments": {"command": "ls"}}</tool_call>', known)
    assert calls[0].arguments == {"command": "ls"}
    _, calls = extract_tool_calls('{"name": "read_file", "parameters": {"path": "b"}}', known)
    assert calls[0].arguments == {"path": "b"}
    _, calls = extract_tool_calls('```json\n{"name": "my-app", "version": "1.0"}\n```', known)
    assert calls == []  # an ordinary JSON code block is not a tool call
    _, calls = extract_tool_calls('```tool\n{"name": "read_file", "arguments": {"path": }\n```', known)
    assert calls and calls[0].parse_error


def test_stream_filter_hides_tool_blocks():
    out = []
    f = _StreamFilter(out.append)
    for chunk in ["Checking the", " file.\n``", "`tool\n{\"name\": \"x\"}", "\n```"]:
        f.feed(chunk)
    f.flush()
    assert "".join(out) == "Checking the file.\n"


def test_prompted_provider_roundtrip():
    from helpers import ScriptedProvider
    from local_agent.llm.base import ChatResponse

    inner = ScriptedProvider([ChatResponse('```tool\n{"name": "read_file", "arguments": {"path": "x"}}\n```')], supports_tools=False)
    wrapped = PromptedToolsProvider(inner)
    msgs = [Message("system", "sys"), Message("user", "go"),
            Message("assistant", "", [ToolCall("c1", "read_file", {"path": "y"})]), Message("tool", "content", tool_call_id="c1", name="read_file")]
    resp = wrapped.chat(msgs, [{"name": "read_file", "description": "", "parameters": {}}])
    assert resp.tool_calls[0].arguments == {"path": "x"}
    sent = inner.seen[0]
    assert "How to call tools" in sent[0].content
    assert sent[-1].role == "user" and sent[-1].content.startswith("TOOL RESULT (read_file)")
    assert "```tool" in sent[2].content


# ---------------------------------------------------------------- context
def test_compaction_keeps_tool_pairs_valid():
    msgs = [Message("system", "s")]
    for i in range(30):
        msgs.append(Message("user", f"q{i}"))
        msgs.append(Message("assistant", "", [ToolCall(f"c{i}", "read_file", {"path": "x"})]))
        msgs.append(Message("tool", "x" * 2000, tool_call_id=f"c{i}", name="read_file"))
        msgs.append(Message("assistant", f"a{i}"))
    before = estimate_tokens(msgs)
    report = compact(msgs, budget_tokens=2000, summarizer=lambda m: "SUMMARY OF WORK")
    assert report["summarized"] and estimate_tokens(msgs) < before
    assert msgs[0].role == "system" and "SUMMARY OF WORK" in msgs[1].content
    ids_called = {c.id for m in msgs if m.role == "assistant" for c in m.tool_calls}
    ids_answered = {m.tool_call_id for m in msgs if m.role == "tool"}
    assert ids_called == ids_answered
    assert msgs[1 + 2].role != "tool"


def test_safe_cut_never_orphans_tool_results():
    msgs = [Message("system", "s"), Message("user", "u"), Message("assistant", "", [ToolCall("a", "x", {})]),
            Message("tool", "r", tool_call_id="a", name="x"), Message("assistant", "done")]
    assert msgs[safe_cut_index(msgs, keep_last=2)].role != "tool"


def test_summarizer_failure_falls_back():
    msgs = [Message("system", "s")] + [Message("user", "x" * 5000) for _ in range(20)]

    def boom(_):
        raise RuntimeError("model down")

    report = compact(msgs, budget_tokens=500, summarizer=boom)
    assert report["summarized"] and "Earlier requests" in msgs[1].content
