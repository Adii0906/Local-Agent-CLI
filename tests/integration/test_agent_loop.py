"""End-to-end agent loop tests with a scripted model and real tools."""

import sys

from helpers import ScriptedProvider, call, respond

from local_agent.agent.events import RecordingUI
from local_agent.cli.session import build_session, resume_into
from local_agent.errors import ProviderError, ProviderUnavailable
from local_agent.llm.base import ChatResponse, ToolCall
from local_agent.permissions.engine import ApprovalResponse, Choice

PY = f'"{sys.executable}"' if " " in sys.executable else sys.executable
TEST_CMD = f"{PY} -m pytest -q -p no:cacheprovider tests"
YES = ApprovalResponse(Choice.ONCE)


def fix_auth_script():
    return [
        respond(call("project_info", "c1")),
        respond(call("search_code", "c2", pattern="check_token")),
        respond(call("read_file", "c3", path="src/auth.py")),
        respond(call("update_plan", "c4", steps=[{"step": "Fix None handling", "status": "in_progress"},
                                                 {"step": "Run tests", "status": "pending"}])),
        respond(call("edit_file", "c5", path="src/auth.py", old_string="        return True", new_string="        return False"),
                text="The None check returns True; it should reject."),
        respond(call("run_command", "c6", command=TEST_CMD, reason="verify the fix")),
        respond(call("remember", "c7", content=f"Run tests with: python -m pytest -q", category="commands", verified=True, evidence="exit 0")),
        ChatResponse("Fixed check_token to reject None in src/auth.py. Tests pass (1 passed)."),
    ]


def test_full_task_workflow_with_approvals_and_memory(make_session, workspace, settings):
    session, provider, ui = make_session(fix_auth_script(), mode="ask", approvals=[YES, YES])
    report = session.agent.run_task("Find the auth bug, fix it and run the tests")

    assert report.status == "completed", report.stop_reason
    assert "return False" in (workspace / "src/auth.py").read_text()
    # Exactly the edit and the test command needed approval; reads/search/plan/memory did not.
    assert [r.tool for r in ui.requests] == ["edit_file", "run_command"]
    assert "None check" in ui.requests[0].reason
    assert "+        return False" in ui.requests[0].assessment.details
    # Evidence comes from real execution.
    assert report.files == [{"path": "src/auth.py", "change": "modified"}]
    assert len(report.commands) == 1 and report.commands[0].exit_code == 0 and report.commands[0].is_test
    assert any("passed" in n for n in report.verification())
    # Tool results were fed back to the model verbatim.
    last_seen = provider.seen[-1]
    tool_msgs = [m for m in last_seen if m.role == "tool"]
    assert any("1 passed" in m.content for m in tool_msgs)
    assert any("src/auth.py:1" in m.content for m in tool_msgs)

    # Restart: a new session in the same project sees the remembered fact and the task history.
    provider2 = ScriptedProvider([ChatResponse("ok")])
    session2 = build_session(settings, workspace, RecordingUI(), provider2, provider2.model_info())
    session2.agent.run_task("How do I run the tests here?")
    system = provider2.seen[0][0].content
    assert "Run tests with: python -m pytest -q" in system and "[verified, commands]" in system
    assert "Find the auth bug" in system  # related past task
    assert session2.memory.tasks()[0]["files_changed"] == ["src/auth.py"]


def test_denied_edit_is_not_performed_and_feedback_reaches_model(make_session, workspace):
    steps = [
        respond(call("read_file", "c1", path="src/auth.py")),
        respond(call("edit_file", "c2", path="src/auth.py", old_string="return True", new_string="return False")),
        ChatResponse("Understood, I did not change the file."),
    ]
    session, provider, ui = make_session(steps, approvals=[ApprovalResponse(Choice.DENY, "use `is None` instead")])
    original = (workspace / "src/auth.py").read_text()
    report = session.agent.run_task("fix it")
    assert (workspace / "src/auth.py").read_text() == original
    denial = [m for m in provider.seen[-1] if m.role == "tool"][-1].content
    assert "DENIED by the user" in denial and "use `is None` instead" in denial
    assert report.denied and report.files == []


def test_plan_mode_never_modifies_or_asks(make_session, workspace):
    steps = [
        respond(call("read_file", "c1", path="src/auth.py")),
        respond(call("edit_file", "c2", path="src/auth.py", old_string="return True", new_string="return False")),
        respond(call("run_command", "c3", command="touch x.txt")),
        respond(call("run_command", "c4", command="git status")),
        ChatResponse("Plan: 1. change return True to return False in src/auth.py 2. run pytest"),
    ]
    original = (workspace / "src/auth.py").read_text()
    session, provider, ui = make_session(steps, mode="plan")
    report = session.agent.run_task("plan the fix")
    assert ui.requests == []
    assert (workspace / "src/auth.py").read_text() == original and not (workspace / "x.txt").exists()
    outputs = [m.content for m in provider.seen[-1] if m.role == "tool"]
    assert sum("Plan mode is read-only" in o for o in outputs) == 2
    assert "PLAN MODE IS ACTIVE" in provider.seen[0][0].content
    assert report.status == "completed"


def test_invalid_calls_are_reported_to_model_not_executed(make_session):
    steps = [
        respond(call("delete_everything", "c1")),
        respond(call("read_file", "c2", file="src/auth.py")),
        ChatResponse("", [ToolCall("c3", "read_file", None, "{bad json", "arguments are not valid JSON")]),
        respond(call("read_file", "c4", path="../../etc/passwd")),
        ChatResponse("done"),
    ]
    session, provider, ui = make_session(steps)
    report = session.agent.run_task("x")
    outs = [m.content for m in provider.seen[-1] if m.role == "tool"]
    assert "Unknown tool 'delete_everything'" in outs[0]
    assert "arguments.path is required" in outs[1] and "file is not a recognized parameter" in outs[1]
    assert "not valid JSON" in outs[2]
    assert "outside the workspace" in outs[3]
    assert report.status == "completed"


def test_repeated_identical_calls_stop_the_loop(make_session):
    steps = [respond(call("list_directory", f"c{i}", path="src")) for i in range(6)]
    session, provider, ui = make_session(steps)
    report = session.agent.run_task("x")
    assert report.status == "stopped" and "repeated the same list_directory call 3 times" in report.stop_reason
    assert report.iterations == 3


def test_iteration_limit(make_session):
    steps = [respond(call("read_file", f"c{i}", path="src/auth.py", start_line=i + 1)) for i in range(10)]
    session, provider, ui = make_session(steps, max_iterations=4)
    report = session.agent.run_task("x")
    assert report.status == "stopped" and "iteration limit reached (4" in report.stop_reason


def test_token_budget(make_session):
    steps = [respond(call("read_file", f"c{i}", path="src/auth.py", start_line=1, max_lines=i + 1)) for i in range(10)]
    session, provider, ui = make_session(steps, max_total_tokens=50)
    report = session.agent.run_task("x")
    assert report.status == "stopped" and "token budget" in report.stop_reason


def test_retryable_provider_error_then_success(make_session, monkeypatch):
    monkeypatch.setattr("local_agent.agent.loop.time.sleep", lambda s: None)
    steps = [ProviderError("overloaded", retryable=True), ChatResponse("recovered")]
    session, provider, ui = make_session(steps)
    report = session.agent.run_task("x")
    assert report.status == "completed" and report.final_message == "recovered"
    assert any("retrying" in e[2] for e in ui.events if e[0] == "notice")


def test_unavailable_model_fails_gracefully(make_session):
    session, provider, ui = make_session([ProviderUnavailable("Cannot reach Ollama")])
    report = session.agent.run_task("x")
    assert report.status == "failed" and "Cannot reach Ollama" in report.stop_reason


def test_cancel_then_resume_after_restart(make_session, workspace, settings):
    def interrupt(_):
        raise KeyboardInterrupt

    steps = [
        respond(call("read_file", "c1", path="src/auth.py")),
        respond(call("edit_file", "c2", path="src/auth.py", old_string="return True", new_string="return False")),
        interrupt,
    ]
    session, provider, ui = make_session(steps, mode="auto")
    report = session.agent.run_task("fix the auth bug")
    assert report.status == "cancelled"
    saved = session.memory.latest_interrupted_session()
    assert saved and saved["task"] == "fix the auth bug" and saved["status"] == "interrupted"

    # New process: restore and continue; the agent's earlier change is still undoable.
    provider2 = ScriptedProvider([ChatResponse("Resumed and finished.")])
    session2 = build_session(settings, workspace, RecordingUI(), provider2, provider2.model_info())
    task = resume_into(session2, saved)
    report2 = session2.agent.run_task(task, resumed=True)
    assert report2.status == "completed"
    assert "interrupted" in provider2.seen[0][-1].content
    assert any(m.role == "tool" and "src/auth.py" in m.content for m in provider2.seen[0])
    results = session2.ctx.tracker.undo()
    assert results and results[0].ok and "return True" in (workspace / "src/auth.py").read_text()


def test_report_flags_unverified_claims(make_session):
    steps = [
        respond(call("write_file", "c1", path="new.py", content="x = 1\n")),
        ChatResponse("Done! All tests pass."),
    ]
    session, provider, ui = make_session(steps, mode="auto")
    report = session.agent.run_task("add new.py")
    notes = " ".join(report.verification())
    assert "UNVERIFIED" in notes
    assert report.final_message == "Done! All tests pass."  # the claim is shown, but the evidence contradicts it


def test_failed_tests_reported_as_failed(make_session):
    steps = [respond(call("run_command", "c1", command=TEST_CMD)), ChatResponse("Tests pass.")]
    session, provider, ui = make_session(steps, mode="full")
    report = session.agent.run_task("run tests")
    assert report.commands[0].exit_code != 0
    assert any("FAILED" in n for n in report.verification())


def test_always_allow_is_scoped_to_command_prefix(make_session):
    steps = [
        respond(call("run_command", "c1", command=f"{PY} -c \"print(1)\"")),
        respond(call("run_command", "c2", command=f"{PY} -c \"print(2)\"")),
        respond(call("run_command", "c3", command="make --version-bogus")),
        ChatResponse("done"),
    ]
    session, provider, ui = make_session(steps, approvals=[ApprovalResponse(Choice.ALWAYS), ApprovalResponse(Choice.DENY)])
    session.agent.run_task("x")
    assert [r.assessment.summary for r in ui.requests] == [f"$ {PY} -c \"print(1)\"", "$ make --version-bogus"]


def test_text_tool_call_salvaged_in_native_mode(make_session):
    steps = [ChatResponse('{"name": "read_file", "arguments": {"path": "src/auth.py"}}'), ChatResponse("It checks tokens.")]
    session, provider, ui = make_session(steps)
    report = session.agent.run_task("what does auth.py do")
    assert report.tool_calls == 1
    assert any("check_token" in m.content for m in provider.seen[-1] if m.role == "tool")


def test_empty_responses_stop(make_session):
    session, provider, ui = make_session([ChatResponse(""), ChatResponse("")])
    report = session.agent.run_task("x")
    assert report.status == "stopped" and "empty" in report.stop_reason


def test_commands_changing_files_are_reported(make_session, workspace):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    steps = [respond(call("run_command", "c1", command=f"{PY} -c \"open('gen.txt','w').write('x')\"")), ChatResponse("done")]
    session, provider, ui = make_session(steps, mode="full")
    report = session.agent.run_task("generate")
    assert "gen.txt" in report.changed_by_commands
    assert any("not covered by /undo" in n for n in report.verification())


def test_compaction_triggers_on_small_context(make_session, workspace):
    (workspace / "big.py").write_text("\n".join(f"x{i} = {i}  # padding padding padding" for i in range(400)))
    steps = [respond(call("read_file", f"c{i}", path="big.py", start_line=1 + i * 100, max_lines=100)) for i in range(4)]
    steps.append(ChatResponse("done"))
    session, provider, ui = make_session(steps, context_window=3000)
    session.agent.context_window = 3000
    report = session.agent.run_task("read big.py")
    assert report.status == "completed"
    assert any("Context compacted" in e[2] for e in ui.events if e[0] == "notice")
