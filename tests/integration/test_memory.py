import json


from local_agent.memory.manager import MemoryManager, project_id
from local_agent.memory.store import SCHEMA_VERSION, JsonDocument
from local_agent.permissions.engine import ApprovalResponse, Choice


def test_persists_across_instances_and_isolates_projects(tmp_path):
    data = tmp_path / "data"
    a, b = tmp_path / "proj-a", tmp_path / "proj-b"
    a.mkdir()
    b.mkdir()
    MemoryManager(data, a).add_fact("Tests run with `make test`", category="commands", verified=True, evidence="exit 0")
    MemoryManager(data, a).add_preference("Prefer pytest fixtures over setUp")

    again = MemoryManager(data, a)
    assert [f["content"] for f in again.facts()] == ["Tests run with `make test`"]
    assert again.facts()[0]["verified"] is True
    other = MemoryManager(data, b)
    assert other.facts() == []  # project memory never leaks between projects
    assert [p["content"] for p in other.preferences()] == ["Prefer pytest fixtures over setUp"]  # preferences are global
    assert project_id(a) != project_id(b)


def test_dedupe_search_delete_clear(tmp_path):
    mem = MemoryManager(tmp_path / "d", tmp_path)
    f1 = mem.add_fact("Auth logic lives in src/auth.py", category="architecture")
    mem.add_fact("auth logic lives in   src/auth.py", verified=True)  # duplicate (normalized)
    assert len(mem.facts()) == 1 and mem.facts()[0]["verified"] is True
    mem.add_fact("Database migrations use alembic", category="architecture")
    mem.record_task({"task": "fix token expiry bug", "status": "completed", "summary": "patched jwt check", "files_changed": ["src/auth.py"]})
    hits = mem.search("auth token")
    assert hits[0]["id"] in (f1["id"],) or hits[0]["kind"] == "task"
    assert {h["kind"] for h in hits} >= {"fact", "task"}
    assert mem.search("zzz-nothing") == []
    assert mem.delete(f1["id"]) and not mem.delete("f-missing")
    assert [f["content"] for f in mem.facts()] == ["Database migrations use alembic"]
    assert mem.clear("all") == ["project facts", "task history", "preferences", "saved sessions"]
    assert mem.facts() == [] and mem.tasks() == []


def test_secrets_are_redacted_before_persisting(tmp_path):
    mem = MemoryManager(tmp_path / "d", tmp_path)
    mem.add_fact("deploy token is ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    mem.record_task({"task": "set OPENAI key sk-proj-abcdefghijklmnopqrstuvwx", "status": "completed"})
    raw = "".join(p.read_text() for p in (tmp_path / "d").rglob("*.json"))
    assert "ghp_abc" not in raw and "sk-proj-abc" not in raw and "[REDACTED]" in raw


def test_context_block_marks_memory_as_data_and_is_bounded(tmp_path):
    mem = MemoryManager(tmp_path / "d", tmp_path)
    for i in range(200):
        mem.add_fact(f"fact number {i} " + "x" * 50, verified=i % 2 == 0)
    block = mem.context_block("fact number 7", budget_chars=1500)
    assert len(block) <= 1600 and "fact number 7" in block
    assert "may be outdated" in block


def test_future_or_corrupt_schema_is_quarantined_not_lost(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 5, "facts": ["keep me"]}))
    doc = JsonDocument(path, lambda: {"facts": []})
    assert doc.load()["facts"] == []
    backups = list(tmp_path.glob("doc.json.bak-*"))
    assert backups and "keep me" in backups[0].read_text() and "newer" in doc.warning
    path.write_text("{not json")
    assert JsonDocument(path, lambda: {"facts": []}).load()["facts"] == []


def test_legacy_unversioned_file_is_upgraded(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"facts": [{"id": "f-1", "content": "x"}]}))
    data = JsonDocument(path, lambda: {"facts": []}).load()
    assert data["schema_version"] == SCHEMA_VERSION and data["facts"][0]["id"] == "f-1"


def test_remember_tool_rejects_secrets(make_session):
    from helpers import call, respond

    from local_agent.llm.base import ChatResponse

    steps = [
        respond(call("remember", "c1", content="The API key is sk-live-0123456789abcdefghijklmnop", verified=True)),
        respond(call("remember", "c2", content="Use 4-space indentation", verified=False, scope="preference")),
        ChatResponse("ok"),
    ]
    session, provider, ui = make_session(steps, approvals=[ApprovalResponse(Choice.DENY)])
    session.agent.run_task("x")
    outs = [m.content for m in provider.seen[-1] if m.role == "tool"]
    assert "Refusing to store" in outs[0]
    assert "DENIED" in outs[1]  # preferences need the user's approval
    assert session.memory.facts() == [] and session.memory.preferences() == []


def test_session_transcripts_are_redacted(make_session, workspace):
    from helpers import call, respond

    from local_agent.llm.base import ChatResponse
    from local_agent.permissions.engine import ApprovalResponse, Choice

    steps = [respond(call("read_file", "c1", path=".env")), ChatResponse("done")]
    session, provider, ui = make_session(steps, approvals=[ApprovalResponse(Choice.ONCE)])
    session.agent.run_task("read env")
    raw = "".join(p.read_text() for p in session.memory.project_dir.rglob("*.json"))
    assert "sk-live-0123456789" not in raw


def test_prune_sessions_keeps_recent(tmp_path):
    import os
    import time

    mem = MemoryManager(tmp_path / "d", tmp_path)
    for i in range(5):
        mem.session_doc(f"s{i}").save({"session_id": f"s{i}", "status": "completed"})
        (mem.checkpoint_dir(f"s{i}") / "blobs").mkdir(parents=True)
        os.utime(mem.project_dir / "sessions" / f"s{i}.json", (time.time() + i, time.time() + i))
    assert mem.prune_sessions(keep=2) == 3
    assert sorted(s["session_id"] for s in mem.list_sessions()) == ["s3", "s4"]
    assert not mem.checkpoint_dir("s0").exists() and mem.checkpoint_dir("s4").exists()
