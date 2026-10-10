"""Provider adapters against a local fake HTTP server (no real model, no network)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from local_agent.config import Settings
from local_agent.errors import ProviderError, ProviderUnavailable
from local_agent.llm.base import Message
from local_agent.llm.factory import create_provider
from local_agent.llm.ollama import OllamaProvider
from local_agent.llm.openai_compat import OpenAICompatProvider


class FakeServer:
    """Scriptable fake of the Ollama and OpenAI-compatible HTTP APIs."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.chat_replies: List[Any] = []  # list of chunk-lists, or an int HTTP status
        self.models = ["qwen2.5-coder:7b"]
        self.capabilities: Any = ["completion", "tools"]
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                server.requests.append({"path": self.path, "headers": dict(self.headers)})
                if self.path == "/api/tags":
                    self._json(200, {"models": [{"name": m} for m in server.models]})
                elif self.path == "/v1/models":
                    self._json(200, {"data": [{"id": m} for m in server.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"path": self.path, "body": payload, "headers": dict(self.headers)})
                if self.path == "/api/show":
                    if payload.get("model") not in server.models:
                        self._json(404, {"error": f"model '{payload.get('model')}' not found"})
                        return
                    info = {"model_info": {"qwen2.context_length": 32768}, "details": {"family": "qwen2", "parameter_size": "7.6B"}}
                    if server.capabilities is not None:
                        info["capabilities"] = server.capabilities
                    self._json(200, info)
                    return
                reply = server.chat_replies.pop(0) if server.chat_replies else []
                if isinstance(reply, int):
                    self._json(reply, {"error": "server says no"})
                    return
                self.send_response(200)
                sse = self.path.endswith("/chat/completions")
                self.send_header("Content-Type", "text/event-stream" if sse else "application/x-ndjson")
                self.end_headers()
                for chunk in reply:
                    line = (f"data: {json.dumps(chunk)}\n\n" if sse else json.dumps(chunk) + "\n").encode()
                    self.wfile.write(line)
                    self.wfile.flush()
                if sse:
                    self.wfile.write(b"data: [DONE]\n\n")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def test_ollama_streaming_tool_calls_and_usage(server):
    server.chat_replies.append([
        {"message": {"role": "assistant", "content": "Let me "}, "done": False},
        {"message": {"role": "assistant", "content": "check.", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 120, "eval_count": 30},
    ])
    p = OllamaProvider(server.url, "qwen2.5-coder:7b")
    info = p.model_info()
    assert info.supports_tools is True and info.context_length == 32768
    streamed: List[str] = []
    resp = p.chat([Message("system", "s"), Message("user", "hi")], [{"name": "read_file", "description": "", "parameters": {}}], on_text=streamed.append)
    assert resp.content == "Let me check." and "".join(streamed) == "Let me check."
    assert resp.tool_calls[0].name == "read_file" and resp.tool_calls[0].arguments == {"path": "a.py"}
    assert resp.usage.prompt_tokens == 120 and resp.usage.completion_tokens == 30
    body = server.requests[-1]["body"]
    assert body["options"]["num_ctx"] == 32768 and body["tools"][0]["type"] == "function"


def test_ollama_errors_are_actionable(server):
    with pytest.raises(ProviderUnavailable, match="not installed") as exc:
        OllamaProvider(server.url, "missing-model").model_info()
    assert "ollama pull missing-model" in exc.value.hint
    with pytest.raises(ProviderUnavailable, match="Cannot reach"):
        OllamaProvider("http://127.0.0.1:9", "m").model_info()
    server.chat_replies.append(503)
    with pytest.raises(ProviderError) as err:
        OllamaProvider(server.url, "qwen2.5-coder:7b").chat([Message("user", "x")])
    assert err.value.retryable
    server.chat_replies.append([{"error": "registry.ollama.ai/library/x does not support tools"}])
    with pytest.raises(ProviderError, match="does not support native tool calling"):
        OllamaProvider(server.url, "qwen2.5-coder:7b").chat([Message("user", "x")])


def test_capability_detection_selects_tool_mode(server):
    s = Settings(model="qwen2.5-coder:7b", base_url=server.url)
    provider, info, why = create_provider(s)
    assert provider.tool_mode == "native"
    server.capabilities = ["completion"]
    provider, info, why = create_provider(s)
    assert provider.tool_mode == "prompted" and "lacks native tool calling" in why
    server.capabilities = None  # old Ollama without capability reporting
    provider, info, why = create_provider(s)
    assert provider.tool_mode == "prompted" and "unknown" in why


def test_openai_compatible_streaming_and_auth(server, monkeypatch):
    server.chat_replies.append([
        {"choices": [{"delta": {"content": "Hi"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_9", "function": {"name": "run_command", "arguments": "{\"comm"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "and\": \"ls\"}"}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    ])
    monkeypatch.setenv("MY_KEY", "sk-test-abcdefghijklmnopqrstuvwxyz")
    s = Settings(provider="openai", model="qwen2.5-coder:7b", base_url=server.url + "/v1", api_key_env="MY_KEY")
    provider, info, why = create_provider(s)
    assert isinstance(provider, OpenAICompatProvider) and "sk-test" not in repr(provider)
    resp = provider.chat([Message("user", "x")], [{"name": "run_command", "description": "", "parameters": {}}])
    assert resp.content == "Hi" and resp.tool_calls[0].arguments == {"command": "ls"} and resp.tool_calls[0].id == "call_9"
    assert resp.usage.prompt_tokens == 10
    assert server.requests[-1]["headers"]["Authorization"] == "Bearer sk-test-abcdefghijklmnopqrstuvwxyz"
    with pytest.raises(ProviderUnavailable, match="not served"):
        create_provider(Settings(provider="openai", model="nope", base_url=server.url + "/v1"))


def test_malformed_tool_arguments_are_flagged(server):
    server.chat_replies.append([
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "read_file", "arguments": "{not json"}}]}}]},
    ])
    p = OpenAICompatProvider(server.url + "/v1", "qwen2.5-coder:7b")
    resp = p.chat([Message("user", "x")])
    assert resp.tool_calls[0].arguments is None and "not valid JSON" in resp.tool_calls[0].parse_error


def test_cli_end_to_end_against_fake_ollama(server, tmp_path, monkeypatch, capsys):
    """The real CLI entry point drives a full task: read, edit (auto mode), run tests, report."""
    import sys

    from local_agent.cli.main import main

    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (ws / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    py = sys.executable
    test_cmd = f"{py} -m pytest -q -p no:cacheprovider test_calc.py"

    def tool(name, **args):
        return [{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}, "done": False},
                {"message": {"content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 5}]

    server.chat_replies += [
        tool("read_file", path="calc.py"),
        tool("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b"),
        tool("run_command", command=test_cmd),
        [{"message": {"content": "Fixed add() in calc.py; test_calc passes."}, "done": True}],
    ]
    monkeypatch.setenv("LOCAL_AGENT_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_AGENT_CONFIG", str(tmp_path / "cfg.json"))
    monkeypatch.setattr("sys.stdin", open(__file__))  # not a TTY: approvals that are needed get denied
    rc = main(["--model", "qwen2.5-coder:7b", "--base-url", server.url, "--mode", "full", "-C", str(ws), "fix the add function"])
    out = capsys.readouterr().out
    assert "return a + b" in (ws / "calc.py").read_text()
    assert rc == 0, out
    assert "calc.py (modified)" in out and "exit 0" in out and "last test run passed" in out
    # The tool results the model received are the real ones.
    chat_bodies = [r["body"] for r in server.requests if r["path"] == "/api/chat"]
    final_msgs = chat_bodies[-1]["messages"]
    assert any(m["role"] == "tool" and "1 passed" in m["content"] for m in final_msgs)


def test_cli_reports_unreachable_model_cleanly(tmp_path, monkeypatch, capsys):
    from local_agent.cli.main import main

    monkeypatch.setenv("LOCAL_AGENT_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("LOCAL_AGENT_CONFIG", str(tmp_path / "cfg.json"))
    rc = main(["--model", "m", "--base-url", "http://127.0.0.1:9", "-C", str(tmp_path), "hello"])
    out = capsys.readouterr().out
    assert rc == 2 and "Model unavailable" in out and "Traceback" not in out
