"""A scriptable fake of the Ollama and OpenAI-compatible HTTP APIs (no network, no models)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

GB = 1024 ** 3


class FakeServer:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.chat_replies: List[Any] = []  # chunk-lists, or an int HTTP status
        self.models = ["qwen2.5-coder:7b"]
        self.sizes: Dict[str, int] = {}
        self.capabilities: Any = ["completion", "tools"]  # default for every model; None = old Ollama
        self.model_caps: Dict[str, Any] = {}  # per-model override
        self.pull_script: Dict[str, Any] = {}  # name -> list of chunks, or int status
        self.version = "0.6.2"
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream(self, chunks, sse=False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if sse else "application/x-ndjson")
                self.end_headers()
                for chunk in chunks:
                    line = (f"data: {json.dumps(chunk)}\n\n" if sse else json.dumps(chunk) + "\n").encode()
                    self.wfile.write(line)
                    self.wfile.flush()
                if sse:
                    self.wfile.write(b"data: [DONE]\n\n")

            def do_GET(self):
                server.requests.append({"path": self.path, "headers": dict(self.headers)})
                if self.path == "/api/version":
                    self._json(200, {"version": server.version})
                elif self.path == "/api/tags":
                    self._json(200, {"models": [
                        {"name": m, "size": server.sizes.get(m, 2 * GB),
                         "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"}}
                        for m in server.models
                    ]})
                elif self.path == "/v1/models":
                    self._json(200, {"data": [{"id": m} for m in server.models]})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"path": self.path, "body": payload, "headers": dict(self.headers)})
                if self.path == "/api/show":
                    name = payload.get("model")
                    if name not in server.models:
                        self._json(404, {"error": f"model '{name}' not found"})
                        return
                    info = {"model_info": {"qwen2.context_length": 32768}, "details": {"family": "qwen2", "parameter_size": "7.6B"}}
                    caps = server.model_caps.get(name, server.capabilities)
                    if caps is not None:
                        info["capabilities"] = caps
                    self._json(200, info)
                    return
                if self.path == "/api/pull":
                    name = payload.get("model")
                    script = server.pull_script.get(name)
                    if isinstance(script, int):
                        self._json(script, {"error": "pull failed"})
                        return
                    if script is None:
                        script = [{"status": "pulling manifest"},
                                  {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 50},
                                  {"status": "pulling abc", "digest": "sha256:abc", "total": 100, "completed": 100},
                                  {"status": "success"}]
                    if any(c.get("status") == "success" for c in script) and name not in server.models:
                        server.models.append(name)
                    self._stream(script)
                    return
                reply = server.chat_replies.pop(0) if server.chat_replies else []
                if isinstance(reply, int):
                    self._json(reply, {"error": "server says no"})
                    return
                self._stream(reply, sse=self.path.endswith("/chat/completions"))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def tool_reply(name: str, **arguments):
    """An Ollama /api/chat stream containing one native tool call."""
    return [{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 5}]


def text_reply(text: str):
    return [{"message": {"content": text}, "done": True, "prompt_eval_count": 10, "eval_count": 5}]
