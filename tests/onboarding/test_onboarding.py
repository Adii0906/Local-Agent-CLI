"""First-run onboarding, subcommands, startup recovery and config persistence.

Everything runs against a local fake Ollama server; answers to prompts are scripted.
"""

import builtins
import io
import json
import sys

import pytest
from fake_ollama import GB, FakeServer, text_reply, tool_reply

from local_agent.cli.prompts import KEY_DOWN, KEY_ENTER, KEY_UP, Option, Prompter, decode_key
from local_agent.cli.ui import Style, TerminalUI
from local_agent.config import CONFIG_SCHEMA_VERSION, Settings, load_settings, read_config_file
from local_agent.errors import ConfigError
from local_agent.onboarding import environment as env
from local_agent.onboarding.models import ModelEntry, recommend_download, recommend_installed
from local_agent.onboarding.validate import validate_model
from local_agent.onboarding.wizard import Wizard


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture
def home(tmp_path, monkeypatch, server):
    """Isolated config/data dirs and an Ollama URL pointing at the fake server."""
    cfg = tmp_path / "cfg" / "config.json"
    monkeypatch.setenv("LOCAL_AGENT_CONFIG", str(cfg))
    monkeypatch.setenv("LOCAL_AGENT_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OLLAMA_HOST", server.url)
    monkeypatch.setenv("LOCAL_AGENT_PLAIN_PROMPTS", "1")
    for var in ("LOCAL_AGENT_MODEL", "LOCAL_AGENT_MODE", "LOCAL_AGENT_PROVIDER", "LOCAL_AGENT_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return cfg


class Answers:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


def make_wizard(answers, settings=None, memory=16 * GB, **kw):
    out = io.StringIO()
    ui = TerminalUI(out=out, interactive=True)
    feed = Answers(answers)
    prompter = Prompter(ui.s, out, input_fn=feed, interactive=False)
    settings = settings or load_settings(lenient=True)
    wiz = Wizard(ui, prompter, settings, memory_bytes=memory, sleep=lambda s: None, **kw)
    return wiz, out, feed


def saved(cfg):
    return json.loads(cfg.read_text())


# ---------------------------------------------------------------- first run
def test_first_run_end_to_end_through_main(home, server, tmp_path, monkeypatch, capsys):
    """`local-agent` with no config: wizard -> saved config -> verified startup -> REPL."""
    from local_agent.cli.main import main

    ws = tmp_path / "my-project"
    ws.mkdir()
    (ws / "app.py").write_text("print('hi')\n")
    server.chat_replies.append(tool_reply("report_status", status="ready"))  # validation request
    feed = Answers(["1", "2", "/exit"])  # model #1, permission "Ask Before Changes", leave the REPL
    monkeypatch.setattr(builtins, "input", feed)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    rc = main(["-C", str(ws)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Welcome! Let's get your agent ready." in out
    for step in ("[1/4] Check your environment", "[2/4] Choose your model", "[3/4] Configure permissions", "[4/4] Start coding"):
        assert step in out
    assert "Ollama running: version 0.6.2" in out
    assert "Model responded: qwen2.5-coder:7b" in out and "native tool calling works" in out
    data = saved(home)
    assert data["schema_version"] == CONFIG_SCHEMA_VERSION and data["model"] == "qwen2.5-coder:7b"
    assert data["mode"] == "ask" and data["onboarding_completed"] is True and data["tool_mode"] == "auto"
    for line in ("Configuration loaded", "Ollama connected: v0.6.2", "Model: qwen2.5-coder:7b (native tool calling)",
                 "Permission mode: Ask Before Changes (ask)", "Project detected: my-project (Python)", "Memory initialized",
                 "Local-Agent-CLI is ready.", "What would you like to work on?"):
        assert line in out, line
    assert "api_key" not in home.read_text()
    assert "base_url" not in data  # OLLAMA_HOST from the environment is not pinned into the config


def test_returning_user_skips_onboarding(home, server, tmp_path, monkeypatch, capsys):
    from local_agent.cli.main import main

    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "qwen2.5-coder:7b", "mode": "auto", "onboarding_completed": True}))
    monkeypatch.setattr(builtins, "input", Answers(["/exit"]))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert main(["-C", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Welcome!" not in out and "Choose your language model" not in out
    assert "Permission mode: Auto Edit (auto)" in out and "Local-Agent-CLI is ready." in out


def test_v1_config_counts_as_onboarded(home, server):
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"model": "qwen2.5-coder:7b", "mode": "ask"}))  # written by 2.0 `--setup`
    s = load_settings()
    assert s.onboarding_completed is True and any("migrated" in w for w in s.load_warnings)
    assert "schema_version" not in json.loads(home.read_text())  # not rewritten just by loading


def test_non_interactive_first_run_does_not_hang(home, server, tmp_path, capsys):
    from local_agent.cli.main import main

    rc = main(["-C", str(tmp_path), "do something"])
    out = capsys.readouterr().out
    assert rc == 2 and "No model configured" in out and "local-agent setup" in out
    assert not home.exists()


# ---------------------------------------------------------------- environment
def test_installed_but_not_running_is_not_reported_as_running(tmp_path):
    state = env.OllamaState(base_url="http://127.0.0.1:9", binary="/usr/bin/ollama", reachable=False, error="refused")
    checks = {c.name: c for c in env.ollama_checks(state, "m")}
    assert checks["Ollama installed"].ok is True
    assert checks["Ollama running"].ok is False and "ollama serve" in checks["Ollama running"].hint
    real = env.probe_ollama("http://127.0.0.1:9")
    assert real.reachable is False and real.error


def test_ollama_missing_then_retry(home, server):
    calls = {"n": 0}

    def probe(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return env.OllamaState(base_url=url, binary=None, reachable=False, error="connection refused")
        return env.probe_ollama(url)

    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, feed = make_wizard(["1", "1", "2"], probe=probe)  # Retry, model #1, ask mode
    assert wiz.run() is not None
    text = out.getvalue()
    assert "Ollama installed: not found" in text and "ollama.com" in text
    assert "Ollama running: no response" in text and calls["n"] >= 2


def test_ollama_down_quit_saves_nothing(home):
    wiz, out, _ = make_wizard(["4"], probe=lambda url: env.OllamaState(base_url=url, reachable=False, error="refused"))
    assert wiz.run() is None
    assert "Setup cancelled. Nothing was saved." in out.getvalue() and not home.exists()


# ---------------------------------------------------------------- models
def test_no_models_download_recommended(home, server):
    server.models = []
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, _ = make_wizard(["1", "y", "2"])  # download recommended, confirm, ask mode
    assert wiz.run() is not None
    text = out.getvalue()
    assert "No language models are installed yet" in text and "Download qwen2.5-coder:7b" in text
    assert "Downloaded: qwen2.5-coder:7b" in text and "100.0%" in text
    assert any(r["path"] == "/api/pull" for r in server.requests)
    assert saved(home)["model"] == "qwen2.5-coder:7b"


def test_download_requires_consent(home, server):
    server.models = []
    wiz, out, _ = make_wizard(["1", "n", "4"])  # download? no -> back to menu -> quit
    assert wiz.run() is None
    assert not any(r["path"] == "/api/pull" for r in server.requests)


def test_download_failure_then_choose_other(home, server):
    server.models = ["llama3.2:3b"]
    server.pull_script["qwen2.5-coder:7b"] = [{"status": "pulling manifest"}, {"error": "connection reset by peer"}]
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    # menu: 1 llama3.2:3b (recommended), 2 Download qwen2.5-coder:7b, 3 custom, 4 quit
    wiz, out, _ = make_wizard(["2", "y", "2", "1", "2"])  # download -> fails -> "Choose a different model" -> llama -> ask
    assert wiz.run() is not None
    text = out.getvalue()
    assert "Download failed" in text and "connection reset" in text
    assert saved(home)["model"] == "llama3.2:3b"


def test_download_cancel_with_ctrl_c(home, server, monkeypatch):
    from local_agent.llm.ollama import OllamaProvider

    server.models = []

    def interrupted(self, name, on_progress=None, cancel_event=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(OllamaProvider, "pull", interrupted)
    wiz, out, _ = make_wizard(["1", "y", "3"])  # download -> Ctrl+C -> quit
    assert wiz.run() is None
    assert "Download cancelled" in out.getvalue() and not home.exists()


def test_unknown_model_name_in_library(home, server):
    server.pull_script["nope:1b"] = [{"error": "pull model manifest: file does not exist"}]
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, _ = make_wizard(["2", "nope:1b", "y", "1", "2"])  # custom name -> not found -> pick installed -> ask
    assert wiz.run() is not None
    assert "was not found in the Ollama library" in out.getvalue()


def test_model_without_tool_support_is_explained(home, server):
    server.models = ["codellama:7b"]
    server.model_caps["codellama:7b"] = ["completion"]
    server.chat_replies.append(text_reply("ready"))
    wiz, out, _ = make_wizard(["1", "2", "2"])  # codellama, "Continue with codellama", ask mode
    assert wiz.run() is not None
    text = out.getvalue()
    assert "no tools" in text and "prompt-based tool calling" in text
    assert saved(home)["tool_mode"] == "prompted"


def test_model_test_failure_offers_alternatives(home, server):
    server.models = ["qwen2.5-coder:7b", "llama3.2:3b"]
    server.chat_replies += [503, tool_reply("report_status", status="ready")]
    wiz, out, _ = make_wizard(["1", "1", "2", "2"])  # qwen -> fails -> choose different -> llama -> ask
    result = wiz.run()
    assert result is not None and "Model test failed" in out.getvalue()


def test_recommendations_use_real_installed_models():
    small = ModelEntry("qwen2.5-coder:3b", 2 * GB, tools=True)
    big = ModelEntry("qwen2.5-coder:32b", 20 * GB, tools=True)
    general = ModelEntry("llama3.2:3b", 2 * GB, tools=True)
    no_tools = ModelEntry("codellama:7b", 4 * GB, tools=False)
    embed = ModelEntry("nomic-embed-text", 1 * GB, embedding=True)
    assert recommend_installed([general, small, no_tools, embed], 16 * GB).name == "qwen2.5-coder:3b"
    assert recommend_installed([small, big], 16 * GB).name == "qwen2.5-coder:3b"  # 32b does not fit in 16 GB
    assert recommend_installed([small, big], 64 * GB).name == "qwen2.5-coder:32b"
    assert recommend_installed([embed], 16 * GB) is None
    assert recommend_download(4 * GB)["name"] == "qwen2.5-coder:1.5b"
    assert recommend_download(16 * GB)["name"] == "qwen2.5-coder:7b"


def test_validation_outcomes(server):
    s = Settings(model="qwen2.5-coder:7b", base_url=server.url)
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    assert validate_model(s).tool_calling == "native"
    server.chat_replies.append(text_reply('{"name": "report_status", "arguments": {"status": "ready"}}'))
    assert validate_model(s).tool_calling == "text"
    server.chat_replies.append(text_reply("ready"))
    r = validate_model(s)
    assert r.responded and r.tool_calling == "none" and r.recommended_tool_mode() == "prompted"
    server.chat_replies.append([{"error": "registry.ollama.ai/library/x does not support tools"}])
    server.chat_replies.append(text_reply("ready"))
    r = validate_model(s)
    assert r.responded and r.tool_calling == "none"
    r = validate_model(Settings(model="x", base_url="http://127.0.0.1:9"))
    assert not r.responded and "Cannot reach" in r.error


# ---------------------------------------------------------------- config
def test_existing_config_is_preserved(home, server):
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "old-model", "max_iterations": 77,
                                "extra_ignore": ["fixtures/"], "future_setting": {"x": 1}, "onboarding_completed": True}))
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, _ = make_wizard(["1", "3", "y"])  # qwen, auto, confirm save
    assert wiz.run(first_run=False) is not None
    data = saved(home)
    assert data["model"] == "qwen2.5-coder:7b" and data["mode"] == "auto"
    assert data["max_iterations"] == 77 and data["extra_ignore"] == ["fixtures/"] and data["future_setting"] == {"x": 1}
    assert "Other settings in the file are kept." in out.getvalue()


def test_declining_save_keeps_config_untouched(home, server):
    home.parent.mkdir(parents=True)
    original = json.dumps({"schema_version": 2, "model": "old-model", "onboarding_completed": True})
    home.write_text(original)
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, _, _ = make_wizard(["1", "2", "n"])
    assert wiz.run(first_run=False) is None
    assert home.read_text() == original


def test_corrupt_config_is_backed_up_and_repaired(home, server):
    home.parent.mkdir(parents=True)
    home.write_text("{this is not json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_settings()
    lenient = load_settings(lenient=True)
    assert lenient.load_warnings
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, _ = make_wizard(["1", "2", "y"], settings=lenient)
    assert wiz.run() is not None
    assert "will be renamed to config.json.bak" in out.getvalue()
    assert saved(home)["model"] == "qwen2.5-coder:7b"
    backups = list(home.parent.glob("config.json.bak-*"))
    assert backups and backups[0].read_text() == "{this is not json"


def test_invalid_values_and_newer_schema(home):
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "mode": "yolo", "model": "m"}))
    with pytest.raises(ConfigError, match="mode"):
        load_settings()
    s = load_settings(lenient=True)
    assert s.mode == "ask" and s.model == "m" and any("yolo" in w for w in s.load_warnings)
    home.write_text(json.dumps({"schema_version": 99, "model": "m"}))
    with pytest.raises(ConfigError, match="newer"):
        load_settings(lenient=True)
    from local_agent.config import save_user_config

    with pytest.raises(ConfigError, match="newer"):
        save_user_config({"model": "x"})
    assert json.loads(home.read_text())["schema_version"] == 99  # never overwritten
    assert read_config_file().error


def test_credentials_never_accepted(home):
    from local_agent.config import save_user_config

    with pytest.raises(ConfigError, match="credential"):
        save_user_config({"api_key": "sk-123"})
    home.parent.mkdir(parents=True, exist_ok=True)
    home.write_text(json.dumps({"schema_version": 2, "token": "abc"}))
    with pytest.raises(ConfigError, match="credential"):
        load_settings(lenient=True)


# ---------------------------------------------------------------- permissions
def test_permission_mode_persisted_and_enforced(home, server, workspace):
    from helpers import ScriptedProvider, call, respond

    from local_agent.agent.events import RecordingUI
    from local_agent.cli.session import build_session
    from local_agent.llm.base import ChatResponse

    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, _, _ = make_wizard(["1", "1"])  # model, Plan Mode
    assert wiz.run() is not None
    settings = load_settings()
    assert settings.mode == "plan"
    provider = ScriptedProvider([respond(call("write_file", "c1", path="new.py", content="x=1")), ChatResponse("done")])
    session = build_session(settings, workspace, RecordingUI(), provider, provider.model_info())
    assert session.engine.mode == "plan"
    session.agent.run_task("create new.py")
    assert not (workspace / "new.py").exists()
    assert "Plan mode is read-only" in [m for m in provider.seen[-1] if m.role == "tool"][0].content


def test_full_control_requires_typed_confirmation(home, server):
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, out, _ = make_wizard(["1", "4", "sure", "2"])  # full -> not 'yes' -> back -> ask
    wiz.run()
    assert saved(home)["mode"] == "ask" and "NOT sandboxed" in out.getvalue()
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    wiz, _, _ = make_wizard(["1", "4", "yes", "y"], settings=load_settings())
    wiz.run(first_run=False)
    assert saved(home)["mode"] == "full"


# ---------------------------------------------------------------- subcommands
def run_cli(argv, monkeypatch, capsys, answers=None, tty=False):
    from local_agent.cli.main import main

    if answers is not None:
        monkeypatch.setattr(builtins, "input", Answers(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
    rc = main(argv)
    return rc, capsys.readouterr().out


def test_models_list_and_use(home, server, monkeypatch, capsys):
    server.models = ["qwen2.5-coder:7b", "llama3.2:3b", "nomic-embed-text:latest"]
    server.model_caps["nomic-embed-text:latest"] = ["embedding"]
    server.sizes["llama3.2:3b"] = 2 * GB
    rc, out = run_cli(["models"], monkeypatch, capsys)
    assert rc == 0 and "qwen2.5-coder:7b" in out and "llama3.2:3b" in out and "2.0 GB" in out
    assert "coding, tools" in out and "nomic-embed" not in out
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    rc, out = run_cli(["models", "use", "llama3.2:3b"], monkeypatch, capsys)
    assert rc == 0 and "Default model: llama3.2:3b" in out
    assert saved(home)["model"] == "llama3.2:3b"
    rc, out = run_cli(["models", "use", "missing:1b"], monkeypatch, capsys)
    assert rc == 1 and "models pull missing:1b" in out and saved(home)["model"] == "llama3.2:3b"
    rc, out = run_cli(["models", "pull", "tiny:1b"], monkeypatch, capsys)
    assert rc == 0 and "Downloaded: tiny:1b" in out
    rc, out = run_cli(["models", "frobnicate"], monkeypatch, capsys)
    assert rc == 2 and "Usage" in out


def test_config_command(home, monkeypatch, capsys):
    rc, out = run_cli(["config", "set", "mode", "auto"], monkeypatch, capsys)
    assert rc == 0 and saved(home)["mode"] == "auto"
    rc, out = run_cli(["config", "set", "max_iterations", "12"], monkeypatch, capsys)
    assert rc == 0 and saved(home)["max_iterations"] == 12
    rc, out = run_cli(["config", "set", "mode", "yolo"], monkeypatch, capsys)
    assert rc == 2 and saved(home)["mode"] == "auto"
    rc, out = run_cli(["config", "set", "api_key", "sk-x"], monkeypatch, capsys)
    assert rc == 2 and "credential" in out and "sk-x" not in home.read_text()
    rc, out = run_cli(["config", "set", "mode", "full"], monkeypatch, capsys)  # non-interactive: refused
    assert rc == 1 and saved(home)["mode"] == "auto"
    rc, out = run_cli(["config", "path"], monkeypatch, capsys)
    assert out.strip() == str(home)
    rc, out = run_cli(["config"], monkeypatch, capsys)
    assert "max_iterations" in out and "(from file)" in out


def test_doctor(home, server, monkeypatch, capsys, tmp_path):
    rc, out = run_cli(["doctor", "-C", str(tmp_path)], monkeypatch, capsys)
    assert rc == 1 and "Configured model: none" in out  # nothing configured yet
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "qwen2.5-coder:7b", "onboarding_completed": True}))
    server.chat_replies.append(tool_reply("report_status", status="ready"))
    rc, out = run_cli(["doctor", "--test", "-C", str(tmp_path)], monkeypatch, capsys)
    assert rc == 0 and "Ollama running: version 0.6.2" in out and "Model test" in out and "No problems found." in out
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    rc, out = run_cli(["doctor", "-C", str(tmp_path)], monkeypatch, capsys)
    assert rc == 1 and "Ollama running: no response" in out


def test_subcommand_words_inside_tasks_are_tasks():
    from local_agent.cli.main import _split_subcommand

    assert _split_subcommand(["setup"]) == ("setup", [])
    assert _split_subcommand(["doctor", "--test"]) == ("doctor", ["--test"])
    assert _split_subcommand(["setup", "the", "database"]) == (None, ["setup", "the", "database"])
    assert _split_subcommand(["models", "use", "x"]) == ("models", ["use", "x"])


# ---------------------------------------------------------------- startup recovery
def test_missing_model_recovery_interactive(home, server, tmp_path, monkeypatch, capsys):
    server.models = ["llama3.2:3b"]
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "gone:7b", "onboarding_completed": True}))
    rc, out = run_cli(["-C", str(tmp_path)], monkeypatch, capsys, answers=["1", "y", "/exit"], tty=True)
    assert rc == 0, out
    assert "Model unavailable" in out and "Default model saved: llama3.2:3b" in out
    assert "Model: llama3.2:3b" in out and saved(home)["model"] == "llama3.2:3b"


def test_missing_model_non_interactive(home, server, tmp_path, monkeypatch, capsys):
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "gone:7b", "onboarding_completed": True}))
    rc, out = run_cli(["-C", str(tmp_path), "do x"], monkeypatch, capsys)
    assert rc == 2 and "gone:7b" in out and "ollama pull gone:7b" in out and "Traceback" not in out


def test_ollama_down_at_startup_offers_recovery(home, tmp_path, monkeypatch, capsys):
    home.parent.mkdir(parents=True)
    home.write_text(json.dumps({"schema_version": 2, "model": "m", "onboarding_completed": True}))
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    monkeypatch.setattr("local_agent.onboarding.system.ollama_binary", lambda: None)
    rc, out = run_cli(["-C", str(tmp_path)], monkeypatch, capsys, answers=["3"], tty=True)  # Retry, Run setup, Quit
    assert rc == 2 and "Ollama not reachable" in out and "Ollama connected" not in out


# ---------------------------------------------------------------- terminal behaviour
@pytest.mark.parametrize("seq,action", [
    ("\x1b[A", KEY_UP), ("\x1b[B", KEY_DOWN), ("\x1bOA", KEY_UP),  # POSIX / application mode
    ("\xe0H", KEY_UP), ("\xe0P", KEY_DOWN), ("\x00H", KEY_UP),     # Windows console scan codes
    ("\r", KEY_ENTER), ("\n", KEY_ENTER), ("3", "digit:3"), ("\x03", "interrupt"), ("x", ""),
])
def test_key_decoding(seq, action):
    assert decode_key(seq) == action


def test_plain_select_and_confirm():
    out = io.StringIO()
    feed = Answers(["", "9", "abc", "2"])
    p = Prompter(Style(False), out, input_fn=feed, interactive=False)
    opts = [Option("A", "a"), Option("B", "b", "hint", "recommended")]
    assert p.select("Pick", opts, default=1) == "b"  # Enter = default
    assert p.select("Pick", opts) == "b"  # invalid answers re-asked
    assert "Please enter a number" in out.getvalue() and "[recommended]" in out.getvalue()
    with pytest.raises(KeyboardInterrupt):
        Prompter(Style(False), out, input_fn=Answers([]), interactive=False).select("Pick", opts)
    assert Prompter(Style(False), out, input_fn=Answers([""]), interactive=False).confirm("ok?", True) is True


def test_ascii_fallback_for_legacy_consoles():
    class Cp437(io.StringIO):
        encoding = "cp437"

    ui = TerminalUI(out=Cp437(), interactive=True)
    ui.box(["hello"])
    ui.check(True, "Configuration loaded")
    text = ui.out.getvalue()
    assert "+---" in text and "| " in text and "  + Configuration loaded" in text
    text.encode("cp437")  # everything printable on a legacy Windows console


def test_platform_config_dirs(monkeypatch, tmp_path):
    from local_agent import config as cfg

    monkeypatch.delenv("LOCAL_AGENT_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(cfg.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert cfg.default_config_path() == tmp_path / "Roaming" / "local-agent" / "config.json"
    monkeypatch.setattr(cfg.sys, "platform", "darwin")
    monkeypatch.setattr(cfg.Path, "home", lambda: tmp_path)
    assert cfg.default_config_path() == tmp_path / "Library" / "Application Support" / "local-agent" / "config.json"
    (tmp_path / ".config" / "local-agent").mkdir(parents=True)  # existing 2.0 location keeps working
    assert cfg.default_config_path() == tmp_path / ".config" / "local-agent" / "config.json"
    monkeypatch.setattr(cfg.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert cfg.default_config_path() == tmp_path / "xdg" / "local-agent" / "config.json"
