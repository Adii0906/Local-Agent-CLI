import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers import ScriptedProvider  # noqa: E402
from local_agent.agent.events import RecordingUI  # noqa: E402
from local_agent.cli.session import build_session  # noqa: E402
from local_agent.config import Settings  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "project"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "auth.py").write_text(
        "def check_token(token):\n    if token == None:\n        return True\n    return token.startswith('tok_')\n",
        encoding="utf-8",
    )
    (ws / "tests").mkdir()
    (ws / "tests" / "test_auth.py").write_text(
        "from src.auth import check_token\n\ndef test_none_rejected():\n    assert check_token(None) is False\n",
        encoding="utf-8",
    )
    (ws / ".env").write_text("API_KEY=sk-live-0123456789abcdefghijklmnop\n", encoding="utf-8")
    return ws


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(model="scripted-model", data_dir=tmp_path / "data")
    s.max_iterations = 15
    s.command_timeout = 30
    return s


@pytest.fixture
def make_session(workspace: Path, settings: Settings):
    def _make(steps, mode: str = "ask", approvals=None, ui=None, **overrides):
        settings.mode = mode
        for k, v in overrides.items():
            setattr(settings, k, v)
        provider = ScriptedProvider(steps)
        ui = ui or RecordingUI(approvals)
        session = build_session(settings, workspace, ui, provider, provider.model_info(), "test")
        return session, provider, ui

    return _make
