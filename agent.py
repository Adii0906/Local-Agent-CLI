#!/usr/bin/env python3
"""Backward-compatible launcher: `python agent.py` runs the packaged CLI.

Prefer `pip install -e .` and the `local-agent` command.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_agent.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
