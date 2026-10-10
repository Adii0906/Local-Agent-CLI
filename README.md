# Local Agent

A local-first coding agent for the terminal. Open a terminal in a project, describe a task in plain language, and the agent inspects the code, plans, edits files, runs tests, reads the failures and iterates. A permission engine you control gates every step, and the final report is built from what actually happened.

```
$ cd my-project
$ local-agent
[ask] > The login tests fail after the token refactor. Find the bug, fix it and run the auth tests.
  → Search for 'check_token' in .
  ✓ Searched 'check_token': 3 match(es) (ripgrep)
  → Read src/auth.py
  ✓ Read src/auth.py lines 1-88 of 88
  Plan:
    [→] Fix None handling in check_token
    [ ] Run tests/test_auth.py

! Permission needed  [ask mode, edit: modifies existing files (undoable with /undo)]
  Action:  Edit src/auth.py (1 replacement)
  Why:     check_token returns True for None; it must reject missing tokens.
    -        return True
    +        return False
  [y] approve once   [a] always allow 'src/auth.py' this session   [n] deny   [f] deny with feedback
  > y
  ✓ Edited src/auth.py
  ...
── Task report (from recorded tool results)
  Status:   completed: the model reported it was done
  Files:    src/auth.py (modified)
  Commands:
    [test] python -m pytest -q tests/test_auth.py  -> exit 0 (1.4s)
  ! The last test run passed: `python -m pytest -q tests/test_auth.py` (exit 0).
  Review with /diff, revert with /undo.
```

## Design priorities

- **You stay in control.** Four permission modes plus a real risk classifier. File edits show a diff before you approve them. Deletes, network access, privileged commands and secret files always ask. The model has no tool for approving its own actions.
- **Transparent execution.** The trace shows only real tool calls and their real results. There are no simulated progress bars.
- **Evidence-based completion.** The task report comes from recorded tool results, not from the model's claims. It flags changes that were never tested, a failing last test run, and files that commands changed behind the file tools.
- **Checkpoints and undo.** Every file is snapshotted before the agent's first change to it. `/undo` restores the originals. It skips any file someone else modified in the meantime instead of overwriting their work. It never runs `git reset`.
- **Persistent, inspectable memory.** Project facts (with a verified/unverified flag), user-approved preferences, task history and resumable sessions survive restarts. Memory is scoped per project, redacted for secrets, and searchable and deletable with `/memory`.
- **Local-first.** Ollama out of the box. Any OpenAI-compatible server (LM Studio, llama.cpp, vLLM) also works. No cloud dependency, and no runtime dependencies beyond the Python standard library.

## Install

Requires Python 3.9+ (tested on 3.11, 3.12 and 3.13), and [Ollama](https://ollama.com) or another OpenAI-compatible model server.

```bash
git clone https://github.com/Adii0906/Local-Agent-CLI.git
cd Local-Agent-CLI
pip install -e .            # installs the `local-agent` command
pip install -e ".[dev]"     # optional: pytest, for running the test suite
```

`python agent.py` still works without installing, for backward compatibility.

Optional: install [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) for fast code search. Without it, a Python fallback is used.

## Set up a model

```bash
ollama serve                       # if not already running
ollama pull qwen2.5-coder:7b       # recommended: coding model with native tool calling
local-agent --setup                # choose provider, model and default permission mode
local-agent --doctor               # verify server, model, context size, tool-calling support
```

The agent works best with a model that supports **native tool calling** (`qwen2.5-coder`, `qwen3`, `llama3.1`, `mistral`). For models without it, the agent switches to prompt-based tool calls, which is less reliable. `--doctor` and `/status` show which mode is active and why.

| Model | Size | Notes |
|---|---|---|
| `qwen2.5-coder:7b` | 4.7 GB | Recommended default |
| `qwen2.5-coder:3b` | 2.0 GB | Modest hardware; simpler tasks |
| `qwen3:8b`, `llama3.1:8b` | ~5 GB | Good tool calling, general reasoning |
| `deepseek-coder-v2:16b` | 9 GB | Better code, needs more RAM |
| `codellama:7b` | 3.8 GB | No native tools (prompt-based only) |

Small local models make mistakes. Review diffs before approving them, and prefer `ask` mode until you trust a model on your codebase.

### OpenAI-compatible servers

```bash
export MY_LLM_KEY=...              # only if your server needs a key
local-agent --provider openai --base-url http://localhost:1234/v1 --model qwen2.5-coder-7b-instruct
# store the variable *name* (never the key) in config:  "api_key_env": "MY_LLM_KEY"
```

## Usage

```bash
local-agent                                         # interactive session in the current directory
local-agent "Explain this codebase"                 # one task, then exit (exit code 0 = completed)
local-agent --plan "Add rate limiting to the API"   # read-only: inspect and propose a plan
local-agent --mode auto "Fix the failing tests"     # edits without asking; commands still ask
local-agent --model qwen3:8b --verbose              # pick a model; show tool output
local-agent --resume                                # continue the last interrupted task
local-agent -C path/to/project "..."                # authorize a different workspace directory
```

When a task needs approval and no interactive terminal is attached (piped input, CI), the action is **denied**, not approved.

### Interactive commands

| Command | Purpose |
|---|---|
| `/help` | List commands |
| `/status` | Model, tool-calling mode, permission mode, context and token usage, changes |
| `/plan [task]` | Switch to plan mode, or run one task in plan mode and switch back |
| `/mode <plan\|ask\|auto\|full>` | Change permission mode (`full` requires typing `yes`) |
| `/permissions [reset]` | Show mode, session rules and recent decisions; `reset` clears rules |
| `/memory` | Show project facts, preferences and recent tasks |
| `/memory search <q>` | Search memory |
| `/memory add <fact>` / `/memory pref <text>` | Save a fact / a global preference |
| `/memory forget <id>` | Delete one item |
| `/memory clear [project\|tasks\|preferences\|sessions\|all]` | Delete memory (asks to confirm) |
| `/diff [path]` | Diff of files the agent changed this session |
| `/undo [--force]` | Revert the agent's file changes |
| `/compact` | Summarize history to free context |
| `/analyze` | Project languages, frameworks and suggested commands |
| `/model [name]` | List models, or switch (offers `ollama pull` if missing) |
| `/resume`, `/history`, `/clear`, `/exit` | Resume interrupted task, task history, reset conversation, quit |

**Ctrl+C** cancels the running task: it stops the model stream and kills the running command's process group. The session continues, and the state is saved so `/resume` can pick it up. Ctrl+D exits.

## Permission modes

| | plan | ask (default) | auto | full |
|---|---|---|---|---|
| Read, search, list, `git status/diff`, read-only commands | ✓ | ✓ | ✓ | ✓ |
| Create / edit files in the workspace | ✗ | ask | ✓ | ✓ |
| Commands with uncertain effects (tests, builds, scripts) | ✗ | ask | ask | ✓ |
| Delete / overwrite (`delete_path`, `rm`, `git reset --hard`, …) | ✗ | ask | ask | ask |
| Network / external side effects (`git push`, `pip install`, `curl`) | ✗ | ask | ask | ask |
| Read or write sensitive files (`.env`, keys, credentials) | ask (read) | ask | ask | ask |
| Privileged (`sudo`, `systemctl`, system package managers) | ✗ | ✗ | ✗ | ask |
| Catastrophic patterns (`rm -rf /`, `curl … \| sh`, `mkfs`, fork bombs) | ✗ | ✗ | ✗ | ✗ |

Approval prompts show the exact action or command, the working directory, the affected paths, the model's stated reason, specific risks, and a diff for file edits. **"Always allow"** is offered only for edits to one specific file or for one command prefix (e.g. `python -m pytest`). It lasts for the session only, and it is never offered for destructive, network, privileged or sensitive actions.

Read [docs/SECURITY.md](docs/SECURITY.md) for exactly what these protections do and do not guarantee. **The permission system is not a sandbox.**

## Memory

Stored locally under `~/.local/share/local-agent/` (Linux/macOS; `$XDG_DATA_HOME` respected), `%APPDATA%\local-agent\` (Windows), or `$LOCAL_AGENT_HOME`:

| Layer | Scope | Written by |
|---|---|---|
| Project facts | One workspace (keyed by its real path) | The agent's `remember` tool (with a verified flag and evidence), or `/memory add` |
| Preferences | All projects | `/memory pref`, or the agent with your explicit approval |
| Task history | One workspace | Automatically: task, status, files changed, commands, verification notes |
| Sessions + checkpoints | One workspace | Automatically, for `/resume` and `/undo` (last 20 kept) |

Relevant memory is injected into the prompt as clearly labelled *reference data*. It is marked possibly outdated and cannot grant permissions. Secrets are redacted before anything is written, and the `remember` tool refuses content that looks like a credential. Files are versioned JSON. A corrupt file, or one from a newer version, is moved aside rather than overwritten. Use `--no-memory` to disable memory entirely.

## Configuration

`~/.config/local-agent/config.json` (or `$LOCAL_AGENT_CONFIG`; `%APPDATA%\local-agent\config.json` on Windows). Precedence: defaults < config file < environment (`LOCAL_AGENT_MODEL`, `LOCAL_AGENT_PROVIDER`, `LOCAL_AGENT_BASE_URL`, `LOCAL_AGENT_MODE`, `OLLAMA_HOST`) < CLI flags.

```json
{
  "provider": "ollama",
  "model": "qwen2.5-coder:7b",
  "mode": "ask",
  "tool_mode": "auto",
  "max_iterations": 40,
  "max_seconds": 1800,
  "max_total_tokens": 0,
  "context_window": 0,
  "command_timeout": 120,
  "max_output_chars": 12000,
  "extra_ignore": ["fixtures/large/"],
  "shell_env_passthrough": ["NPM_TOKEN"]
}
```

A config file inside the project is **never** read, so a repository cannot change your permissions or model endpoint. Credentials in the config file are rejected. Add a `.local-agentignore` (gitignore syntax) to hide paths from listing and search. An `AGENTS.md` or `LOCAL-AGENT.md` at the project root is shown to the model as informational project notes. It cannot change rules or permissions.

## How it works

```
user task ─► context (project discovery + relevant memory + notes)
          ─► model ─► tool call ─► schema validation ─► risk assessment ─► permission engine
                 ▲                                                    allow │ ask (you) │ deny
                 └──────────────── real tool result ◄── execution ◄─────────┘
          ─► stops on: model done │ iteration/time/token limit │ repeated identical call
                       │ repeated failures │ Ctrl+C │ model error
          ─► evidence-based report + memory update + saved session
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module layout and the audit of the original prototype.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest            # 180 tests: unit, integration, security
```

The tests never call a real or paid model. They use a scripted fake model, plus a local fake Ollama/OpenAI HTTP server for the adapters and a full CLI run. Shell tests only run the current Python interpreter with harmless snippets.

## Feature status

| Area | Status |
|---|---|
| Agent loop with native and prompt-based tool calling | Working; tested with scripted models and a fake Ollama server |
| File tools (read/page, targeted edit, write, move, delete, mkdir), search, find | Working |
| Command execution: timeouts, Ctrl+C, output limits, env scrubbing | Working on Linux; macOS/Windows code paths written but **not tested** yet |
| Permission engine, risk classifier, sensitive-file and workspace protection | Working; heuristic for shell commands (see SECURITY.md) |
| Persistent memory, task history, resume, checkpoints / undo | Working |
| Ollama adapter | Tested against a protocol-faithful fake server, **not yet against a live Ollama** |
| OpenAI-compatible adapter | Tested against a fake server only |
| Context management (elision, summarization, read de-duplication) | Working; token counts are estimates unless the server reports usage |
| OS-level sandboxing (containers, seccomp, network isolation) | **Not implemented**; approval + heuristics only |

## Troubleshooting

- **"Model unavailable: Cannot reach Ollama"**: start it with `ollama serve`, or point `--base-url` / `OLLAMA_HOST` at it.
- **"Model 'x' is not installed"**: `ollama pull x`, or choose one from `/model`.
- **The agent writes JSON instead of acting, or loops**: the model's tool calling is weak. Use a tool-capable model (`--doctor` shows support), or set `--tool-mode prompted`/`native` explicitly.
- **Answers get cut off or the model forgets early context**: Ollama defaults to a small context. The agent requests up to 32k (`num_ctx`), capped by the model. Set `context_window` if you have more RAM, or use `/compact`.
- **"iteration limit reached"**: the task needed more steps. Raise `--max-iterations` or split the task.
- **A command seems stuck**: stdin is closed, so interactive tools fail fast, and commands are killed at `command_timeout` (120s by default; Ctrl+C kills sooner).
- Use `--verbose` to see tool output, auto-approval reasons and tracebacks for internal errors.

## Acknowledgements

Started as a learning prototype built on [Ollama](https://ollama.com) and [Rich](https://github.com/Textualize/rich). Thanks to Ollama for making free local models easy to run.
