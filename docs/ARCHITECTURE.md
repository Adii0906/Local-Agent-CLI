# Architecture

## Audit of the original prototype (v1.0)

| Area | Finding | Resolution |
|---|---|---|
| Structure | One 980-line `agent.py` holding UI, model calls, file ops and analysis | Split into focused packages (below) |
| Model integration | `subprocess.run(["ollama", "run", model, prompt])` per message: no conversation history, no tool calling, no streaming, 120–180s hard timeouts | Ollama `/api/chat` over HTTP with streaming, native tool calls and capability detection. OpenAI-compatible adapter added. |
| "Build" mode | Regex-parsed `PATH:`/`CONTENT:` blocks from free text, then bulk file creation | Replaced by a real tool loop (`write_file`, `edit_file`, …) with per-action permissions. Nothing is applied from unparsed prose. |
| Path safety | `str(resolved).startswith(str(workspace))`, so `/ws-evil` passed as inside `/ws`. `..` was removed by string replacement, mangling names like `a..b`. | Component-wise containment after `realpath`, with symlink escape tests |
| Secrets | Analyze/search read every file, `.env` included | Sensitive-file detection, search filtering and redaction everywhere |
| Errors | Bare `except: pass` in several places; generic errors printed raw | Typed error hierarchy. User-facing messages with hints, tracebacks only with `--verbose`. |
| Bugs | `time.sleep(0.5)` at class-body level ran at import. `requirements.txt` listed `flask` but not `rich`. Fake `sleep(0.05)` progress animation. | Removed |
| Tests / docs | None / prototype README | 180 tests and a full README, security and architecture docs |

Preserved: the Ollama focus, the model catalog (`local_agent/llm/catalog.py`), model listing and `ollama pull` offering (`/model`, `--setup`), the banner, the codebase analysis (`/analyze`, now backed by `project/discovery.py`), the Windows UTF-8 console fix, and `python agent.py` as an entry point.

## Layout

```
local_agent/
├── cli/              main.py (argparse, setup, doctor, one-shot), repl.py (slash commands),
│                     ui.py (trace, approvals, diffs, report), session.py (wiring), banner.py
├── agent/            loop.py (tool loop, limits, retries, session persistence),
│                     context.py (system prompt, token estimate, elision + summarization),
│                     checkpoint.py (snapshots, diff, safe undo), report.py (evidence),
│                     events.py (UI interface; RecordingUI for tests)
├── llm/              base.py (types, Provider ABC, HTTP helpers), ollama.py, openai_compat.py,
│                     prompted.py (text tool-call protocol + parser), factory.py (capability → mode)
├── tools/            base.py (Tool, Risk, Assessment, schema validation), filesystem.py, search.py,
│                     shell.py, git.py, agent_tools.py (plan, project_info, remember), registry.py
├── permissions/      engine.py (mode × risk matrix, session rules), commands.py (shell classifier)
├── memory/           store.py (versioned atomic JSON), manager.py (facts, prefs, tasks, sessions)
├── project/          discovery.py (languages, frameworks, commands), ignore.py (gitignore engine)
├── security/         paths.py (workspace boundary, sensitive files), secrets.py (redaction)
├── config.py         settings, precedence, validation
└── errors.py         AgentError hierarchy
```

Dependency direction: `cli → agent → (llm, tools, permissions, memory, project) → security/config/errors`. Tools never import the agent loop, and the permission engine knows nothing about specific tools. It sees only an `Assessment(risk, summary, paths, …)`.

## The tool call pipeline

1. **Parse.** Native tool calls come from the provider. In prompted mode, or when a model writes a tool call as text, `extract_tool_calls` parses fenced/tagged JSON, accepting only known tool names.
2. **Validate.** `validate_args` checks the tool's JSON schema. Unknown parameters are rejected.
3. **Repeat guard.** An identical call made while the state is unchanged gets a warning the second time and stops the task the third time. The "state version" increments after any state-changing tool.
4. **Assess.** `tool.assess()` resolves paths through `Workspace` and returns the risk, the exact action, the affected paths, a diff preview, warnings and the scope key. Invalid paths fail here, before any prompt.
5. **Decide.** `PermissionEngine.evaluate()` returns allow, ask or deny. *Ask* goes to `AgentUI.approve()`, the human.
6. **Run.** `tool.run()` executes. File writes are snapshotted first, then written atomically.
7. **Return.** The real result text, truncated if huge, goes back to the model as a `tool` message. The trace shows the same result.

## Context management

- The system prompt holds the rules, project overview, `AGENTS.md` notes (fenced as untrusted) and relevant memory, all within a bounded budget.
- `read_file` pages large files and returns "unchanged since last read" instead of re-sending identical content that is still in context.
- Command output keeps head and tail, since errors are usually at the end. Diff output is capped.
- When the estimated history exceeds 75% of the context window (minus tool schemas), old large tool outputs are replaced with stubs first. If that is not enough, older turns are summarized by the model. If summarization fails, a deterministic summary is used. Cuts never orphan a tool result from its call.

## Extending

- **New tool:** subclass `Tool` with `name`, `description`, `parameters`, `assess()` (pick an honest `Risk`) and `run()`, then register it in `tools/registry.py`. Add a security test if it touches files or processes.
- **New provider:** implement `Provider.chat()` (streaming text callback, `ToolCall`s with parsed arguments or a `parse_error`) and `model_info()`, then wire it in `llm/factory.py`.
