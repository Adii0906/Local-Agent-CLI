# Security model and limitations

Local Agent runs LLM-chosen actions on your machine. This document states plainly what the safeguards do and what they do not do.

## Threats considered

1. **A mistaken model**: wrong edits, destructive commands, loops.
2. **Prompt injection**: instructions hidden in repository files, command output or memory that try to make the agent act against you.
3. **Secret exposure**: credentials sent to the model, printed, or persisted.
4. **Workspace escape**: reading or writing outside the directory you authorized.

## Enforced protections

| Protection | Mechanism |
|---|---|
| Workspace boundary | Every file-tool path is resolved with `realpath`, symlinks included, and checked for containment by path components, not string prefixes. `..`, absolute paths, `~`, NUL bytes and symlinks pointing outside are rejected before any prompt. Directory walking does not follow symlinked directories. |
| Permission engine | `(mode, risk)` → allow / ask / deny ([`local_agent/permissions/engine.py`](../local_agent/permissions/engine.py)). Only the human approval callback can turn *ask* into *approve*. The model has no approval tool, and unknown arguments such as `approved: true` fail schema validation. |
| Narrow session rules | "Always allow" covers one file path or one command prefix, lasts for the session only, can only turn *ask* into *allow*, and is never offered for destructive, network, privileged or sensitive actions. |
| Hard blocks | `rm -rf /`/`~`, `--no-preserve-root`, `mkfs`, raw device writes, fork bombs, `curl … \| sh`, writes inside `.git/`, and deleting the workspace root are refused in every mode. |
| Sensitive files | `.env*` (except `.example`/`.sample`/`.template`), keys (`*.pem`, `id_rsa`, …), credential files and directories (`.ssh`, `.aws`, …) need explicit approval to read or write, in every mode including `full`. `search_code` hides their matches. Shell commands that name them are escalated to *ask*. |
| Secret redaction | Command output, session transcripts, memory and task history are redacted with pattern-based detection (API keys, tokens, private keys, URL credentials, `PASSWORD=…`). The `remember` tool refuses secret-looking content. |
| Scrubbed child environment | Commands run without environment variables whose names look like secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSW*`, …) unless you list them in `shell_env_passthrough`. |
| Non-interactive safety | With no TTY, every action that needs approval is denied. |
| Untrusted content | File contents, command output, `AGENTS.md` and memory are labelled as data in the prompt. Project-level config files are never read, so a repository cannot change modes or rules. |
| Bounded execution | Iteration, wall-clock and token limits. Repeated identical calls stop the loop. Commands have timeouts, `stdin` is closed, the process group is killed on timeout or Ctrl+C, and output is capped. |
| Recoverability | Files are snapshotted before the agent's first change. `/undo` refuses to overwrite files modified by someone else unless you pass `--force`. |

## Limitations

- **This is not a sandbox.** An approved command runs with **your user's privileges**, with full filesystem and network access. The shell risk classifier is a conservative *heuristic* that decides whether to ask. It cannot prove what a command will do. A test suite, build script or `npm install` hook can do anything your user can do. Read commands before approving them, especially in `full` mode.
- **Interpreters and build tools are opaque.** `python script.py`, `make`, `npm test` and similar are classified as "uncertain" and ask in `ask`/`auto` mode, because their effects depend on code the agent may have just written.
- **Read-only commands are allowlisted, not verified.** `cat`, `grep`, `git log` and the like run without asking in every mode. They cannot modify files, but they can read any file your user can read outside the workspace if given an absolute path. Absolute paths outside the workspace, `~`, unquoted globs and `$` expansions all escalate to *ask*, because their targets cannot be checked in advance.
- **Changes made by commands are not undoable.** `/undo` covers only the agent's file tools. The task report lists files that commands changed (via `git status`) so you can see them.
- **Secret detection is pattern-based.** A secret without a recognizable shape can still be read (after approval) and sent to the model. With a local model, it stays on your machine. With a remote OpenAI-compatible endpoint, it leaves it.
- **Prompt injection is mitigated, not solved.** Policy is enforced outside the model, so injected text cannot grant permissions. It can still persuade the model to *propose* harmful actions, which you will then see in an approval prompt (or which run automatically in `auto`/`full` mode within those modes' limits).
- **TOCTOU.** Paths are validated before execution. A concurrent process that swaps a symlink between validation and use is out of scope.
- **Windows and macOS.** The process-group kill, path handling and console code have Windows/macOS branches, but this release was only tested on Linux.

## Recommendations

- Use `ask` (default) or `plan` mode on unfamiliar code. Use `auto` for routine edits under version control. Use `full` only in throwaway environments.
- For untrusted repositories, run the agent inside a container or VM with no credentials mounted. That is the only way to get real isolation.
- Commit, or at least stage, before large tasks, so `git diff` shows everything, including changes made by commands.
