"""Static risk classification for shell commands.

This is a *conservative heuristic*: it decides whether a command can run
without asking. When it cannot be sure a command is read-only, it reports
``Risk.SHELL`` (uncertain effects), so the permission engine asks in the default
modes. It does not sandbox anything. An approved command runs with the user's
privileges.

Rules of thumb:
* Only a short allowlist of inspection commands counts as read-only.
* Any command substitution, process substitution or ``eval`` disqualifies a
  command from being read-only.
* A reference to a sensitive file (``.env``, keys) or to a path outside the
  workspace escalates the risk.
* A small set of catastrophic patterns is blocked outright in every mode.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from local_agent.security.paths import Workspace, is_within, sensitive_reason
from local_agent.tools.base import Risk

_SEVERITY = [
    Risk.SHELL_READONLY,
    Risk.READ_SENSITIVE,
    Risk.SHELL,
    Risk.DESTRUCTIVE,
    Risk.NETWORK,
    Risk.PRIVILEGED,
    Risk.BLOCKED,
]


def _max_risk(a: Risk, b: Risk) -> Risk:
    return a if _SEVERITY.index(a) >= _SEVERITY.index(b) else b


_BLOCKED = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*(-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\s+(-[a-zA-Z-]+\s+)*(/|/\*|~/?|\$HOME/?|\.\./?\.\.)(\s|$)"), "recursive delete of root, home or a parent tree"),
    (re.compile(r"--no-preserve-root"), "rm --no-preserve-root"),
    (re.compile(r":\s*\(\s*\)\s*\{.*\|.*&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "filesystem formatting"),
    (re.compile(r"\bdd\b.*\bof=/dev/"), "raw write to a device"),
    (re.compile(r">\s*/dev/(sd|hd|nvme|disk)"), "raw write to a disk device"),
    (re.compile(r"\bchmod\s+(-[a-zA-Z]+\s+)*[0-7]*777\s+/(\s|$)"), "making the root filesystem world-writable"),
    (re.compile(r"(?i)\bformat\s+[a-z]:"), "formatting a drive"),
    (re.compile(r"(?i)\b(rd|rmdir|del)\s+/s\s+(/q\s+)?[a-z]:\\?(\s|$)"), "recursive delete of a drive"),
    (re.compile(r"(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b"), "piping a downloaded script into a shell"),
]

_READONLY = {
    "ls", "dir", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "grep", "egrep", "fgrep", "rg", "ag",
    "tree", "which", "where", "whereis", "file", "stat", "du", "df", "uniq", "cut", "diff", "cmp", "basename",
    "dirname", "realpath", "readlink", "date", "uname", "whoami", "hostname", "id", "true", "false", "test",
    "nl", "less", "more", "column", "tr", "comm", "md5sum", "sha256sum", "sha1sum", "type",
}
_GIT_READONLY = {"status", "diff", "log", "show", "rev-parse", "ls-files", "blame", "describe", "shortlog", "grep", "reflog", "cat-file", "ls-tree", "merge-base"}
_GIT_NETWORK = {"push", "pull", "fetch", "clone", "ls-remote", "submodule", "remote-update"}
_PRIVILEGED = {"sudo", "su", "doas", "pkexec", "runas", "mount", "umount", "systemctl", "service", "launchctl",
               "shutdown", "reboot", "halt", "poweroff", "passwd", "useradd", "userdel", "usermod", "visudo",
               "iptables", "ufw", "apt", "apt-get", "yum", "dnf", "pacman", "zypper", "apk", "snap", "crontab",
               "chroot", "insmod", "modprobe", "sysctl", "kill", "killall", "pkill"}
_NETWORK = {"curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat", "telnet", "ftp", "ping",
            "npx", "pnpx", "bunx", "gh", "aws", "az", "gcloud", "gsutil", "heroku", "vercel", "netlify", "fly",
            "kubectl", "helm", "terraform", "ansible", "brew", "choco", "winget", "scoop", "http", "httpie"}
_PKG_NETWORK_SUBCMDS = {
    "pip": {"install", "download", "wheel"},
    "pip3": {"install", "download", "wheel"},
    "uv": {"add", "sync", "pip", "lock", "tool"},
    "poetry": {"add", "install", "update", "lock", "publish"},
    "pipenv": {"install", "update", "lock"},
    "npm": {"install", "i", "ci", "add", "update", "upgrade", "publish", "login", "exec", "x", "create", "init"},
    "pnpm": {"install", "i", "add", "update", "up", "publish", "dlx", "create"},
    "yarn": {"install", "add", "upgrade", "publish", "dlx", "create"},
    "bun": {"install", "i", "add", "update", "publish", "x", "create"},
    "cargo": {"install", "publish", "add", "update", "fetch", "search", "login"},
    "go": {"get", "install", "mod"},
    "gem": {"install", "push", "update"},
    "bundle": {"install", "update"},
    "composer": {"install", "require", "update"},
    "docker": {"pull", "push", "login", "build", "run", "compose"},
    "podman": {"pull", "push", "login", "build", "run"},
    "dotnet": {"add", "restore", "nuget"},
    "mvn": set(), "gradle": set(),
}
_DESTRUCTIVE = {"rm", "rmdir", "shred", "unlink", "truncate", "del", "erase", "rd"}
_WRAPPERS = {"env", "time", "nice", "nohup", "command", "builtin", "exec", "stdbuf", "ionice"}
_SEGMENT_OPS = {";", "&&", "||", "|", "&", "\n", "|&"}
_REDIRECT_OPS = {">", ">>", "<", "<<", "<<<", ">&", "&>", ">|", "<>", "&>>"}
_SHELL_META = re.compile(r"[|&;<>()$`*?\[\]{}~\n]")


@dataclass
class CommandAssessment:
    risk: Risk
    warnings: List[str] = field(default_factory=list)
    scope_key: Optional[str] = None
    block_reason: str = ""
    needs_shell: bool = False
    programs: List[str] = field(default_factory=list)


def _unquoted(command: str) -> str:
    text = re.sub(r"'[^']*'", "", command)
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: "".join(ch for ch in m.group(0) if ch in "$`"), text)


def needs_shell(command: str) -> bool:
    """Whether the command uses shell syntax (pipes, redirects, globs, expansion).

    Characters inside single quotes are literal. Inside double quotes only ``$``
    and backticks still expand, so only those count there.
    """
    return bool(_SHELL_META.search(_unquoted(command)))


def _tokenize(command: str) -> List[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _split_segments(tokens: Sequence[str]) -> Tuple[List[List[str]], List[str]]:
    segments: List[List[str]] = [[]]
    redirect_targets: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _SEGMENT_OPS:
            segments.append([])
        elif tok in _REDIRECT_OPS or (tok.endswith((">", ">>")) and tok.rstrip(">").isdigit()):
            if tok.startswith(">") or tok.endswith(">") or tok.endswith(">>") or tok in ("&>", "&>>", ">|", "<>"):
                if i + 1 < len(tokens):
                    redirect_targets.append(tokens[i + 1])
            i += 1  # skip the redirect target / source
        elif tok in ("(", ")"):
            segments.append([])
        else:
            segments[-1].append(tok)
        i += 1
    return [s for s in segments if s], redirect_targets


def _program(segment: List[str]) -> Tuple[str, List[str]]:
    rest = list(segment)
    while rest and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", rest[0]):
        rest.pop(0)
    while rest and os.path.basename(rest[0]).lower() in _WRAPPERS:
        name = os.path.basename(rest[0]).lower()
        rest.pop(0)
        while rest and (rest[0].startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", rest[0]) or (name == "nice" and rest[0].lstrip("-").isdigit())):
            rest.pop(0)
    if rest and os.path.basename(rest[0]).lower() == "timeout":
        rest.pop(0)
        while rest and (rest[0].startswith("-") or re.match(r"^\d+[smhd]?$", rest[0])):
            rest.pop(0)
    if not rest:
        return "", []
    prog = os.path.basename(rest[0]).lower()
    if prog.endswith(".exe"):
        prog = prog[:-4]
    return prog, rest[1:]


def _first_positional(args: Sequence[str]) -> str:
    for a in args:
        if not a.startswith("-"):
            return a
    return ""


def _classify_segment(prog: str, args: List[str]) -> Tuple[Risk, List[str]]:
    warnings: List[str] = []
    if not prog:
        return Risk.SHELL_READONLY, warnings
    sub = _first_positional(args)
    if prog in _PRIVILEGED:
        return Risk.PRIVILEGED, [f"'{prog}' is a privileged or system-level command"]
    if prog in _NETWORK:
        return Risk.NETWORK, [f"'{prog}' accesses the network or external services"]
    if prog == "git":
        if sub in _GIT_NETWORK:
            risk = Risk.NETWORK
            if sub == "push" and any(a in ("-f", "--force", "--force-with-lease", "--mirror", "--delete") or a.startswith("+") for a in args):
                warnings.append("force/delete push rewrites remote history")
            return risk, warnings + [f"'git {sub}' contacts a remote"]
        if sub in ("reset", "clean", "checkout", "restore", "rm", "stash", "branch", "tag", "rebase", "filter-branch", "gc", "prune", "worktree", "update-ref"):
            hard = ("--hard" in args or sub in ("clean", "filter-branch", "update-ref", "prune")
                    or (sub == "checkout" and ("--" in args or "." in args or "-f" in args))
                    or (sub == "restore" and "--staged" not in args)
                    or (sub == "rm") or (sub == "stash" and any(a in ("drop", "clear") for a in args))
                    or (sub == "branch" and any(a in ("-D", "-d", "--delete", "-M", "-m") for a in args))
                    or (sub == "tag" and "-d" in args))
            if hard:
                return Risk.DESTRUCTIVE, [f"'git {sub}' can discard work"]
            listing_flags = {"-a", "--all", "-r", "--remotes", "-v", "-vv", "--list", "-l", "--show-current"}
            if sub in ("branch", "tag") and all(a == sub or a in listing_flags for a in args):
                return Risk.SHELL_READONLY, warnings
            return Risk.SHELL, warnings
        if sub in _GIT_READONLY:
            return Risk.SHELL_READONLY, warnings
        if sub == "remote" and all(a in ("remote", "-v", "--verbose", "show") for a in args):
            return Risk.SHELL_READONLY, warnings
        if sub == "config" and any(a in ("--get", "--list", "-l", "--get-all") for a in args):
            return Risk.SHELL_READONLY, warnings
        return Risk.SHELL, warnings
    if prog in _PKG_NETWORK_SUBCMDS:
        subs = _PKG_NETWORK_SUBCMDS[prog]
        if prog in ("mvn", "gradle"):
            return Risk.SHELL, ["build tool may download dependencies"]
        if sub in subs:
            return Risk.NETWORK, [f"'{prog} {sub}' downloads or publishes packages"]
        return Risk.SHELL, warnings
    if prog in ("python", "python3", "py") and len(args) >= 3 and args[0] == "-m" and args[1] == "pip" and args[2] in ("install", "download"):
        return Risk.NETWORK, ["'pip install' downloads packages"]
    if prog in _DESTRUCTIVE:
        return Risk.DESTRUCTIVE, [f"'{prog}' deletes data"]
    if prog == "find":
        if any(a in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls", "-fprintf") for a in args):
            return Risk.DESTRUCTIVE if "-delete" in args else Risk.SHELL, ["find with -delete/-exec"]
        return Risk.SHELL_READONLY, warnings
    if prog in ("sort",) and any(a == "-o" or a.startswith("--output") for a in args):
        return Risk.SHELL, warnings
    if prog in ("grep", "egrep", "fgrep") and any(a in ("-r", "-R", "--recursive", "--dereference-recursive") or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a)) for a in args):
        return Risk.SHELL, ["recursive grep can read secrets in ignored or hidden files (use search_code instead)"]
    if prog in ("rg", "ag") and any(a in ("--hidden", "--no-ignore", "-u", "-uu", "-uuu", "--unrestricted", "-.") for a in args):
        return Risk.SHELL, ["searching hidden/ignored files can expose secrets"]
    if prog in ("echo", "printf", "true", "false", "test"):
        return Risk.SHELL_READONLY, warnings
    if prog in _READONLY:
        return Risk.SHELL_READONLY, warnings
    if args and all(a in ("--version", "-V", "version", "--help", "-h") for a in args):
        return Risk.SHELL_READONLY, warnings
    return Risk.SHELL, warnings


def _path_like(token: str) -> bool:
    return (
        token.startswith(("/", "~", "./", "../", ".\\", "..\\"))
        or "/" in token
        or "\\" in token
        or token.startswith(".")
        or bool(re.match(r"^[A-Za-z]:[\\/]", token))
    )


def _scope_key(prog: str, args: List[str]) -> str:
    if prog in ("python", "python3", "py") and len(args) >= 2 and args[0] == "-m":
        return f"cmd:{prog} -m {args[1]}"
    sub = _first_positional(args)
    if prog in ("npm", "pnpm", "yarn", "bun", "cargo", "go", "make", "dotnet", "mvn", "gradle", "./gradlew", "poetry", "uv") and sub:
        if sub == "run" and len(args) > 1:
            return f"cmd:{prog} run {_first_positional(args[1:])}"
        return f"cmd:{prog} {sub}"
    return f"cmd:{prog}"


def classify_command(command: str, workspace: Workspace) -> CommandAssessment:
    if not command or not command.strip():
        return CommandAssessment(Risk.BLOCKED, block_reason="Empty command")
    for pattern, reason in _BLOCKED:
        if pattern.search(command):
            return CommandAssessment(Risk.BLOCKED, block_reason=f"Blocked by safety policy: {reason}")

    result = CommandAssessment(Risk.SHELL_READONLY, needs_shell=needs_shell(command))
    if re.search(r"`|\$\(|<\(|>\(", command):
        result.risk = Risk.SHELL
        result.warnings.append("uses command or process substitution; effects cannot be predicted")
    try:
        tokens = _tokenize(command)
    except ValueError as exc:
        result.risk = _max_risk(result.risk, Risk.SHELL)
        result.warnings.append(f"could not parse command ({exc})")
        return result

    if re.search(r"[*?\[$]", _unquoted(command)) and result.risk is Risk.SHELL_READONLY:
        result.risk = Risk.SHELL
        result.warnings.append("uses globs or variable expansion; the files it touches cannot be checked in advance")
    segments, redirect_targets = _split_segments(tokens)
    for target in redirect_targets:
        if target in ("/dev/null", "&1", "&2", "1", "2", "NUL", "nul"):
            continue
        result.risk = _max_risk(result.risk, Risk.SHELL)
        result.warnings.append(f"writes output to '{target}'")

    for seg in segments:
        prog, args = _program(seg)
        if prog:
            result.programs.append(prog)
        if prog in ("eval", "source", ".", "sh", "bash", "zsh", "dash", "fish", "pwsh", "powershell", "cmd", "xargs"):
            result.risk = _max_risk(result.risk, Risk.SHELL)
            result.warnings.append(f"'{prog}' runs arbitrary nested commands")
            continue
        risk, warnings = _classify_segment(prog, args)
        result.risk = _max_risk(result.risk, risk)
        result.warnings.extend(warnings)

    expect_program = True
    for token in tokens:
        is_program = False
        if token in _SEGMENT_OPS or token in ("(", ")"):
            expect_program = True
            continue
        if expect_program and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", token):
            is_program, expect_program = True, False
        if token in _REDIRECT_OPS or token.startswith("-") and "=" not in token:
            continue
        candidate = token.split("=", 1)[1] if token.startswith("-") else token
        if not candidate or "://" in candidate:
            continue
        name = candidate[2:] if candidate.startswith(("./", ".\\")) else candidate
        reason = sensitive_reason(name)
        if reason:
            result.risk = _max_risk(result.risk, Risk.READ_SENSITIVE)
            result.warnings.append(f"references sensitive file '{candidate}' ({reason})")
        if not _path_like(candidate) or is_program:
            continue  # the executable itself may live anywhere (e.g. /usr/bin/python)
        if candidate.startswith("~") or candidate.startswith("$HOME"):
            result.risk = _max_risk(result.risk, Risk.SHELL)
            result.warnings.append(f"references '{candidate}' outside the workspace")
            continue
        try:
            target = Path(candidate) if os.path.isabs(candidate) else workspace.root / candidate
            resolved = Path(os.path.realpath(str(target)))
        except (OSError, ValueError):
            continue
        if not is_within(resolved, workspace.root) and not str(resolved).startswith(("/dev/null",)):
            result.risk = _max_risk(result.risk, Risk.SHELL)
            result.warnings.append(f"references '{candidate}' outside the workspace")

    if result.risk is Risk.SHELL and len(segments) == 1 and not result.needs_shell:
        prog, args = _program(segments[0])
        if prog:
            result.scope_key = _scope_key(prog, args)
    # De-duplicate warnings, keep order.
    result.warnings = list(dict.fromkeys(result.warnings))
    return result
