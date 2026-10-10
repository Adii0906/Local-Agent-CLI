"""Interactive prompts: arrow-key menus with a plain numbered fallback.

* POSIX terminals: cbreak mode via ``termios`` (Ctrl+C still raises KeyboardInterrupt).
* Windows consoles: ``msvcrt.getwch`` (arrow keys arrive as a 0xE0/0x00 prefix).
* Anything else (pipes, CI, ``TERM=dumb``, ``LOCAL_AGENT_PLAIN_PROMPTS=1``):
  a numbered list read with ``input()``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, TextIO

from local_agent.cli.ui import Style


@dataclass
class Option:
    label: str
    value: object
    hint: str = ""
    badge: str = ""


KEY_UP, KEY_DOWN, KEY_ENTER, KEY_CANCEL = "up", "down", "enter", "cancel"


def decode_key(seq: str) -> str:
    """Map a raw key sequence (POSIX escape codes or Windows scan codes) to an action."""
    if seq in ("\x1b[A", "\x1bOA", "\xe0H", "\x00H", "k", "K"):
        return KEY_UP
    if seq in ("\x1b[B", "\x1bOB", "\xe0P", "\x00P", "j", "J", "\t"):
        return KEY_DOWN
    if seq in ("\r", "\n", "\r\n", " "):
        return KEY_ENTER
    if seq in ("\x03",):
        return "interrupt"
    if seq in ("\x1b", "q", "Q", "\x04"):
        return KEY_CANCEL
    if len(seq) == 1 and seq.isdigit() and seq != "0":
        return f"digit:{seq}"
    return ""


def _read_key_posix(fd: int) -> str:
    import select

    ch = os.read(fd, 1).decode("utf-8", "replace")
    if ch == "\x1b":
        seq = ch
        while select.select([fd], [], [], 0.03)[0] and len(seq) < 4:
            seq += os.read(fd, 1).decode("utf-8", "replace")
        return seq
    return ch


def _read_key_windows() -> str:  # pragma: no cover - Windows only
    import msvcrt

    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        return ch + msvcrt.getwch()
    return ch


class Prompter:
    def __init__(
        self,
        style: Style,
        out: Optional[TextIO] = None,
        input_fn: Optional[Callable[[str], str]] = None,
        interactive: Optional[bool] = None,
        symbols: Optional[dict] = None,
    ) -> None:
        self.s = style
        self.out = out if out is not None else sys.stdout
        self.input_fn = input_fn or (lambda prompt: input(prompt))
        if interactive is None:
            interactive = (
                sys.stdin.isatty()
                and hasattr(self.out, "isatty")
                and self.out.isatty()
                and os.environ.get("TERM") != "dumb"
                and not os.environ.get("LOCAL_AGENT_PLAIN_PROMPTS")
            )
        self.arrow_keys = interactive
        self.sym = symbols or {"pointer": "›", "on": "●", "off": "○"}

    def _w(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    # -- select ------------------------------------------------------------
    def select(self, title: str, options: Sequence[Option], default: int = 0, footer: str = "") -> object:
        """Return the chosen option's value. Raises KeyboardInterrupt on Ctrl+C / cancel."""
        if not options:
            raise ValueError("no options")
        default = max(0, min(default, len(options) - 1))
        if self.arrow_keys:
            try:
                return self._select_arrows(title, options, default, footer)
            except (ImportError, OSError, ValueError):
                self.arrow_keys = False  # e.g. no termios on this platform/terminal
        return self._select_plain(title, options, default, footer)

    def _render(self, title: str, options: Sequence[Option], idx: int, footer: str) -> List[str]:
        lines = [self.s.bold(title), ""]
        width = max(len(o.label) for o in options)
        for i, o in enumerate(options):
            mark = self.s.cyan(self.sym["on"]) if i == idx else self.s.dim(self.sym["off"])
            label = o.label.ljust(width)
            label = self.s.bold(label) if i == idx else label
            extra = "  " + self.s.dim(o.hint) if o.hint else ""
            badge = "  " + self.s.green(o.badge) if o.badge else ""
            lines.append(f"  {mark} {label}{extra}{badge}")
        lines.append("")
        if footer:
            lines.append(footer)
        lines.append(self.s.dim("Use ↑/↓ to move, Enter to select, Ctrl+C to cancel."))
        return lines

    def _select_arrows(self, title: str, options: Sequence[Option], default: int, footer: str) -> object:
        idx = default
        drawn = 0

        def draw() -> None:
            nonlocal drawn
            if drawn:
                self._w(f"\033[{drawn}F\033[J")
            lines = self._render(title, options, idx, footer)
            self._w("\n".join(lines) + "\n")
            drawn = len(lines)

        if sys.platform == "win32":  # pragma: no cover - Windows only
            reader: Callable[[], str] = _read_key_windows
            restore: Callable[[], None] = lambda: None  # noqa: E731
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            reader = lambda: _read_key_posix(fd)  # noqa: E731
            restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)  # noqa: E731
        self._w("\033[?25l")
        try:
            draw()
            while True:
                action = decode_key(reader())
                if action == KEY_UP:
                    idx = (idx - 1) % len(options)
                elif action == KEY_DOWN:
                    idx = (idx + 1) % len(options)
                elif action.startswith("digit:"):
                    n = int(action.split(":")[1]) - 1
                    if n < len(options):
                        idx = n
                elif action == KEY_ENTER:
                    break
                elif action in (KEY_CANCEL, "interrupt"):
                    raise KeyboardInterrupt
                else:
                    continue
                draw()
        finally:
            restore()
            self._w("\033[?25h")
        # Collapse the menu to a one-line summary of the choice.
        self._w(f"\033[{drawn}F\033[J")
        self._w(f"{self.s.bold(title)} {self.s.cyan(options[idx].label)}\n")
        return options[idx].value

    def _select_plain(self, title: str, options: Sequence[Option], default: int, footer: str) -> object:
        self._w(self.s.bold(title) + "\n")
        for i, o in enumerate(options, 1):
            extra = f"  {o.hint}" if o.hint else ""
            badge = f"  [{o.badge}]" if o.badge else ""
            self._w(f"  {i}. {o.label}{extra}{badge}\n")
        if footer:
            self._w(footer + "\n")
        while True:
            try:
                answer = self.input_fn(f"Choose 1-{len(options)} [{default + 1}]: ").strip()
            except EOFError:
                raise KeyboardInterrupt from None
            if not answer:
                return options[default].value
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1].value
            if answer.lower() in ("q", "quit", "cancel"):
                raise KeyboardInterrupt
            self._w(self.s.dim(f"  Please enter a number from 1 to {len(options)}.\n"))

    # -- simple prompts --------------------------------------------------------
    def confirm(self, question: str, default: bool = False) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            try:
                answer = self.input_fn(f"{question} {suffix} ").strip().lower()
            except EOFError:
                return False
            if not answer:
                return default
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False

    def text(self, question: str, default: str = "") -> str:
        hint = f" [{default}]" if default else ""
        try:
            answer = self.input_fn(f"{question}{hint}: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        return answer or default
