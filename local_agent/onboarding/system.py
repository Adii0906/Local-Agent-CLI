"""Host facts used for recommendations (memory) and environment checks."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from typing import Optional


def total_memory_bytes() -> Optional[int]:
    """Physical memory in bytes, or None if it cannot be determined."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return int(out.stdout.strip())
        elif sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
    except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
        return None
    return None


def ollama_binary() -> Optional[str]:
    return shutil.which("ollama")


def install_instructions() -> str:
    if sys.platform == "darwin":
        return "Download the app from https://ollama.com/download  (or: brew install ollama)"
    if sys.platform == "win32":
        return "Download and run the installer from https://ollama.com/download"
    return "curl -fsSL https://ollama.com/install.sh | sh   (or see https://ollama.com/download)"


def start_instructions() -> str:
    if sys.platform in ("darwin", "win32"):
        return "Open the Ollama app, or run `ollama serve` in another terminal."
    return "Run `ollama serve` in another terminal (or `systemctl start ollama` if installed as a service)."


def start_ollama_background(binary: str) -> bool:
    """Start `ollama serve` detached from this process. Returns True if it launched."""
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([binary, "serve"], env=dict(os.environ), **kwargs)  # noqa: S603 - fixed argv, user-approved
        return True
    except OSError:
        return False
