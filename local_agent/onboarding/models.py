"""Model discovery helpers: classification, sizes and recommendations.

Everything shown as "installed" comes from the Ollama server. The catalog is
only used to suggest models to download.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

GB = 1024 ** 3

_CODING = re.compile(r"coder|codestral|devstral|starcoder|codegemma|codellama|granite-code|codeqwen|deepseek-code|code-", re.I)
_EMBEDDING = re.compile(r"embed|bge-|nomic-embed|all-minilm|mxbai", re.I)

# Downloadable suggestions, smallest first. All support native tool calling in Ollama.
SUGGESTIONS = [
    {"name": "qwen2.5-coder:1.5b", "size": "1.0 GB", "min_ram_gb": 0, "note": "very small; basic edits only"},
    {"name": "qwen2.5-coder:3b", "size": "1.9 GB", "min_ram_gb": 8, "note": "coding, light hardware"},
    {"name": "qwen2.5-coder:7b", "size": "4.7 GB", "min_ram_gb": 16, "note": "coding, recommended"},
    {"name": "qwen2.5-coder:14b", "size": "9.0 GB", "min_ram_gb": 32, "note": "coding, stronger, needs more RAM"},
]


@dataclass
class ModelEntry:
    name: str
    size: int = 0
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    tools: Optional[bool] = None  # None = unknown
    embedding: bool = False

    @property
    def coding(self) -> bool:
        return bool(_CODING.search(self.name))

    @property
    def is_chat(self) -> bool:
        return not self.embedding and not _EMBEDDING.search(self.name)

    def tags(self) -> List[str]:
        tags = ["coding" if self.coding else "general"]
        if self.tools is True:
            tags.append("tools")
        elif self.tools is False:
            tags.append("no tools")
        return tags


def human_size(n: int) -> str:
    if n < 0:
        return "?"
    if n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit in ("GB", "TB") else f"{n:.0f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return "?"


def recommend_installed(models: List[ModelEntry], memory_bytes: Optional[int]) -> Optional[ModelEntry]:
    """Pick the best installed chat model for coding on this machine."""
    budget = memory_bytes * 0.6 if memory_bytes else None
    best, best_score = None, float("-inf")
    for m in models:
        if not m.is_chat:
            continue
        score = 0.0
        score += 4 if m.tools is True else (0 if m.tools is None else -6)
        score += 3 if m.coding else 0
        if budget and m.size:
            score += 2 if m.size <= budget else -8
        score += min(m.size / GB, 10) * 0.3  # larger is better while it fits
        if score > best_score:
            best, best_score = m, score
    return best


def recommend_download(memory_bytes: Optional[int]) -> dict:
    """The largest suggested coding model that comfortably fits in memory."""
    ram_gb = (memory_bytes or 8 * GB) / GB
    chosen = SUGGESTIONS[0]
    for s in SUGGESTIONS:
        if ram_gb >= s["min_ram_gb"]:
            chosen = s
    if chosen["name"].endswith(":14b"):  # 7b is the better default speed/quality trade-off
        chosen = SUGGESTIONS[2]
    return chosen
