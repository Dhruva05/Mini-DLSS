from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_markdown_table(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n")


def format_metrics_md(metrics: Dict[str, float]) -> str:
    items = [f"- {k}: {v:.4f}" for k, v in sorted(metrics.items())]
    return "\n".join(items) + "\n"
