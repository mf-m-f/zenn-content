#!/usr/bin/env python3
"""Fail if notebook outputs contain local absolute paths (privacy leak)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Home-directory style paths that should not appear in committed outputs.
FORBIDDEN_PATTERNS = [
    re.compile(r"/Users/[^\s\"']+"),
    re.compile(r"/home/[^\s\"']+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+"),
    re.compile(r"Library/CloudStorage/"),
]


def iter_output_texts(notebook: dict) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    for cell_idx, cell in enumerate(notebook.get("cells", [])):
        for out_idx, output in enumerate(cell.get("outputs", [])):
            parts: list[str] = []
            if "text" in output:
                text = output["text"]
                parts.append(text if isinstance(text, str) else "".join(text))
            if "traceback" in output:
                parts.append("\n".join(output["traceback"]))
            if "data" in output:
                for value in output["data"].values():
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, list):
                        parts.append("".join(value))
            content = "\n".join(parts)
            if content:
                hits.append((cell_idx, f"output[{out_idx}]", content))
    return hits


def main() -> int:
    errors: list[str] = []
    checked = 0

    for path in sorted(NOTEBOOKS_DIR.rglob("*.ipynb")):
        checked += 1
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell_idx, label, text in iter_output_texts(notebook):
            for pattern in FORBIDDEN_PATTERNS:
                match = pattern.search(text)
                if match:
                    rel = path.relative_to(REPO_ROOT)
                    snippet = match.group(0)[:80]
                    errors.append(
                        f"{rel} cell[{cell_idx}] {label}: local path leaked ({snippet}...)"
                    )
                    break

    if errors:
        print("notebook check FAILED", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "対処: warnings.filterwarnings('ignore') を追加し、"
            "該当セルの出力をクリアしてから再実行・commit してください。",
            file=sys.stderr,
        )
        return 1

    print(f"notebook check OK ({checked} notebooks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
