#!/usr/bin/env python3
"""Verify articles/ matches article-manifest.json (prevents duplicate slug mistakes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "article-manifest.json"
ARTICLES_DIR = REPO_ROOT / "articles"


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    allowed = {entry["slug"] for entry in manifest["articles"]}
    on_disk = {path.stem for path in ARTICLES_DIR.glob("*.md")}

    errors: list[str] = []
    missing = allowed - on_disk
    extra = on_disk - allowed

    if missing:
        errors.append(f"manifest にあるが articles/ にない: {sorted(missing)}")
    if extra:
        errors.append(
            "articles/ にあるが manifest にない（重複の原因になります）: "
            f"{sorted(extra)}"
        )

    if errors:
        print("article check FAILED", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        print(file=sys.stderr)
        print("対処: article-manifest.json と Zenn の slug を確認してください。", file=sys.stderr)
        return 1

    print(f"article check OK ({len(allowed)} articles)")
    for entry in manifest["articles"]:
        print(f"  - {entry['slug']}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
