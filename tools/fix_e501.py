#!/usr/bin/env python3
"""Wrap long lines reported by ruff E501 (first-party paths only)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LEN = 88
VENDOR_PREFIX = "videotuna/models/wan/wan/"


def wrap_line(line: str) -> list[str]:
    body = line.rstrip("\n")
    if len(body) <= MAX_LEN:
        return [body]

    indent_len = len(body) - len(body.lstrip())
    indent = body[:indent_len]
    content = body[indent_len:]

    if content.startswith('"""') or content.startswith("'''"):
        return [body]

    # Preserve closing quotes on short trailing docstring lines.
    wrap_width = MAX_LEN - indent_len
    if wrap_width < 20:
        return [body]

    if content.startswith("#"):
        prefix = "# "
        text = content[2:] if content.startswith("# ") else content[1:]
        chunks = textwrap.wrap(
            text,
            width=wrap_width - len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        )
        if not chunks:
            return [body]
        out = [indent + prefix + chunks[0]]
        for chunk in chunks[1:]:
            out.append(indent + prefix + chunk)
        return out

    chunks = textwrap.wrap(
        content,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(chunks) <= 1:
        return [body]

    out = [indent + chunks[0]]
    cont = indent + (" " * 4)
    for chunk in chunks[1:]:
        out.append(cont + chunk)
    return out


def main() -> int:
    proc = subprocess.run(
        ["poetry", "run", "ruff", "check", "--output-format=json", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    violations = [
        v
        for v in json.loads(proc.stdout)
        if v["code"] == "E501"
        and VENDOR_PREFIX not in v["filename"].split("VideoTuna/")[-1]
    ]
    by_file: dict[str, list[int]] = defaultdict(list)
    for v in violations:
        rel = v["filename"].split("VideoTuna/")[-1]
        by_file[rel].append(v["location"]["row"])

    for rel, rows in by_file.items():
        path = ROOT / rel
        lines = path.read_text().splitlines()
        for row in sorted(rows, reverse=True):
            idx = row - 1
            wrapped = wrap_line(lines[idx] + "\n")
            if len(wrapped) == 1 and len(wrapped[0]) > MAX_LEN:
                continue
            lines[idx : idx + 1] = wrapped
        path.write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
