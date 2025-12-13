#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


KEEP_C_KEYWORDS = ("TODO", "FIXME", "NOTE", "HACK")


def iter_files(root: Path) -> list[Path]:
    skip_dirs = {
        ".git",
        "__jaicache__",
        "vscode-extension-jai",
        "assets",
    }
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for name in filenames:
            p = dp / name
            if p.name in {"LICENSE"}:
                continue
            if p.suffix.lower() in {".jai", ".c", ".h", ".m"}:
                out.append(p)
    return out


def strip_jai_comments(text: str, keep_test_header: bool) -> str:
    lines = text.splitlines(True)

    header_end = 0
    if keep_test_header:
        i = 0
        while i < len(lines):
            s = lines[i].lstrip()
            if s.startswith("#"):
                header_end = i + 1
                i += 1
                continue
            if s.strip() == "":
                header_end = i + 1
                i += 1
                continue
            break

    def strip_inline(line: str) -> str:
        in_string = False
        escaped = False
        for i, ch in enumerate(line):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == "#":
                    return line[:i].rstrip() + ("\n" if line.endswith("\n") else "")
        return line

    out: list[str] = []
    out.extend(lines[:header_end])

    for line in lines[header_end:]:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(strip_inline(line))

    # Collapse excessive blank lines (keep at most 1).
    collapsed: list[str] = []
    blank = 0
    for line in out:
        if line.strip() == "":
            blank += 1
            if blank <= 1:
                collapsed.append("\n" if line.endswith("\n") else line)
        else:
            blank = 0
            collapsed.append(line)

    return "".join(collapsed)


def strip_c_comments(text: str) -> str:
    # Keep comments that contain key tokens; strip the rest.
    out: list[str] = []
    i = 0
    n = len(text)

    in_str = False
    in_chr = False
    esc = False

    def keep_comment(comment: str) -> bool:
        upper = comment.upper()
        return any(k in upper for k in KEEP_C_KEYWORDS)

    while i < n:
        ch = text[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if in_chr:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "'":
                in_chr = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "'":
            in_chr = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "/":
                j = i + 2
                while j < n and text[j] != "\n":
                    j += 1
                comment = text[i:j]
                if keep_comment(comment):
                    out.append(comment)
                i = j
                continue
            if nxt == "*":
                j = i + 2
                while j + 1 < n and not (text[j] == "*" and text[j + 1] == "/"):
                    j += 1
                j = min(n, j + 2)
                comment = text[i:j]
                if keep_comment(comment):
                    out.append(comment)
                i = j
                continue

        out.append(ch)
        i += 1

    cleaned = "".join(out)
    # Collapse 3+ blank lines to 2 to avoid huge whitespace damage.
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes in place")
    ap.add_argument("--root", default=".", help="Repo root")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = iter_files(root)
    changed = 0

    for p in files:
        rel = p.relative_to(root).as_posix()

        if rel.startswith("docs/"):
            continue

        original = p.read_text(encoding="utf-8", errors="ignore")

        if p.suffix.lower() == ".jai":
            keep_header = rel.startswith("test/") and ("/checks/" in rel)
            updated = strip_jai_comments(original, keep_test_header=keep_header)
        else:
            updated = strip_c_comments(original)

        if updated != original:
            changed += 1
            if args.apply:
                p.write_text(updated, encoding="utf-8")

    print(f"files_changed={changed} apply={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

