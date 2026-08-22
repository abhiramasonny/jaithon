#!/usr/bin/env python3
"""Fail when a function jaicv exports has no caller in any of its tests.

The recorded OpenCV cases cover what OpenCV was asked to record, so a function
nobody recorded is a function nobody ran. `test_reachable.jai` exists to close
that gap by calling every export once, and it worked -- `find_homography` with
RANSAC, OpenCV's own default, raised `OverflowError` for as long as it had
existed. But nothing kept that file complete, and it had drifted: seventeen
exports had no caller when this script was written.

Text only, so it costs nothing to run on every `make test`.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "packages" / "jaicv" / "src" / "jaicv"
TESTS = ROOT / "packages" / "jaicv" / "tests"

DEFINITION = re.compile(r"^pub fn ([a-z_][a-z_0-9]*)", re.MULTILINE)

#: Opening a window and waiting for a key needs a person in front of it, so
#: these are the one group a headless suite cannot reach.
UNREACHABLE = {"highgui/window.jai"}


def main() -> int:
    exported = (SRC / "mod.jai").read_text()
    called = "\n".join(path.read_text() for path in sorted(TESTS.glob("*.jai")))

    missing = []
    for path in sorted(SRC.rglob("*.jai")):
        if "__jaicache__" in str(path) or path.name == "mod.jai":
            continue
        where = path.relative_to(SRC).as_posix()
        if where in UNREACHABLE:
            continue
        for name in DEFINITION.findall(path.read_text()):
            word = r"\b" + re.escape(name) + r"\b"
            #: Named in mod.jai is what makes it somebody else's to call.
            if not re.search(word, exported):
                continue
            if re.search(word + r"\s*\(", called):
                continue
            missing.append(f"{where}: {name}")

    if missing:
        print("exports with no caller in packages/jaicv/tests:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nCall each one somewhere -- test_reachable.jai is where the cheap "
            "ones go.",
            file=sys.stderr,
        )
        return 1
    print("exports reachable: every jaicv export has a caller in its tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
