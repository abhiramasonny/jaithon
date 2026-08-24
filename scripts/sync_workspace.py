"""Rewrite jaithon.workspace.json from the packages that are actually here.

The members list is one shared line in one shared file, and every author of a
new package has to add themselves to it. With several authors working at once
that file is a contention point: one of them rewrites it from what they can see
and silently drops the three packages they did not know about, and the next
`make package-check` fails somewhere unrelated to what anybody changed.

Nothing about the list is a judgement -- it is exactly the set of directories
under packages/ that carry a manifest. So derive it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    members = sorted(
        f"packages/{child.name}"
        for child in (root / "packages").iterdir()
        if child.is_dir() and (child / "jaithon.package.json").is_file()
    )
    path = root / "jaithon.workspace.json"
    data = json.loads(path.read_text()) if path.is_file() else {"schema": 1}
    if data.get("members") == members:
        return 0
    before = set(data.get("members", []))
    data["members"] = members
    path.write_text(json.dumps(data, indent=2) + "\n")
    added = sorted(set(members) - before)
    dropped = sorted(before - set(members))
    if added:
        print(f"workspace: added {', '.join(added)}")
    if dropped:
        print(f"workspace: dropped {', '.join(dropped)} (no manifest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
