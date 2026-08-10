#!/bin/sh
# Reseed after a change to the compiler's own sources.
#
# A change under lib/jaithon changes that file's source hash, so the seed no
# longer matches and the binary cannot start -- and the only thing that could
# rebuild the seed is the binary. Editing the compiler in place is therefore
# impossible without a second compiler.
#
# The second compiler is the committed tree: HEAD's lib is consistent with the
# seed built into ./jaithon. Snapshot it, point the running compiler at the
# snapshot, and have it compile the *working tree's* sources by path. Nothing
# imports the new sources -- they are compiled as data -- so the old compiler
# never has to load a file it cannot understand.
#
# usage: scripts/stage0_reseed.sh
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x ./jaithon ] || { echo "build jaithon first" >&2; exit 2; }

stage0=$(mktemp -d) || exit 2
trap 'rm -rf "$stage0"' EXIT INT TERM

# HEAD's lib is the compiler that still matches the seed in ./jaithon.
git archive HEAD lib | tar -x -C "$stage0"
[ -d "$stage0/lib/jaithon/compile" ] || {
    echo "stage0: HEAD has no lib/jaithon/compile" >&2
    exit 1
}

cat >"$stage0/build_all.jai" <<'JAI'
#: Compile every module of the *working tree* using the stage0 compiler this
#: script is run under. Paths come in on the command line so that nothing here
#: imports the tree being compiled.
from jaithon.compile.mod import build_file
from std.io import Path
from std.os import argv

fn main() -> int {
    let root = argv()[1]
    var built = 0
    var refused = 0
    for entry in Path(root).walk() {
        let source = f"{entry}"
        if not source.ends_with(".jai") { continue }
        if source.contains("__jaicache__") { continue }
        let path = Path(source)
        let name = path.name()
        let out = f"{path.parent()}/__jaicache__/{name[0:name.len() - 4]}.jaic"
        Path(out).parent().mkdir(parents: true, exist_ok: true)
        if build_file(source, out).ok() { built += 1 } else { refused += 1 }
    }
    print(f"stage0: {built} compiled, {refused} refused")
    return if refused > 0 { 1 } else { 0 }
}
JAI

find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null || true

# The compiler runs from the snapshot; the tree it compiles is the working one.
JAITHON_PATH="$stage0/lib" JAITHON_NO_DEFAULT_PATH=1 \
    ./jaithon run "$stage0/build_all.jai" "$ROOT/lib"

python3 scripts/gen_seed.py lib boot/seed.c lib
echo "stage0: seed rebuilt from the working tree; rebuild to pick it up"
