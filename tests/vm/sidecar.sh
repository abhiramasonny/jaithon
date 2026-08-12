#!/usr/bin/env bash
# The .jaid debug sidecar: a --release image carries no line tables, and the
# spans come back from the file beside it.
#
# Three things have to hold, and only the first is obvious:
#   1. With the sidecar, a release traceback names the same lines as a debug one.
#   2. Without it, the program still runs and still reports the same error --
#      it just loses the line numbers. A release install may ship no .jaid.
#   3. A corrupt sidecar behaves like a missing one. It is read before anything
#      is attached, so a bad file must never take the load down with it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cat > "$work/boom.jai" <<'EOF'
import std.io

fn inner(n: int) -> int {
    if n > 2 { throw ValueError("boom") }
    return n
}

fn outer() -> int { return inner(5) }

print(outer())
EOF

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

# The debug build is the reference: whatever lines it reports are the truth.
"$JAITHON" run "$work/boom.jai" >/dev/null 2>&1
"$JAITHON" run "$work/boom.jai" >/dev/null 2>&1
debug_lines=$("$JAITHON" run "$work/boom.jai" 2>&1 | grep -o 'line [0-9]*' | tr '\n' ' ')
rm -rf "$work/__jaicache__"

# Release: the image must shrink and a sidecar must appear.
"$JAITHON" --release run "$work/boom.jai" >/dev/null 2>&1
jaic="$work/__jaicache__/boom.jaic"
jaid="$work/__jaicache__/boom.jaid"
if [ -f "$jaid" ]; then note ok "release writes a .jaid"; else
    note FAIL "release wrote no .jaid"; exit 1
fi

release_lines=$("$JAITHON" --release run "$work/boom.jai" 2>&1 | grep -o 'line [0-9]*' | tr '\n' ' ')
if [ "$release_lines" = "$debug_lines" ]; then
    note ok "release spans match debug ($debug_lines)"
else
    note FAIL "release spans '$release_lines' != debug '$debug_lines'"
fi

# A debug rebuild must remove the stale sidecar: its records would not line up.
rm -rf "$work/__jaicache__"
"$JAITHON" run "$work/boom.jai" >/dev/null 2>&1
if [ -f "$jaid" ]; then
    note FAIL "a debug build left a .jaid behind"
else
    note ok "debug build writes no .jaid"
fi

# Missing and corrupt sidecars: the program still runs and still throws.
rm -rf "$work/__jaicache__"
"$JAITHON" --release run "$work/boom.jai" >/dev/null 2>&1
cp "$jaid" "$work/good.jaid"

check_survives() {
    local label="$1"
    local out; out=$("$JAITHON" --release run "$work/boom.jai" 2>&1 | tail -1)
    if [ "$out" = "ValueError: boom" ]; then
        note ok "$label"
    else
        note FAIL "$label: got '$out'"
    fi
}

rm -f "$jaid";                                     check_survives "missing sidecar"
cp "$work/good.jaid" "$jaid"; : > "$jaid";         check_survives "empty sidecar"
head -c 12 "$work/good.jaid" > "$jaid";            check_survives "truncated sidecar"
head -c 200 /dev/urandom > "$jaid";                check_survives "garbage sidecar"
cp "$work/good.jaid" "$jaid"
printf 'XXXX' | dd of="$jaid" bs=1 seek=0 conv=notrunc 2>/dev/null
check_survives "wrong magic"

exit $fail
