#!/usr/bin/env bash
# A corrupt or truncated .jaic must be an ordinary cache miss, never a crash
# and never a load failure.
#
# The 8-byte header probe used to be what guaranteed this: it opened the file,
# read the magic and version, and declined before the deserialiser ever saw the
# bytes. Merging that probe into the single full read moved the guarantee into
# the reader, so this pins the behaviour on both sides of that change.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$ROOT/tests/golden/cache_short_file.jai" "$work/prog.jai"

# Warm the cache, then confirm an image was actually written -- without one
# every case below would trivially "pass" by never reading a cache at all.
"$JAITHON" run "$work/prog.jai" >/dev/null 2>&1 || { echo "FAIL: warm run errored"; exit 1; }
cache=$(find "$work/__jaicache__" -name '*.jaic' 2>/dev/null | head -1)
[ -n "$cache" ] || { echo "FAIL: no cache image was written"; exit 1; }

fail=0
check() {
    local label="$1" expected="recompiled and ran"
    local out rc
    out=$("$JAITHON" run "$work/prog.jai" 2>&1); rc=$?
    if [ $rc -ne 0 ] || [ "$out" != "$expected" ]; then
        echo "FAIL $label: rc=$rc out=$out"
        fail=1
    else
        echo "ok $label"
    fi
}

: > "$cache";                             check "empty file"
printf 'JAI' > "$cache";                  check "3 bytes, short of the header"
printf 'XXXX\x0b\x00\x15\x00' > "$cache"; check "wrong magic"
printf 'JAIC\xff\xff\x15\x00' > "$cache"; check "unknown version"
head -c 64 /dev/urandom > "$cache";       check "random bytes"
head -c 7 /dev/urandom > "$cache";        check "7 bytes, one short of the header"

exit $fail
