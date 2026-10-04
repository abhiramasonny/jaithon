#!/usr/bin/env bash
# Source that is not UTF-8 is refused, not compiled.
#
# The lexer works over the source's bytes and slices the `str` by offsets it
# derives from them, which is only sound for well-formed UTF-8. A `str` read by
# the runtime may hold any bytes, so the lexer validates first. Without that a
# stray continuation byte passed as part of a name and the program ran, and a
# lone lead byte made the lexer slice through a sequence and fail inside the VM.
# Each case below is one of those shapes; the last is valid and must still run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail=0
note() { echo "$1 $2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

refused() {
    local label="$1" file="$work/$2.jai"
    local err
    err=$("$JAITHON" check --no-cache "$file" 2>&1 >/dev/null)
    local rc=$?
    if [ $rc -eq 0 ] || ! grep -q 'invalid UTF-8' <<< "$err"; then
        note FAIL "$label: check exited $rc: $(tail -1 <<< "$err")"
        return
    fi
    local out
    out=$("$JAITHON" run --no-cache "$file" 2>/dev/null)
    if [ -n "$out" ]; then
        note FAIL "$label: run printed '$out'"
        return
    fi
    note ok "$label"
}

printf 'let x\x80 = 41\nprint(x\x80 + 1)\n' > "$work/continuation.jai"
printf 'let a = "\xff"\nprint(a)\n' > "$work/bad_lead.jai"
printf 'let \xed\xa0\x80 = 1\nprint(1)\n' > "$work/surrogate.jai"
printf 'print("\xc0\xaf")\n' > "$work/overlong.jai"
printf 'print(1)\n\xe2\x82' > "$work/truncated.jai"
printf '# lone lead \xc3\nprint(1)\n' > "$work/comment.jai"
printf 'print("\xf4\x90\x80\x80")\n' > "$work/beyond_max.jai"

refused "continuation byte in a name" continuation
refused "invalid lead byte in a string" bad_lead
refused "encoded surrogate" surrogate
refused "overlong encoding" overlong
refused "truncated sequence at end of file" truncated
refused "lone lead byte in a comment" comment
refused "code point past U+10FFFF" beyond_max

printf 'let caf\xc3\xa9 = 1 # \xe2\x82\xac\nprint("\xf0\x9f\x98\x80", caf\xc3\xa9)\n' > "$work/valid.jai"
out=$("$JAITHON" run --no-cache "$work/valid.jai" 2>&1)
if [ "$out" = "$(printf '\xf0\x9f\x98\x80 1')" ]; then
    note ok "valid non-ASCII source still runs"
else
    note FAIL "valid non-ASCII source still runs: got '$out'"
fi

exit $fail
