#!/usr/bin/env bash
# The line table must answer identically before and after a change to how it is
# encoded.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
export JAI_DISASM_SPANS=1
JAITHON="${JAITHON:-$ROOT/jaithon}"
GOLDEN="$ROOT/tests/golden/linetable.golden"

corpus() {
    { find "$ROOT/lib" -name '*.jai'
      find "$ROOT/tests/golden" -name '*.jai'
      find "$ROOT/examples" -name '*.jai'
    } | LC_ALL=C sort
}

manifest() {
    local files=0 insts=0
    while IFS= read -r f; do
        local out hash n
        out="$("$JAITHON" disasm "$f" 2>/dev/null)"
        if [[ -z "$out" ]]; then
            hash="<<did-not-disassemble>>"
            n=0
        else
            hash="$(printf '%s\n' "$out" | tail -n +2 | shasum -a 256 |
                    cut -d' ' -f1)"
            n="$(printf '%s\n' "$out" | grep -c '^[0-9][0-9][0-9][0-9]  ')"
        fi
        printf '%s  %s  %s\n' "$hash" "$n" "${f#$ROOT/}"
        files=$((files + 1))
        insts=$((insts + n))
    done < <(corpus)
    printf '# %d files, %d instructions probed\n' "$files" "$insts"
}

case "${1:-check}" in
capture)
    manifest > "$GOLDEN"
    tail -1 "$GOLDEN" | sed 's/^# /captured: /'
    ;;
check)
    if [[ ! -f "$GOLDEN" ]]; then
        echo "FAIL: no golden at ${GOLDEN#$ROOT/}; run 'capture' first"
        exit 1
    fi
    tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
    manifest > "$tmp"
    if diff -q "$GOLDEN" "$tmp" > /dev/null; then
        tail -1 "$tmp" | sed 's/^# /linetable golden ok: /'
        exit 0
    fi
    echo "linetable golden MISMATCH -- a span now resolves differently:"
    diff -u "$GOLDEN" "$tmp" | grep '^[-+][^-+]' | head -20
    echo "(re-run with: scripts/linetable_golden.sh dump <file> on both sides)"
    exit 1
    ;;
dump)
    [[ $# -ge 2 ]] || { echo "usage: $0 dump <file>"; exit 2; }
    "$JAITHON" disasm "$2"
    ;;
*)
    echo "usage: $0 [capture|check|dump <file>]"
    exit 2
    ;;
esac
