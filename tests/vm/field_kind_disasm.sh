#!/usr/bin/env bash
# The disassembler must name each field's declared kind.
#
# Nothing else exposes bits 4-7 of OP_FIELD_DEF's info byte -- a golden's stdout
# cannot see them -- and there is no second front end to differ from, so this is
# the only thing standing between the mapping and a silent wrong answer. Every
# branch of the mapping appears below, including the three that must come out
# ANY: nullable, unannotated, and a bare generic parameter.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JAITHON_PATH="$ROOT/lib"
JAITHON="${JAITHON:-$ROOT/jaithon}"

out=$("$JAITHON" disasm "$ROOT/tests/golden/field_kind.jai" 2>&1)
fail=0

# field  expected-kind
while read -r field kind; do
    [ -n "$field" ] || continue
    if ! printf '%s\n' "$out" | grep -q "OP_FIELD_DEF.*\"$field\".*$kind"; then
        echo "FAIL: no OP_FIELD_DEF line for \"$field\" naming kind $kind"
        printf '%s\n' "$out" | grep "OP_FIELD_DEF.*\"$field\"" | sed 's/^/      got: /'
        fail=1
    else
        echo "ok $field -> $kind"
    fi
done <<'PAIRS'
count int
ratio float
name str
flag bool
items list
table dict
maybe any
loose any
item any
many list
PAIRS

exit $fail
