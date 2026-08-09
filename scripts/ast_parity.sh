#!/bin/sh
# Compare the s-expression tree each front end prints for the same source.
#
# The C's `jaiAstPrint` (src/lang/ast.c) is the reference while it still exists.
# The self-hosted printer replaces it, and `jaithon ast` has no golden of its own
# -- only `tests/repl/meta_ast` pins three small trees -- so this is what says the
# port is faithful over real programs rather than over the examples in a comment.
#
# Delete this script when src/lang goes: with one front end there is nothing to
# compare against, and a script that compares a thing with itself always passes.
#
# usage: scripts/ast_parity.sh [path ...]        (default: lib tests examples)
set -u

JAI=./jaithon
[ -x "$JAI" ] || { echo "build jaithon first" >&2; exit 2; }
JAITHON_PATH=${JAITHON_PATH:-$PWD/lib}
export JAITHON_PATH

work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT INT TERM

same=0 differ=0 skipped=0

# Unquoted on purpose: `"${@:-a b c}"` expands the default as ONE word, so the
# no-argument run compared two paths that do not exist and reported no
# differences. The positional parameters are still quoted where they matter.
if [ "$#" -eq 0 ]; then
    set -- lib tests examples
fi

for arg in "$@"; do
    if [ -d "$arg" ]; then
        files=$(find "$arg" -name '*.jai' | sort)
    else
        files=$arg
    fi

    for src in $files; do
        if ! $JAI --front=c ast "$src" >"$work/c.txt" 2>/dev/null; then
            skipped=$((skipped + 1)); continue
        fi
        if ! $JAI run scripts/ast_sexp_dump.jai "$src" >"$work/j.txt" 2>/dev/null; then
            differ=$((differ + 1))
            printf 'FAILED  %s (self-hosted printer raised or refused)\n' "$src"
            continue
        fi
        if cmp -s "$work/c.txt" "$work/j.txt"; then
            same=$((same + 1))
        else
            differ=$((differ + 1))
            printf 'differs %s\n' "$src"
            diff "$work/c.txt" "$work/j.txt" | head -6
        fi
    done
done

printf '%d trees compared, %d identical, %d differing, %d skipped\n' \
    $((same + differ)) "$same" "$differ" "$skipped"

# A skipped file is one the C front end would not parse, which is not this
# script's business. A differing one is the port being wrong.
[ "$differ" -eq 0 ] || exit 1
