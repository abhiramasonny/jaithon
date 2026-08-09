#!/bin/sh
# Compare the .jaic image each front end produces for the same source.
#
# `--bootstrap-verify` compares FuncProtos: arity, flags, frame size, captures,
# defaults, the code stream and the constant pool. It never opens the serialised
# container, so it reports `ok, 0 differing` for a file whose images differ --
# which is exactly what it did while the self-hosted front end was emitting a
# shorter line table than the C for the same source.
#
# This is a DIAGNOSTIC, not a spec gate, and the distinction cost a wrong
# conclusion once. Phase 8's fixpoint gate is `stage1.jaic == stage2.jaic` --
# two *self-hosted* outputs -- which this does not measure. C-vs-self-hosted
# image equality was never required by the spec.
#
# It earns its place anyway: it is what found the self-hosted front end emitting
# a shorter line table than the C for identical source, a real bug that
# `--bootstrap-verify` reported as `ok`.
#
# One difference here is EXPECTED and not a defect: serialize.c:439 writes
# `module->exports`, which is empty at `build` time under the C front end
# because exports populate when the module body runs, while the self-hosted
# compiler computes them statically. Every stdlib file differs for that reason
# alone.
#
# Two things the comparison has to control for, both learned by getting them
# wrong:
#
#   - The source path is embedded in the image, so the two builds must compile
#     the *same* path. Building into two directories makes every image differ at
#     the path and tells you nothing.
#   - Both images must be read the same way. Diffing a compile-path disassembly
#     against an image-path one measures the two pipelines, not the two files.
#     This script diffs the bytes, which has no such ambiguity.
#
# usage: scripts/image_parity.sh [path ...]        (default: lib/std)
set -u

JAI=./jaithon
[ -x "$JAI" ] || { echo "build jaithon first" >&2; exit 2; }
JAITHON_PATH=${JAITHON_PATH:-$PWD/lib}
export JAITHON_PATH

work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT INT TERM

same=0 differ=0 skipped=0

for arg in "${@:-lib/std}"; do
    if [ -d "$arg" ]; then
        files=$(find "$arg" -name '*.jai' | sort)
    else
        files=$arg
    fi

    for src in $files; do
        base=$(basename "$src")
        stem=${base%.jai}
        cp "$src" "$work/$base" || { skipped=$((skipped + 1)); continue; }

        # Same path both times, so the embedded path cannot be the difference.
        rm -rf "$work/__jaicache__"
        if ! $JAI --front=c build "$work/$base" >/dev/null 2>&1 ||
           [ ! -f "$work/__jaicache__/$stem.jaic" ]; then
            skipped=$((skipped + 1)); rm -f "$work/$base"; continue
        fi
        cp "$work/__jaicache__/$stem.jaic" "$work/c.jaic"

        rm -rf "$work/__jaicache__"
        if ! $JAI --front=jai build "$work/$base" >/dev/null 2>&1 ||
           [ ! -f "$work/__jaicache__/$stem.jaic" ]; then
            skipped=$((skipped + 1)); rm -f "$work/$base"; continue
        fi
        cp "$work/__jaicache__/$stem.jaic" "$work/j.jaic"

        if cmp -s "$work/c.jaic" "$work/j.jaic"; then
            same=$((same + 1))
        else
            differ=$((differ + 1))
            at=$(cmp "$work/c.jaic" "$work/j.jaic" 2>&1 | sed 's/.*char //; s/,.*//')
            cs=$(wc -c < "$work/c.jaic" | tr -d ' ')
            js=$(wc -c < "$work/j.jaic" | tr -d ' ')
            printf 'differs %-28s c=%s jai=%s first=%s\n' "$src" "$cs" "$js" "$at"
        fi
        rm -f "$work/$base"
    done
done

printf '%d images compared, %d identical, %d differing, %d skipped\n' \
    $((same + differ)) "$same" "$differ" "$skipped"

# A skipped file is not a pass. It means one front end refused the file, and a
# shrinking compared-count is how coverage quietly disappears.
#
# Differences do not fail this script: the exports gap above makes every module
# differ, and treating a diagnostic as a gate is what made it look like Phase 8
# was blocked when it was not. Read the output; do not gate on it.
exit 0

# See also: the gate that DOES block Phase 8 is self-vs-self determinism --
# compile one source twice with --front=jai and compare. `make fixpoint-check`.
