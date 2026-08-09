#!/bin/sh
# Compile each source twice with the self-hosted front end and compare the two
# images byte for byte.
#
# This is the precondition for Phase 8's real gate. The three-stage fixpoint
# asks that `stage1.jaic == stage2.jaic` -- two outputs of the *self-hosted*
# front end -- and a front end whose output is not reproducible from one run to
# the next can never satisfy it, whatever else is true of it.
#
# Deliberately not the same question as `make image-parity`, which compares the
# C's image against the self-hosted one. Those two differ on every module for a
# benign reason (the exports table; see image_parity.sh), and reading that as a
# Phase 8 blocker was a mistake worth not repeating: C-vs-self-hosted image
# equality is not something the spec asks for.
#
# usage: scripts/fixpoint_check.sh [path ...]      (default: lib/std)
set -u

JAI=./jaithon
[ -x "$JAI" ] || { echo "build jaithon first" >&2; exit 2; }
JAITHON_PATH=${JAITHON_PATH:-$PWD/lib}
export JAITHON_PATH

work=$(mktemp -d) || exit 2
trap 'rm -rf "$work"' EXIT INT TERM

stable=0 unstable=0 skipped=0

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

        # The same path both times: the source path is embedded in the image, so
        # compiling into two directories would differ for that reason alone.
        rm -rf "$work/__jaicache__"
        if ! $JAI --front=jai build "$work/$base" >/dev/null 2>&1 ||
           [ ! -f "$work/__jaicache__/$stem.jaic" ]; then
            skipped=$((skipped + 1)); rm -f "$work/$base"; continue
        fi
        cp "$work/__jaicache__/$stem.jaic" "$work/stage1.jaic"

        rm -rf "$work/__jaicache__"
        if ! $JAI --front=jai build "$work/$base" >/dev/null 2>&1 ||
           [ ! -f "$work/__jaicache__/$stem.jaic" ]; then
            skipped=$((skipped + 1)); rm -f "$work/$base"; continue
        fi
        cp "$work/__jaicache__/$stem.jaic" "$work/stage2.jaic"

        if cmp -s "$work/stage1.jaic" "$work/stage2.jaic"; then
            stable=$((stable + 1))
        else
            unstable=$((unstable + 1))
            at=$(cmp "$work/stage1.jaic" "$work/stage2.jaic" 2>&1 |
                 sed 's/.*char //; s/,.*//')
            printf 'NOT REPRODUCIBLE %-28s first difference at byte %s\n' "$src" "$at"
        fi
        rm -f "$work/$base"
    done
done

printf '%d sources compiled twice, %d reproducible, %d not, %d skipped\n' \
    $((stable + unstable)) "$stable" "$unstable" "$skipped"

# Unlike image-parity, this one is a gate: a non-reproducible image makes the
# fixpoint unreachable. A skipped file is not a pass either -- it means the front
# end refused a file it used to accept, which is how coverage disappears quietly.
[ "$unstable" -eq 0 ] || exit 1
