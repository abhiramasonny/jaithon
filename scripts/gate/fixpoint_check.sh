#!/bin/sh
# Compile each source twice with the self-hosted front end and compare the two
# images byte for byte.
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

[ "$unstable" -eq 0 ] || exit 1
