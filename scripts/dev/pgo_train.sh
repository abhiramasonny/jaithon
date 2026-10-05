#!/usr/bin/env bash
# Regenerate boot/jaithon.profdata, the profile release builds optimise with.
#
# Builds an instrumented binary in a tree of its own (so ./jaithon and build/
# are untouched), runs the training set below with every process writing its
# own .profraw, and merges them. The set is what the binary spends its life
# doing: the language benchmarks at medium size, the self-hosted compiler
# checking and formatting lib/jaithon (interpreter, collector, tables), the
# language suites, and the CPU package suites for the natives.
#
# Run it after any change that moves where the runtime spends its time, then
# `make` -- a retrained profile changes CC_ID and rebuilds every object. The
# profile is generated data belonging to its commit, like the seed: regenerate
# it, never hand-edit or merge it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GEN_ROOT=build-pgogen
GEN_BIN=jaithon-pgogen
RAW="$(mktemp -d)"
trap 'rm -rf "$RAW" "$ROOT/$GEN_BIN"' EXIT

echo "  PGO     building instrumented binary"
make -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 8)" PGO=0 BUILD_ROOT="$GEN_ROOT" \
     TARGET="$GEN_BIN" EXTRA_CFLAGS=-fprofile-instr-generate \
     EXTRA_LDFLAGS=-fprofile-instr-generate >/dev/null

export LLVM_PROFILE_FILE="$RAW/%p-%m.profraw"
export JAITHON_PATH="$ROOT/lib"
run() { "./$GEN_BIN" "$@" >/dev/null 2>&1 || true; }

echo "  PGO     training: language benchmarks"
for dir in tests/bench/*/; do
    name="$(basename "$dir")"
    case "$name" in jaicv|jaitensor|jainum|jaiframe|shapes|__pycache__) continue ;; esac
    [[ -f "$dir$name.jai" ]] && BENCH_LEVEL=medium run run "$dir$name.jai"
done
for probe in tests/bench/shapes/*.jai; do run run "$probe"; done

echo "  PGO     training: compiler"
run check --no-cache lib/jaithon
run fmt --check lib/jaithon
run run --no-cache examples/hello.jai

echo "  PGO     training: suites"
# One process per file: forked test workers exit without writing a .profraw.
JAITHON_TEST_FORK=0 run test tests/lang tests/stdlib tests/checker
JAITHON_TEST_FORK=0 run test packages/jainum/tests packages/jaiframe/tests

PROFDATA="$(command -v llvm-profdata || echo "xcrun llvm-profdata")"
$PROFDATA merge -o boot/jaithon.profdata "$RAW"/*.profraw
echo "  PGO     wrote boot/jaithon.profdata ($(ls "$RAW" | wc -l | tr -d ' ') runs)"
