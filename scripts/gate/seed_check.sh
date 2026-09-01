#!/bin/sh
# Check that the seed can bootstrap the compiler from a cold cache and is required to do so.
set -u

JAI=./jaithon
[ -x "$JAI" ] || { echo "build jaithon first" >&2; exit 2; }
JAITHON_PATH=${JAITHON_PATH:-$PWD/lib}
export JAITHON_PATH
export JAITHON_NO_DEFAULT_PATH=1

probe=${1:-lib/std}

find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null
if ! $JAI check "$probe" >/dev/null 2>&1; then
    echo "FAIL: the seed cannot bootstrap the compiler on a cold cache"
    exit 1
fi

find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null
if (JAITHON_NO_SEED=1 $JAI check "$probe" >/dev/null 2>&1); then
    echo "FAIL: the run succeeded with the seed disabled, so it proves nothing"
    exit 1
fi

find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null
$JAI check "$probe" >/dev/null 2>&1

echo "seed bootstraps the compiler from a cold cache, and is required to"
