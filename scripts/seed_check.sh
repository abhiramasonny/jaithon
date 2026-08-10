#!/bin/sh
# The seed alone must be able to bootstrap the compiler.
#
# This went untested for a long time and was broken the whole while: the C front
# end quietly compiled whatever the seed failed to serve, so nothing noticed
# until that front end was deleted and a clean checkout stopped building.
#
# Two things this has to control for, both of which produced a false pass:
#
#   - `/usr/local/share/jaithon` and `/opt/homebrew/share/jaithon` are searched
#     as a fallback, so on a machine with jaithon installed the run succeeds
#     using the installed copy. JAITHON_NO_DEFAULT_PATH=1 shuts that off.
#   - a populated __jaicache__ makes the seed irrelevant, so it is wiped first.
#
# And it checks the negative: with the seed disabled the same run must FAIL. A
# test that only checks the success case cannot tell "the seed works" from "the
# seed was never needed".
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
# The abort is the expected outcome, so the shell's own "Abort trap" notice is
# noise; a subshell keeps it out of the build log.
if (JAITHON_NO_SEED=1 $JAI check "$probe" >/dev/null 2>&1); then
    echo "FAIL: the run succeeded with the seed disabled, so it proves nothing"
    exit 1
fi

# Leave a warm cache behind: every other target expects one.
find lib -name '__jaicache__' -type d -exec rm -rf {} + 2>/dev/null
$JAI check "$probe" >/dev/null 2>&1

echo "seed bootstraps the compiler from a cold cache, and is required to"
