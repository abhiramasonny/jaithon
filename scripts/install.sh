#!/usr/bin/env bash
# Install Jaithon.
#
#   scripts/install.sh              -> /usr/local
#   PREFIX=~/.local scripts/install.sh
#
# Installs the binary to $PREFIX/bin and the standard library to
# $PREFIX/share/jaithon/lib. The binary locates its library relative to its own
# path first, so a relocated install keeps working.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"
LIB_DIR="$PREFIX/share/jaithon"

cd "$ROOT"

if [[ ! -x ./jaithon ]]; then
    echo "==> building"
    make -s
fi

echo "==> installing to $PREFIX"

need_sudo=""
if [[ ! -w "$(dirname "$BIN_DIR")" ]]; then
    need_sudo="sudo"
    echo "    (elevating: $PREFIX is not writable)"
fi

$need_sudo install -d "$BIN_DIR" "$LIB_DIR"
$need_sudo install -m 755 ./jaithon "$BIN_DIR/jaithon"
$need_sudo rm -rf "$LIB_DIR/lib"
$need_sudo cp -R ./lib "$LIB_DIR/lib"

echo "==> installed"
echo "    binary : $BIN_DIR/jaithon"
echo "    library: $LIB_DIR/lib"

if ! command -v jaithon >/dev/null 2>&1; then
    echo
    echo "note: $BIN_DIR is not on your PATH. Add this to your shell profile:"
    echo "      export PATH=\"$BIN_DIR:\$PATH\""
fi

echo
"$BIN_DIR/jaithon" --version
