#!/usr/bin/env bash
set -e

ROOT=${PREFIX:-/usr/local/share/jaithon}
BIN_DIR=${BINDIR:-/usr/local/bin}

if [ ! -x "./jaithon" ]; then
    echo "Build ./jaithon first (run 'make')."
    exit 1
fi

mkdir -p "$ROOT"
mkdir -p "$ROOT/lib"
cp -R lib/* "$ROOT/lib/"
mkdir -p "$BIN_DIR"
install -m 755 ./jaithon "$BIN_DIR/jaithon"

echo "Installed jaithon to $BIN_DIR/jaithon"
echo "Libraries are in $ROOT"
