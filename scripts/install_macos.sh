#!/usr/bin/env bash
# Jaithon macOS installer
# - Builds the jaithon binary
# - Installs it under /usr/local/lib/jaithon with the bundled stdlib modules
# - Creates a symlink at /usr/local/bin/jaithon
set -euo pipefail

PREFIX="/usr/local"
INSTALL_DIR="${PREFIX}/lib/jaithon"
BIN_LINK="${PREFIX}/bin/jaithon"

echo "==> Building jaithon..."
make -s

echo "==> Creating install directory at ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

echo "==> Copying binary and stdlib modules..."
cp -f jaithon "${INSTALL_DIR}/"
rsync -a --delete lib/ "${INSTALL_DIR}/lib/"

echo "==> Creating launcher ${BIN_LINK}"
mkdir -p "${PREFIX}/bin"
cat > "${BIN_LINK}" <<'LAUNCH'
#!/usr/bin/env bash
# Jaithon launcher: points to installed prefix and sets JAITHON_LIB for imports
export JAITHON_LIB="${JAITHON_LIB:-/usr/local/lib/jaithon}"
exec "/usr/local/lib/jaithon/jaithon" "$@"
LAUNCH
chmod +x "${BIN_LINK}"

cat <<'DONE'
Jaithon installed.

Binary:      /usr/local/lib/jaithon/jaithon
Stdlib:      /usr/local/lib/jaithon/lib
CLI launcher: /usr/local/bin/jaithon (sets JAITHON_LIB automatically)

Imports resolve relative to the executable, so the stdlib will be found automatically.
DONE
