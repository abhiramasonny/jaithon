#!/usr/bin/env bash
# Install the Jaithon VS Code extension.
#
#   editors/vscode/install.sh
#   EDITOR_CLI=cursor editors/vscode/install.sh
#
# VS Code no longer loads a folder dropped into ~/.vscode/extensions: the
# registry in extensions.json is authoritative, and anything on disk that is
# not listed there is logged as "Marked extension as removed" and ignored. So
# the extension is packaged and installed through the CLI, which writes that
# entry. The install is a copy — run this again after changing the source.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(node -p "require('$ROOT/package.json').version")"
VSIX="$(mktemp -d)/jaithon-$VERSION.vsix"

cli() {
    if [[ -n "${EDITOR_CLI:-}" ]]; then
        command -v "$EDITOR_CLI" || return 1
        return 0
    fi
    for candidate in code code-insiders \
        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"; do
        command -v "$candidate" 2>/dev/null && return 0
        [[ -x $candidate ]] && { echo "$candidate"; return 0; }
    done
    return 1
}

CODE="$(cli)" || {
    echo "no VS Code CLI found; set EDITOR_CLI, or install the 'code' command" >&2
    exit 1
}

echo "==> packaging $VERSION"
cd "$ROOT"
npx --yes @vscode/vsce package \
    --no-dependencies --allow-missing-repository --skip-license \
    --out "$VSIX" >/dev/null

echo "==> installing"
"$CODE" --install-extension "$VSIX" --force

rm -rf "$(dirname "$VSIX")"

echo
echo "Installed. Reload the window (Developer: Reload Window) to pick it up."
