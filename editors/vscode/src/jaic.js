// The .jaic viewer.
//
// A .jaic is a bytecode image, so opening one normally shows binary garbage.
// This registers a read-only custom editor that renders `jaithon disasm` output
// instead — the header fields, then the disassembly of every function in the
// image. Read-only because an image is generated: editing one is meaningless,
// and the source it names is what you actually want to change.

const vscode = require('vscode');
const path = require('path');
const tool = require('./tool');

class JaicDocument {
    constructor(uri) { this.uri = uri; }
    dispose() {}
}

function escapeHtml(text) {
    return text.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function render(fsPath, text) {
    // Colour the things worth picking out at a glance: header and comment
    // lines, the `== name ==` function banners, opcode mnemonics, the constant
    // pool indices, and jump targets.
    const highlighted = escapeHtml(text)
        .replace(/^(==.*==)$/gm, '<span class="fn">$1</span>')
        .replace(/^(;.*)$/gm, '<span class="cmt">$1</span>')
        .replace(/\b(OP_[A-Z0-9_]+)\b/g, '<span class="op">$1</span>')
        .replace(/^(\s*)(\d{4,})/gm, '$1<span class="addr">$2</span>')
        .replace(/(&#39;|&quot;|")([^"\n]*)\1/g, '<span class="str">$1$2$1</span>');

    return `<!DOCTYPE html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
<style>
  body { font-family: var(--vscode-editor-font-family, monospace);
         font-size: var(--vscode-editor-font-size, 12px);
         color: var(--vscode-editor-foreground);
         background: var(--vscode-editor-background);
         padding: 12px 16px; }
  h1 { font-size: 1.1em; font-weight: 600; margin: 0 0 4px; }
  .sub { opacity: 0.6; margin-bottom: 14px; font-size: 0.9em; }
  pre { margin: 0; white-space: pre; overflow-x: auto; line-height: 1.45; tab-size: 4; }
  .cmt  { color: var(--vscode-descriptionForeground, #888); }
  .fn   { color: var(--vscode-symbolIcon-functionForeground, #b180d7); font-weight: 600; }
  .op   { color: var(--vscode-symbolIcon-keywordForeground, #569cd6); }
  .addr { color: var(--vscode-descriptionForeground, #888); }
  .str  { color: var(--vscode-debugTokenExpression-string, #ce9178); }
</style></head><body>
<h1>${escapeHtml(path.basename(fsPath))}</h1>
<div class="sub">Jaithon bytecode image &middot; read-only</div>
<pre>${highlighted}</pre>
</body></html>`;
}

function register(context) {
    context.subscriptions.push(vscode.window.registerCustomEditorProvider(
        'jaithon.jaicViewer',
        {
            openCustomDocument(uri) { return new JaicDocument(uri); },

            async resolveCustomEditor(document, panel) {
                panel.webview.options = { enableScripts: false };

                const dir = path.dirname(document.uri.fsPath);
                const result = await tool.run(['disasm', document.uri.fsPath], { cwd: dir });

                let body;
                if (result.spawnFailed) {
                    body = 'jaithon not found.\n\nSet "jaithon.path", or build it with \'make\'.';
                } else {
                    // disasm writes the listing to stdout and any complaint to
                    // stderr; show both, since a version mismatch is reported on
                    // stderr but the header still prints.
                    body = `${result.stdout || ''}${result.stderr || ''}`;
                    if (!body.trim()) body = '(empty image)';
                }
                panel.webview.html = render(document.uri.fsPath, body);
            },
        },
        { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: true },
    ));
}

module.exports = { register };
