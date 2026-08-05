// Jaithon VS Code extension.
//
// Deliberately thin: the compiler already produces precise, positioned
// diagnostics and a canonical formatter, so the extension's whole job is to
// shell out to `jaithon` and translate its output. There is no reimplemented
// parser here to drift out of sync with the real one.

const vscode = require('vscode');
const { execFile } = require('child_process');
const path = require('path');

/** Diagnostic lines look like:  path:line:col: error[E0301]: message */
const DIAG_RE = /^(.*?):(\d+):(\d+):\s+(error|warning)\[([EW]\d{4})\]:\s+(.*)$/;

let diagnostics;
let output;

function config() {
    return vscode.workspace.getConfiguration('jaithon');
}

function workspaceDir(document) {
    if (document) {
        const folder = vscode.workspace.getWorkspaceFolder(document.uri);
        if (folder) return folder.uri.fsPath;
        return path.dirname(document.uri.fsPath);
    }
    return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
}

// Resolve the interpreter. The setting wins, with ${workspaceFolder} expanded
// because VS Code does not substitute variables in arbitrary string settings.
// Otherwise prefer a binary built in the workspace over one on PATH, so working
// on the compiler tests the compiler you just built.
function binary(document) {
    const root = workspaceDir(document);
    const configured = config().get('path');

    if (configured && configured !== 'jaithon') {
        return configured.replace(/\$\{workspaceFolder\}/g, root);
    }
    const local = path.join(root, 'jaithon');
    try {
        require('fs').accessSync(local, require('fs').constants.X_OK);
        return local;
    } catch {
        return 'jaithon';
    }
}

function runJaithon(args, cwd, document) {
    return new Promise((resolve) => {
        execFile(binary(document), args, { cwd, maxBuffer: 16 * 1024 * 1024 },
            (error, stdout, stderr) => {
                resolve({
                    code: error && typeof error.code === 'number' ? error.code : 0,
                    stdout,
                    stderr,
                    spawnFailed: Boolean(error && error.code === 'ENOENT'),
                });
            });
    });
}

function reportMissingBinary() {
    vscode.window.showErrorMessage(
        `Jaithon: '${binary()}' not found. Build it with 'make', or set ` +
        `"jaithon.path" in settings.`);
}

async function checkDocument(document) {
    if (document.languageId !== 'jaithon') return;

    const args = ['check'];
    if (config().get('strict')) args.push('--strict');
    args.push('--color=never', document.uri.fsPath);

    const result = await runJaithon(args, workspaceDir(document), document);
    if (result.spawnFailed) return;

    const byFile = new Map();
    for (const line of (result.stderr + result.stdout).split('\n')) {
        const match = DIAG_RE.exec(line.trim());
        if (!match) continue;

        const [, file, lineNo, colNo, severity, code, message] = match;
        const position = new vscode.Position(Number(lineNo) - 1, Number(colNo) - 1);
        // The compiler reports a start position; extend to the end of the word
        // so the squiggle is visible rather than a zero-width caret.
        const range = document.getWordRangeAtPosition(position)
            || new vscode.Range(position, position.translate(0, 1));

        const diagnostic = new vscode.Diagnostic(
            range,
            message,
            severity === 'error'
                ? vscode.DiagnosticSeverity.Error
                : vscode.DiagnosticSeverity.Warning);
        diagnostic.code = code;
        diagnostic.source = 'jaithon';

        const key = path.resolve(workspaceDir(document), file);
        if (!byFile.has(key)) byFile.set(key, []);
        byFile.get(key).push(diagnostic);
    }

    diagnostics.set(document.uri, byFile.get(document.uri.fsPath) || []);
}

function terminal() {
    const existing = vscode.window.terminals.find((t) => t.name === 'Jaithon');
    return existing || vscode.window.createTerminal('Jaithon');
}

function runInTerminal(args, document) {
    const term = terminal();
    term.show(true);
    term.sendText(`"${binary(document)}" ${args.join(' ')}`);
}

async function showInPanel(title, args, cwd, language, document) {
    const result = await runJaithon(args, cwd, document);
    if (result.spawnFailed) return reportMissingBinary();
    const doc = await vscode.workspace.openTextDocument({
        content: result.stdout || result.stderr,
        language: language || 'plaintext',
    });
    await vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.Beside });
}


// ---------------------------------------------------------------------------
// .jaic viewer
//
// A .jaic is a bytecode image, so opening one normally shows binary garbage.
// This registers a read-only custom editor that renders `jaithon disasm` output
// instead — the header fields, then the disassembly of every function in the
// image. Read-only because an image is generated: editing one is meaningless,
// and the source it names is what you actually want to change.
// ---------------------------------------------------------------------------

class JaicDocument {
    constructor(uri) { this.uri = uri; }
    dispose() {}
}

function registerJaicViewer(context) {
    return vscode.window.registerCustomEditorProvider(
        'jaithon.jaicViewer',
        {
            openCustomDocument(uri) { return new JaicDocument(uri); },

            async resolveCustomEditor(document, panel) {
                panel.webview.options = { enableScripts: false };

                const dir = path.dirname(document.uri.fsPath);
                const result = await runJaithon(['disasm', document.uri.fsPath], dir);

                let body;
                if (result.spawnFailed) {
                    body = `jaithon not found.\n\n`
                         + `Set "jaithon.path", or build it with 'make'.`;
                } else {
                    // disasm writes the listing to stdout and any complaint to
                    // stderr; show both, since a version mismatch is reported
                    // on stderr but the header still prints.
                    body = (result.stdout || '') + (result.stderr || '');
                    if (!body.trim()) body = '(empty image)';
                }
                panel.webview.html = renderListing(document.uri.fsPath, body);
            },
        },
        { webviewOptions: { retainContextWhenHidden: true }, supportsMultipleEditorsPerDocument: true },
    );
}

function escapeHtml(text) {
    return text.replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function renderListing(fsPath, text) {
    // Colour the three things worth picking out at a glance: the `;` header and
    // comment lines, the `== name ==` function banners, and opcode mnemonics.
    const highlighted = escapeHtml(text)
        .replace(/^(==.*==)$/gm, '<span class="fn">$1</span>')
        .replace(/^(;.*)$/gm, '<span class="cmt">$1</span>')
        .replace(/\b(OP_[A-Z0-9_]+)\b/g, '<span class="op">$1</span>');

    return `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body { font-family: var(--vscode-editor-font-family, monospace);
         font-size: var(--vscode-editor-font-size, 12px);
         color: var(--vscode-editor-foreground);
         background: var(--vscode-editor-background);
         padding: 12px; }
  h1 { font-size: 1.1em; font-weight: 600; margin: 0 0 4px; }
  .sub { opacity: 0.6; margin-bottom: 14px; font-size: 0.9em; }
  pre { margin: 0; white-space: pre; overflow-x: auto; line-height: 1.45; }
  .cmt { color: var(--vscode-descriptionForeground, #888); }
  .fn  { color: var(--vscode-symbolIcon-functionForeground, #b180d7); font-weight: 600; }
  .op  { color: var(--vscode-symbolIcon-keywordForeground, #569cd6); }
</style></head><body>
<h1>${escapeHtml(path.basename(fsPath))}</h1>
<div class="sub">Jaithon bytecode image &middot; read-only</div>
<pre>${highlighted}</pre>
</body></html>`;
}

function activate(context) {
    diagnostics = vscode.languages.createDiagnosticCollection('jaithon');
    output = vscode.window.createOutputChannel('Jaithon');
    context.subscriptions.push(diagnostics, output);

    const currentFile = () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.document.languageId !== 'jaithon') {
            vscode.window.showWarningMessage('Jaithon: no Jaithon file is active.');
            return null;
        }
        return editor.document;
    };

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.run', async () => {
            const doc = currentFile();
            if (!doc) return;
            await doc.save();
            runInTerminal(['run', `"${doc.uri.fsPath}"`], doc);
        }),

        vscode.commands.registerCommand('jaithon.check', async () => {
            const doc = currentFile();
            if (!doc) return;
            await doc.save();
            await checkDocument(doc);
            const found = diagnostics.get(doc.uri) || [];
            vscode.window.showInformationMessage(
                found.length === 0 ? 'Jaithon: no problems found.'
                                   : `Jaithon: ${found.length} problem(s).`);
        }),

        vscode.commands.registerCommand('jaithon.test', () => {
            const folder = vscode.workspace.workspaceFolders?.[0];
            runInTerminal(['test', folder ? `"${folder.uri.fsPath}"` : '']);
        }),

        vscode.commands.registerCommand('jaithon.repl', () => runInTerminal(['repl'])),

        vscode.commands.registerCommand('jaithon.disasm', async () => {
            const doc = currentFile();
            if (!doc) return;
            await doc.save();
            await showInPanel('bytecode', ['disasm', doc.uri.fsPath], workspaceDir(doc),
                              undefined, doc);
        }),

        vscode.commands.registerCommand('jaithon.ast', async () => {
            const doc = currentFile();
            if (!doc) return;
            await doc.save();
            await showInPanel('ast', ['ast', '--json', doc.uri.fsPath], workspaceDir(doc),
                              'json', doc);
        }),
    );

    // The formatter is canonical, so wiring it as the document formatter needs
    // no options plumbing at all.
    context.subscriptions.push(registerJaicViewer(context));

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.disasmImage', async (uri) => {
            const target = uri || vscode.window.activeTextEditor?.document.uri;
            if (!target) {
                vscode.window.showWarningMessage('Jaithon: no .jaic file selected.');
                return;
            }
            await vscode.commands.executeCommand(
                'vscode.openWith', target, 'jaithon.jaicViewer');
        }));

    context.subscriptions.push(
        vscode.languages.registerDocumentFormattingEditProvider('jaithon', {
            async provideDocumentFormattingEdits(document) {
                const result = await runJaithon(['fmt', '--stdout', document.uri.fsPath],
                                                workspaceDir(document), document);
                if (result.spawnFailed || result.code !== 0 || !result.stdout) return [];
                const whole = new vscode.Range(
                    document.positionAt(0),
                    document.positionAt(document.getText().length));
                return [vscode.TextEdit.replace(whole, result.stdout)];
            },
        }));

    context.subscriptions.push(
        vscode.workspace.onDidSaveTextDocument((doc) => {
            if (config().get('checkOnSave')) checkDocument(doc);
        }),
        vscode.workspace.onDidOpenTextDocument(checkDocument),
        vscode.workspace.onDidCloseTextDocument((doc) => diagnostics.delete(doc.uri)),
    );

    vscode.workspace.textDocuments.forEach(checkDocument);
}

function deactivate() {
    if (diagnostics) diagnostics.dispose();
}

module.exports = { activate, deactivate };
