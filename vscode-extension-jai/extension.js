const vscode = require('vscode');
const path = require('path');
const fs = require('fs');
const os = require('os');

/** @type {string[]} */
let modulePaths = [];
/** @type {Map<string, { kind: 'func'|'class'|'namespace', locations: vscode.Location[] }>} */
let symbolIndex = new Map();

function config() {
    return vscode.workspace.getConfiguration('jaithon');
}

function isExecutable(p) {
    try {
        fs.accessSync(p, fs.constants.X_OK);
        return true;
    } catch {
        return false;
    }
}

function resolveInterpreterPath() {
    const cfg = config();
    const configured = cfg.get('interpreterPath');
    const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const envBin = process.env.JAITHON_BIN;

    const candidates = [];
    if (configured) {
        candidates.push(folder ? configured.replace(/\$\{workspaceFolder\}/g, folder) : configured);
    }
    if (envBin) candidates.push(envBin);
    if (folder) {
        candidates.push(path.join(folder, 'jaithon'));
        candidates.push(path.join(folder, 'build', 'jaithon'));
        candidates.push(path.join(folder, 'bin', 'jaithon'));
    }
    candidates.push('/usr/local/bin/jaithon');
    candidates.push('/opt/homebrew/bin/jaithon');
    candidates.push('/usr/bin/jaithon');

    const pathDirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean);
    for (const dir of pathDirs) {
        candidates.push(path.join(dir, 'jaithon'));
    }

    for (const c of candidates) {
        if (c && fs.existsSync(c) && isExecutable(c)) return c;
    }

    if (configured) return configured;
    return folder ? path.join(folder, 'jaithon') : 'jaithon';
}

function terminal() {
    const name = 'Jaithon';
    const existing = vscode.window.terminals.find(t => t.name === name);
    return existing ?? vscode.window.createTerminal({ name });
}

function quoteIfNeeded(s) {
    if (typeof s !== 'string') return '';
    if (/^[A-Za-z0-9_.\\/-:=]+$/.test(s)) return s;
    return `"${s.replace(/"/g, '\\"')}"`;
}

async function collectModules() {
    if (!vscode.workspace.workspaceFolders) return;
    const roots = vscode.workspace.workspaceFolders.map(f => f.uri.fsPath);
    const patterns = config().get('moduleGlobs') || ['lib/modules/**/*.jai', 'modules/**/*.jai'];
    const exclude = config().get('excludeGlob') || '**/{node_modules,__pycache__,build,out,__jaicache__}/**';
    const files = [];
    for (const pat of patterns) {
        const uris = await vscode.workspace.findFiles(pat, exclude);
        files.push(...uris);
    }
    modulePaths = files.map(uri => {
        for (const root of roots) {
            const rel = path.relative(root, uri.fsPath);
            if (!rel.startsWith('..')) {
                return rel.replace(/\.jai$/i, '').split(path.sep).join('/');
            }
        }
        return uri.fsPath.replace(/\.jai$/i, '').split(path.sep).join('/');
    });
}

function parseDefinitions(text) {
    /** @type {{name: string, kind: 'func'|'class'|'namespace', line: number, col: number}[]} */
    const defs = [];
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const m = /^\s*(func|class|namespace)\s+([A-Za-z_][A-Za-z0-9_]*)\b/.exec(line);
        if (m) {
            defs.push({ kind: m[1], name: m[2], line: i, col: line.indexOf(m[2]) });
        }
    }
    return defs;
}

async function collectSymbols() {
    if (!vscode.workspace.workspaceFolders) return;
    const patterns = config().get('indexGlobs') || ['lib/std.jai', 'lib/modules/**/*.jai', 'modules/**/*.jai'];
    const exclude = config().get('excludeGlob') || '**/{node_modules,__pycache__,build,out,__jaicache__}/**';
    /** @type {Map<string, { kind: 'func'|'class'|'namespace', locations: vscode.Location[] }>} */
    const next = new Map();

    const uris = [];
    for (const pat of patterns) {
        const found = await vscode.workspace.findFiles(pat, exclude);
        uris.push(...found);
    }

    for (const uri of uris) {
        let text = '';
        try {
            text = fs.readFileSync(uri.fsPath, 'utf8');
        } catch {
            continue;
        }
        const defs = parseDefinitions(text);
        for (const d of defs) {
            const loc = new vscode.Location(uri, new vscode.Position(d.line, Math.max(0, d.col)));
            const entry = next.get(d.name);
            if (entry) {
                entry.locations.push(loc);
            } else {
                next.set(d.name, { kind: d.kind, locations: [loc] });
            }
        }
    }

    symbolIndex = next;
}

function registerCompletion(context) {
    const importProvider = vscode.languages.registerCompletionItemProvider(
        { language: 'jaithon' },
        {
            provideCompletionItems(document, position) {
                const line = document.lineAt(position).text.slice(0, position.character);
                if (!/^\s*import\s+[A-Za-z0-9_./-]*$/.test(line)) return undefined;
                const items = modulePaths.map(mod => {
                    const item = new vscode.CompletionItem(mod, vscode.CompletionItemKind.Module);
                    item.insertText = mod;
                    item.detail = 'Jaithon module';
                    return item;
                });
                return items;
            }
        },
        '/' // trigger on slash to help nested modules
    );
    context.subscriptions.push(importProvider);

    const symbolProvider = vscode.languages.registerCompletionItemProvider(
        { language: 'jaithon' },
        {
            provideCompletionItems(document, position) {
                const line = document.lineAt(position).text;
                if (/^\s*import\b/.test(line)) return undefined;

                const range = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_]*/);
                const prefix = range ? document.getText(range) : '';
                const items = [];
                for (const [name, entry] of symbolIndex.entries()) {
                    if (prefix && !name.startsWith(prefix)) continue;
                    const kind =
                        entry.kind === 'class'
                            ? vscode.CompletionItemKind.Class
                            : entry.kind === 'namespace'
                              ? vscode.CompletionItemKind.Module
                              : vscode.CompletionItemKind.Function;
                    const item = new vscode.CompletionItem(name, kind);
                    const loc = entry.locations[0];
                    item.detail = loc ? path.basename(loc.uri.fsPath) : 'Jaithon symbol';
                    items.push(item);
                    if (items.length > 500) break;
                }
                return items.length ? items : undefined;
            }
        }
    );
    context.subscriptions.push(symbolProvider);
}

function registerDefinitions(context) {
    const provider = vscode.languages.registerDefinitionProvider(
        { language: 'jaithon' },
        {
            async provideDefinition(document, position) {
                const lineText = document.lineAt(position.line).text;
                const importMatch = /^\s*import\s+([A-Za-z0-9_./-]+)\s*$/.exec(lineText);
                if (importMatch) {
                    const modulePath = importMatch[1].replace(/\.jai$/i, '');
                    const candidate = modulePath.endsWith('.jai') ? modulePath : `${modulePath}.jai`;
                    const roots = vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath) || [];
                    for (const root of roots) {
                        const full = path.join(root, ...candidate.split('/'));
                        if (fs.existsSync(full)) {
                            return new vscode.Location(vscode.Uri.file(full), new vscode.Position(0, 0));
                        }
                    }
                }

                const wordRange = document.getWordRangeAtPosition(position, /[A-Za-z_][A-Za-z0-9_]*/);
                if (!wordRange) return undefined;
                const word = document.getText(wordRange);

                const inDocRegex = new RegExp(`^\\s*(func|class|namespace)\\s+${escapeRegex(word)}\\b`, 'm');
                const text = document.getText();
                const m = inDocRegex.exec(text);
                if (m) {
                    const before = text.slice(0, m.index);
                    const line = before.split(/\r?\n/).length - 1;
                    return new vscode.Location(document.uri, new vscode.Position(line, m[0].indexOf(word)));
                }

                const entry = symbolIndex.get(word);
                if (entry?.locations?.length) return entry.locations;
                return undefined;
            }
        }
    );
    context.subscriptions.push(provider);
}

function registerDocumentSymbols(context) {
    const provider = vscode.languages.registerDocumentSymbolProvider(
        { language: 'jaithon' },
        {
            provideDocumentSymbols(document) {
                const defs = parseDefinitions(document.getText());
                return defs.map(d => {
                    const kind =
                        d.kind === 'class'
                            ? vscode.SymbolKind.Class
                            : d.kind === 'namespace'
                              ? vscode.SymbolKind.Namespace
                              : vscode.SymbolKind.Function;
                    const pos = new vscode.Position(d.line, Math.max(0, d.col));
                    const range = new vscode.Range(pos, pos);
                    return new vscode.DocumentSymbol(d.name, d.kind, kind, range, range);
                });
            }
        }
    );
    context.subscriptions.push(provider);
}

function escapeRegex(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function registerCommands(context) {
    async function resolveTargetUri(uri) {
        if (uri && uri.fsPath) return uri;
        const editor = vscode.window.activeTextEditor;
        return editor?.document?.uri;
    }

    async function runUri(uri, extraArgs = []) {
        const target = await resolveTargetUri(uri);
        if (!target) return;
        const doc = await vscode.workspace.openTextDocument(target);
        await doc.save();
        await runCommand([...extraArgs, doc.uri.fsPath], { cwd: path.dirname(doc.uri.fsPath) });
    }

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.runFile', async uri => {
            await runUri(uri);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.runFileDebug', async uri => {
            await runUri(uri, ['--debug']);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.runSelection', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;
            const selection = editor.document.getText(editor.selection);
            if (!selection.trim()) return;
            const tmp = path.join(os.tmpdir(), `jaithon_selection_${Date.now()}.jai`);
            fs.writeFileSync(tmp, selection, 'utf8');
            await runCommand([tmp], { cwd: path.dirname(editor.document.uri.fsPath) });
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.repl', async () => {
            await runCommand(['--shell']);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.runTestFile', async uri => {
            await runUri(uri);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('jaithon.runTestSuite', async () => {
            const python = config().get('pythonPath') || 'python3';
            const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
            if (!folder) return;
            const script = path.join(folder, 'test_runner.py');
            await runCommand([], {
                cwd: folder,
                terminalCommandOverride: `${python} ${quoteIfNeeded(script)}`
            });
        })
    );
}

function registerCodeLens(context) {
    const provider = vscode.languages.registerCodeLensProvider({ language: 'jaithon' }, {
        provideCodeLenses(document) {
            const lenses = [];
            const top = new vscode.Range(0, 0, 0, 0);
            lenses.push(new vscode.CodeLens(top, { title: 'Run', command: 'jaithon.runFile', arguments: [document.uri] }));
            lenses.push(new vscode.CodeLens(top, { title: 'Run (Debug)', command: 'jaithon.runFileDebug', arguments: [document.uri] }));

            const p = document.uri.fsPath.split(path.sep).join('/');
            if (p.includes('/test/checks/')) {
                lenses.push(new vscode.CodeLens(top, { title: 'Run Test Suite', command: 'jaithon.runTestSuite' }));
            }

            return lenses;
        }
    });
    context.subscriptions.push(provider);
}

async function runCommand(args, opts = {}) {
    const cfg = config();
    const interpreter = resolveInterpreterPath();
    const defaultArgs = cfg.get('defaultArgs') || [];
    const useTerminal = cfg.get('useTerminal');
    const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const cwd = opts.cwd || workspaceRoot;

    const terminalCommandOverride = opts.terminalCommandOverride;
    const fullArgs = [...defaultArgs, ...args].filter(Boolean);

    if (useTerminal) {
        const t = terminal();
        t.show(true);
        const cmd = terminalCommandOverride
            ? terminalCommandOverride
            : [interpreter, ...fullArgs].map(quoteIfNeeded).join(' ');
        t.sendText(cmd, true);
        return;
    }

    const output = vscode.window.createOutputChannel('Jaithon');
    output.show(true);
    if (terminalCommandOverride) {
        output.appendLine(terminalCommandOverride);
        const cp = require('child_process').spawn(terminalCommandOverride, {
            cwd,
            shell: true
        });
        cp.stdout.on('data', d => output.append(d.toString()));
        cp.stderr.on('data', d => output.append(d.toString()));
        cp.on('close', code => output.appendLine(`\n[exit ${code}]`));
        return;
    }

    output.appendLine([interpreter, ...fullArgs].join(' '));
    const cp = require('child_process').spawn(interpreter, fullArgs, {
        cwd,
        shell: false
    });
    cp.stdout.on('data', d => output.append(d.toString()));
    cp.stderr.on('data', d => output.append(d.toString()));
    cp.on('close', code => output.appendLine(`\n[exit ${code}]`));
}

function activate(context) {
    collectModules();
    collectSymbols();
    registerCompletion(context);
    registerDefinitions(context);
    registerDocumentSymbols(context);
    registerCommands(context);
    registerCodeLens(context);
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('jaithon')) {
                collectModules();
                collectSymbols();
            }
        })
    );

    const watcher = vscode.workspace.createFileSystemWatcher('**/*.jai');
    watcher.onDidCreate(collectModules);
    watcher.onDidDelete(collectModules);
    watcher.onDidChange(collectModules);
    context.subscriptions.push(watcher);

    const libWatcher = vscode.workspace.createFileSystemWatcher('lib/**/*.jai');
    libWatcher.onDidCreate(collectSymbols);
    libWatcher.onDidDelete(collectSymbols);
    libWatcher.onDidChange(collectSymbols);
    context.subscriptions.push(libWatcher);

    const modulesWatcher = vscode.workspace.createFileSystemWatcher('modules/**/*.jai');
    modulesWatcher.onDidCreate(collectSymbols);
    modulesWatcher.onDidDelete(collectSymbols);
    modulesWatcher.onDidChange(collectSymbols);
    context.subscriptions.push(modulesWatcher);
}

function deactivate() {}

module.exports = { activate, deactivate };
