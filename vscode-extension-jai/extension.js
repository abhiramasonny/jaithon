const vscode = require('vscode');
const path = require('path');

/** @type {string[]} */
let modulePaths = [];

async function collectModules() {
    if (!vscode.workspace.workspaceFolders) return;
    const roots = vscode.workspace.workspaceFolders.map(f => f.uri.fsPath);
    const patterns = ['lib/modules/**/*.jai', 'modules/**/*.jai'];
    const exclude = '**/{node_modules,__pycache__,build,out,__jaicache__}/**';
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

function registerCompletion(context) {
    const provider = vscode.languages.registerCompletionItemProvider(
        { language: 'jaithon' },
        {
            provideCompletionItems(document, position) {
                const line = document.lineAt(position).text.slice(0, position.character);
                if (!/^\s*import\s+[\w\/]*$/.test(line)) return undefined;
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
    context.subscriptions.push(provider);
}

function activate(context) {
    collectModules();
    registerCompletion(context);

    const watcher = vscode.workspace.createFileSystemWatcher('**/*.jai');
    watcher.onDidCreate(collectModules);
    watcher.onDidDelete(collectModules);
    watcher.onDidChange(() => {}); // noop; we only care about adds/removes
    context.subscriptions.push(watcher);
}

function deactivate() {}

module.exports = { activate, deactivate };
