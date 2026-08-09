// Jaithon VS Code extension.
//
// There is no reimplemented parser here, and no language server either. The
// compiler already produces positioned diagnostics, a canonical formatter, a
// JSON syntax tree and a dump of every type the checker inferred, so the whole
// job is to run it and index what it says. Everything the editor offers is one
// of those four answers turned into a VS Code provider; see src/analysis.js.

const vscode = require('vscode');
const tool = require('./src/tool');
const { Checker } = require('./src/diagnostics');
const { Workspace } = require('./src/analysis');
const navigation = require('./src/navigation');
const completion = require('./src/completion');
const hints = require('./src/hints');
const actions = require('./src/actions');
const jaic = require('./src/jaic');

let diagnostics;
let output;
let checker;
let workspace;
let status;

function activeDocument({ quiet } = {}) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'jaithon') {
        if (!quiet) vscode.window.showWarningMessage('Jaithon: no Jaithon file is active.');
        return null;
    }
    return editor.document;
}

function terminal() {
    return vscode.window.terminals.find((item) => item.name === 'Jaithon')
        || vscode.window.createTerminal('Jaithon');
}

function runInTerminal(args, document) {
    const term = terminal();
    term.show(true);
    term.sendText(`"${tool.binary(document)}" ${args.join(' ')}`);
}

async function showInPanel(args, cwd, language, document) {
    const result = await tool.run(args, { cwd, document });
    if (result.spawnFailed) return;
    const doc = await vscode.workspace.openTextDocument({
        content: result.stdout || result.stderr,
        language: language || 'plaintext',
    });
    await vscode.window.showTextDocument(doc, { preview: true, viewColumn: vscode.ViewColumn.Beside });
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatus(document) {
    if (!status) return;
    if (!document || document.languageId !== 'jaithon') { status.hide(); return; }

    const found = diagnostics.get(document.uri) || [];
    const errors = found.filter((item) => item.severity === vscode.DiagnosticSeverity.Error).length;
    const warnings = found.length - errors;

    status.text = errors ? `$(error) ${errors}` : warnings ? `$(warning) ${warnings}` : '$(check) Jaithon';
    status.tooltip = errors || warnings
        ? `Jaithon: ${errors} error(s), ${warnings} warning(s)`
        : 'Jaithon: no problems found';
    status.command = 'workbench.actions.view.problems';
    status.show();
}

// ---------------------------------------------------------------------------

function registerCommands(context) {
    const command = (name, handler) =>
        context.subscriptions.push(vscode.commands.registerCommand(name, handler));

    command('jaithon.run', async (uri) => {
        const document = uri
            ? await vscode.workspace.openTextDocument(uri)
            : activeDocument();
        if (!document) return;
        await document.save();
        runInTerminal(['run', `"${document.uri.fsPath}"`], document);
    });

    command('jaithon.check', async () => {
        const document = activeDocument();
        if (!document) return;
        const found = await checker.check(document);
        vscode.window.showInformationMessage(
            found.length === 0 ? 'Jaithon: no problems found.'
                               : `Jaithon: ${found.length} problem(s).`);
    });

    command('jaithon.checkWorkspace', async () => {
        const folder = vscode.workspace.workspaceFolders?.[0];
        if (!folder) return;
        runInTerminal(['check', `"${folder.uri.fsPath}"`]);
    });

    command('jaithon.test', () => {
        const folder = vscode.workspace.workspaceFolders?.[0];
        runInTerminal(['test', folder ? `"${folder.uri.fsPath}"` : '']);
    });

    command('jaithon.testFile', async (uri) => {
        const document = uri ? await vscode.workspace.openTextDocument(uri) : activeDocument();
        if (!document) return;
        await document.save();
        runInTerminal(['test', `"${document.uri.fsPath}"`], document);
    });

    command('jaithon.bench', () => {
        const folder = vscode.workspace.workspaceFolders?.[0];
        runInTerminal(['bench', folder ? `"${folder.uri.fsPath}"` : '']);
    });

    command('jaithon.repl', () => runInTerminal(['repl']));

    command('jaithon.disasm', async () => {
        const document = activeDocument();
        if (!document) return;
        await document.save();
        await showInPanel(['disasm', document.uri.fsPath],
                          tool.workspaceDir(document), undefined, document);
    });

    command('jaithon.ast', async () => {
        const document = activeDocument();
        if (!document) return;
        await document.save();
        await showInPanel(['ast', '--json', document.uri.fsPath],
                          tool.workspaceDir(document), 'json', document);
    });

    command('jaithon.tokens', async () => {
        const document = activeDocument();
        if (!document) return;
        await document.save();
        await showInPanel(['tokens', document.uri.fsPath],
                          tool.workspaceDir(document), undefined, document);
    });

    command('jaithon.disasmImage', async (uri) => {
        const target = uri || vscode.window.activeTextEditor?.document.uri;
        if (!target) {
            vscode.window.showWarningMessage('Jaithon: no .jaic file selected.');
            return;
        }
        await vscode.commands.executeCommand('vscode.openWith', target, 'jaithon.jaicViewer');
    });

    command('jaithon.restart', async () => {
        workspace.invalidateAll();
        for (const document of vscode.workspace.textDocuments) {
            if (document.languageId === 'jaithon') checker.schedule(document, 0);
        }
        const found = await tool.version();
        vscode.window.showInformationMessage(
            found ? `Jaithon: reloaded (compiler ${found}).` : 'Jaithon: reloaded.');
    });
}

// ---------------------------------------------------------------------------
// Tasks
//
// Every one of these is a thing you would otherwise type; declaring them means
// Ctrl+Shift+B builds, and the problem matcher puts the compiler's own output
// in the Problems panel with the same spans the editor shows inline.
// ---------------------------------------------------------------------------

const TASKS = [
    { command: 'check', args: ['.'], group: vscode.TaskGroup.Build, name: 'check' },
    { command: 'test', args: [], group: vscode.TaskGroup.Test, name: 'test' },
    { command: 'fmt', args: ['--check', '.'], group: undefined, name: 'fmt --check' },
    { command: 'bench', args: [], group: undefined, name: 'bench' },
];

function taskProvider() {
    const build = (definition) => {
        const folder = vscode.workspace.workspaceFolders?.[0];
        const task = new vscode.Task(
            { type: 'jaithon', command: definition.command, args: definition.args },
            folder || vscode.TaskScope.Workspace,
            definition.name || definition.command,
            'jaithon',
            new vscode.ShellExecution(tool.binary(), [definition.command, ...(definition.args || [])]),
            '$jaithon');
        if (definition.group) task.group = definition.group;
        return task;
    };
    return {
        provideTasks: () => TASKS.map(build),
        resolveTask: (task) => (task.definition.command ? build(task.definition) : undefined),
    };
}

function activate(context) {
    diagnostics = vscode.languages.createDiagnosticCollection('jaithon');
    output = vscode.window.createOutputChannel('Jaithon');
    checker = new Checker(diagnostics);
    workspace = new Workspace(output);
    status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 90);
    context.subscriptions.push(diagnostics, output, checker, workspace, status);

    registerCommands(context);
    jaic.register(context);
    navigation.register(context, workspace);
    completion.register(context, workspace);
    hints.register(context, workspace);
    actions.register(context, workspace);

    context.subscriptions.push(
        vscode.tasks.registerTaskProvider('jaithon', taskProvider()),
    );

    context.subscriptions.push(
        checker.onDidCheck(() => updateStatus(vscode.window.activeTextEditor?.document)),

        vscode.workspace.onDidChangeTextDocument((event) => {
            if (event.document.languageId !== 'jaithon') return;
            workspace.invalidate(event.document.uri.fsPath);
            if (tool.config().get('checkOnType') !== false) checker.schedule(event.document);
        }),

        vscode.workspace.onDidSaveTextDocument(async (document) => {
            if (document.languageId !== 'jaithon') return;
            workspace.invalidate(document.uri.fsPath);
            if (tool.config().get('checkOnSave') !== false) checker.schedule(document, 0);
        }),

        vscode.workspace.onDidOpenTextDocument((document) => checker.schedule(document, 0)),
        vscode.workspace.onDidCloseTextDocument((document) => {
            checker.forget(document);
            workspace.invalidate(document.uri.fsPath);
        }),

        vscode.window.onDidChangeActiveTextEditor((editor) => updateStatus(editor?.document)),

        vscode.workspace.onDidChangeConfiguration((event) => {
            if (!event.affectsConfiguration('jaithon')) return;
            tool.resetBinaryWarning();
            workspace.invalidateAll();
            for (const document of vscode.workspace.textDocuments) {
                if (document.languageId === 'jaithon') checker.schedule(document, 0);
            }
        }),
    );

    // Files changed outside the editor — a rebuild, a branch switch — invalidate
    // the index, since a definition may now live somewhere else.
    const watcher = vscode.workspace.createFileSystemWatcher('**/*.jai');
    context.subscriptions.push(
        watcher,
        watcher.onDidChange((uri) => workspace.invalidate(uri.fsPath)),
        watcher.onDidCreate((uri) => workspace.invalidate(uri.fsPath)),
        watcher.onDidDelete((uri) => workspace.invalidate(uri.fsPath)),
    );

    vscode.workspace.textDocuments.forEach((document) => checker.schedule(document, 0));
    updateStatus(vscode.window.activeTextEditor?.document);
}

function deactivate() {
    tool.cleanup();
}

module.exports = { activate, deactivate };
