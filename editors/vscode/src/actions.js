// Formatting and code actions.

const vscode = require('vscode');
const fs = require('fs');
const os = require('os');
const path = require('path');
const tool = require('./tool');
const { byteOffsetAt } = require('./navigation');

// ---------------------------------------------------------------------------
// Formatting
//
// `jaithon fmt` rewrites a file in place and takes no options, so formatting is
// a round trip through a scratch copy. The result is trimmed to the lines that
// actually changed: replacing the whole document would move every cursor,
// folding state and decoration in the file for the sake of one re-indented line.
// ---------------------------------------------------------------------------

let scratchTick = 0;

async function formatted(document, token) {
    const scratch = path.join(os.tmpdir(), `jaithon-fmt-${process.pid}-${scratchTick++}.jai`);
    try {
        fs.writeFileSync(scratch, document.getText(), 'utf8');
        const result = await tool.run(['fmt', '--color=never', scratch],
                                      { cwd: path.dirname(document.uri.fsPath), document, token });
        if (result.spawnFailed) return null;
        const text = fs.readFileSync(scratch, 'utf8');
        // A parse error leaves the file untouched and reports on stderr; there
        // is nothing to apply, and overwriting with the original is not an edit.
        return text === document.getText() ? null : text;
    } catch {
        return null;
    } finally {
        try { fs.unlinkSync(scratch); } catch { /* never written */ }
    }
}

/** One replacement covering exactly the run of lines that differ. */
function minimalEdit(document, next) {
    const previous = document.getText();
    if (previous === next) return [];

    const before = previous.split('\n');
    const after = next.split('\n');

    let head = 0;
    while (head < before.length && head < after.length && before[head] === after[head]) head++;

    let tail = 0;
    while (tail < before.length - head && tail < after.length - head
           && before[before.length - 1 - tail] === after[after.length - 1 - tail]) tail++;

    const start = new vscode.Position(head, 0);
    const endLine = before.length - tail;
    const end = endLine >= before.length
        ? document.lineAt(before.length - 1).range.end
        : new vscode.Position(endLine, 0);

    const replacement = after.slice(head, after.length - tail).join('\n')
        + (endLine >= before.length ? '' : '\n');
    return [vscode.TextEdit.replace(new vscode.Range(start, end), replacement)];
}

function formattingProvider() {
    return {
        async provideDocumentFormattingEdits(document, options, token) {
            const next = await formatted(document, token);
            return next === null ? [] : minimalEdit(document, next);
        },
    };
}

function rangeFormattingProvider() {
    return {
        async provideDocumentRangeFormattingEdits(document, range, options, token) {
            // The formatter is canonical and whole-file; the honest thing to do
            // for a selection is to format everything and hand back only the
            // part of the result that falls inside it.
            const next = await formatted(document, token);
            if (next === null) return [];
            return minimalEdit(document, next).filter((edit) => edit.range.intersection(range));
        },
    };
}

// ---------------------------------------------------------------------------
// Code actions
// ---------------------------------------------------------------------------

const DID_YOU_MEAN = /did you mean `([^`]+)`/;

/** Where a new import belongs: after the last one, else above the first code. */
function importAnchor(document) {
    let line = 0;
    for (let index = 0; index < document.lineCount; index++) {
        const text = document.lineAt(index).text.trim();
        if (/^(import|from)\s/.test(text)) line = index + 1;
        else if (text !== '' && !text.startsWith('#') && line === 0) return { line: index, blankAfter: true };
    }
    return { line, blankAfter: false };
}

const EXPORT_RE = (name) =>
    new RegExp(`^\\s*pub\\s+(?:static\\s+)?(fn|class|trait|enum|type|const|let|var)\\s+${name}(?![A-Za-z0-9_])`, 'm');

/** Modules that export `name`, as dotted paths. */
async function exporters(workspace, document, name, token) {
    const out = [];
    const pattern = EXPORT_RE(name);
    for (const fsPath of await workspace.candidates(name, token)) {
        if (fsPath === document.uri.fsPath) continue;
        let text;
        const open = workspace.documentFor(fsPath);
        if (open) text = open.getText();
        else { try { text = fs.readFileSync(fsPath, 'utf8'); } catch { continue; } }
        if (!pattern.test(text)) continue;

        const dotted = dottedNameFor(fsPath, document);
        if (dotted) out.push(dotted);
    }
    return out;
}

/** Turn a file path back into the dotted name an import would spell. */
function dottedNameFor(fsPath, document) {
    for (const dir of tool.searchPath(document, path.dirname(document.uri.fsPath))) {
        const relative = path.relative(dir, fsPath);
        if (relative.startsWith('..') || path.isAbsolute(relative)) continue;
        const withoutExtension = relative.replace(/\.jai$/, '').replace(/[\\/]mod$/, '');
        return withoutExtension.split(path.sep).join('.');
    }
    return null;
}

function codeActionProvider(workspace) {
    return {
        async provideCodeActions(document, range, context, token) {
            const actions = [];

            for (const diagnostic of context.diagnostics) {
                if (diagnostic.source !== 'jaithon') continue;
                const text = document.getText(diagnostic.range);

                const suggestion = DID_YOU_MEAN.exec(diagnostic.jaithon?.help || '');
                if (suggestion) {
                    actions.push(replacement(document, diagnostic,
                        `Change to \`${suggestion[1]}\``, diagnostic.range, suggestion[1]));
                }

                if (diagnostic.code === 'W0101') {
                    actions.push(replacement(document, diagnostic,
                        `Prefix with \`_\``, diagnostic.range, `_${text}`));
                }

                if (/`var`/.test(diagnostic.jaithon?.help || '')) {
                    const line = document.lineAt(diagnostic.range.start.line);
                    const at = line.text.indexOf('let');
                    if (at >= 0) {
                        const letRange = new vscode.Range(line.lineNumber, at, line.lineNumber, at + 3);
                        actions.push(replacement(document, diagnostic,
                            'Declare with `var`', letRange, 'var'));
                    }
                }

                if (diagnostic.code === 'E0200' && /^[A-Za-z_][A-Za-z0-9_]*$/.test(text)) {
                    for (const module of await exporters(workspace, document, text, token)) {
                        const action = new vscode.CodeAction(
                            `Import \`${text}\` from \`${module}\``,
                            vscode.CodeActionKind.QuickFix);
                        const anchor = importAnchor(document);
                        action.edit = new vscode.WorkspaceEdit();
                        action.edit.insert(document.uri, new vscode.Position(anchor.line, 0),
                                           `from ${module} import ${text}\n${anchor.blankAfter ? '\n' : ''}`);
                        action.diagnostics = [diagnostic];
                        actions.push(action);
                    }
                }
            }

            const analysis = await workspace.analyze(document, token);
            if (analysis) {
                actions.push(...(await refactors(workspace, analysis, document, range, token)));
                const organize = organizeImports(analysis, document);
                if (organize) actions.push(organize);
            }
            return actions;
        },
    };
}

function replacement(document, diagnostic, title, range, text) {
    const action = new vscode.CodeAction(title, vscode.CodeActionKind.QuickFix);
    action.edit = new vscode.WorkspaceEdit();
    action.edit.replace(document.uri, range, text);
    action.diagnostics = [diagnostic];
    action.isPreferred = true;
    return action;
}

async function refactors(workspace, analysis, document, range, token) {
    const out = [];
    const offset = byteOffsetAt(analysis, range.start);
    const hit = analysis.at(offset);

    const symbol = hit?.symbol
        || (hit?.ref ? (await workspace.definitionOf(analysis, hit.ref, token))?.symbol : null);

    // The checker already knows the type; writing it down is a pure annotation.
    if (symbol && ['variable', 'parameter'].includes(symbol.kind)
        && !symbol.declaredType && symbol.type && symbol.type !== 'any'
        && !symbol.type.startsWith('module ')) {
        const action = new vscode.CodeAction(
            `Annotate as \`${symbol.type}\``, vscode.CodeActionKind.RefactorRewrite);
        action.edit = new vscode.WorkspaceEdit();
        action.edit.insert(document.uri, analysis.offsets.position(symbol.end), `: ${symbol.type}`);
        out.push(action);
    }

    if (symbol && symbol.kind === 'variable' && symbol.mutable !== undefined) {
        const line = document.lineAt(analysis.offsets.position(symbol.fullStart).line);
        const from = symbol.mutable ? 'var' : 'let';
        const to = symbol.mutable ? 'let' : 'var';
        const at = line.text.indexOf(from);
        if (at >= 0) {
            const action = new vscode.CodeAction(
                `Change to \`${to}\``, vscode.CodeActionKind.RefactorRewrite);
            action.edit = new vscode.WorkspaceEdit();
            action.edit.replace(document.uri,
                new vscode.Range(line.lineNumber, at, line.lineNumber, at + 3), to);
            out.push(action);
        }
    }

    // A class that names a trait owes it every method the trait declares.
    const owner = symbol && symbol.members !== undefined ? symbol : null;
    if (owner && (owner.traits || []).length) {
        const missing = [];
        for (const traitName of owner.traits) {
            const found = await workspace.typeSymbol(analysis, traitName, token);
            if (!found) continue;
            for (const method of found.analysis.membersOf(found.symbol)) {
                if (method.kind !== 'method') continue;
                if (analysis.memberNamed(owner, method.name)) continue;
                missing.push({ trait: traitName, method });
            }
        }
        if (missing.length) {
            const end = analysis.offsets.position(owner.fullEnd);
            const stubs = missing.map(({ trait, method }) => {
                const params = (method.params || []).map((p) => p.name).join(', ');
                const signature = method.detail || `fn ${method.name}(${params})`;
                return `\n    #: Required by ${trait}.\n    ${signature.replace(/^fn /, 'fn ')} {\n        throw RuntimeError("${method.name} is not implemented")\n    }\n`;
            }).join('');

            const action = new vscode.CodeAction(
                missing.length === 1
                    ? `Implement \`${missing[0].method.name}\``
                    : `Implement ${missing.length} missing trait methods`,
                vscode.CodeActionKind.QuickFix);
            action.edit = new vscode.WorkspaceEdit();
            action.edit.insert(document.uri, new vscode.Position(end.line, Math.max(0, end.character - 1)), stubs);
            out.push(action);
        }
    }
    return out;
}

function organizeImports(analysis, document) {
    const lines = [];
    for (let index = 0; index < document.lineCount; index++) {
        const text = document.lineAt(index).text;
        if (/^\s*(import|from)\s/.test(text)) lines.push({ index, text: text.trim() });
        else if (lines.length && text.trim() !== '') break;
    }
    if (lines.length < 2) return null;

    const sorted = [...new Set(lines.map((line) => line.text))].sort((a, b) => {
        const rank = (text) => (text.includes(' .') || /\s\./.test(text) ? 2 : text.includes('std.') ? 0 : 1);
        return rank(a) - rank(b) || a.localeCompare(b);
    });
    if (sorted.join('\n') === lines.map((line) => line.text).join('\n')) return null;

    const action = new vscode.CodeAction('Organize imports', vscode.CodeActionKind.SourceOrganizeImports);
    action.edit = new vscode.WorkspaceEdit();
    action.edit.replace(document.uri,
        new vscode.Range(lines[0].index, 0, lines[lines.length - 1].index,
                         document.lineAt(lines[lines.length - 1].index).text.length),
        sorted.join('\n'));
    return action;
}

// ---------------------------------------------------------------------------

function register(context, workspace) {
    const selector = { language: 'jaithon', scheme: 'file' };
    context.subscriptions.push(
        vscode.languages.registerDocumentFormattingEditProvider(selector, formattingProvider()),
        vscode.languages.registerDocumentRangeFormattingEditProvider(selector, rangeFormattingProvider()),
        vscode.languages.registerCodeActionsProvider(selector, codeActionProvider(workspace), {
            providedCodeActionKinds: [
                vscode.CodeActionKind.QuickFix,
                vscode.CodeActionKind.RefactorRewrite,
                vscode.CodeActionKind.SourceOrganizeImports,
            ],
        }),
    );
}

module.exports = { register, formatted, minimalEdit };
