// Completion and signature help.
//
// Both fire on text that does not parse — `xs.` and `f(` are syntax errors, so
// the tree they need does not exist yet. The analyser keeps the last tree that
// did parse, and every byte before the cursor still has the offset it had then,
// so the receiver's type and the callee's identity are both still answerable.
// What cannot come from the tree is read off the line directly.

const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const tool = require('./tool');
const builtins = require('./builtins');
const { baseTypeName, moduleOfType } = require('./analysis');
const { byteOffsetAt, scopeIdAt } = require('./navigation');

const ITEM_KIND = {
    function: vscode.CompletionItemKind.Function,
    method: vscode.CompletionItemKind.Method,
    class: vscode.CompletionItemKind.Class,
    trait: vscode.CompletionItemKind.Interface,
    enum: vscode.CompletionItemKind.Enum,
    variant: vscode.CompletionItemKind.EnumMember,
    field: vscode.CompletionItemKind.Field,
    variable: vscode.CompletionItemKind.Variable,
    parameter: vscode.CompletionItemKind.Variable,
    self: vscode.CompletionItemKind.Variable,
    typealias: vscode.CompletionItemKind.TypeParameter,
    typeParameter: vscode.CompletionItemKind.TypeParameter,
    module: vscode.CompletionItemKind.Module,
    import: vscode.CompletionItemKind.Reference,
};

function itemFor(symbol, sortGroup) {
    const item = new vscode.CompletionItem(
        symbol.name, ITEM_KIND[symbol.kind] ?? vscode.CompletionItemKind.Variable);
    item.detail = symbol.detail || symbol.type || '';
    if (symbol.doc) item.documentation = new vscode.MarkdownString(symbol.doc);
    item.sortText = `${sortGroup}${symbol.name}`;

    if (['function', 'method'].includes(symbol.kind)) {
        const takesArguments = (symbol.params || []).some((param) => param.name !== 'self');
        item.insertText = new vscode.SnippetString(`${symbol.name}(${takesArguments ? '$0' : ''})`);
    }
    return item;
}

// ---------------------------------------------------------------------------
// Reading backwards from the cursor
// ---------------------------------------------------------------------------

function memberDot(line, character) {
    let index = character - 1;
    while (index >= 0 && /[A-Za-z0-9_]/.test(line[index])) index--;
    if (index < 0 || line[index] !== '.') return -1;
    if (index > 0 && /[0-9]/.test(line[index - 1]) && !/[A-Za-z_)\]]/.test(line[index - 1])) return -1;
    return index;
}

function chainStart(line, end) {
    let index = end;
    let depth = 0;
    while (index > 0) {
        const ch = line[index - 1];
        if (ch === ')' || ch === ']') { depth++; index--; continue; }
        if (ch === '(' || ch === '[') {
            if (depth === 0) break;
            depth--; index--; continue;
        }
        if (depth > 0) { index--; continue; }
        if (/[A-Za-z0-9_.]/.test(ch)) { index--; continue; }
        break;
    }
    return index;
}

/**
 * The open paren of the call the cursor sits inside, with its argument index.
 * Bounded, because a cursor that is inside no call at all would otherwise scan
 * the whole file on every keystroke.
 */
const CALL_SCAN_LIMIT = 8192;

function enclosingCall(text, offset) {
    let depth = 0;
    let commas = 0;
    const stop = Math.max(0, offset - CALL_SCAN_LIMIT);
    for (let index = offset - 1; index >= stop; index--) {
        const ch = text[index];
        if (ch === ')' || ch === ']' || ch === '}') { depth++; continue; }
        if (ch === '(' && depth === 0) return { open: index, argument: commas };
        if (ch === '[' || ch === '{') { if (depth === 0) return null; depth--; continue; }
        if (ch === '(') { depth--; continue; }
        if (ch === ',' && depth === 0) { commas++; continue; }
        if (ch === '\n' && depth === 0) {
            continue;
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Module paths
// ---------------------------------------------------------------------------

function dottedModules(document, fromFile) {
    const out = new Map();
    const seen = new Set();

    const visit = (dir, prefix, depth) => {
        if (depth > 4) return;
        let real;
        try { real = fs.realpathSync(dir); } catch { return; }
        if (seen.has(real)) return;
        seen.add(real);

        let entries;
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
        for (const entry of entries) {
            if (entry.name.startsWith('.') || entry.name === '__jaicache__') continue;
            const full = path.join(dir, entry.name);

            let isDirectory = entry.isDirectory();
            if (entry.isSymbolicLink()) {
                try { isDirectory = fs.statSync(full).isDirectory(); } catch { continue; }
            }
            if (isDirectory) {
                visit(full, `${prefix}${entry.name}.`, depth + 1);
            } else if (entry.name.endsWith('.jai')) {
                const base = entry.name.slice(0, -4);
                const dotted = base === 'mod' ? prefix.replace(/\.$/, '') : `${prefix}${base}`;
                if (dotted && !out.has(dotted)) out.set(dotted, full);
            }
        }
    };
    for (const dir of tool.searchPath(document, path.dirname(fromFile))) visit(dir, '', 0);
    return out;
}

// ---------------------------------------------------------------------------
// Completion
// ---------------------------------------------------------------------------

function completionProvider(workspace) {
    return {
        async provideCompletionItems(document, position, token) {
            const line = document.lineAt(position.line).text;
            const before = line.slice(0, position.character);

            const importMatch = /^\s*(?:import|from)\s+([A-Za-z0-9_.]*)$/.exec(before);
            if (importMatch) return moduleCompletions(document, importMatch[1]);

            const fromMatch = /^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+([A-Za-z0-9_,\s]*)$/.exec(before);
            if (fromMatch) return exportCompletions(workspace, document, fromMatch[1], token);

            const analysis = await workspace.analyze(document, token);
            if (!analysis) return [];

            const dot = memberDot(before, position.character);
            if (dot >= 0) return memberCompletions(workspace, analysis, line, position, dot, token);

            return scopeCompletions(analysis, byteOffsetAt(analysis, position));
        },
    };
}

function moduleCompletions(document, prefix) {
    const modules = dottedModules(document, document.uri.fsPath);
    const out = [];
    for (const [dotted, file] of modules) {
        if (prefix && !dotted.startsWith(prefix.split('.').slice(0, -1).join('.'))) continue;
        const item = new vscode.CompletionItem(dotted, vscode.CompletionItemKind.Module);
        item.detail = vscode.workspace.asRelativePath(file);
        item.sortText = dotted.startsWith('std.') ? `1${dotted}` : `0${dotted}`;
        out.push(item);
    }
    return out;
}

async function exportCompletions(workspace, document, modulePath, token) {
    const file = workspace.moduleFile(modulePath, document.uri.fsPath);
    if (!file) return [];
    const analysis = await workspace.analyze(file, token);
    if (!analysis) return [];

    const out = [];
    for (const symbol of analysis.symbols) {
        if (symbol.scope !== analysis.moduleScope) continue;
        if (symbol.visibility !== 'pub') continue;
        out.push(itemFor(symbol, '0'));
    }
    return out;
}

/**
 * The type of the expression the cursor is taking a member of. The receiver
 * ends where the dot begins, and that offset is unchanged by the dot itself, so
 * the type the checker recorded for it in the last parse still applies.
 */
function receiverType(analysis, lineNumber, lineText, dotColumn) {
    const start = chainStart(lineText, dotColumn);
    const startByte = byteOffsetAt(analysis, new vscode.Position(lineNumber, start));
    const dotByte = byteOffsetAt(analysis, new vscode.Position(lineNumber, dotColumn));

    const checked = analysis.types.get(`${startByte}:${dotByte}`)
        || analysis.narrowestType(Math.max(startByte, dotByte - 1))?.type;
    if (checked) return checked;

    // No checker type is ever available yet (no --dump-sema equivalent). A
    // bare identifier naming an import still resolves syntactically: it IS
    // its module, no inference needed.
    const identifier = lineText.slice(start, dotColumn);
    if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(identifier)) {
        const symbol = analysis.lookup(identifier, scopeIdAt(analysis, startByte));
        if (symbol?.kind === 'module') return `module ${symbol.modulePath}`;
    }
    return null;
}

async function memberCompletions(workspace, analysis, line, position, dotColumn, token) {
    const type = receiverType(analysis, position.line, line, dotColumn);
    const out = [];

    const moduleName = moduleOfType(type);
    if (moduleName) {
        const file = workspace.moduleFile(moduleName, analysis.fsPath);
        const other = file ? await workspace.analyze(file, token) : null;
        if (other) {
            for (const symbol of other.symbols) {
                if (symbol.scope !== other.moduleScope || symbol.visibility !== 'pub') continue;
                out.push(itemFor(symbol, '0'));
            }
        }
        return out;
    }

    const owner = await workspace.typeSymbol(analysis, type, token);
    if (owner) {
        let current = owner;
        const seen = new Set();
        let group = 0;
        while (current) {
            for (const member of current.analysis.membersOf(current.symbol)) {
                if (member.kind === 'self') continue;
                out.push(itemFor(member, String(group)));
            }
            const parent = current.symbol.superclass;
            if (!parent || seen.has(parent)) break;
            seen.add(parent);
            group += 1;
            current = await workspace.typeSymbol(current.analysis, parent, token);
        }
    }

    for (const name of builtins.methodsFor(type)) {
        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Method);
        item.detail = `${baseTypeName(type) || type}.${name}()`;
        item.insertText = new vscode.SnippetString(`${name}($0)`);
        item.sortText = `5${name}`;
        out.push(item);
    }
    return out;
}

function scopeCompletions(analysis, offset) {
    const out = [];
    const seen = new Set();

    let scopeId = analysis.moduleScope;
    for (const scope of analysis.scopes) {
        if (offset < scope.start || offset > scope.end) continue;
        if (!scope.lexical) continue;
        if (analysis.scopes[scopeId].start <= scope.start) scopeId = scope.id;
    }

    let depth = 0;
    let scope = analysis.scopes[scopeId];
    while (scope) {
        for (const index of scope.names.values()) {
            const symbol = analysis.symbols[index];
            if (seen.has(symbol.name)) continue;
            seen.add(symbol.name);
            out.push(itemFor(symbol, String(depth)));
        }
        scope = scope.parent === null ? null : analysis.scopes[scope.parent];
        depth = Math.min(depth + 1, 8);
    }

    const enclosing = analysis.enclosing(offset);

    if (enclosing && enclosing.members !== undefined) {
        for (const name of builtins.DUNDERS) {
            if (analysis.memberNamed(enclosing, name)) continue;
            const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Method);
            item.detail = 'operator method';
            item.insertText = new vscode.SnippetString(`fn ${name}(self${name === '__init__' ? ', $1' : ''}) {\n    $0\n}`);
            item.sortText = `4${name}`;
            out.push(item);
        }
    }

    if (enclosing && enclosing.container !== null) {
        for (const member of analysis.membersOf(analysis.symbols[enclosing.container])) {
            if (seen.has(`self.${member.name}`)) continue;
            seen.add(`self.${member.name}`);
            const item = itemFor(member, '3');
            item.label = { label: member.name, description: `self.${member.name}` };
            item.filterText = member.name;
            item.insertText = `self.${member.name}`;
            out.push(item);
        }
    }

    for (const [name, builtin] of Object.entries(builtins.FUNCTIONS)) {
        if (seen.has(name)) continue;
        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Function);
        item.detail = builtin.signature;
        item.documentation = new vscode.MarkdownString(builtin.doc);
        item.insertText = new vscode.SnippetString(`${name}($0)`);
        item.sortText = `7${name}`;
        out.push(item);
    }
    for (const name of Object.keys(builtins.EXCEPTIONS)) {
        if (seen.has(name)) continue;
        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Class);
        item.detail = 'runtime exception';
        item.sortText = `8${name}`;
        out.push(item);
    }
    for (const name of [...builtins.PRIMITIVE_TYPES, ...builtins.PRELUDE_TRAITS]) {
        if (seen.has(name)) continue;
        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Class);
        item.sortText = `8${name}`;
        out.push(item);
    }
    for (const keyword of builtins.KEYWORDS) {
        const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
        item.sortText = `9${keyword}`;
        out.push(item);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Signature help
// ---------------------------------------------------------------------------

function signatureHelpProvider(workspace) {
    return {
        async provideSignatureHelp(document, position, token) {
            const analysis = await workspace.analyze(document, token);
            if (!analysis) return null;

            const text = document.getText();
            const call = enclosingCall(text, document.offsetAt(position));
            if (!call) return null;

            const line = document.positionAt(call.open).line;
            const column = call.open - document.offsetAt(new vscode.Position(line, 0));
            const lineText = document.lineAt(line).text;

            const nameEnd = column;
            let nameStart = nameEnd;
            while (nameStart > 0 && /[A-Za-z0-9_]/.test(lineText[nameStart - 1])) nameStart--;
            const name = lineText.slice(nameStart, nameEnd);
            if (!name) return null;

            const dot = nameStart > 0 && lineText[nameStart - 1] === '.' ? nameStart - 1 : -1;
            let resolved = null;

            if (dot >= 0) {
                const type = receiverType(analysis, line, lineText, dot);
                const owner = await workspace.typeSymbol(analysis, type, token);
                if (owner) {
                    const member = owner.analysis.memberNamed(owner.symbol, name);
                    if (member) resolved = { analysis: owner.analysis, symbol: member };
                }
                const moduleName = moduleOfType(type);
                if (!resolved && moduleName) {
                    const file = workspace.moduleFile(moduleName, analysis.fsPath);
                    const other = file ? await workspace.analyze(file, token) : null;
                    const found = other && other.lookup(name, other.moduleScope);
                    if (found) resolved = { analysis: other, symbol: found };
                }
            } else {
                const startByte = byteOffsetAt(analysis, new vscode.Position(line, nameStart));
                const scopeRef = { name, scope: scopeIdAt(analysis, startByte), kind: 'value' };
                resolved = await workspace.definitionOf(analysis, scopeRef, token);
            }

            if (!resolved) {
                const builtin = builtins.FUNCTIONS[name];
                if (!builtin) return null;
                const help = new vscode.SignatureHelp();
                help.signatures = [new vscode.SignatureInformation(
                    builtin.signature, new vscode.MarkdownString(builtin.doc))];
                help.activeSignature = 0;
                help.activeParameter = call.argument;
                return help;
            }

            let symbol = resolved.symbol;
            if (symbol.members !== undefined) {
                const init = resolved.analysis.memberNamed(symbol, 'init');
                if (!init) return null;
                symbol = init;
            }
            if (!symbol.params) return null;

            const params = symbol.params.filter((param) => param.name !== 'self');
            const information = new vscode.SignatureInformation(
                symbol.detail || `${symbol.name}(…)`,
                symbol.doc ? new vscode.MarkdownString(symbol.doc) : undefined);
            information.parameters = params.map(
                (param) => new vscode.ParameterInformation(param.name));

            const help = new vscode.SignatureHelp();
            help.signatures = [information];
            help.activeSignature = 0;
            help.activeParameter = Math.min(call.argument, Math.max(params.length - 1, 0));
            return help;
        },
    };
}

// ---------------------------------------------------------------------------

function register(context, workspace) {
    const selector = { language: 'jaithon', scheme: 'file' };
    context.subscriptions.push(
        vscode.languages.registerCompletionItemProvider(
            selector, completionProvider(workspace), '.', ' '),
        vscode.languages.registerSignatureHelpProvider(
            selector, signatureHelpProvider(workspace), '(', ',', ':'),
    );
}

module.exports = { register, memberDot, chainStart, enclosingCall };
