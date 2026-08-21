// Moving around: definitions, references, rename, outlines, call hierarchy.

const vscode = require('vscode');
const fs = require('fs');
const path = require('path');
const builtins = require('./builtins');
const { walk } = require('./analysis');

const SYMBOL_KIND = {
    function: vscode.SymbolKind.Function,
    method: vscode.SymbolKind.Method,
    class: vscode.SymbolKind.Class,
    trait: vscode.SymbolKind.Interface,
    enum: vscode.SymbolKind.Enum,
    variant: vscode.SymbolKind.EnumMember,
    field: vscode.SymbolKind.Field,
    variable: vscode.SymbolKind.Variable,
    parameter: vscode.SymbolKind.Variable,
    self: vscode.SymbolKind.Variable,
    typealias: vscode.SymbolKind.TypeParameter,
    typeParameter: vscode.SymbolKind.TypeParameter,
    module: vscode.SymbolKind.Module,
    import: vscode.SymbolKind.Module,
};

function symbolKind(symbol) {
    return SYMBOL_KIND[symbol.kind] ?? vscode.SymbolKind.Variable;
}

function byteOffsetAt(analysis, position) {
    const starts = analysis.offsets.lineStarts;
    if (position.line >= starts.length) return analysis.offsets.byte(analysis.text.length);
    return analysis.offsets.byte(starts[position.line] + position.character);
}

/** The deepest lexical scope whose extent contains `offset`. */
function scopeIdAt(analysis, offset) {
    let best = analysis.moduleScope;
    for (const scope of analysis.scopes) {
        if (offset < scope.start || offset > scope.end) continue;
        if (analysis.scopes[best].start <= scope.start) best = scope.id;
    }
    return best;
}

async function contextAt(workspace, document, position, token) {
    const analysis = await workspace.analyze(document, token);
    if (!analysis) return null;
    const offset = byteOffsetAt(analysis, position);
    return { analysis, offset, hit: analysis.at(offset) };
}

function locationOf(analysis, symbol) {
    return new vscode.Location(analysis.uri, analysis.offsets.range(symbol.start, symbol.end));
}

// ---------------------------------------------------------------------------
// Definition, type definition, implementations
// ---------------------------------------------------------------------------

function definitionProvider(workspace) {
    return {
        async provideDefinition(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const { analysis, hit } = context;

            const modulePath = hit.ref?.modulePath || hit.symbol?.modulePath;
            if (modulePath && (hit.ref?.kind === 'module-path' || hit.symbol?.kind === 'module')) {
                const file = workspace.moduleFile(modulePath, analysis.fsPath);
                if (!file) return null;
                return new vscode.Location(vscode.Uri.file(file), new vscode.Position(0, 0));
            }

            if (hit.symbol) {
                const followed = await workspace.follow(analysis, hit.symbol, token);
                if (!followed) return null;
                if (followed.analysis === analysis && followed.symbol === hit.symbol) return null;
                return locationOf(followed.analysis, followed.symbol);
            }

            const found = await workspace.definitionOf(analysis, hit.ref, token);
            if (!found) return null;
            return new vscode.Location(
                found.analysis.uri,
                found.analysis.offsets.range(found.symbol.start, found.symbol.end));
        },
    };
}

function typeDefinitionProvider(workspace) {
    return {
        async provideTypeDefinition(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const { analysis, hit, offset } = context;

            const type = hit.symbol?.type || hit.symbol?.declaredType
                || analysis.narrowestType(offset)?.type;
            const found = await workspace.typeSymbol(analysis, type, token);
            if (!found) return null;
            return locationOf(found.analysis, found.symbol);
        },
    };
}

function implementationProvider(workspace) {
    return {
        async provideImplementation(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const { analysis, hit } = context;

            const target = hit.symbol || (await workspace.definitionOf(analysis, hit.ref, token))?.symbol;
            if (!target) return null;

            const owner = target.kind === 'method' && target.container !== null
                ? analysis.symbols[target.container] : target;
            if (!owner || owner.members === undefined) return null;

            const out = [];
            for (const fsPath of await workspace.candidates(owner.name, token)) {
                const other = await workspace.analyze(fsPath, token);
                if (!other) continue;
                for (const symbol of other.symbols) {
                    if (symbol.members === undefined) continue;
                    const inherits = symbol.superclass === owner.name
                        || (symbol.traits || []).includes(owner.name);
                    if (!inherits) continue;
                    const member = target.kind === 'method'
                        ? other.memberNamed(symbol, target.name) : null;
                    out.push(locationOf(other, member || symbol));
                }
            }
            return out;
        },
    };
}

// ---------------------------------------------------------------------------
// References and rename
// ---------------------------------------------------------------------------

async function referencesTo(workspace, home, symbol, token) {
    if (symbol.kind === 'import' && symbol.aliased) {
        return home.refs
            .filter((ref) => ref.name === symbol.name && ref.kind === 'value'
                && home.lookup(ref.name, ref.scope)?.index === symbol.index)
            .map((ref) => ({ analysis: home, start: ref.start, end: ref.end }));
    }

    const scoped = ['variable', 'parameter', 'self', 'typeParameter'].includes(symbol.kind);
    const files = scoped
        ? [home.fsPath]
        : await workspace.candidates(symbol.name, token);

    const out = [];
    for (const fsPath of files) {
        if (token?.isCancellationRequested) break;
        const analysis = fsPath === home.fsPath ? home : await workspace.analyze(fsPath, token);
        if (!analysis) continue;

        for (const ref of analysis.refs) {
            if (ref.name !== symbol.name) continue;
            const bound = ref.kind === 'value' ? analysis.lookup(ref.name, ref.scope) : null;
            if (bound && bound.aliased && bound.index !== symbol.index) continue;

            const found = await workspace.definitionOf(analysis, ref, token);
            if (!found) continue;
            if (found.analysis.fsPath !== home.fsPath || found.symbol.index !== symbol.index) continue;
            out.push({ analysis, start: ref.start, end: ref.end });
        }
        for (const other of analysis.symbols) {
            if (other.kind !== 'import' || other.aliased || other.name !== symbol.name) continue;
            const followed = await workspace.follow(analysis, other, token);
            if (followed && followed.analysis.fsPath === home.fsPath
                && followed.symbol.index === symbol.index) {
                out.push({ analysis, start: other.start, end: other.end });
            }
        }
    }
    return out;
}

function referenceProvider(workspace) {
    return {
        async provideReferences(document, position, context, token) {
            const found = await contextAt(workspace, document, position, token);
            if (!found || !found.hit) return null;

            const resolved = found.hit.symbol
                ? { analysis: found.analysis, symbol: found.hit.symbol }
                : await workspace.definitionOf(found.analysis, found.hit.ref, token);
            if (!resolved) return null;

            const uses = await referencesTo(workspace, resolved.analysis, resolved.symbol, token);
            const out = uses.map((use) => new vscode.Location(
                use.analysis.uri, use.analysis.offsets.range(use.start, use.end)));
            if (context.includeDeclaration) out.unshift(locationOf(resolved.analysis, resolved.symbol));
            return out;
        },
    };
}

function renameTarget(analysis, hit) {
    if (hit.symbol) return { analysis, symbol: hit.symbol };
    if (hit.ref?.kind !== 'value') return null;
    const bound = analysis.lookup(hit.ref.name, hit.ref.scope);
    return bound && bound.aliased ? { analysis, symbol: bound } : null;
}

function renameProvider(workspace) {
    return {
        async prepareRename(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) throw new Error('Nothing to rename here.');
            const { analysis, hit } = context;

            if (hit.ref?.kind === 'module-path') throw new Error('A module path is a file name; rename the file instead.');

            const resolved = renameTarget(analysis, hit)
                || await workspace.definitionOf(analysis, hit.ref, token);
            if (!resolved) throw new Error('That name is not declared in this workspace.');
            if (builtins.FUNCTIONS[resolved.symbol.name] || builtins.isException(resolved.symbol.name)) {
                throw new Error(`\`${resolved.symbol.name}\` is defined by the runtime and cannot be renamed.`);
            }

            const range = analysis.offsets.range(hit.start, hit.end);
            return { range, placeholder: resolved.symbol.name };
        },

        async provideRenameEdits(document, position, newName, token) {
            if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(newName)) {
                throw new Error(`\`${newName}\` is not a valid Jaithon identifier.`);
            }
            if (builtins.KEYWORDS.includes(newName)) {
                throw new Error(`\`${newName}\` is a keyword.`);
            }

            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const resolved = renameTarget(context.analysis, context.hit)
                || await workspace.definitionOf(context.analysis, context.hit.ref, token);
            if (!resolved) return null;

            const edit = new vscode.WorkspaceEdit();
            const seen = new Set();
            const add = (analysis, start, end) => {
                const key = `${analysis.fsPath}:${start}`;
                if (seen.has(key)) return;
                seen.add(key);
                edit.replace(analysis.uri, analysis.offsets.range(start, end), newName);
            };

            add(resolved.analysis, resolved.symbol.start, resolved.symbol.end);
            for (const use of await referencesTo(workspace, resolved.analysis, resolved.symbol, token)) {
                add(use.analysis, use.start, use.end);
            }
            return edit;
        },
    };
}

function highlightProvider(workspace) {
    return {
        async provideDocumentHighlights(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const { analysis, hit } = context;

            const resolved = hit.symbol
                ? { analysis, symbol: hit.symbol }
                : await workspace.definitionOf(analysis, hit.ref, token);
            if (!resolved) return null;

            const out = [];
            if (resolved.analysis.fsPath === analysis.fsPath) {
                out.push(new vscode.DocumentHighlight(
                    analysis.offsets.range(resolved.symbol.start, resolved.symbol.end),
                    vscode.DocumentHighlightKind.Write));
            }
            for (const ref of analysis.refs) {
                if (ref.name !== resolved.symbol.name) continue;
                const found = await workspace.definitionOf(analysis, ref, token);
                if (!found) continue;
                if (found.analysis.fsPath !== resolved.analysis.fsPath) continue;
                if (found.symbol.index !== resolved.symbol.index) continue;
                out.push(new vscode.DocumentHighlight(
                    analysis.offsets.range(ref.start, ref.end),
                    vscode.DocumentHighlightKind.Read));
            }
            return out;
        },
    };
}

// ---------------------------------------------------------------------------
// Outlines
// ---------------------------------------------------------------------------

function documentSymbolProvider(workspace) {
    return {
        async provideDocumentSymbols(document, token) {
            const analysis = await workspace.analyze(document, token);
            if (!analysis) return [];

            const nodes = new Map();
            const roots = [];
            const interesting = new Set(['function', 'method', 'class', 'trait',
                                         'enum', 'variant', 'field', 'typealias']);

            for (const symbol of analysis.symbols) {
                if (!interesting.has(symbol.kind)) continue;
                const full = analysis.offsets.range(symbol.fullStart, symbol.fullEnd);
                const name = analysis.offsets.range(symbol.start, symbol.end);
                const item = new vscode.DocumentSymbol(
                    symbol.name, symbol.detail || '', symbolKind(symbol),
                    full, full.contains(name) ? name : full);
                nodes.set(symbol.index, item);
            }
            for (const symbol of analysis.symbols) {
                const item = nodes.get(symbol.index);
                if (!item) continue;
                const parent = symbol.container !== null ? nodes.get(symbol.container) : null;
                if (parent) parent.children.push(item); else roots.push(item);
            }

            for (const symbol of analysis.symbols) {
                if (symbol.kind !== 'variable' || symbol.scope !== analysis.moduleScope) continue;
                roots.push(new vscode.DocumentSymbol(
                    symbol.name, symbol.detail || '',
                    symbol.constant ? vscode.SymbolKind.Constant : vscode.SymbolKind.Variable,
                    analysis.offsets.range(symbol.fullStart, symbol.fullEnd),
                    analysis.offsets.range(symbol.start, symbol.end)));
            }
            roots.sort((a, b) => a.range.start.compareTo(b.range.start));
            return roots;
        },
    };
}

const DECLARATION_RE =
    /^([ \t]*)(?:(pub|prot)\s+)?(?:(static)\s+)?(fn|class|trait|enum|type|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)/;

const DECLARATION_KIND = {
    fn: vscode.SymbolKind.Function,
    class: vscode.SymbolKind.Class,
    trait: vscode.SymbolKind.Interface,
    enum: vscode.SymbolKind.Enum,
    type: vscode.SymbolKind.TypeParameter,
    const: vscode.SymbolKind.Constant,
    let: vscode.SymbolKind.Variable,
    var: vscode.SymbolKind.Variable,
};

function workspaceSymbolProvider(workspace) {
    return {
        async provideWorkspaceSymbols(query, token) {
            if (!query) return [];
            const pattern = new RegExp(
                [...query.replace(/[^A-Za-z0-9_]/g, '')].map((ch) => ch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*'),
                'i');

            const out = [];
            for (const fsPath of await workspace.files(token)) {
                if (token?.isCancellationRequested || out.length > 500) break;
                const document = workspace.documentFor(fsPath);
                let text;
                if (document) text = document.getText();
                else { try { text = fs.readFileSync(fsPath, 'utf8'); } catch { continue; } }

                const uri = vscode.Uri.file(fsPath);
                const lines = text.split('\n');
                let container = path.basename(fsPath, '.jai');
                for (let line = 0; line < lines.length; line++) {
                    const match = DECLARATION_RE.exec(lines[line]);
                    if (!match) continue;
                    const [, indent, , , keyword, name] = match;
                    if (indent.length === 0 && ['class', 'trait', 'enum'].includes(keyword)) container = name;
                    if (!pattern.test(name)) continue;
                    const column = lines[line].indexOf(name, match[0].length - name.length);
                    out.push(new vscode.SymbolInformation(
                        name, DECLARATION_KIND[keyword] || vscode.SymbolKind.Variable,
                        indent.length > 0 ? container : path.basename(fsPath, '.jai'),
                        new vscode.Location(uri, new vscode.Range(line, column, line, column + name.length))));
                }
            }
            return out;
        },
    };
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

function foldingProvider(workspace) {
    return {
        async provideFoldingRanges(document, context, token) {
            const analysis = await workspace.analyze(document, token);
            const ranges = [];
            const add = (startLine, endLine, kind) => {
                if (endLine > startLine) ranges.push(new vscode.FoldingRange(startLine, endLine, kind));
            };

            if (analysis) {
                const FOLDABLE = new Set([
                    'AST_BLOCK', 'AST_CLASS_DECL', 'AST_TRAIT_DECL', 'AST_ENUM_DECL',
                    'AST_MATCH', 'AST_MATCH_EXPR', 'AST_LIST_LIT', 'AST_DICT_LIT',
                    'AST_SET_LIT', 'AST_TRY',
                ]);
                walk(analysis.ast, (node) => {
                    if (!FOLDABLE.has(node.kind) || !node.span) return true;
                    add(analysis.offsets.position(node.span.start).line,
                        analysis.offsets.position(node.span.end).line - 1);
                    return true;
                });
            }

            let commentStart = -1;
            let importStart = -1;
            for (let line = 0; line < document.lineCount; line++) {
                const text = document.lineAt(line).text.trim();
                const isComment = text.startsWith('#');
                const isImport = /^(import|from)\s/.test(text);

                if (isComment && commentStart < 0) commentStart = line;
                if (!isComment && commentStart >= 0) {
                    add(commentStart, line - 1, vscode.FoldingRangeKind.Comment);
                    commentStart = -1;
                }
                if (isImport && importStart < 0) importStart = line;
                if (!isImport && importStart >= 0) {
                    add(importStart, line - 1, vscode.FoldingRangeKind.Imports);
                    importStart = -1;
                }
            }
            if (commentStart >= 0) add(commentStart, document.lineCount - 1, vscode.FoldingRangeKind.Comment);
            if (importStart >= 0) add(importStart, document.lineCount - 1, vscode.FoldingRangeKind.Imports);
            return ranges;
        },
    };
}

function selectionRangeProvider(workspace) {
    return {
        async provideSelectionRanges(document, positions, token) {
            const analysis = await workspace.analyze(document, token);
            if (!analysis) return null;

            return positions.map((position) => {
                const offset = byteOffsetAt(analysis, position);
                const enclosing = [];
                walk(analysis.ast, (node) => {
                    if (!node.span) return true;
                    if (offset < node.span.start || offset > node.span.end) return false;
                    enclosing.push(node.span);
                    return true;
                });
                enclosing.sort((a, b) => (b.end - b.start) - (a.end - a.start));

                let range = null;
                for (const span of enclosing) {
                    range = new vscode.SelectionRange(analysis.offsets.range(span.start, span.end), range);
                }
                return range || new vscode.SelectionRange(new vscode.Range(position, position));
            });
        },
    };
}

// ---------------------------------------------------------------------------
// Call hierarchy
// ---------------------------------------------------------------------------

function callItem(analysis, symbol) {
    const item = new vscode.CallHierarchyItem(
        symbolKind(symbol), symbol.name, symbol.detail || '',
        analysis.uri,
        analysis.offsets.range(symbol.fullStart, symbol.fullEnd),
        analysis.offsets.range(symbol.start, symbol.end));
    item.jaithon = { fsPath: analysis.fsPath, index: symbol.index };
    return item;
}

function callHierarchyProvider(workspace) {
    return {
        async prepareCallHierarchy(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context || !context.hit) return null;
            const resolved = context.hit.symbol
                ? { analysis: context.analysis, symbol: context.hit.symbol }
                : await workspace.definitionOf(context.analysis, context.hit.ref, token);
            if (!resolved || !['function', 'method'].includes(resolved.symbol.kind)) return null;
            return [callItem(resolved.analysis, resolved.symbol)];
        },

        async provideCallHierarchyIncomingCalls(item, token) {
            const home = await workspace.analyze(item.jaithon.fsPath, token);
            if (!home) return [];
            const symbol = home.symbols[item.jaithon.index];

            const byCaller = new Map();
            for (const use of await referencesTo(workspace, home, symbol, token)) {
                const caller = use.analysis.enclosing(use.start);
                if (!caller) continue;
                const key = `${use.analysis.fsPath}:${caller.index}`;
                if (!byCaller.has(key)) {
                    byCaller.set(key, new vscode.CallHierarchyIncomingCall(
                        callItem(use.analysis, caller), []));
                }
                byCaller.get(key).fromRanges.push(use.analysis.offsets.range(use.start, use.end));
            }
            return [...byCaller.values()];
        },

        async provideCallHierarchyOutgoingCalls(item, token) {
            const home = await workspace.analyze(item.jaithon.fsPath, token);
            if (!home) return [];
            const symbol = home.symbols[item.jaithon.index];

            const byCallee = new Map();
            for (const ref of home.refs) {
                if (ref.start < symbol.fullStart || ref.end > symbol.fullEnd) continue;
                if (ref.kind === 'module-path' || ref.kind === 'type') continue;
                const found = await workspace.definitionOf(home, ref, token);
                if (!found || !['function', 'method'].includes(found.symbol.kind)) continue;
                const key = `${found.analysis.fsPath}:${found.symbol.index}`;
                if (!byCallee.has(key)) {
                    byCallee.set(key, new vscode.CallHierarchyOutgoingCall(
                        callItem(found.analysis, found.symbol), []));
                }
                byCallee.get(key).fromRanges.push(home.offsets.range(ref.start, ref.end));
            }
            return [...byCallee.values()];
        },
    };
}

// ---------------------------------------------------------------------------

function register(context, workspace) {
    const selector = { language: 'jaithon', scheme: 'file' };
    context.subscriptions.push(
        vscode.languages.registerDefinitionProvider(selector, definitionProvider(workspace)),
        vscode.languages.registerTypeDefinitionProvider(selector, typeDefinitionProvider(workspace)),
        vscode.languages.registerImplementationProvider(selector, implementationProvider(workspace)),
        vscode.languages.registerReferenceProvider(selector, referenceProvider(workspace)),
        vscode.languages.registerRenameProvider(selector, renameProvider(workspace)),
        vscode.languages.registerDocumentHighlightProvider(selector, highlightProvider(workspace)),
        vscode.languages.registerDocumentSymbolProvider(selector, documentSymbolProvider(workspace),
                                                        { label: 'Jaithon' }),
        vscode.languages.registerWorkspaceSymbolProvider(workspaceSymbolProvider(workspace)),
        vscode.languages.registerFoldingRangeProvider(selector, foldingProvider(workspace)),
        vscode.languages.registerSelectionRangeProvider(selector, selectionRangeProvider(workspace)),
        vscode.languages.registerCallHierarchyProvider(selector, callHierarchyProvider(workspace)),
    );
}

module.exports = { register, contextAt, byteOffsetAt, referencesTo, scopeIdAt };
