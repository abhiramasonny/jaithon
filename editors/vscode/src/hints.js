// What the editor shows without being asked: hover, inlay hints, code lenses,
// and semantic tokens.

const vscode = require('vscode');
const tool = require('./tool');
const builtins = require('./builtins');
const { walk, baseTypeName, moduleOfType } = require('./analysis');
const { contextAt, byteOffsetAt, referencesTo } = require('./navigation');

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

function fence(text) {
    return ['```jaithon', text, '```'].join('\n');
}

function describe(symbol) {
    const parts = [];
    if (symbol.visibility) parts.push(symbol.visibility);
    if (symbol.static) parts.push('static');
    parts.push(symbol.detail || symbol.name);
    return parts.join(' ');
}

function hoverProvider(workspace) {
    return {
        async provideHover(document, position, token) {
            const context = await contextAt(workspace, document, position, token);
            if (!context) return null;
            const { analysis, offset, hit } = context;

            const markdown = new vscode.MarkdownString();
            markdown.isTrusted = false;

            if (hit) {
                const resolved = hit.symbol
                    ? { analysis, symbol: hit.symbol }
                    : await workspace.definitionOf(analysis, hit.ref, token);

                if (resolved) {
                    const { symbol } = resolved;
                    markdown.appendMarkdown(fence(describe(symbol)));

                    const type = symbol.declaredType || symbol.type;
                    if (type && !String(symbol.detail || '').includes(type)) {
                        markdown.appendMarkdown(`\n\n\`${type}\``);
                    }
                    if (symbol.superclass) markdown.appendMarkdown(`\n\nextends \`${symbol.superclass}\``);
                    if (symbol.traits?.length) markdown.appendMarkdown(`\n\nimplements ${symbol.traits.map((t) => `\`${t}\``).join(', ')}`);
                    if (symbol.doc) markdown.appendMarkdown(`\n\n---\n\n${symbol.doc}`);
                    if (resolved.analysis.fsPath !== analysis.fsPath) {
                        markdown.appendMarkdown(`\n\n*declared in ${vscode.workspace.asRelativePath(resolved.analysis.uri)}*`);
                    }
                    return new vscode.Hover(markdown, analysis.offsets.range(hit.start, hit.end));
                }

                // Not declared in any file the workspace can see: the runtime
                // registers it, so answer from the builtin tables instead.
                const name = hit.ref?.name || hit.symbol?.name;
                const builtin = builtins.FUNCTIONS[name];
                if (builtin && hit.ref?.kind !== 'member') {
                    markdown.appendMarkdown(fence(builtin.signature));
                    markdown.appendMarkdown(`\n\n---\n\n${builtin.doc}`);
                    return new vscode.Hover(markdown, analysis.offsets.range(hit.start, hit.end));
                }
                if (builtins.isException(name)) {
                    const chain = builtins.exceptionChain(name);
                    markdown.appendMarkdown(fence(`class ${name}`));
                    if (chain.length > 1) markdown.appendMarkdown(`\n\n${chain.join(' → ')}`);
                    const members = builtins.exceptionMembers().map((m) => `\`${m}\``).join(', ');
                    markdown.appendMarkdown(`\n\n---\n\nRegistered by the runtime. Members: ${members}.`);
                    return new vscode.Hover(markdown, analysis.offsets.range(hit.start, hit.end));
                }
            }

            // Anything else the checker gave a type gets that type. This is what
            // makes hovering the middle of an expression useful.
            const narrowest = analysis.narrowestType(offset);
            if (!narrowest) return null;
            markdown.appendMarkdown(fence(narrowest.type));
            return new vscode.Hover(markdown, analysis.offsets.range(narrowest.start, narrowest.end));
        },
    };
}

// ---------------------------------------------------------------------------
// Inlay hints
// ---------------------------------------------------------------------------

function inlayHintProvider(workspace) {
    return {
        async provideInlayHints(document, range, token) {
            const settings = tool.config();
            const wantTypes = settings.get('inlayHints.types') !== false;
            const wantParameters = settings.get('inlayHints.parameterNames') !== false;
            if (!wantTypes && !wantParameters) return [];

            const analysis = await workspace.analyze(document, token);
            if (!analysis) return [];

            const from = byteOffsetAt(analysis, range.start);
            const to = byteOffsetAt(analysis, range.end);
            const hints = [];

            if (wantTypes) {
                for (const symbol of analysis.symbols) {
                    if (symbol.start < from || symbol.end > to) continue;
                    if (symbol.declaredType) continue;
                    if (!['variable', 'parameter'].includes(symbol.kind)) continue;
                    const type = symbol.type;
                    if (!type || type === 'any' || type.startsWith('module ')) continue;

                    const hint = new vscode.InlayHint(
                        analysis.offsets.position(symbol.end), `: ${type}`,
                        vscode.InlayHintKind.Type);
                    hint.paddingLeft = false;
                    hints.push(hint);
                }
            }

            if (wantParameters) {
                const calls = [];
                walk(analysis.ast, (node) => {
                    if (node.kind === 'AST_CALL' && node.span.start >= from && node.span.end <= to) {
                        calls.push(node);
                    }
                    return true;
                });

                for (const call of calls.slice(0, 200)) {
                    if (token?.isCancellationRequested) break;
                    const params = await parametersOf(workspace, analysis, call, token);
                    if (!params) continue;
                    call.args.forEach((arg, index) => {
                        // A named argument already says what it binds to, and a
                        // literal is the only case where the name adds anything.
                        if (arg.name || index >= params.length) return;
                        const param = params[index];
                        if (!param || param.isVariadic || param.name === 'self') return;
                        if (!isLiteral(arg.value)) return;
                        const hint = new vscode.InlayHint(
                            analysis.offsets.position(arg.span.start), `${param.name}:`,
                            vscode.InlayHintKind.Parameter);
                        hint.paddingRight = true;
                        hints.push(hint);
                    });
                }
            }
            return hints;
        },
    };
}

function isLiteral(node) {
    return node && ['AST_INT_LIT', 'AST_FLOAT_LIT', 'AST_STR_LIT', 'AST_BOOL_LIT',
                    'AST_NULL_LIT', 'AST_UNARY'].includes(node.kind);
}

/** The parameter list of whatever a call's callee resolves to. */
async function parametersOf(workspace, analysis, call, token) {
    const callee = call.callee;
    if (!callee || !callee.span) return null;

    const ref = analysis.refs.find(
        (candidate) => candidate.start === callee.span.start
            || (candidate.kind === 'member' && candidate.end === callee.span.end));
    if (!ref) return null;

    const found = await workspace.definitionOf(analysis, ref, token);
    if (!found) return null;
    const { symbol } = found;
    if (symbol.params) return symbol.params.filter((param) => param.name !== 'self');
    // Constructing a class calls its `init`.
    if (symbol.members !== undefined) {
        const init = found.analysis.memberNamed(symbol, 'init');
        if (init?.params) return init.params.filter((param) => param.name !== 'self');
    }
    return null;
}

// ---------------------------------------------------------------------------
// Code lenses
// ---------------------------------------------------------------------------

function codeLensProvider(workspace) {
    return {
        async provideCodeLenses(document, token) {
            const settings = tool.config();
            const wantRun = settings.get('codeLens.run') !== false;
            const wantReferences = settings.get('codeLens.references') === true;
            if (!wantRun && !wantReferences) return [];

            const analysis = await workspace.analyze(document, token);
            if (!analysis) return [];

            const lenses = [];
            for (const symbol of analysis.symbols) {
                const range = analysis.offsets.range(symbol.start, symbol.end);

                if (wantRun && symbol.kind === 'function' && symbol.name === 'main') {
                    lenses.push(new vscode.CodeLens(range, {
                        title: '$(play) Run', command: 'jaithon.run',
                        arguments: [document.uri],
                    }));
                }
                if (wantRun && /^test_/.test(symbol.name) && ['function', 'method'].includes(symbol.kind)) {
                    lenses.push(new vscode.CodeLens(range, {
                        title: '$(beaker) Run test', command: 'jaithon.testFile',
                        arguments: [document.uri, symbol.name],
                    }));
                }
                if (wantReferences && ['function', 'method', 'class', 'trait', 'enum'].includes(symbol.kind)) {
                    const lens = new vscode.CodeLens(range);
                    lens.jaithon = { fsPath: analysis.fsPath, index: symbol.index, uri: document.uri, range };
                    lenses.push(lens);
                }
            }
            return lenses;
        },

        async resolveCodeLens(lens, token) {
            if (!lens.jaithon) return lens;
            const analysis = await workspace.analyze(lens.jaithon.fsPath, token);
            if (!analysis) return lens;
            const symbol = analysis.symbols[lens.jaithon.index];
            const uses = await referencesTo(workspace, analysis, symbol, token);
            const title = uses.length === 1 ? '1 reference' : `${uses.length} references`;
            lens.command = uses.length === 0
                ? { title, command: '' }
                : {
                    title,
                    command: 'editor.action.showReferences',
                    arguments: [lens.jaithon.uri, lens.jaithon.range.start,
                                uses.map((use) => new vscode.Location(
                                    use.analysis.uri, use.analysis.offsets.range(use.start, use.end)))],
                };
            return lens;
        },
    };
}

// ---------------------------------------------------------------------------
// Semantic tokens
//
// TextMate colours a file by shape; this colours it by what the compiler
// decided each name actually is. A class and a variable that happen to be
// spelled alike stop looking alike, and a call through a variable holding a
// function stops being painted as a call to a function of that name.
// ---------------------------------------------------------------------------

const TOKEN_TYPES = [
    'namespace', 'class', 'enum', 'interface', 'typeParameter', 'type',
    'parameter', 'variable', 'property', 'enumMember', 'function', 'method',
];
const TOKEN_MODIFIERS = [
    'declaration', 'definition', 'readonly', 'static', 'defaultLibrary', 'abstract',
];
const LEGEND = new vscode.SemanticTokensLegend(TOKEN_TYPES, TOKEN_MODIFIERS);

const SYMBOL_TOKEN = {
    function: 'function',
    method: 'method',
    class: 'class',
    trait: 'interface',
    enum: 'enum',
    variant: 'enumMember',
    field: 'property',
    variable: 'variable',
    parameter: 'parameter',
    typealias: 'type',
    typeParameter: 'typeParameter',
    module: 'namespace',
    import: 'variable',
    self: 'parameter',
};

function modifiersOf(symbol, declaration) {
    const out = [];
    if (declaration) out.push('declaration', 'definition');
    if (symbol.static) out.push('static');
    if (symbol.abstract) out.push('abstract');
    if (symbol.kind === 'variable' && symbol.mutable === false) out.push('readonly');
    if (symbol.constant) out.push('readonly');
    return out;
}

/**
 * The checker types a member access with the type of what it yields, so a
 * function type is how you tell `canvas.clear(…)` from `game.food` without
 * resolving anything.
 */
function memberToken(analysis, ref) {
    const span = ref.node?.span;
    const accessed = span ? analysis.typeAt(span.start, span.end) : null;
    return accessed && accessed.trim().startsWith('fn(') ? 'method' : 'property';
}

/**
 * Resolve within this file. Names that come from another module are resolved
 * once per file by the caller and handed in as `imported`, because an import
 * that names a class must not be painted as a variable — two types written
 * side by side would come out two different colours.
 */
function classify(analysis, ref, imported) {
    // An import line reads as one thing, so the name being imported is coloured
    // like the path it comes from rather than guessed at.
    if (ref.kind === 'module-path' || ref.kind === 'imported') {
        return { type: 'namespace', modifiers: [] };
    }

    // A named argument labels a parameter, and reads best as one.
    if (ref.kind === 'argument') return { type: 'parameter', modifiers: [] };

    if (ref.kind === 'variant') {
        const owner = analysis.lookup(ref.owner, analysis.moduleScope);
        const member = owner && analysis.memberNamed(owner, ref.name);
        return { type: 'enumMember', modifiers: member ? modifiersOf(member, false) : [] };
    }

    if (ref.kind === 'member') {
        const owner = analysis.lookup(baseTypeName(ref.receiver) || '', analysis.moduleScope);
        const member = owner && analysis.memberNamed(owner, ref.name);
        if (member) return { type: SYMBOL_TOKEN[member.kind], modifiers: modifiersOf(member, false) };
        if (builtins.methodsFor(ref.receiver).includes(ref.name)) {
            return { type: 'method', modifiers: ['defaultLibrary'] };
        }
        if (moduleOfType(ref.receiver)) return { type: memberToken(analysis, ref), modifiers: [] };
        return { type: memberToken(analysis, ref), modifiers: [] };
    }

    const symbol = analysis.lookup(ref.name, ref.scope);
    if (symbol) {
        // An import stands for a declaration elsewhere; paint it as whatever
        // that declaration is, not as the local binding that names it.
        if (symbol.kind === 'import') {
            const resolved = imported?.get(symbol.name);
            if (resolved) return { type: resolved, modifiers: [] };
            return { type: ref.kind === 'type' ? 'class' : 'variable', modifiers: [] };
        }
        const type = SYMBOL_TOKEN[symbol.kind];
        return type ? { type, modifiers: modifiersOf(symbol, false) } : null;
    }
    if (ref.name === builtins.PRIMITIVE_NAMESPACE) return { type: 'namespace', modifiers: ['defaultLibrary'] };
    if (builtins.FUNCTIONS[ref.name]) return { type: 'function', modifiers: ['defaultLibrary'] };
    if (builtins.isException(ref.name)) return { type: 'class', modifiers: ['defaultLibrary'] };
    if (builtins.PRELUDE_TRAITS.includes(ref.name)) return { type: 'interface', modifiers: ['defaultLibrary'] };
    if (ref.kind === 'type') {
        return builtins.PRIMITIVE_TYPES.includes(ref.name)
            ? { type: 'type', modifiers: ['defaultLibrary'] }
            : { type: 'type', modifiers: [] };
    }
    return null;
}

function encodeModifiers(modifiers) {
    let bits = 0;
    for (const name of modifiers || []) {
        const index = TOKEN_MODIFIERS.indexOf(name);
        if (index >= 0) bits |= 1 << index;
    }
    return bits;
}

function semanticTokenProvider(workspace) {
    return {
        async provideDocumentSemanticTokens(document, token) {
            const analysis = await workspace.analyze(document, token);
            if (!analysis) return null;

            const builder = new vscode.SemanticTokensBuilder(LEGEND);
            const emit = (start, end, type, modifiers) => {
                if (!type || end <= start) return;
                const position = analysis.offsets.position(start);
                const width = analysis.offsets.position(end).character - position.character;
                if (width <= 0) return;
                builder.push(position.line, position.character, width, TOKEN_TYPES.indexOf(type),
                             encodeModifiers(modifiers));
            };

            // What each imported name really is, resolved once per file rather
            // than once per use: there are far fewer imports than references.
            const imported = new Map();
            for (const symbol of analysis.symbols) {
                if (symbol.kind !== 'import' || imported.has(symbol.name)) continue;
                const followed = await workspace.follow(analysis, symbol, token);
                if (followed && followed.symbol !== symbol) {
                    imported.set(symbol.name, SYMBOL_TOKEN[followed.symbol.kind] || 'variable');
                }
            }
            if (token?.isCancellationRequested) return null;

            const emissions = [];
            for (const symbol of analysis.symbols) {
                if (symbol.end <= symbol.start) continue;
                const type = symbol.kind === 'import'
                    ? imported.get(symbol.name) || SYMBOL_TOKEN[symbol.kind]
                    : SYMBOL_TOKEN[symbol.kind];
                emissions.push({
                    start: symbol.start, end: symbol.end,
                    type, modifiers: modifiersOf(symbol, true),
                });
            }
            for (const ref of analysis.refs) {
                const classified = classify(analysis, ref, imported);
                if (classified) emissions.push({ start: ref.start, end: ref.end, ...classified });
            }

            // The builder demands source order, and the two lists interleave.
            emissions.sort((a, b) => a.start - b.start);
            let previousEnd = -1;
            for (const item of emissions) {
                if (item.start < previousEnd) continue;
                emit(item.start, item.end, item.type, item.modifiers);
                previousEnd = item.end;
            }
            return builder.build();
        },
    };
}

// ---------------------------------------------------------------------------

function register(context, workspace) {
    const selector = { language: 'jaithon', scheme: 'file' };
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(selector, hoverProvider(workspace)),
        vscode.languages.registerInlayHintsProvider(selector, inlayHintProvider(workspace)),
        vscode.languages.registerCodeLensProvider(selector, codeLensProvider(workspace)),
        vscode.languages.registerDocumentSemanticTokensProvider(
            selector, semanticTokenProvider(workspace), LEGEND),
    );
}

module.exports = { register, LEGEND };
