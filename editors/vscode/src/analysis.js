// The language service.
//
// There is no reimplemented parser here. `jaithon ast --json` gives a syntax
// tree with a byte span on every node, and `jaithon check --dump-sema` gives
// the type the real checker assigned to each of those spans. Everything the
// editor offers — definitions, references, rename, hover, completion — is those
// two answers indexed.
//
// What this file does add is lexical scoping, because neither dump records
// which declaration a name resolved to. Member access does not need it: the
// receiver's type comes from the sema dump, so `p.norm` is answered by the
// checker, not by a guess.

const vscode = require('vscode');
const fs = require('fs');
const os = require('os');
const path = require('path');
const tool = require('./tool');

// ---------------------------------------------------------------------------
// Offsets
//
// AST spans are byte offsets; VS Code positions are UTF-16. They agree for
// ASCII, which is the overwhelmingly common case, so the conversion table is
// only built when a file actually needs one.
// ---------------------------------------------------------------------------

class Offsets {
    constructor(text) {
        this.text = text;
        this.ascii = Buffer.byteLength(text, 'utf8') === text.length;
        this.byteToChar = this.ascii ? null : Offsets.buildTable(text);

        this.lineStarts = [0];
        for (let i = 0; i < text.length; i++) {
            if (text.charCodeAt(i) === 10) this.lineStarts.push(i + 1);
        }
    }

    static buildTable(text) {
        const table = new Int32Array(Buffer.byteLength(text, 'utf8') + 1);
        let byte = 0;
        for (let i = 0; i < text.length;) {
            const code = text.codePointAt(i);
            const width = code > 0xffff ? 2 : 1;
            const size = code < 0x80 ? 1 : code < 0x800 ? 2 : code < 0x10000 ? 3 : 4;
            for (let b = 0; b < size; b++) table[byte + b] = i;
            byte += size;
            i += width;
        }
        table[byte] = text.length;
        return table;
    }

    char(byteOffset) {
        if (this.ascii) return Math.min(Math.max(byteOffset, 0), this.text.length);
        const clamped = Math.min(Math.max(byteOffset, 0), this.byteToChar.length - 1);
        return this.byteToChar[clamped];
    }

    byte(charOffset) {
        if (this.ascii) return charOffset;
        return Buffer.byteLength(this.text.slice(0, charOffset), 'utf8');
    }

    lineOf(charOffset) {
        let low = 0;
        let high = this.lineStarts.length - 1;
        while (low < high) {
            const mid = (low + high + 1) >> 1;
            if (this.lineStarts[mid] <= charOffset) low = mid; else high = mid - 1;
        }
        return low;
    }

    position(byteOffset) {
        const at = this.char(byteOffset);
        const line = this.lineOf(at);
        return new vscode.Position(line, at - this.lineStarts[line]);
    }

    range(startByte, endByte) {
        return new vscode.Range(this.position(startByte), this.position(endByte));
    }

    lineText(line) {
        const start = this.lineStarts[line];
        if (start === undefined) return '';
        const end = this.lineStarts[line + 1];
        return this.text.slice(start, end === undefined ? this.text.length : end - 1).replace(/\r$/, '');
    }
}

// ---------------------------------------------------------------------------
// Small helpers over the tree
// ---------------------------------------------------------------------------

function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * A declaration node's span covers its modifiers and often its whole body, so
 * the identifier has to be found inside it. Searching for the name as a whole
 * word from the front of the declaration lands on the declared name, because
 * the only thing that precedes it is `pub`, `static`, `fn`, `class` and friends.
 */
function locateName(text, from, to, name) {
    if (!name) return null;
    const region = text.slice(from, to);
    const match = new RegExp(`(?:^|[^A-Za-z0-9_])(${escapeRegExp(name)})(?![A-Za-z0-9_])`).exec(region);
    if (!match) return null;
    const start = from + match.index + match[0].length - name.length;
    return { start, end: start + name.length };
}

/** Every node beneath `node`, including itself, in source order. */
function walk(node, visit) {
    if (Array.isArray(node)) {
        for (const item of node) walk(item, visit);
        return;
    }
    if (!node || typeof node !== 'object') return;
    if (typeof node.kind === 'string' && visit(node) === false) return;
    for (const key of Object.keys(node)) {
        if (key === 'kind' || key === 'span') continue;
        walk(node[key], visit);
    }
}

function renderType(node) {
    if (!node) return null;
    switch (node.kind) {
        case 'TYPE_OPTIONAL':
            return `${renderType(node.inner) || 'any'}?`;
        case 'TYPE_FN': {
            const args = (node.args || []).map(renderType).join(', ');
            const ret = renderType(node.inner);
            return `fn(${args})${ret ? ` -> ${ret}` : ''}`;
        }
        default: {
            const args = node.args || [];
            return args.length ? `${node.name}[${args.map(renderType).join(', ')}]` : node.name;
        }
    }
}

function renderParam(param) {
    let text = param.isVariadic ? `...${param.name}` : param.name;
    if (param.isKwRest) text = `**${param.name}`;
    const type = renderType(param.type);
    if (type) text += `: ${type}`;
    if (param.defaultValue) text += ' = …';
    return text;
}

function signatureOf(fn) {
    const params = (fn.params || []).filter((p) => p.name !== 'self').map(renderParam).join(', ');
    const ret = renderType(fn.returnType);
    const generics = (fn.generics || []).length ? `[${fn.generics.join(', ')}]` : '';
    return `fn ${fn.name || ''}${generics}(${params})${ret ? ` -> ${ret}` : ''}`;
}

const VISIBILITY = {
    AST_VIS_PUBLIC: 'pub',
    AST_VIS_PROTECTED: 'prot',
    AST_VIS_PRIVATE: null,
};

// ---------------------------------------------------------------------------
// One file
// ---------------------------------------------------------------------------

class FileAnalysis {
    constructor(fsPath, text, ast, types) {
        this.fsPath = fsPath;
        this.uri = vscode.Uri.file(fsPath);
        this.text = text;
        this.offsets = new Offsets(text);
        this.ast = ast;
        this.types = types;             // "start:end" -> type string

        this.symbols = [];
        this.scopes = [];
        this.refs = [];
        this.imports = [];
        this.moduleScope = this.pushScope(null, 0, Buffer.byteLength(text, 'utf8'), 'module', true);

        if (ast) {
            this.declare(ast, this.moduleScope);
            this.resolve(ast, this.moduleScope);
            this.refs.sort((a, b) => a.start - b.start);
        }
    }

    // -- construction -------------------------------------------------------

    pushScope(parent, start, end, kind, lexical) {
        const scope = {
            id: this.scopes.length, parent, start, end, kind,
            lexical: lexical !== false, names: new Map(), owner: null,
        };
        this.scopes.push(scope);
        return scope.id;
    }

    addSymbol(spec) {
        const symbol = { index: this.symbols.length, container: null, ...spec };
        // The sema dump is keyed by whole-node spans; a pattern bind is the one
        // declaration whose name span is a node of its own.
        symbol.type = symbol.type
            || this.typeAt(symbol.start, symbol.end)
            || this.typeAt(symbol.fullStart, symbol.fullEnd);
        this.symbols.push(symbol);
        const scope = this.scopes[symbol.scope];
        if (scope && !scope.names.has(symbol.name)) scope.names.set(symbol.name, symbol.index);
        return symbol;
    }

    typeAt(start, end) {
        return this.types.get(`${start}:${end}`) || null;
    }

    /** Contiguous `#:` lines immediately above `byteOffset`. */
    docAbove(byteOffset) {
        const line = this.offsets.lineOf(this.offsets.char(byteOffset));
        const out = [];
        for (let i = line - 1; i >= 0; i--) {
            const text = this.offsets.lineText(i).trim();
            if (text.startsWith('#:')) { out.unshift(text.slice(2).replace(/^ /, '')); continue; }
            if (text === '' && out.length === 0) continue;   // one blank line is allowed
            break;
        }
        return out.length ? out.join('\n') : null;
    }

    span(node) {
        return node && node.span ? node.span : { start: 0, end: 0 };
    }

    /** Pass one: create scopes and declare every name they introduce. */
    declare(node, scopeId, container) {
        if (Array.isArray(node)) {
            for (const item of node) this.declare(item, scopeId, container);
            return;
        }
        if (!node || typeof node !== 'object' || typeof node.kind !== 'string') return;

        const { start, end } = this.span(node);
        const recurse = (child, id, owner) => this.declare(child, id === undefined ? scopeId : id,
                                                           owner === undefined ? container : owner);

        switch (node.kind) {
            case 'AST_PROGRAM':
                return recurse(node.stmts);

            case 'AST_BLOCK': {
                const inner = this.pushScope(scopeId, start, end, 'block');
                return recurse(node.stmts, inner);
            }

            case 'AST_FN_DECL': case 'AST_LAMBDA': case 'AST_ANON_FN': {
                const at = node.name ? locateName(this.text, start, this.span(node.body).start || end, node.name) : null;
                const fnScope = this.pushScope(scopeId, start, end, 'function');
                let symbol = null;
                if (node.name) {
                    symbol = this.addSymbol({
                        name: node.name,
                        kind: container && this.symbols[container]?.members !== undefined ? 'method' : 'function',
                        start: at ? at.start : start, end: at ? at.end : start,
                        fullStart: start, fullEnd: end,
                        detail: signatureOf(node),
                        doc: this.docAbove(start),
                        scope: container !== undefined && container !== null && this.symbols[container]?.members !== undefined
                            ? this.symbols[container].members
                            : scopeId,
                        container: container ?? null,
                        visibility: VISIBILITY[node.visibility] || null,
                        static: Boolean(node.isStatic),
                        params: node.params || [],
                        typeSpan: [start, end],
                        bodyScope: fnScope,
                    });
                }
                this.scopes[fnScope].owner = symbol ? symbol.index : null;

                for (const param of node.params || []) {
                    const pspan = this.span(param);
                    this.addSymbol({
                        name: param.name, kind: param.name === 'self' ? 'self' : 'parameter',
                        start: pspan.start, end: pspan.start + Buffer.byteLength(param.name, 'utf8'),
                        fullStart: pspan.start, fullEnd: pspan.end,
                        detail: renderParam(param), doc: null,
                        scope: fnScope, container: symbol ? symbol.index : (container ?? null),
                        declaredType: renderType(param.type),
                    });
                    recurse(param.defaultValue, fnScope, container);
                }
                recurse(node.returnType, fnScope, container);
                return recurse(node.body, fnScope, symbol ? symbol.index : container);
            }

            case 'AST_CLASS_DECL': case 'AST_TRAIT_DECL': case 'AST_ENUM_DECL': {
                const kind = node.kind === 'AST_CLASS_DECL' ? 'class'
                           : node.kind === 'AST_TRAIT_DECL' ? 'trait' : 'enum';
                const at = locateName(this.text, start, end, node.name);
                const members = this.pushScope(scopeId, start, end, kind, false);
                const symbol = this.addSymbol({
                    name: node.name, kind,
                    start: at ? at.start : start, end: at ? at.end : start,
                    fullStart: start, fullEnd: end,
                    detail: `${kind} ${node.name}${(node.generics || []).length ? `[${node.generics.join(', ')}]` : ''}`,
                    doc: this.docAbove(start),
                    scope: scopeId, container: container ?? null,
                    visibility: VISIBILITY[node.visibility] || null,
                    members,
                    abstract: Boolean(node.isAbstract),
                    superclass: renderType(node.superclass),
                    traits: (node.traits || []).map(renderType),
                });
                this.scopes[members].owner = symbol.index;

                for (const field of node.fields || []) {
                    const fspan = this.span(field);
                    const nameAt = locateName(this.text, fspan.start, fspan.end, field.name);
                    this.addSymbol({
                        name: field.name, kind: 'field',
                        start: nameAt ? nameAt.start : fspan.start, end: nameAt ? nameAt.end : fspan.start,
                        fullStart: fspan.start, fullEnd: fspan.end,
                        detail: `${field.isLet ? 'let' : 'var'} ${field.name}${renderType(field.type) ? `: ${renderType(field.type)}` : ''}`,
                        doc: this.docAbove(fspan.start),
                        scope: members, container: symbol.index,
                        visibility: VISIBILITY[field.visibility] || null,
                        static: Boolean(field.isStatic),
                        declaredType: renderType(field.type),
                    });
                    recurse(field.type, scopeId, symbol.index);
                    recurse(field.defaultValue, scopeId, symbol.index);
                }

                for (const variant of node.variants || []) {
                    const vspan = this.span(variant);
                    const params = (variant.params || []).map(renderParam).join(', ');
                    this.addSymbol({
                        name: variant.name, kind: 'variant',
                        start: vspan.start, end: vspan.start + Buffer.byteLength(variant.name, 'utf8'),
                        fullStart: vspan.start, fullEnd: vspan.end,
                        detail: `${node.name}.${variant.name}${params ? `(${params})` : ''}`,
                        doc: this.docAbove(vspan.start),
                        scope: members, container: symbol.index,
                        params: variant.params || [],
                    });
                    for (const param of variant.params || []) recurse(param.type, scopeId, symbol.index);
                }

                recurse(node.superclass, scopeId, container);
                recurse(node.traits, scopeId, container);
                recurse(node.supers, scopeId, container);
                for (const accessor of [...(node.getters || []), ...(node.setters || [])]) {
                    recurse(accessor, scopeId, symbol.index);
                }
                return recurse(node.methods, scopeId, symbol.index);
            }

            case 'AST_TYPE_DECL': {
                const at = locateName(this.text, start, end, node.name);
                this.addSymbol({
                    name: node.name, kind: 'typealias',
                    start: at ? at.start : start, end: at ? at.end : start,
                    fullStart: start, fullEnd: end,
                    detail: `type ${node.name} = ${renderType(node.aliased) || '…'}`,
                    doc: this.docAbove(start),
                    scope: scopeId, container: container ?? null,
                    visibility: VISIBILITY[node.visibility] || null,
                });
                return recurse(node.aliased);
            }

            case 'AST_VAR_DECL': {
                const declared = renderType(node.declaredType);
                for (const bind of this.patternBinds(node.pattern)) {
                    this.addSymbol({
                        name: bind.name, kind: 'variable',
                        start: bind.span.start, end: bind.span.end,
                        fullStart: start, fullEnd: end,
                        detail: `${node.varDeclKind === 'VD_VAR' ? 'var' : node.varDeclKind === 'VD_CONST' ? 'const' : 'let'} ${bind.name}`,
                        doc: this.docAbove(start),
                        scope: scopeId, container: container ?? null,
                        mutable: node.varDeclKind === 'VD_VAR',
                        constant: node.varDeclKind === 'VD_CONST',
                        declaredType: declared || renderType(bind.type),
                        visibility: VISIBILITY[node.visibility] || null,
                    });
                }
                recurse(node.declaredType);
                return recurse(node.init);
            }

            case 'AST_FOR': {
                const inner = this.pushScope(scopeId, start, end, 'for');
                recurse(node.iterable);
                for (const bind of this.patternBinds(node.pattern)) {
                    this.addSymbol({
                        name: bind.name, kind: 'variable',
                        start: bind.span.start, end: bind.span.end,
                        fullStart: bind.span.start, fullEnd: bind.span.end,
                        detail: `for ${bind.name}`, doc: null,
                        scope: inner, container: container ?? null,
                    });
                }
                return recurse(node.body, inner);
            }

            case 'AST_COMPREHENSION': {
                const inner = this.pushScope(scopeId, start, end, 'comprehension');
                for (const clause of node.clauses || []) {
                    recurse(clause.iterable, inner);
                    for (const bind of this.patternBinds(clause.pattern)) {
                        this.addSymbol({
                            name: bind.name, kind: 'variable',
                            start: bind.span.start, end: bind.span.end,
                            fullStart: bind.span.start, fullEnd: bind.span.end,
                            detail: `for ${bind.name}`, doc: null,
                            scope: inner, container: container ?? null,
                        });
                    }
                    recurse(clause.conditions, inner);
                }
                recurse(node.keyExpr, inner);
                return recurse(node.element, inner);
            }

            case 'AST_MATCH': case 'AST_MATCH_EXPR': {
                recurse(node.subject);
                for (const arm of node.arms || []) {
                    const aspan = this.span(arm);
                    const inner = this.pushScope(scopeId, aspan.start, aspan.end, 'arm');
                    for (const bind of this.patternBinds(arm.pattern)) {
                        this.addSymbol({
                            name: bind.name, kind: 'variable',
                            start: bind.span.start, end: bind.span.end,
                            fullStart: bind.span.start, fullEnd: bind.span.end,
                            detail: `case ${bind.name}`, doc: null,
                            scope: inner, container: container ?? null,
                        });
                    }
                    recurse(arm.pattern, inner);
                    recurse(arm.guard, inner);
                    recurse(arm.body, inner);
                }
                return;
            }

            case 'AST_TRY': {
                recurse(node.body);
                for (const clause of node.catches || []) {
                    const bodySpan = this.span(clause.body);
                    const inner = this.pushScope(scopeId, bodySpan.start, bodySpan.end, 'catch');
                    if (clause.name) {
                        const anchor = clause.types && clause.types.length
                            ? this.span(clause.types[0]).start : bodySpan.start;
                        const at = locateName(this.text, Math.max(0, anchor - 64), anchor, clause.name)
                                || locateName(this.text, Math.max(0, bodySpan.start - 64), bodySpan.start, clause.name);
                        this.addSymbol({
                            name: clause.name, kind: 'variable',
                            start: at ? at.start : bodySpan.start, end: at ? at.end : bodySpan.start,
                            fullStart: at ? at.start : bodySpan.start, fullEnd: at ? at.end : bodySpan.start,
                            detail: `catch ${clause.name}${clause.types?.length ? `: ${clause.types.map(renderType).join(' | ')}` : ''}`,
                            doc: null, scope: inner, container: container ?? null,
                            declaredType: clause.types?.length ? renderType(clause.types[0]) : null,
                        });
                    }
                    recurse(clause.types, inner);
                    recurse(clause.body, inner);
                }
                return recurse(node.finallyBlock);
            }

            case 'AST_IMPORT': {
                const bound = node.alias || node.path.split('.').filter(Boolean).pop();
                const at = locateName(this.text, start, end, bound);
                const symbol = this.addSymbol({
                    name: bound, kind: 'module',
                    start: at ? at.start : start, end: at ? at.end : start,
                    fullStart: start, fullEnd: end,
                    detail: `import ${node.path}${node.alias ? ` as ${node.alias}` : ''}`,
                    doc: null, scope: scopeId, container: container ?? null,
                    modulePath: node.path,
                });
                this.imports.push({ path: node.path, node, symbol: symbol.index, items: null });
                this.addModulePathRef(node.path, start, end, scopeId);
                return;
            }

            case 'AST_FROM_IMPORT': {
                const record = { path: node.path, node, symbol: null, items: [] };
                for (const item of node.items || []) {
                    const ispan = this.span(item);
                    const bound = item.alias || item.name;
                    const at = item.alias ? locateName(this.text, ispan.start, ispan.end + 32, item.alias) : null;
                    const symbol = this.addSymbol({
                        name: bound, kind: 'import',
                        start: at ? at.start : ispan.start,
                        end: at ? at.end : ispan.start + Buffer.byteLength(item.name, 'utf8'),
                        fullStart: ispan.start, fullEnd: ispan.end,
                        detail: `from ${node.path} import ${item.name}${item.alias ? ` as ${item.alias}` : ''}`,
                        doc: null, scope: scopeId, container: container ?? null,
                        modulePath: node.path, importedName: item.name,
                    });
                    record.items.push(symbol.index);
                }
                this.imports.push(record);
                this.addModulePathRef(node.path, start, end, scopeId);
                return;
            }

            default: {
                for (const key of Object.keys(node)) {
                    if (key === 'kind' || key === 'span') continue;
                    recurse(node[key]);
                }
            }
        }
    }

    /** Makes the dotted path in an import statement clickable. */
    addModulePathRef(modulePath, start, end, scopeId) {
        const at = this.text.indexOf(modulePath, this.offsets.char(start));
        if (at < 0 || at > this.offsets.char(end)) return;
        const byteStart = this.offsets.byte(at);
        this.refs.push({
            name: modulePath, kind: 'module-path', scope: scopeId,
            start: byteStart, end: byteStart + Buffer.byteLength(modulePath, 'utf8'),
            modulePath,
        });
    }

    patternBinds(pattern) {
        const out = [];
        walk(pattern, (node) => {
            if (node.kind === 'AST_PAT_BIND' && node.name) out.push({ name: node.name, span: node.span, type: node.type });
        });
        return out;
    }

    // -- pass two: references ----------------------------------------------

    resolve(node, scopeId) {
        const scopeFor = (offset) => {
            // Deepest lexical scope whose extent contains the offset.
            let best = scopeId;
            for (const scope of this.scopes) {
                if (offset < scope.start || offset > scope.end) continue;
                if (this.scopes[best].start <= scope.start) best = scope.id;
            }
            return best;
        };

        walk(node, (child) => {
            const { start, end } = this.span(child);
            switch (child.kind) {
                case 'AST_IDENT':
                    this.refs.push({
                        name: child.name, start, end, kind: 'value',
                        scope: scopeFor(start), node: child,
                    });
                    break;
                case 'AST_MEMBER': case 'AST_OPT_MEMBER': {
                    const objectSpan = this.span(child.object);
                    const nameLength = Buffer.byteLength(child.name || '', 'utf8');
                    this.refs.push({
                        name: child.name, start: end - nameLength, end, kind: 'member',
                        scope: scopeFor(start), node: child,
                        receiver: this.typeAt(objectSpan.start, objectSpan.end),
                    });
                    break;
                }
                case 'TYPE_NAME': case 'TYPE_GENERIC':
                    if (child.name) {
                        this.refs.push({
                            name: child.name,
                            start, end: start + Buffer.byteLength(child.name, 'utf8'),
                            kind: 'type', scope: scopeFor(start), node: child,
                        });
                    }
                    break;
                case 'AST_PAT_ENUM': case 'AST_PAT_CLASS':
                    if (child.typeName) {
                        this.refs.push({
                            name: child.typeName,
                            start, end: start + Buffer.byteLength(child.typeName, 'utf8'),
                            kind: 'type', scope: scopeFor(start), node: child,
                        });
                    }
                    break;
                default:
                    break;
            }
            return true;
        });
    }

    /** Walk the lexical chain for `name`, starting at `scopeId`. */
    lookup(name, scopeId) {
        let scope = this.scopes[scopeId];
        while (scope) {
            if (scope.lexical || scope.id === scopeId) {
                const found = scope.names.get(name);
                if (found !== undefined) return this.symbols[found];
            }
            scope = scope.parent === null ? null : this.scopes[scope.parent];
        }
        return null;
    }

    /** Members declared directly on a class, trait or enum symbol. */
    membersOf(symbol) {
        if (!symbol || symbol.members === undefined) return [];
        const scope = this.scopes[symbol.members];
        return [...scope.names.values()].map((index) => this.symbols[index]);
    }

    memberNamed(symbol, name) {
        if (!symbol || symbol.members === undefined) return null;
        const index = this.scopes[symbol.members].names.get(name);
        return index === undefined ? null : this.symbols[index];
    }

    /** The declaration whose body encloses `byteOffset`, innermost first. */
    enclosing(byteOffset) {
        let best = null;
        for (const symbol of this.symbols) {
            if (symbol.fullStart === undefined) continue;
            if (byteOffset < symbol.fullStart || byteOffset > symbol.fullEnd) continue;
            if (!['function', 'method', 'class', 'trait', 'enum'].includes(symbol.kind)) continue;
            if (!best || symbol.fullStart >= best.fullStart) best = symbol;
        }
        return best;
    }

    /** The declaration or reference the cursor is inside. */
    at(byteOffset) {
        for (const symbol of this.symbols) {
            if (byteOffset >= symbol.start && byteOffset <= symbol.end && symbol.end > symbol.start) {
                return { symbol, ref: null, start: symbol.start, end: symbol.end };
            }
        }
        for (const ref of this.refs) {
            if (byteOffset >= ref.start && byteOffset <= ref.end) {
                return { symbol: null, ref, start: ref.start, end: ref.end };
            }
        }
        return null;
    }

    /** Type the checker gave the smallest node covering `byteOffset`. */
    narrowestType(byteOffset) {
        let best = null;
        for (const key of this.types.keys()) {
            const split = key.indexOf(':');
            const start = Number(key.slice(0, split));
            const end = Number(key.slice(split + 1));
            if (byteOffset < start || byteOffset > end) continue;
            if (!best || end - start < best.end - best.start) best = { start, end, type: this.types.get(key) };
        }
        return best;
    }
}

// ---------------------------------------------------------------------------
// Cross-file
// ---------------------------------------------------------------------------

/**
 * Reduce a checker type to the name it is really about: `list[int]` to `list`,
 * `Point?` to `Point`, `fn() -> Point` to `Point`. A type from another module
 * is qualified — `helpers.Box` — and the qualifier is kept separately, because
 * it says which file to look in.
 */
function splitTypeName(type) {
    if (!type) return null;
    let text = type.trim();
    if (text.startsWith('module ')) return null;
    const arrow = text.lastIndexOf('->');
    if (text.startsWith('fn(') && arrow >= 0) text = text.slice(arrow + 2).trim();
    text = text.replace(/\?+$/, '');
    const bracket = text.indexOf('[');
    if (bracket >= 0) text = text.slice(0, bracket);
    if (!/^[A-Za-z_][A-Za-z0-9_.]*$/.test(text)) return null;

    const parts = text.split('.');
    return { name: parts.pop(), qualifier: parts.join('.') || null };
}

function baseTypeName(type) {
    return splitTypeName(type)?.name || null;
}

function moduleOfType(type) {
    const match = /^module\s+(\S+)$/.exec((type || '').trim());
    return match ? match[1] : null;
}

class Workspace {
    static tick = 0;

    constructor(output) {
        this.output = output;
        this.cache = new Map();       // fsPath -> {key, analysis}
        this.inflight = new Map();
    }

    dispose() {
        this.cache.clear();
    }

    invalidate(fsPath) {
        this.cache.delete(fsPath);
    }

    invalidateAll() {
        this.cache.clear();
    }

    documentFor(fsPath) {
        return vscode.workspace.textDocuments.find(
            (doc) => doc.uri.scheme === 'file' && doc.uri.fsPath === fsPath) || null;
    }

    cacheKey(fsPath, document) {
        if (document) return `v${document.version}`;
        try {
            const stat = fs.statSync(fsPath);
            return `m${stat.mtimeMs}:${stat.size}`;
        } catch {
            return 'missing';
        }
    }

    /**
     * Analyse a file, reusing the last result while its content is unchanged.
     * Returns null when the file does not parse — callers fall back to whatever
     * they can do without a tree rather than showing nothing.
     */
    async analyze(target, token) {
        const document = target.uri ? target : this.documentFor(target);
        const fsPath = document ? document.uri.fsPath : target;
        if (!fsPath) return null;

        const key = this.cacheKey(fsPath, document);
        const cached = this.cache.get(fsPath);
        if (cached && cached.key === key) return cached.analysis;

        const pendingKey = `${fsPath} ${key}`;
        if (this.inflight.has(pendingKey)) return this.inflight.get(pendingKey);

        const job = this.run(fsPath, document, token).then((analysis) => {
            this.inflight.delete(pendingKey);
            // Keep the previous tree when this revision does not parse, so that
            // navigation survives a half-typed line.
            if (!analysis) return cached ? cached.analysis : null;
            this.cache.set(fsPath, { key, analysis });
            return analysis;
        }).catch((error) => {
            this.inflight.delete(pendingKey);
            this.output?.appendLine(`analysis failed for ${fsPath}: ${error.message}`);
            return cached ? cached.analysis : null;
        });

        this.inflight.set(pendingKey, job);
        return job;
    }

    async run(fsPath, document, token) {
        let text;
        if (document) {
            text = document.getText();
        } else {
            try { text = fs.readFileSync(fsPath, 'utf8'); } catch { return null; }
        }

        const snapshot = document
            ? tool.snapshot(document)
            : { path: fsPath, mirrored: false, dir: path.dirname(fsPath), dispose() {} };
        const include = snapshot.mirrored ? tool.includeArgs(document, snapshot.dir) : [];
        // Written to a scratch directory, never beside the source: analysis runs
        // on every keystroke and must leave the workspace untouched.
        const semaOut = path.join(os.tmpdir(), `jaithon-sema-${process.pid}-${Workspace.tick++}.txt`);

        const [astResult, semaResult] = await Promise.all([
            tool.run(['ast', '--json', '--color=never', ...include, snapshot.path],
                     { cwd: snapshot.dir, document, token }),
            tool.run(['check', '--color=never', '--dump-sema', semaOut, ...include, snapshot.path],
                     { cwd: snapshot.dir, document, token }),
        ]);

        let semaText = '';
        try { semaText = fs.readFileSync(semaOut, 'utf8'); } catch { /* no dump written */ }
        try { fs.unlinkSync(semaOut); } catch { /* never created */ }
        snapshot.dispose();
        void semaResult;

        if (astResult.spawnFailed) return null;

        let ast = null;
        const trimmed = astResult.stdout.trim();
        if (trimmed.startsWith('{')) {
            try { ast = JSON.parse(trimmed); } catch { ast = null; }
        }
        if (!ast) return null;

        const types = new Map();
        for (const line of semaText.split('\n')) {
            const match = /^(\d+) (\d+) (\S+) type (.*)$/.exec(line);
            if (match) types.set(`${match[1]}:${match[2]}`, match[4]);
        }

        return new FileAnalysis(fsPath, text, ast, types);
    }

    /** The file a dotted import path names, resolved as the compiler would. */
    moduleFile(modulePath, fromFile) {
        return tool.resolveModule(modulePath, fromFile, this.documentFor(fromFile));
    }

    /**
     * Resolve a reference to its declaration, following imports into other
     * files. Returns `{analysis, symbol}` or null.
     */
    async definitionOf(analysis, ref, token) {
        if (!ref) return null;

        if (ref.kind === 'member') {
            return this.memberDefinition(analysis, ref, token);
        }

        const local = analysis.lookup(ref.name, ref.scope);
        if (!local) return null;
        return this.follow(analysis, local, token);
    }

    /** An `import`ed name stands for a declaration in another file. */
    async follow(analysis, symbol, token) {
        if (symbol.kind !== 'import') return { analysis, symbol };

        const file = this.moduleFile(symbol.modulePath, analysis.fsPath);
        if (!file) return { analysis, symbol };
        const other = await this.analyze(file, token);
        if (!other) return { analysis, symbol };

        const target = other.lookup(symbol.importedName, other.moduleScope);
        return target ? { analysis: other, symbol: target } : { analysis, symbol };
    }

    async memberDefinition(analysis, ref, token) {
        const moduleName = moduleOfType(ref.receiver);
        if (moduleName) {
            const file = this.moduleFile(moduleName, analysis.fsPath);
            if (!file) return null;
            const other = await this.analyze(file, token);
            if (!other) return null;
            const target = other.lookup(ref.name, other.moduleScope);
            return target ? { analysis: other, symbol: target } : null;
        }

        const owner = await this.typeSymbol(analysis, ref.receiver, token);
        if (!owner) return null;

        let current = owner;
        const seen = new Set();
        while (current) {
            const member = current.analysis.memberNamed(current.symbol, ref.name);
            if (member) return { analysis: current.analysis, symbol: member };
            const parent = current.symbol.superclass;
            if (!parent || seen.has(parent)) break;
            seen.add(parent);
            current = await this.typeSymbol(current.analysis, parent, token);
        }
        return null;
    }

    /** The class, trait or enum a checker type string names. */
    async typeSymbol(analysis, type, token) {
        const split = splitTypeName(type);
        if (!split) return null;
        const { name, qualifier } = split;

        // A qualified type says which module declared it, so go straight there.
        if (qualifier) {
            const bound = analysis.lookup(qualifier.split('.')[0], analysis.moduleScope);
            const modulePath = bound?.modulePath || qualifier;
            const file = this.moduleFile(modulePath, analysis.fsPath);
            const other = file ? await this.analyze(file, token) : null;
            const found = other && other.lookup(name, other.moduleScope);
            if (found && found.members !== undefined) return { analysis: other, symbol: found };
        }

        const local = analysis.lookup(name, analysis.moduleScope);
        if (local) {
            const followed = await this.follow(analysis, local, token);
            if (followed && followed.symbol.members !== undefined) return followed;
        }
        for (const record of analysis.imports) {
            const file = this.moduleFile(record.path, analysis.fsPath);
            if (!file) continue;
            const other = await this.analyze(file, token);
            const found = other && other.lookup(name, other.moduleScope);
            if (found && found.members !== undefined) return { analysis: other, symbol: found };
        }
        return null;
    }

    /** Every `.jai` file in the workspace, respecting the exclude setting. */
    async files(token) {
        const exclude = tool.config().get('excludeGlob') || '**/{__jaicache__,node_modules,build,.git}/**';
        const found = await vscode.workspace.findFiles('**/*.jai', exclude, 4000, token);
        return found.map((uri) => uri.fsPath);
    }

    /**
     * Files that could mention `name`. Reading bytes is far cheaper than
     * parsing, so a textual pre-filter keeps a workspace-wide rename to the
     * handful of files that can possibly be involved.
     */
    async candidates(name, token) {
        const all = await this.files(token);
        const needle = new RegExp(`(?:^|[^A-Za-z0-9_])${escapeRegExp(name)}(?![A-Za-z0-9_])`);
        const out = [];
        for (const fsPath of all) {
            if (token?.isCancellationRequested) break;
            const document = this.documentFor(fsPath);
            let text;
            if (document) {
                text = document.getText();
            } else {
                try { text = fs.readFileSync(fsPath, 'utf8'); } catch { continue; }
            }
            if (needle.test(text)) out.push(fsPath);
        }
        return out;
    }
}

module.exports = {
    Workspace, FileAnalysis, Offsets,
    renderType, signatureOf, walk, locateName,
    baseTypeName, splitTypeName, moduleOfType, escapeRegExp,
};
