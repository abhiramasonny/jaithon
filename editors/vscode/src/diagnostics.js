// Diagnostics.
//
// The compiler renders rustc-style blocks, not one line per problem:
//
//     error[E0200]: undefined name `nope`
//       --> src/main.jai:2:11
//        |
//      2 |   let x = nope
//        |           ^^^^ the primary span
//        |       ---- a secondary span, possibly in another file
//        |
//     help: ...
//     note: ...
//
// Parsing the whole block rather than the header alone is what makes the
// squiggle the right width, puts secondary spans in Problems as related
// information, and gives the quick fixes something to work from.

const vscode = require('vscode');
const path = require('path');
const tool = require('./tool');

const TAB_WIDTH = 4;   // src/common/diag.c: JAI_TAB_WIDTH

const HEADER_RE = /^(error|warning)(?:\[([EW]\d+)\])?:\s*(.*)$/;
const ARROW_RE = /^\s*-->\s+(.*?):(\d+):(\d+)\s*$/;
const SOURCE_RE = /^\s*(\d+)\s\|(.*)$/;
const MARKER_RE = /^\s*\|\s*(\^+|-+)(?:\s+(.*?))?\s*$/;
const TRAILER_RE = /^(help|note):\s*(.*)$/;

/**
 * Display column -> UTF-16 index. The compiler counts one column per codepoint
 * and advances tabs to the next multiple of four, so the inverse has to walk
 * the same line the same way.
 */
function displayColumnToUtf16(line, column) {
    let col = 0;
    let index = 0;
    for (const ch of line) {
        if (col >= column) break;
        col = ch === '\t' ? Math.floor(col / TAB_WIDTH + 1) * TAB_WIDTH : col + 1;
        index += ch.length;
    }
    return index;
}

/**
 * Split the compiler's output into diagnostics.
 *
 * Spans carry the source line as the compiler printed it, with tabs already
 * expanded. That is enough to place a caret on its own, and is replaced by the
 * real line when the file is open — see `rangeOf`.
 */
function parse(text) {
    const out = [];
    let current = null;
    let file = null;
    let lastSourceLine = 0;
    let lastSourceText = '';

    const flush = () => {
        if (current) out.push(current);
        current = null;
        file = null;
    };

    for (const raw of text.split('\n')) {
        const line = raw.replace(/\x1b\[[0-9;]*m/g, '').replace(/\r$/, '');

        const header = HEADER_RE.exec(line);
        if (header) {
            flush();
            current = {
                severity: header[1],
                code: header[2] || null,
                message: header[3],
                file: null, line: 0, column: 1,
                primary: null, secondary: [],
                help: null, note: null,
            };
            continue;
        }
        if (!current) continue;

        const arrow = ARROW_RE.exec(line);
        if (arrow) {
            file = arrow[1];
            // The first arrow anchors the diagnostic; a later one opens a block
            // for another file, whose markers become related information.
            if (current.file === null) {
                current.file = file;
                current.line = Number(arrow[2]);
                current.column = Number(arrow[3]);
            }
            continue;
        }

        const source = SOURCE_RE.exec(line);
        if (source) {
            lastSourceLine = Number(source[1]);
            lastSourceText = source[2].replace(/^ /, '');
            continue;
        }

        const marker = MARKER_RE.exec(line);
        if (marker && lastSourceLine > 0) {
            const run = marker[1];
            const bar = line.indexOf('|');
            const span = {
                file,
                line: lastSourceLine,
                column: Math.max(0, line.indexOf(run, bar) - (bar + 2)),
                width: run.length,
                sourceText: lastSourceText,
                label: marker[2] || null,
                primary: run[0] === '^',
            };
            if (span.primary && !current.primary && span.file === current.file) {
                current.primary = span;
            } else {
                current.secondary.push(span);
            }
            continue;
        }

        const trailer = TRAILER_RE.exec(line);
        if (trailer) current[trailer[1]] = trailer[2];
    }
    flush();
    return out;
}

/**
 * The range a span covers. `lineText` is the file's real line when it can be
 * had; without it the compiler's tab-expanded copy is used, which agrees for
 * every line that does not contain a tab.
 */
function rangeOf(span, lineText) {
    const text = lineText !== undefined && lineText !== null ? lineText : span.sourceText;
    const line = Math.max(0, span.line - 1);
    const start = displayColumnToUtf16(text, span.column);
    const end = displayColumnToUtf16(text, span.column + span.width);
    return new vscode.Range(line, start, line, Math.max(end, start + 1));
}

/** The text of `line` (1-based) in an open document, or null. */
function openLine(fsPath, line) {
    for (const doc of vscode.workspace.textDocuments) {
        if (doc.uri.fsPath !== fsPath) continue;
        if (line - 1 >= doc.lineCount) return null;
        return doc.lineAt(line - 1).text;
    }
    return null;
}

function toVSCode(diag, resolvePath) {
    const fsPath = resolvePath(diag.file);
    const uri = vscode.Uri.file(fsPath);

    const range = diag.primary
        ? rangeOf(diag.primary, openLine(fsPath, diag.primary.line))
        : new vscode.Range(diag.line - 1, Math.max(0, diag.column - 1),
                           diag.line - 1, Math.max(0, diag.column));

    const item = new vscode.Diagnostic(
        range, diag.message,
        diag.severity === 'error' ? vscode.DiagnosticSeverity.Error
                                  : vscode.DiagnosticSeverity.Warning);
    item.source = 'jaithon';
    if (diag.code) item.code = diag.code;

    const related = [];
    if (diag.primary && diag.primary.label) {
        related.push(new vscode.DiagnosticRelatedInformation(
            new vscode.Location(uri, range), diag.primary.label));
    }
    for (const span of diag.secondary) {
        if (!span.file) continue;
        const secondaryPath = resolvePath(span.file);
        related.push(new vscode.DiagnosticRelatedInformation(
            new vscode.Location(vscode.Uri.file(secondaryPath),
                                rangeOf(span, openLine(secondaryPath, span.line))),
            span.label || 'related to this'));
    }
    for (const kind of ['help', 'note']) {
        if (diag[kind]) {
            related.push(new vscode.DiagnosticRelatedInformation(
                new vscode.Location(uri, range), `${kind}: ${diag[kind]}`));
        }
    }
    if (related.length) item.relatedInformation = related;

    // An unused binding should fade rather than shout: the whole content of the
    // warning is that the code does nothing.
    const tags = [];
    if (diag.code === 'W0101') tags.push(vscode.DiagnosticTag.Unnecessary);
    if (/\bdeprecated\b/i.test(diag.message)) tags.push(vscode.DiagnosticTag.Deprecated);
    if (tags.length) item.tags = tags;

    item.jaithon = diag;
    return { uri, item };
}

// ---------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------

class Checker {
    constructor(collection) {
        this.collection = collection;
        this.pending = new Map();     // uri -> timer
        this.running = new Map();     // uri -> CancellationTokenSource
        this.owned = new Map();       // uri -> the uris its last run wrote to
        this.emitter = new vscode.EventEmitter();
        this.onDidCheck = this.emitter.event;
    }

    dispose() {
        for (const timer of this.pending.values()) clearTimeout(timer);
        for (const source of this.running.values()) source.cancel();
        this.emitter.dispose();
    }

    /** Schedule a check, coalescing keystrokes into a single compiler run. */
    schedule(document, delay) {
        if (document.languageId !== 'jaithon') return;
        const key = document.uri.toString();
        clearTimeout(this.pending.get(key));
        this.pending.set(key, setTimeout(() => {
            this.pending.delete(key);
            this.check(document);
        }, delay ?? tool.config().get('checkDelay') ?? 300));
    }

    async check(document) {
        if (document.languageId !== 'jaithon') return [];
        const key = document.uri.toString();

        this.running.get(key)?.cancel();
        const source = new vscode.CancellationTokenSource();
        this.running.set(key, source);

        const snapshot = tool.snapshot(document);
        const args = ['check'];
        if (tool.config().get('strict')) args.push('--strict');
        args.push('--color=never');
        if (snapshot.mirrored) args.push('--no-cache', ...tool.includeArgs(document, snapshot.dir));
        args.push(snapshot.path);

        const result = await tool.run(args, { cwd: snapshot.dir, document, token: source.token });
        snapshot.dispose();

        if (source.token.isCancellationRequested) return [];
        this.running.delete(key);
        if (result.spawnFailed) return [];

        // The mirror is what the compiler quotes; map it back so the squiggle
        // lands on the buffer the user is actually editing.
        const mirrored = path.resolve(snapshot.path);
        const resolvePath = (quoted) => {
            if (!quoted) return document.uri.fsPath;
            const absolute = path.resolve(snapshot.dir, quoted);
            return absolute === mirrored ? document.uri.fsPath : absolute;
        };

        const byFile = new Map();
        const items = [];
        for (const diag of parse(`${result.stderr}\n${result.stdout}`)) {
            if (!diag.file) { diag.file = snapshot.path; diag.line = diag.line || 1; }
            const { uri, item } = toVSCode(diag, resolvePath);
            const fileKey = uri.toString();
            if (!byFile.has(fileKey)) byFile.set(fileKey, { uri, items: [] });
            byFile.get(fileKey).items.push(item);
            items.push(item);
        }

        for (const stale of this.owned.get(key) || []) {
            if (!byFile.has(stale)) this.collection.delete(vscode.Uri.parse(stale));
        }
        for (const entry of byFile.values()) this.collection.set(entry.uri, entry.items);
        if (!byFile.has(key)) this.collection.set(document.uri, []);
        this.owned.set(key, [...byFile.keys()]);

        this.emitter.fire({ uri: document.uri, diagnostics: items });
        return items;
    }

    forget(document) {
        const key = document.uri.toString();
        clearTimeout(this.pending.get(key));
        this.pending.delete(key);
        this.running.get(key)?.cancel();
        this.running.delete(key);
        for (const stale of this.owned.get(key) || []) {
            this.collection.delete(vscode.Uri.parse(stale));
        }
        this.owned.delete(key);
        this.collection.delete(document.uri);
    }
}

module.exports = { parse, rangeOf, toVSCode, Checker, displayColumnToUtf16 };
