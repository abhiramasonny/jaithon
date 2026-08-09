/* diag.c — source registry, diagnostic collection, and rendering.
 *
 * The rendered shape is normative; see spec/LANGUAGE.md §13. Nothing here
 * writes to stderr on its own — the caller picks the stream.
 */
#include "diag.h"

#include <stdlib.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Small string helpers                                                */
/* ------------------------------------------------------------------ */

/* jaiRealloc needs the old size to keep its byte accounting honest, and every
 * string we own is NUL-terminated, so strlen is the authority. */
static void freeStr(char *s) {
    if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
}

static char *formatV(const char *fmt, va_list ap) {
    va_list probe;
    va_copy(probe, ap);
    int n = vsnprintf(NULL, 0, fmt, probe);
    va_end(probe);
    if (n < 0) return jaiStrdup("<malformed diagnostic message>");

    char *buf = JAI_ALLOC(char, (size_t)n + 1);
    (void)vsnprintf(buf, (size_t)n + 1, fmt, ap);
    return buf;
}

/* ------------------------------------------------------------------ */
/* Source registry                                                     */
/* ------------------------------------------------------------------ */

static JAI_VEC(JaiSourceFile) gSources;

/* One pass to count lines, one to fill: the index is exact-sized and never
 * grown again, which matters because spans index it on every diagnostic. */
static void buildLineIndex(JaiSourceFile *f) {
    int lines = 1;
    for (size_t i = 0; i < f->length; i++) {
        if (f->source[i] == '\n') lines++;
    }
    f->lineStarts = JAI_ALLOC(uint32_t, lines);
    f->lineCount = lines;

    int n = 0;
    f->lineStarts[n++] = 0;
    for (size_t i = 0; i < f->length; i++) {
        if (f->source[i] == '\n' && n < lines) f->lineStarts[n++] = (uint32_t)(i + 1);
    }
}

int jaiSourceAdd(const char *path, char *source, size_t length) {
    JaiSourceFile f;
    memset(&f, 0, sizeof f);
    f.id = gSources.count;
    f.path = jaiStrdup(path != NULL ? path : "<unknown>");
    f.source = source;
    f.length = length;
    buildLineIndex(&f);
    JAI_VEC_PUSH(JaiSourceFile, &gSources, f);
    return f.id;
}

JaiSourceFile *jaiSourceGet(int id) {
    if (id < 0 || id >= gSources.count) return NULL;
    return &gSources.data[id];
}

void jaiSourceFreeAll(void) {
    for (int i = 0; i < gSources.count; i++) {
        JaiSourceFile *f = &gSources.data[i];
        freeStr(f->path);
        if (f->source != NULL) (void)jaiRealloc(f->source, f->length + 1, 0);
        JAI_FREE_ARRAY(uint32_t, f->lineStarts, f->lineCount);
        f->path = NULL;
        f->source = NULL;
        f->lineStarts = NULL;
        f->lineCount = 0;
    }
    JAI_VEC_FREE(JaiSourceFile, &gSources);
}

/* Index of the line containing `offset`; the offset is assumed clamped. */
static int lineIndexFor(const JaiSourceFile *f, uint32_t offset) {
    int lo = 0, hi = f->lineCount - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (f->lineStarts[mid] <= offset) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

void jaiSourceLineCol(int fileId, uint32_t offset, int *line, int *col) {
    if (line != NULL) *line = 0;
    if (col != NULL) *col = 0;

    JaiSourceFile *f = jaiSourceGet(fileId);
    if (f == NULL) return;
    if (offset > f->length) offset = (uint32_t)f->length;

    int idx = lineIndexFor(f, offset);
    uint32_t start = f->lineStarts[idx];
    if (line != NULL) *line = idx + 1;
    if (col != NULL) {
        /* Columns count Unicode scalars, per spec §1.1. */
        *col = (int)jaiUtf8Length(f->source + start, offset - start) + 1;
    }
}

const char *jaiSourceLineText(int fileId, uint32_t offset, size_t *outLen,
                              int *outLine) {
    if (outLen != NULL) *outLen = 0;
    if (outLine != NULL) *outLine = 0;

    JaiSourceFile *f = jaiSourceGet(fileId);
    if (f == NULL) return NULL;
    if (offset > f->length) offset = (uint32_t)f->length;

    int idx = lineIndexFor(f, offset);
    size_t start = f->lineStarts[idx];
    size_t end = (idx + 1 < f->lineCount) ? f->lineStarts[idx + 1] - 1 : f->length;
    if (end > start && f->source[end - 1] == '\r') end--;

    if (outLen != NULL) *outLen = end - start;
    if (outLine != NULL) *outLine = idx + 1;
    return f->source + start;
}

/* ------------------------------------------------------------------ */
/* Spans                                                               */
/* ------------------------------------------------------------------ */

JaiSpan jaiSpanJoin(JaiSpan a, JaiSpan b) {
    if (a.file < 0) return b;
    if (b.file < 0) return a;
    if (a.file != b.file) return a;   /* cross-file joins keep the first */

    JaiSpan out;
    out.file = a.file;
    out.start = a.start < b.start ? a.start : b.start;
    out.end = a.end > b.end ? a.end : b.end;
    if (out.end < out.start) out.end = out.start;
    return out;
}

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */

#define JAI_WARNING_BASE 10000

bool jaiDiagCodeIsWarning(JaiDiagCode code) {
    return (int)code >= JAI_WARNING_BASE;
}

const char *jaiDiagCodeString(JaiDiagCode code) {
    /* A small ring so that two codes can be live in one printf argument list. */
    static char bufs[4][16];
    static int next = 0;

    if (code == JAI_OK) return "";
    char *b = bufs[next];
    next = (next + 1) & 3;

    if (jaiDiagCodeIsWarning(code)) {
        (void)snprintf(b, sizeof bufs[0], "W%04d", (int)code - JAI_WARNING_BASE + 100);
    } else {
        (void)snprintf(b, sizeof bufs[0], "E%04d", (int)code);
    }
    return b;
}

/* "E0402" or "W0101" back to the code it names, or JAI_OK.
 *
 * The inverse of jaiDiagCodeString, and arithmetic for the same reason that is:
 * an error code *is* its enum value and a warning code is its offset from
 * JAI_WARNING_BASE. No table to keep in step.
 *
 * Needed because the self-hosted front end spells codes as strings -- Jaithon
 * has no C enum to share -- and its diagnostics have to become JaiDiag before
 * anything can render them properly. */
JaiDiagCode jaiDiagCodeFromString(const char *text) {
    if (text == NULL || (text[0] != 'E' && text[0] != 'W')) return JAI_OK;
    int value = 0;
    for (const char *p = text + 1; *p != '\0'; p++) {
        if (*p < '0' || *p > '9') return JAI_OK;
        value = value * 10 + (*p - '0');
    }
    if (text[0] == 'W') return (JaiDiagCode)(JAI_WARNING_BASE + value - 100);
    return (JaiDiagCode)value;
}

/* ------------------------------------------------------------------ */
/* "did you mean" suggestions                                          */
/* ------------------------------------------------------------------ */

int jaiNameDistance(const char *a, const char *b, int max) {
    if (a == NULL || b == NULL || max < 0) return max + 1;
    size_t la = strlen(a), lb = strlen(b);
    if (la > JAI_SUGGEST_MAX_LEN || lb > JAI_SUGGEST_MAX_LEN) return max + 1;
    /* The length gap alone is already a lower bound on the distance. */
    size_t gap = la > lb ? la - lb : lb - la;
    if ((int)gap > max) return max + 1;

    /* Three rows, because a transposition looks back two rows. */
    int prev2[JAI_SUGGEST_MAX_LEN + 1];
    int prev[JAI_SUGGEST_MAX_LEN + 1];
    int cur[JAI_SUGGEST_MAX_LEN + 1];
    for (size_t j = 0; j <= lb; j++) prev[j] = (int)j;

    for (size_t i = 1; i <= la; i++) {
        cur[0] = (int)i;
        int best = cur[0];
        for (size_t j = 1; j <= lb; j++) {
            int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            int d = prev[j] + 1;                      /* delete */
            if (cur[j - 1] + 1 < d) d = cur[j - 1] + 1;   /* insert */
            if (prev[j - 1] + cost < d) d = prev[j - 1] + cost; /* substitute */
            /* Adjacent transposition costs one, not two. */
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] &&
                prev2[j - 2] + 1 < d)
                d = prev2[j - 2] + 1;
            cur[j] = d;
            if (d < best) best = d;
        }
        /* Every remaining row can only add to the running minimum. */
        if (best > max) return max + 1;
        memcpy(prev2, prev, sizeof(int) * (lb + 1));
        memcpy(prev, cur, sizeof(int) * (lb + 1));
    }
    return prev[lb];
}

bool jaiNameIsCloser(const char *name, const char *candidate, int *best) {
    if (name == NULL || candidate == NULL || best == NULL) return false;
    if (strcmp(name, candidate) == 0) return false;

    size_t len = strlen(name);
    int limit = (int)((len > 3 ? len : 3) / 3);
    if (limit > 3) limit = 3;

    int d = jaiNameDistance(name, candidate, limit);
    if (d > limit || d >= *best) return false;
    *best = d;
    return true;
}

/* ------------------------------------------------------------------ */
/* The bag                                                             */
/* ------------------------------------------------------------------ */

JaiDiagBag gDiags;

/* Handed back once the error limit is hit so callers can keep attaching
 * labels to a diagnostic that will never be rendered. */
static JaiDiag gSink;

#define JAI_DEFAULT_MAX_ERRORS 25

static bool colorAvailable(void) {
    if (getenv("NO_COLOR") != NULL) return false;
    const char *term = getenv("TERM");
    if (term != NULL && strcmp(term, "dumb") == 0) return false;
    return isatty(2) != 0;
}

void jaiDiagInit(JaiDiagBag *bag) {
    if (bag == NULL) return;
    memset(bag, 0, sizeof *bag);
    JAI_VEC_INIT(&bag->diags);
    bag->maxErrors = JAI_DEFAULT_MAX_ERRORS;
    bag->colorOutput = colorAvailable();
}

static void freeDiag(JaiDiag *d) {
    freeStr(d->message);
    d->message = NULL;
    for (int i = 0; i < d->secondaryCount; i++) {
        freeStr(d->secondary[i].label);
        d->secondary[i].label = NULL;
    }
    d->secondaryCount = 0;
    freeStr(d->help);
    d->help = NULL;
    freeStr(d->note);
    d->note = NULL;
}

void jaiDiagReset(JaiDiagBag *bag) {
    if (bag == NULL) return;
    for (int i = 0; i < bag->diags.count; i++) freeDiag(&bag->diags.data[i]);
    bag->diags.count = 0;
    bag->errorCount = 0;
    bag->warningCount = 0;
}

void jaiDiagFree(JaiDiagBag *bag) {
    if (bag == NULL) return;
    jaiDiagReset(bag);
    JAI_VEC_FREE(JaiDiag, &bag->diags);
}

static JaiDiag *addDiag(JaiDiagCode code, JaiSeverity severity, JaiSpan span,
                        char *message) {
    JaiDiagBag *bag = &gDiags;

    if (severity == JAI_SEV_ERROR) {
        /* errorCount keeps rising past the limit; flush compares it against
         * the number of errors it actually rendered to detect truncation. */
        if (bag->maxErrors > 0 && bag->errorCount >= bag->maxErrors) {
            bag->errorCount++;
            freeStr(message);
            memset(&gSink, 0, sizeof gSink);
            gSink.code = code;
            gSink.severity = severity;
            gSink.primary = span;
            return &gSink;
        }
        bag->errorCount++;
    } else {
        bag->warningCount++;
    }

    JaiDiag d;
    memset(&d, 0, sizeof d);
    d.code = code;
    d.severity = severity;
    d.message = message;
    d.primary = span;
    JAI_VEC_PUSH(JaiDiag, &bag->diags, d);
    return &bag->diags.data[bag->diags.count - 1];
}

JaiDiag *jaiDiagError(JaiDiagCode code, JaiSpan span, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *msg = formatV(fmt, ap);
    va_end(ap);
    return addDiag(code, JAI_SEV_ERROR, span, msg);
}

JaiDiag *jaiDiagWarn(JaiDiagCode code, JaiSpan span, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *msg = formatV(fmt, ap);
    va_end(ap);
    return addDiag(code, JAI_SEV_WARNING, span, msg);
}

void jaiDiagAddLabel(JaiDiag *d, JaiSpan span, const char *fmt, ...) {
    if (d == NULL || d == &gSink) return;
    /* secondary[] is fixed at four by the header; a fifth label is dropped
     * rather than displacing an earlier, usually more relevant one. */
    if (d->secondaryCount >= (int)(sizeof d->secondary / sizeof d->secondary[0])) return;

    char *label = NULL;
    if (fmt != NULL) {
        va_list ap;
        va_start(ap, fmt);
        label = formatV(fmt, ap);
        va_end(ap);
    }
    d->secondary[d->secondaryCount].span = span;
    d->secondary[d->secondaryCount].label = label;
    d->secondaryCount++;
}

void jaiDiagAddHelp(JaiDiag *d, const char *fmt, ...) {
    if (d == NULL || d == &gSink || fmt == NULL) return;
    va_list ap;
    va_start(ap, fmt);
    char *text = formatV(fmt, ap);
    va_end(ap);
    freeStr(d->help);          /* last writer wins; the old text is not leaked */
    d->help = text;
}

void jaiDiagAddNote(JaiDiag *d, const char *fmt, ...) {
    if (d == NULL || d == &gSink || fmt == NULL) return;
    va_list ap;
    va_start(ap, fmt);
    char *text = formatV(fmt, ap);
    va_end(ap);
    freeStr(d->note);
    d->note = text;
}

bool jaiDiagHasErrors(const JaiDiagBag *bag) {
    if (bag == NULL) return false;
    if (bag->errorCount > 0) return true;
    return bag->warningsAsErrors && bag->warningCount > 0;
}

/* ------------------------------------------------------------------ */
/* Rendering                                                           */
/* ------------------------------------------------------------------ */

#define ANSI_RESET  "\x1b[0m"
#define ANSI_BOLD   "\x1b[1m"
#define ANSI_RED    "\x1b[1;31m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_BLUE   "\x1b[34m"

#define JAI_TAB_WIDTH 4

static const char *pick(bool color, const char *code) { return color ? code : ""; }

static const char *severityName(JaiSeverity s) {
    switch (s) {
        case JAI_SEV_ERROR:   return "error";
        case JAI_SEV_WARNING: return "warning";
        case JAI_SEV_NOTE:    return "note";
        case JAI_SEV_HELP:    return "help";
    }
    return "error";
}

static const char *severityColor(JaiSeverity s, bool color) {
    if (!color) return "";
    switch (s) {
        case JAI_SEV_ERROR:   return ANSI_RED;
        case JAI_SEV_WARNING: return ANSI_YELLOW;
        case JAI_SEV_NOTE:
        case JAI_SEV_HELP:    return ANSI_BOLD;
    }
    return ANSI_RED;
}

/* Display column of a byte offset within a line: scalars count one, tabs
 * advance to the next multiple of JAI_TAB_WIDTH. */
static int displayColumn(const char *line, size_t len, size_t byteOffset) {
    if (byteOffset > len) byteOffset = len;
    int col = 0;
    size_t i = 0;
    while (i < byteOffset) {
        if (line[i] == '\t') {
            col = (col / JAI_TAB_WIDTH + 1) * JAI_TAB_WIDTH;
            i++;
            continue;
        }
        int n = 1;
        (void)jaiUtf8Decode(line + i, line + len, &n);
        if (n <= 0) n = 1;
        i += (size_t)n;
        col++;
    }
    return col;
}

/* Writes the line with the same tab expansion displayColumn assumes, so the
 * markers below it land under the right characters. */
static void writeExpanded(FILE *out, const char *line, size_t len) {
    int col = 0;
    size_t i = 0;
    while (i < len) {
        if (line[i] == '\t') {
            int target = (col / JAI_TAB_WIDTH + 1) * JAI_TAB_WIDTH;
            while (col < target) { fputc(' ', out); col++; }
            i++;
            continue;
        }
        if (line[i] == '\r') { i++; continue; }
        int n = 1;
        (void)jaiUtf8Decode(line + i, line + len, &n);
        if (n <= 0) n = 1;
        if (i + (size_t)n > len) n = (int)(len - i);
        (void)fwrite(line + i, 1, (size_t)n, out);
        i += (size_t)n;
        col++;
    }
}

typedef struct {
    int32_t     file;
    int         line;        /* 1-based */
    const char *text;        /* line text, newline excluded */
    size_t      textLen;
    int         startCol;    /* display column, 0-based, tabs expanded */
    int         col;         /* 1-based scalar column, as jaiSourceLineCol reports */
    int         width;       /* caret run length, >= 1 */
    const char *label;       /* borrowed from the diagnostic; may be NULL */
    bool        isPrimary;
} Marker;

#define JAI_MAX_MARKERS 5

static bool spanEqual(JaiSpan a, JaiSpan b) {
    return a.file == b.file && a.start == b.start && a.end == b.end;
}

static bool makeMarker(JaiSpan span, const char *label, bool isPrimary, Marker *m) {
    if (!jaiSpanValid(span)) return false;
    JaiSourceFile *f = jaiSourceGet(span.file);
    if (f == NULL) return false;

    size_t len = 0;
    int line = 0;
    const char *text = jaiSourceLineText(span.file, span.start, &len, &line);
    if (text == NULL) return false;

    size_t lineStart = (size_t)(text - f->source);
    size_t s = span.start > lineStart ? span.start - lineStart : 0;
    size_t e = span.end > lineStart ? span.end - lineStart : 0;
    if (s > len) s = len;
    if (e > len) e = len;          /* multi-line spans stop at the line end */
    if (e < s) e = s;

    m->file = span.file;
    m->line = line;
    m->text = text;
    m->textLen = len;
    /* The caret sits at the tab-expanded column, but the reported column is
     * the scalar count so that it agrees with jaiSourceLineCol (spec §1.1). */
    m->startCol = displayColumn(text, len, s);
    m->col = (int)jaiUtf8Length(text, s) + 1;
    int endCol = displayColumn(text, len, e);
    m->width = endCol - m->startCol;
    if (m->width < 1) m->width = 1;
    m->label = label;
    m->isPrimary = isPrimary;
    return true;
}

static void sortMarkers(Marker *m, int n) {
    for (int i = 1; i < n; i++) {
        Marker key = m[i];
        int j = i - 1;
        while (j >= 0 && (m[j].line > key.line ||
                          (m[j].line == key.line && m[j].startCol > key.startCol))) {
            m[j + 1] = m[j];
            j--;
        }
        m[j + 1] = key;
    }
}

static int digitCount(int n) {
    int d = 1;
    while (n >= 10) { n /= 10; d++; }
    return d;
}

static void renderGutterBar(FILE *out, int gw, const char *blue, const char *reset) {
    fprintf(out, "%s%*s |%s\n", blue, gw, "", reset);
}

/* One location block: the arrow, then every marker line for this file in
 * source order, elided with "..." across gaps. */
static void renderBlock(FILE *out, const Marker *m, int n, int gw,
                        JaiSeverity severity, bool color) {
    const char *blue = pick(color, ANSI_BLUE);
    const char *reset = pick(color, ANSI_RESET);
    const char *sev = severityColor(severity, color);

    JaiSourceFile *f = jaiSourceGet(m[0].file);
    const char *path = (f != NULL && f->path != NULL) ? f->path : "<unknown>";

    /* The arrow points at the primary marker even when a secondary sorts
     * ahead of it. */
    const Marker *anchor = &m[0];
    for (int i = 0; i < n; i++) {
        if (m[i].isPrimary) { anchor = &m[i]; break; }
    }
    fprintf(out, "%s%*s-->%s %s:%d:%d\n", blue, gw, "", reset, path, anchor->line,
            anchor->col);
    renderGutterBar(out, gw, blue, reset);

    int prevLine = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0 && m[i].line != prevLine) {
            /* The elision sits in the gutter, right-aligned with the bar. */
            if (m[i].line - prevLine > 1) fprintf(out, "%s%*s%s\n", blue, gw + 2, "...", reset);
        }
        if (m[i].line != prevLine) {
            fprintf(out, "%s%*d |%s ", blue, gw, m[i].line, reset);
            writeExpanded(out, m[i].text, m[i].textLen);
            fputc('\n', out);
            prevLine = m[i].line;
        }

        const char *mc = m[i].isPrimary ? sev : blue;
        fprintf(out, "%s%*s |%s ", blue, gw, "", reset);
        for (int c = 0; c < m[i].startCol; c++) fputc(' ', out);
        fputs(mc, out);
        for (int c = 0; c < m[i].width; c++) fputc(m[i].isPrimary ? '^' : '-', out);
        if (m[i].label != NULL) fprintf(out, " %s", m[i].label);
        fprintf(out, "%s\n", reset);
    }
}

void jaiDiagRender(const JaiDiag *d, FILE *out, bool color) {
    if (d == NULL || out == NULL) return;
    if (d == &gSink) return;   /* dropped by the error limit; never shown */

    const char *reset = pick(color, ANSI_RESET);
    const char *bold = pick(color, ANSI_BOLD);
    const char *blue = pick(color, ANSI_BLUE);
    const char *sev = severityColor(d->severity, color);
    const char *message = d->message != NULL ? d->message : "";

    if (d->code != JAI_OK) {
        fprintf(out, "%s%s[%s]%s: %s%s%s\n", sev, severityName(d->severity),
                jaiDiagCodeString(d->code), reset, bold, message, reset);
    } else {
        fprintf(out, "%s%s%s: %s%s%s\n", sev, severityName(d->severity), reset,
                bold, message, reset);
    }

    /* JaiDiag.primary is a bare span, so the only way to label the caret run
     * is a label attached to exactly that span; it folds into the primary
     * marker instead of drawing a second underline on the same columns. */
    const char *primaryLabel = NULL;
    for (int i = 0; i < d->secondaryCount; i++) {
        if (spanEqual(d->secondary[i].span, d->primary) && d->secondary[i].label != NULL) {
            primaryLabel = d->secondary[i].label;
            break;
        }
    }

    Marker markers[JAI_MAX_MARKERS];
    int n = 0;
    if (makeMarker(d->primary, primaryLabel, true, &markers[n])) n++;
    for (int i = 0; i < d->secondaryCount && n < JAI_MAX_MARKERS; i++) {
        if (spanEqual(d->secondary[i].span, d->primary)) continue;
        if (makeMarker(d->secondary[i].span, d->secondary[i].label, false, &markers[n])) n++;
    }

    if (n > 0) {
        int maxLine = 1;
        for (int i = 0; i < n; i++) {
            if (markers[i].line > maxLine) maxLine = markers[i].line;
        }
        int gw = digitCount(maxLine) + 1;

        /* Group by file, first appearance first, so the primary's file leads. */
        bool taken[JAI_MAX_MARKERS] = {false};
        for (int i = 0; i < n; i++) {
            if (taken[i]) continue;
            Marker group[JAI_MAX_MARKERS];
            int gn = 0;
            for (int j = i; j < n; j++) {
                if (!taken[j] && markers[j].file == markers[i].file) {
                    group[gn++] = markers[j];
                    taken[j] = true;
                }
            }
            sortMarkers(group, gn);
            renderBlock(out, group, gn, gw, d->severity, color);
        }

        if (d->help != NULL || d->note != NULL) renderGutterBar(out, gw, blue, reset);
    }

    if (d->help != NULL) fprintf(out, "%shelp:%s %s\n", bold, reset, d->help);
    if (d->note != NULL) fprintf(out, "%snote:%s %s\n", bold, reset, d->note);
}

bool jaiDiagFlush(JaiDiagBag *bag, FILE *out) {
    if (bag == NULL) return false;
    bool hadErrors = jaiDiagHasErrors(bag);

    if (!bag->quiet && out != NULL) {
        int renderedErrors = 0;
        for (int i = 0; i < bag->diags.count; i++) {
            if (i > 0) fputc('\n', out);
            jaiDiagRender(&bag->diags.data[i], out, bag->colorOutput);
            if (bag->diags.data[i].severity == JAI_SEV_ERROR) renderedErrors++;
        }
        if (bag->errorCount > renderedErrors) {
            if (bag->diags.count > 0) fputc('\n', out);
            fprintf(out, "%serror%s: %stoo many errors emitted, stopping now%s\n",
                    severityColor(JAI_SEV_ERROR, bag->colorOutput),
                    pick(bag->colorOutput, ANSI_RESET),
                    pick(bag->colorOutput, ANSI_BOLD),
                    pick(bag->colorOutput, ANSI_RESET));
        }
        fflush(out);
    }

    jaiDiagReset(bag);
    return hadErrors;
}

/* ------------------------------------------------------------------ */
/* Runtime tracebacks                                                  */
/* ------------------------------------------------------------------ */

void jaiPrintTraceback(FILE *out, const JaiFrameInfo *frames, int count,
                       const char *excType, const char *excMessage, bool color) {
    if (out == NULL) return;

    const char *reset = pick(color, ANSI_RESET);
    const char *bold = pick(color, ANSI_BOLD);
    const char *blue = pick(color, ANSI_BLUE);
    const char *red = pick(color, ANSI_RED);

    if (frames != NULL && count > 0) {
        fprintf(out, "%sTraceback (most recent call last):%s\n", bold, reset);
        /* frames[0] is the outermost call — "most recent call last". */
        for (int i = 0; i < count; i++) {
            const JaiFrameInfo *fr = &frames[i];
            const char *name = fr->functionName != NULL ? fr->functionName : "<module>";
            const char *path = fr->modulePath;
            int line = 0;

            JaiSourceFile *sf = jaiSpanValid(fr->span) ? jaiSourceGet(fr->span.file) : NULL;
            if (sf != NULL) {
                int col = 0;
                jaiSourceLineCol(sf->id, fr->span.start, &line, &col);
                if (path == NULL) path = sf->path;
            }
            if (path == NULL) path = "<native>";

            if (line > 0) {
                fprintf(out, "  %sFile \"%s\", line %d, in %s%s\n", blue, path, line,
                        name, reset);
            } else {
                fprintf(out, "  %sFile \"%s\", in %s%s\n", blue, path, name, reset);
            }

            if (sf == NULL) continue;
            size_t len = 0;
            const char *text = jaiSourceLineText(sf->id, fr->span.start, &len, NULL);
            if (text == NULL) continue;
            /* Re-indent to a fixed four spaces, as Python does, so deeply
             * nested source stays readable in the traceback. */
            size_t s = 0;
            while (s < len && (text[s] == ' ' || text[s] == '\t')) s++;
            if (s < len) {
                fputs("    ", out);
                writeExpanded(out, text + s, len - s);
                fputc('\n', out);
            }
        }
    }

    const char *type = (excType != NULL && excType[0] != '\0') ? excType : "Error";
    if (excMessage != NULL && excMessage[0] != '\0') {
        fprintf(out, "%s%s%s: %s%s%s\n", red, type, reset, bold, excMessage, reset);
    } else {
        fprintf(out, "%s%s%s\n", red, type, reset);
    }
    fflush(out);
}
