// diag.c has source registry, diagnostic collection, and rendering.

#include "common/diag.h"
#include <stdlib.h>
#include <unistd.h>

static void freeStr(char *s) {
    if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
}

static char *formatV(const char *fmt, va_list ap) {
    char stack[256];

    va_list probe;
    va_copy(probe, ap);
    int n = vsnprintf(stack, sizeof stack, fmt, probe);
    va_end(probe);

    if (n < 0)
        return jaiStrdup("<malformed diagnostic message>");

    const size_t length = (size_t)n;
    char *buf = JAI_ALLOC(char, length + 1);

    if (length < sizeof stack) {
        memcpy(buf, stack, length + 1);
        return buf;
    }

    va_list render;
    va_copy(render, ap);
    (void)vsnprintf(buf, length + 1, fmt, render);
    va_end(render);
    return buf;
}

static JAI_VEC(JaiSourceFile) gSources;

static void buildLineIndex(JaiSourceFile *f) {
    const char *const source = f->source;
    const char *const end = source + f->length;

    int lines = 1;
    const char *p = source;

    while (p < end) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        if (nl == NULL) break;
        ++lines;
        p = nl + 1;
    }

    f->lineStarts = JAI_ALLOC(uint32_t, lines);
    f->lineCount = lines;

    int n = 0;
    f->lineStarts[n++] = 0;
    p = source;

    while (p < end && n < lines) {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        if (nl == NULL) break;
        f->lineStarts[n++] = (uint32_t)((nl - source) + 1);
        p = nl + 1;
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

static inline int lineIndexFor(const JaiSourceFile *f, uint32_t offset) {
    const int count = f->lineCount;
    if (count <= 1) return 0;

    const uint32_t *const starts = f->lineStarts;

    if (offset < starts[1])
        return 0;

    const int last = count - 1;
    if (offset >= starts[last])
        return last;

    int lo = 1;
    int hi = last - 1;

    while (lo <= hi) {
        const int mid = lo + ((hi - lo) >> 1);

        if (starts[mid] <= offset) {
            if (starts[mid + 1] > offset)
                return mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return hi < 0 ? 0 : hi;
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
    if (col != NULL) { *col = (int)jaiUtf8Length(f->source + start, offset - start) + 1; }
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

JaiSpan jaiSpanJoin(JaiSpan a, JaiSpan b) {
    if (a.file < 0) return b;
    if (b.file < 0) return a;
    if (a.file != b.file) return a;

    JaiSpan out;
    out.file = a.file;
    out.start = a.start < b.start ? a.start : b.start;
    out.end = a.end > b.end ? a.end : b.end;
    if (out.end < out.start) out.end = out.start;
    return out;
}

//error codes

#define JAI_WARNING_BASE 10000

bool jaiDiagCodeIsWarning(JaiDiagCode code) {
    return (int)code >= JAI_WARNING_BASE;
}

const char *jaiDiagCodeString(JaiDiagCode code) {
    static char bufs[4][16];
    static unsigned next = 0;

    if (code == JAI_OK) return "";

    char *const b = bufs[next];
    next = (next + 1u) & 3u;

    int value;
    char prefix;

    if (jaiDiagCodeIsWarning(code)) {
        prefix = 'W';
        value = (int)code - JAI_WARNING_BASE + 100;
    } else {
        prefix = 'E';
        value = (int)code;
    }

    if ((unsigned)value <= 9999u) {
        b[0] = prefix;
        b[1] = (char)('0' + (value / 1000) % 10);
        b[2] = (char)('0' + (value / 100) % 10);
        b[3] = (char)('0' + (value / 10) % 10);
        b[4] = (char)('0' + value % 10);
        b[5] = '\0';
        return b;
    }

    (void)snprintf(b, sizeof bufs[0], "%c%04d", prefix, value);
    return b;
}

// "E0402" or "W0101" back to the code it names, or JAI_OK
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

static int nameDistanceSized(const char *a, size_t la,
                             const char *b, size_t lb, int max) {
    if (max < 0) return max + 1;
    if (la > JAI_SUGGEST_MAX_LEN || lb > JAI_SUGGEST_MAX_LEN)
        return max + 1;

    const size_t gap = la > lb ? la - lb : lb - la;
    if ((int)gap > max)
        return max + 1;

    const int inf = max + 1;
    int row0[JAI_SUGGEST_MAX_LEN + 1];
    int row1[JAI_SUGGEST_MAX_LEN + 1];
    int row2[JAI_SUGGEST_MAX_LEN + 1];

    int *prev2 = row0;
    int *prev = row1;
    int *cur = row2;

    for (size_t j = 0; j <= lb; ++j) {
        prev2[j] = inf;
        prev[j] = j <= (size_t)max ? (int)j : inf;
        cur[j] = inf;
    }

    for (size_t i = 1; i <= la; ++i) {
        size_t lo = i > (size_t)max ? i - (size_t)max : 1u;
        size_t hi = i + (size_t)max;
        if (hi > lb) hi = lb;

        cur[0] = i <= (size_t)max ? (int)i : inf;

        if (lo > 1)
            cur[lo - 1] = inf;

        int best = lo == 1 ? cur[0] : inf;

        for (size_t j = lo; j <= hi; ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;

            int d = prev[j] + 1;

            const int insertion = cur[j - 1] + 1;
            if (insertion < d) d = insertion;

            const int substitution = prev[j - 1] + cost;
            if (substitution < d) d = substitution;

            if (i > 1 && j > 1 &&
                a[i - 1] == b[j - 2] &&
                a[i - 2] == b[j - 1]) {
                const int transpose = prev2[j - 2] + 1;
                if (transpose < d) d = transpose;
            }

            cur[j] = d;
            if (d < best) best = d;
        }

        if (hi < lb)
            cur[hi + 1] = inf;

        if (best > max)
            return max + 1;

        int *tmp = prev2;
        prev2 = prev;
        prev = cur;
        cur = tmp;
    }

    return prev[lb] <= max ? prev[lb] : max + 1;
}

int jaiNameDistance(const char *a, const char *b, int max) {
    if (a == NULL || b == NULL)
        return max + 1;

    return nameDistanceSized(a, strlen(a), b, strlen(b), max);
}

bool jaiNameIsCloser(const char *name, const char *candidate, int *best) {
    if (name == NULL || candidate == NULL || best == NULL)
        return false;

    const size_t nameLen = strlen(name);
    const size_t candidateLen = strlen(candidate);

    if (nameLen == candidateLen &&
        memcmp(name, candidate, nameLen) == 0)
        return false;

    int limit = (int)((nameLen > 3 ? nameLen : 3) / 3);
    if (limit > 3) limit = 3;

    const int d =
        nameDistanceSized(name, nameLen, candidate, candidateLen, limit);

    if (d > limit || d >= *best)
        return false;

    *best = d;
    return true;
}

JaiDiagBag gDiags;
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
    freeStr(d->help);
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
    return bag != NULL &&
           (bag->errorCount > 0 ||
            (bag->warningsAsErrors && bag->warningCount > 0));
}

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

static int displayColumn(const char *line, size_t len, size_t byteOffset) {
    if (byteOffset > len) byteOffset = len;

    int col = 0;
    size_t i = 0;

    while (i < byteOffset) {
        const unsigned char c = (unsigned char)line[i];

        if (c == '\t') {
            col = (col / JAI_TAB_WIDTH + 1) * JAI_TAB_WIDTH;
            ++i;
            continue;
        }

        if (c < 0x80u) {
            ++i;
            ++col;
            continue;
        }

        int n = 1;
        (void)jaiUtf8Decode(line + i, line + len, &n);
        if (n <= 0) n = 1;

        i += (size_t)n;
        ++col;
    }

    return col;
}

static void writeExpanded(FILE *out, const char *line, size_t len) {
    int col = 0;
    size_t i = 0;

    while (i < len) {
        const unsigned char c = (unsigned char)line[i];

        if (c == '\t') {
            const int target =
                (col / JAI_TAB_WIDTH + 1) * JAI_TAB_WIDTH;

            while (col < target) {
                fputc(' ', out);
                ++col;
            }

            ++i;
            continue;
        }

        if (c == '\r') {
            ++i;
            continue;
        }

        if (c < 0x80u) {
            const size_t start = i;

            do {
                ++i;
            } while (i < len &&
                     (unsigned char)line[i] < 0x80u &&
                     line[i] != '\t' &&
                     line[i] != '\r');

            const size_t run = i - start;
            (void)fwrite(line + start, 1, run, out);
            col += (int)run;
            continue;
        }

        int n = 1;
        (void)jaiUtf8Decode(line + i, line + len, &n);
        if (n <= 0) n = 1;
        if (i + (size_t)n > len)
            n = (int)(len - i);

        (void)fwrite(line + i, 1, (size_t)n, out);
        i += (size_t)n;
        ++col;
    }
}

typedef struct {
    int32_t     file;
    int         line;
    const char *text;
    size_t      textLen;
    int         startCol;
    int         col;
    int         width;
    const char *label;
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
    if (e > len) e = len;
    if (e < s) e = s;

    m->file = span.file;
    m->line = line;
    m->text = text;
    m->textLen = len;

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

static inline int digitCount(int n) {
    if (n < 0) n = -n;
    if (n == 0) return 1;
    return (int)floor(log10(n)) + 1;
}

static void renderGutterBar(FILE *out, int gw, const char *blue, const char *reset) {
    fprintf(out, "%s%*s |%s\n", blue, gw, "", reset);
}

static void writeRepeatedChar(FILE *out, char c, int count) {
    if (count <= 0) return;

    char block[64];
    memset(block, (unsigned char)c, sizeof block);

    while (count >= (int)sizeof block) {
        (void)fwrite(block, 1, sizeof block, out);
        count -= (int)sizeof block;
    }

    if (count > 0)
        (void)fwrite(block, 1, (size_t)count, out);
}

static void renderBlock(FILE *out, const Marker *m, int n, int gw,
                        JaiSeverity severity, bool color) {
    const char *blue = pick(color, ANSI_BLUE);
    const char *reset = pick(color, ANSI_RESET);
    const char *sev = severityColor(severity, color);

    JaiSourceFile *f = jaiSourceGet(m[0].file);
    const char *path = (f != NULL && f->path != NULL) ? f->path : "<unknown>";

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
        writeRepeatedChar(out, ' ', m[i].startCol);
        fputs(mc, out);
        writeRepeatedChar(out, m[i].isPrimary ? '^' : '-', m[i].width);
        if (m[i].label != NULL) fprintf(out, " %s", m[i].label);
        fprintf(out, "%s\n", reset);
    }
}

void jaiDiagRender(const JaiDiag *d, FILE *out, bool color) {
    if (d == NULL || out == NULL) return;
    if (d == &gSink) return;

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

void jaiPrintTraceback(FILE *out, const JaiFrameInfo *frames, int count,
                       const char *excType, const char *excMessage, bool color) {
    if (out == NULL) return;

    const char *reset = pick(color, ANSI_RESET);
    const char *bold = pick(color, ANSI_BOLD);
    const char *blue = pick(color, ANSI_BLUE);
    const char *red = pick(color, ANSI_RED);

    if (frames != NULL && count > 0) {
        fprintf(out, "%sTraceback (most recent call last):%s\n", bold, reset);
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
