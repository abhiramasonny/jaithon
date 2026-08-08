/* semadump.c — the writer described in semadump.h.
 *
 * Every record is (span, kind, field, value). Records are sorted before they
 * are written, so the same program always produces the same bytes: a format
 * whose ordering depends on the order the checker happened to visit nodes in
 * cannot be diffed against a second implementation that visits them in
 * another, which is the only thing this file exists to allow.
 */
#include "semadump.h"

#include <stdlib.h>   /* qsort */

typedef struct {
    int   start;
    int   end;
    const char *kind;     /* interned by jaiAstKindName; not owned */
    const char *field;    /* a literal; not owned */
    char *value;          /* owned */
} DumpRecord;

static struct {
    JAI_VEC(DumpRecord) records;
    char *armedPath;      /* owned; where a dump goes when one is begun */
    char *sourcePath;     /* owned */
    bool  active;
} sDump;

static char *dupString(const char *s) {
    if (s == NULL) s = "";
    size_t n = strlen(s) + 1;
    char *copy = JAI_ALLOC(char, n);
    memcpy(copy, s, n);
    return copy;
}

static void freeString(char *s) {
    if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
}

bool jaiSemaDumpActive(void) { return sDump.active; }
bool jaiSemaDumpArmed(void) { return sDump.armedPath != NULL; }

void jaiSemaDumpArm(const char *outPath) {
    freeString(sDump.armedPath);
    sDump.armedPath = outPath == NULL ? NULL : dupString(outPath);
}

void jaiSemaDumpBegin(const char *sourcePath) {
    JAI_VEC_INIT(&sDump.records);
    freeString(sDump.sourcePath);
    sDump.sourcePath = NULL;
    sDump.active = false;
    if (sDump.armedPath == NULL) return;

    sDump.sourcePath = dupString(sourcePath);
    sDump.active = true;
}

/* A span with no file is a synthesised node — a desugaring, a default thunk —
 * and it has no counterpart in the other front end's tree to be compared
 * against, so recording it would report a mismatch that means nothing. */
static void record(JaiSpan span, AstKind kind, const char *field,
                   const char *value) {
    if (!sDump.active || !jaiSpanValid(span)) return;
    DumpRecord r;
    r.start = (int)span.start;
    r.end = (int)span.end;
    r.kind = jaiAstKindName(kind);
    r.field = field;
    r.value = dupString(value);
    JAI_VEC_PUSH(DumpRecord, &sDump.records, r);
}

void jaiSemaDumpFold(JaiSpan span, AstKind kind, const char *rendered) {
    record(span, kind, "fold", rendered);
}

void jaiSemaDumpCast(JaiSpan span, AstKind kind, const JaiType *from,
                     const JaiType *to, bool widen) {
    if (!sDump.active) return;
    /* jaiTypeToString hands back a buffer it owns until the next call, so the
     * first rendering is copied before the second overwrites it. */
    char *fromText = dupString(from == NULL ? "?" : jaiTypeToString((JaiType *)from));
    const char *toText = to == NULL ? "?" : jaiTypeToString((JaiType *)to);

    char rendered[256];
    snprintf(rendered, sizeof rendered, "%s->%s", fromText, toText);
    freeString(fromText);

    record(span, kind, widen ? "widen" : "guard", rendered);
}

static bool recordType(AstNode *node, void *userData) {
    (void)userData;
    if (node != NULL && node->type != NULL) {
        record(node->span, node->kind, "type", jaiTypeToString(node->type));
    }
    return true;
}

void jaiSemaDumpTypes(AstNode *program) {
    if (!sDump.active || program == NULL) return;
    jaiAstWalk(program, recordType, NULL, NULL);
}

static int compareRecords(const void *a, const void *b) {
    const DumpRecord *x = a, *y = b;
    if (x->start != y->start) return x->start < y->start ? -1 : 1;
    if (x->end != y->end) return x->end < y->end ? -1 : 1;
    int kind = strcmp(x->kind, y->kind);
    if (kind != 0) return kind;
    int field = strcmp(x->field, y->field);
    if (field != 0) return field;
    return strcmp(x->value, y->value);
}

/* Two nodes can cover the same span with the same field — a cast wraps its
 * operand in place and both carry the operand's span — and an exact duplicate
 * says nothing the first one did not. Dropping them keeps the file a set of
 * decisions rather than a log of visits. */
static bool sameRecord(const DumpRecord *a, const DumpRecord *b) {
    return a->start == b->start && a->end == b->end &&
           strcmp(a->kind, b->kind) == 0 && strcmp(a->field, b->field) == 0 &&
           strcmp(a->value, b->value) == 0;
}

/* The armed path outlives one dump: a driver arms once and may check several
 * files, and re-arming per file would put the destination back in the hands of
 * whatever ran last. */
static void releaseAll(void) {
    for (int i = 0; i < sDump.records.count; i++) {
        freeString(sDump.records.data[i].value);
    }
    JAI_VEC_FREE(DumpRecord, &sDump.records);
    freeString(sDump.sourcePath);
    sDump.sourcePath = NULL;
    sDump.active = false;
}

bool jaiSemaDumpEnd(void) {
    if (!sDump.active) return true;

    if (sDump.records.count > 1) {
        qsort(sDump.records.data, (size_t)sDump.records.count,
              sizeof(DumpRecord), compareRecords);
    }

    FILE *out = fopen(sDump.armedPath, "wb");
    if (out == NULL) {
        releaseAll();
        return false;
    }

    fprintf(out, "# jaithon sema dump v%d\n", JAI_SEMADUMP_VERSION);
    fprintf(out, "# file %s\n", sDump.sourcePath);
    for (int i = 0; i < sDump.records.count; i++) {
        const DumpRecord *r = &sDump.records.data[i];
        if (i > 0 && sameRecord(r, &sDump.records.data[i - 1])) continue;
        fprintf(out, "%d %d %s %s %s\n", r->start, r->end, r->kind, r->field,
                r->value);
    }

    bool ok = ferror(out) == 0;
    if (fclose(out) != 0) ok = false;
    releaseAll();
    return ok;
}
