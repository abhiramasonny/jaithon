#include "trace.h"

#include <string.h>

#include "vm/object/object.h"

static JaiTraceSession gSession;
static JaiTraceSession gCached;
static bool            gReady;
static int             gDepth;

void jaiTraceInit(void) {
    memset(&gSession, 0, sizeof gSession);
    memset(&gCached, 0, sizeof gCached);
    gReady = false;
    gDepth = 0;
}

static bool shapesEqual(const JaiTraceOp *a, const int *shape, int shape_len) {
    if (a->shape_len != shape_len) return false;
    for (int i = 0; i < shape_len; i++) {
        int got = shape != NULL ? shape[i] : 0;
        if (a->shape[i] != got) return false;
    }
    return true;
}

void jaiTraceEnter(ObjFunction *fn) {
    if (fn == NULL || !(fn->flags & FN_TRACE)) return;

    /* Nested `@trace` callees keep the outermost session: ML kernels are
     * marked so they JIT sooner, but the training-step function owns the
     * recorded graph. */
    if (gDepth > 0) {
        gDepth++;
        return;
    }
    gDepth = 1;

    if (gReady && gCached.fn == fn && gCached.op_count > 0) {
        gSession = gCached;
        gSession.run_id++;
        gSession.shapes_match = true;
        gSession.replay_at = 0;
        gSession.replaying = true;
        return;
    }

    memset(&gSession, 0, sizeof gSession);
    gSession.fn = fn;
    gSession.run_id = 1;
    gSession.shapes_match = false;
    gSession.replay_at = 0;
    gSession.replaying = false;
}

void jaiTraceLeave(ObjFunction *fn) {
    if (fn == NULL || !(fn->flags & FN_TRACE)) return;
    if (gDepth > 1) {
        gDepth--;
        return;
    }
    if (gDepth == 0) return;
    gDepth = 0;
    if (gSession.fn != fn) {
        memset(&gSession, 0, sizeof gSession);
        return;
    }

    if (gSession.replaying) {
        if (!gSession.shapes_match || gSession.replay_at != gCached.op_count) {
            gReady = false;
            memset(&gCached, 0, sizeof gCached);
        }
    } else if (gSession.op_count > 0) {
        gCached = gSession;
        gReady = true;
    }

    memset(&gSession, 0, sizeof gSession);
}

bool jaiTraceIsActive(void) {
    return gSession.fn != NULL;
}

uint32_t jaiTraceRunId(void) {
    return gSession.run_id;
}

bool jaiTraceRecordOp(const char *name, const int *shape, int shape_len) {
    if (!jaiTraceIsActive() || name == NULL) return false;
    if (shape_len > 4) shape_len = 4;
    if (shape_len < 0) shape_len = 0;

    if (gSession.shapes_match) {
        if (gSession.replay_at >= gCached.op_count) {
            gSession.shapes_match = false;
            gReady = false;
            return false;
        }
        const JaiTraceOp *expected = &gCached.ops[gSession.replay_at];
        if (strcmp(expected->name, name) != 0 ||
            !shapesEqual(expected, shape, shape_len)) {
            gSession.shapes_match = false;
            gReady = false;
            return false;
        }
        gSession.replay_at++;
        return true;
    }

    if (gSession.op_count >= JAI_TRACE_MAX_OPS) return false;

    JaiTraceOp *op = &gSession.ops[gSession.op_count++];
    memset(op, 0, sizeof *op);
    strncpy(op->name, name, sizeof op->name - 1);
    op->shape_len = shape_len;
    if (shape != NULL) {
        for (int i = 0; i < shape_len; i++) op->shape[i] = shape[i];
    }
    return true;
}

int jaiTraceOpCount(void) {
    return gSession.op_count;
}

bool jaiTraceReplay(void) {
    return gSession.shapes_match && gSession.op_count > 0;
}
