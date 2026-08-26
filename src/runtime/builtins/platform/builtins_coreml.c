/* builtins_coreml.c — `__prim__.coreml_*`, running a model on the neural
 * accelerator.
 *
 * The accelerator is separate silicon from the GPU and the two overlap almost
 * perfectly, so the reason to reach it is to run a model at the same time as
 * GPU work rather than instead of it. See src/native/apple/coreml.m. */
#include "runtime/runtime.h"
#include "runtime/handles.h"
#include <stdint.h>
#include <string.h>

#include "native/native.h"
#include "vm/gc.h"

#define COREML_ERROR_BUFFER 1024

static bool requireModel(Value v, int index, const char *fnName, JaiCoreMLModel **out) {
    void *ptr = NULL;
    if (!jaiHandleGet(v, index, HANDLE_COREML_MODEL, fnName, &ptr)) return false;
    *out = (JaiCoreMLModel *)ptr;
    return true;
}

static bool nCoreMLOpen(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_STRING(args[0]))
        return jaiThrow(vm.cTypeError, "coreml_open() argument 1 is %s, expected a string",
                        jaiTypeNameStatic(args[0]));
    int64_t units;
    if (!jaiArgInt(args[1], 2, "coreml_open", &units)) return false;
    if (units < 0 || units > 3)
        return jaiThrow(vm.cValueError,
                        "coreml_open(): compute units must be 0 to 3, got %lld",
                        (long long)units);

    char trouble[COREML_ERROR_BUFFER];
    JaiCoreMLModel *model = jaiCoreMLOpen(AS_CSTRING(args[0]), (int)units,
                                          trouble, sizeof(trouble));
    if (model == NULL) {
        return jaiThrow(vm.cRuntimeError, "coreml_open(): %s",
                        trouble[0] != '\0' ? trouble : "no CoreML on this machine");
    }
    *out = INT_VAL(jaiHandleAdd(HANDLE_COREML_MODEL, model));
    return true;
}

static bool nCoreMLClose(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCoreMLModel *model;
    if (!requireModel(args[0], 1, "coreml_close", &model)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    jaiCoreMLClose(model);
    *out = NULL_VAL;
    return true;
}

/* `coreml_names(model, outputs)` -- what the model calls each of its inputs
 * or outputs, in the order every other call here expects them. */
static bool nCoreMLNames(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCoreMLModel *model;
    int64_t outputs;
    if (!requireModel(args[0], 1, "coreml_names", &model)) return false;
    if (!jaiArgInt(args[1], 2, "coreml_names", &outputs)) return false;
    const int count = jaiCoreMLCount(model, outputs != 0);
    if (count < 0) return jaiThrow(vm.cRuntimeError, "coreml_names(): the model has no such side");

    ObjList *list = jaiListNew(count);
    if (list == NULL) return false;
    jaiGCPushRoot(OBJ_VAL(list));
    for (int i = 0; i < count; i++) {
        const char *name = jaiCoreMLName(model, outputs != 0, i);
        ObjString *held = jaiStringInternC(name != NULL ? name : "");
        if (held == NULL) {
            jaiGCPopRoot();
            return false;
        }
        jaiListPush(list, OBJ_VAL(held));
    }
    jaiGCPopRoot();
    *out = OBJ_VAL(list);
    return true;
}

/* `coreml_shape(model, outputs, index)` -- the declared shape, or an empty
 * list when the model leaves it open. */
static bool nCoreMLShape(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCoreMLModel *model;
    int64_t outputs, index;
    if (!requireModel(args[0], 1, "coreml_shape", &model)) return false;
    if (!jaiArgInt(args[1], 2, "coreml_shape", &outputs)) return false;
    if (!jaiArgInt(args[2], 3, "coreml_shape", &index)) return false;

    int64_t dims[8];
    const int rank = jaiCoreMLShape(model, outputs != 0, (int)index, dims, 8);
    ObjList *list = jaiListNew(rank > 0 ? rank : 0);
    if (list == NULL) return false;
    if (rank > 0) {
        if (!jaiListReserveExact(list, rank)) return false;
        for (int i = 0; i < rank; i++) jaiListPut(list, i, INT_VAL(dims[i]));
        list->count = rank;
        list->version++;
    }
    *out = OBJ_VAL(list);
    return true;
}

/* Device buffers and the shapes to read them with, for one side of a run. */
typedef struct {
    JaiGpuBuffer **buffers;
    size_t        *offsets;
    int64_t       *shapes;
    int           *ranks;
    int            count;
    int            dims;
} CoreMLSide;

static void releaseSide(CoreMLSide *side) {
    if (side->buffers != NULL) JAI_FREE_ARRAY(JaiGpuBuffer *, side->buffers, (size_t)side->count);
    if (side->offsets != NULL) JAI_FREE_ARRAY(size_t, side->offsets, (size_t)side->count);
    if (side->shapes != NULL) JAI_FREE_ARRAY(int64_t, side->shapes, (size_t)side->dims);
    if (side->ranks != NULL) JAI_FREE_ARRAY(int, side->ranks, (size_t)side->count);
    side->buffers = NULL;
    side->offsets = NULL;
    side->shapes = NULL;
    side->ranks = NULL;
}

/* `buffers` is a list of handles, `shapes` a list of dimension lists. */
static bool readSide(Value handles, Value shapes, int index, const char *fnName,
                     CoreMLSide *side) {
    ObjList *held;
    ObjList *sizes;
    if (!jaiArgList(handles, index, fnName, &held)) return false;
    if (!jaiArgList(shapes, index + 1, fnName, &sizes)) return false;
    if (held->count != sizes->count)
        return jaiThrow(vm.cValueError, "%s(): %d buffers against %d shapes",
                        fnName, held->count, sizes->count);

    side->count = held->count;
    side->dims = 0;
    for (int i = 0; i < sizes->count; i++) {
        if (!IS_LIST(jaiListGet(sizes, i)))
            return jaiThrow(vm.cTypeError, "%s(): shape %d is %s, expected a list",
                            fnName, i, jaiTypeNameStatic(jaiListGet(sizes, i)));
        side->dims += AS_LIST(jaiListGet(sizes, i))->count;
    }
    if (side->count == 0) return true;

    side->buffers = JAI_ALLOC(JaiGpuBuffer *, (size_t)side->count);
    side->offsets = JAI_ALLOC(size_t, (size_t)side->count);
    side->ranks = JAI_ALLOC(int, (size_t)side->count);
    side->shapes = side->dims > 0 ? JAI_ALLOC(int64_t, (size_t)side->dims) : NULL;

    int at = 0;
    for (int i = 0; i < side->count; i++) {
        JaiGpuBuffer *native = NULL;
        int64_t origin = 0;
        if (!jaiGpuBufferOf(jaiListGet(held, i), index, fnName, &native, &origin)) {
            releaseSide(side);
            return false;
        }
        side->buffers[i] = native;
        side->offsets[i] = (size_t)origin;
        ObjList *shape = AS_LIST(jaiListGet(sizes, i));
        side->ranks[i] = shape->count;
        for (int d = 0; d < shape->count; d++) {
            if (!IS_INT(jaiListGet(shape, d))) {
                releaseSide(side);
                return jaiThrow(vm.cTypeError, "%s(): shape %d holds a non-integer", fnName, i);
            }
            side->shapes[at++] = AS_INT(jaiListGet(shape, d));
        }
    }
    return true;
}

static bool nCoreMLRun(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCoreMLModel *model;
    if (!requireModel(args[0], 1, "coreml_run", &model)) return false;

    CoreMLSide ins = {0};
    CoreMLSide outs = {0};
    if (!readSide(args[1], args[2], 2, "coreml_run", &ins)) return false;
    if (!readSide(args[3], args[4], 4, "coreml_run", &outs)) {
        releaseSide(&ins);
        return false;
    }

    char trouble[COREML_ERROR_BUFFER];
    const bool ok = jaiCoreMLRun(model, ins.buffers, ins.offsets, ins.shapes, ins.ranks,
                                 outs.buffers, outs.offsets, outs.shapes, outs.ranks,
                                 trouble, sizeof(trouble));
    releaseSide(&ins);
    releaseSide(&outs);
    if (!ok) {
        return jaiThrow(vm.cRuntimeError, "coreml_run(): %s",
                        trouble[0] != '\0' ? trouble : "the model did not run");
    }
    *out = NULL_VAL;
    return true;
}

/* `coreml_start(...)` returns a ticket; `coreml_wait(ticket)` blocks on it.
 * Between the two the caller can queue GPU work, and the two pieces of silicon
 * run at once -- which is the only reason the accelerator is worth reaching. */
static bool nCoreMLStart(int argc, Value *args, Value *out) {
    (void)argc;
    JaiCoreMLModel *model;
    if (!requireModel(args[0], 1, "coreml_start", &model)) return false;

    CoreMLSide ins = {0};
    CoreMLSide outs = {0};
    if (!readSide(args[1], args[2], 2, "coreml_start", &ins)) return false;
    if (!readSide(args[3], args[4], 4, "coreml_start", &outs)) {
        releaseSide(&ins);
        return false;
    }
    char trouble[COREML_ERROR_BUFFER];
    JaiCoreMLTicket *ticket = jaiCoreMLStart(model, ins.buffers, ins.offsets, ins.shapes,
                                             ins.ranks, outs.buffers, outs.offsets,
                                             outs.shapes, outs.ranks, trouble, sizeof(trouble));
    releaseSide(&ins);
    releaseSide(&outs);
    if (ticket == NULL) {
        return jaiThrow(vm.cRuntimeError, "coreml_start(): %s",
                        trouble[0] != '\0' ? trouble : "the model would not start");
    }
    *out = INT_VAL(jaiHandleAdd(HANDLE_COREML_TICKET, ticket));
    return true;
}

static bool nCoreMLWait(int argc, Value *args, Value *out) {
    (void)argc;
    void *ptr = NULL;
    if (!jaiHandleGet(args[0], 1, HANDLE_COREML_TICKET, "coreml_wait", &ptr)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    char trouble[COREML_ERROR_BUFFER];
    const bool ok = jaiCoreMLWait((JaiCoreMLTicket *)ptr, trouble, sizeof(trouble));
    if (!ok) {
        return jaiThrow(vm.cRuntimeError, "coreml_wait(): %s",
                        trouble[0] != '\0' ? trouble : "the model did not finish");
    }
    *out = NULL_VAL;
    return true;
}

void jaiRegisterCoreMLPrimitives(void) {
    jaiDefineNative("__prim__.coreml_open",  nCoreMLOpen,  2, 2);
    jaiDefineNative("__prim__.coreml_close", nCoreMLClose, 1, 1);
    jaiDefineNative("__prim__.coreml_names", nCoreMLNames, 2, 2);
    jaiDefineNative("__prim__.coreml_shape", nCoreMLShape, 3, 3);
    jaiDefineNative("__prim__.coreml_run",   nCoreMLRun,   5, 5);
    jaiDefineNative("__prim__.coreml_start", nCoreMLStart, 5, 5);
    jaiDefineNative("__prim__.coreml_wait",  nCoreMLWait,  1, 1);
}
