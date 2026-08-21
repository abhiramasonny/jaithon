/* builtins_graph.c — the primitives behind whole-network graph compilation.
 *
 * jaicv builds a graph node by node from an imported model, hands the lot to
 * `graph_compile`, and runs the plan that comes back. Tensors are named by the
 * small integers the builder returns, so nothing here has to know what a
 * tensor is; the builder and the plan are handles, freed by their owner.
 *
 * Every builder call refuses with -1 rather than raising, and a -1 poisons
 * everything downstream that names it. That lets a caller build a whole graph
 * optimistically and check once at the end -- which is what an importer wants,
 * since an operator it cannot express is a reason to fall back to the
 * interpreter rather than an error to report. */
#include "runtime/runtime.h"
#include "runtime/handles.h"
#include <stdint.h>

#include "native/native.h"
#include "vm/gc.h"

static bool requireBuilder(Value v, int index, const char *fnName, JaiGraphBuilder **out) {
    void *ptr = NULL;
    if (!jaiHandleGet(v, index, HANDLE_GRAPH_BUILDER, fnName, &ptr)) return false;
    *out = (JaiGraphBuilder *)ptr;
    return true;
}

static bool requirePlan(Value v, int index, const char *fnName, JaiGraphPlan **out) {
    void *ptr = NULL;
    if (!jaiHandleGet(v, index, HANDLE_GRAPH_PLAN, fnName, &ptr)) return false;
    *out = (JaiGraphPlan *)ptr;
    return true;
}

/* A list of integers as a freshly allocated array. Returns false and raises
 * when an element is not an integer; `*out` is NULL for an empty list, which
 * every caller treats as "no elements" rather than as a failure. */
static bool intsOf(Value v, int index, const char *fnName, int64_t **out, int *count) {
    ObjList *list;
    if (!jaiArgList(v, index, fnName, &list)) return false;
    *count = list->count;
    if (list->count == 0) {
        *out = NULL;
        return true;
    }
    int64_t *values = JAI_ALLOC(int64_t, (size_t)list->count);
    for (int i = 0; i < list->count; i++) {
        if (!IS_INT(list->items[i])) {
            JAI_FREE_ARRAY(int64_t, values, (size_t)list->count);
            return jaiThrow(vm.cTypeError,
                            "%s(): element %d of argument %d is %s, expected an integer",
                            fnName, i, index, jaiTypeNameStatic(list->items[i]));
        }
        values[i] = AS_INT(list->items[i]);
    }
    *out = values;
    return true;
}

static void freeInts(int64_t *values, int count) {
    if (values != NULL) JAI_FREE_ARRAY(int64_t, values, (size_t)count);
}

static bool nGraphNew(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    JaiGraphBuilder *builder = jaiGraphNew();
    if (builder == NULL)
        return jaiThrow(vm.cRuntimeError, "graph_new(): no device to build a graph for");
    *out = INT_VAL(jaiHandleAdd(HANDLE_GRAPH_BUILDER, builder));
    return true;
}

static bool nGraphFree(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    if (!requireBuilder(args[0], 1, "graph_free", &builder)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    jaiGraphFree(builder);
    *out = NULL_VAL;
    return true;
}

static bool nGraphInput(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    if (!requireBuilder(args[0], 1, "graph_input", &builder)) return false;
    int64_t *dims;
    int rank;
    if (!intsOf(args[1], 2, "graph_input", &dims, &rank)) return false;
    *out = INT_VAL(jaiGraphInput(builder, dims, rank));
    freeInts(dims, rank);
    return true;
}

static bool nGraphConstant(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    if (!requireBuilder(args[0], 1, "graph_constant", &builder)) return false;
    ObjList *values;
    if (!jaiArgList(args[1], 2, "graph_constant", &values)) return false;
    int64_t *dims;
    int rank;
    if (!intsOf(args[2], 3, "graph_constant", &dims, &rank)) return false;

    float *floats = values->count > 0 ? JAI_ALLOC(float, (size_t)values->count) : NULL;
    for (int i = 0; i < values->count; i++) {
        Value item = values->items[i];
        if (IS_FLOAT(item)) floats[i] = (float)AS_FLOAT(item);
        else if (IS_INT(item)) floats[i] = (float)AS_INT(item);
        else {
            if (floats != NULL) JAI_FREE_ARRAY(float, floats, (size_t)values->count);
            freeInts(dims, rank);
            return jaiThrow(vm.cTypeError,
                            "graph_constant(): element %d is %s, expected a number",
                            i, jaiTypeNameStatic(item));
        }
    }
    *out = INT_VAL(jaiGraphConstant(builder, floats, dims, rank));
    if (floats != NULL) JAI_FREE_ARRAY(float, floats, (size_t)values->count);
    freeInts(dims, rank);
    return true;
}

static bool nGraphUnary(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, op;
    if (!requireBuilder(args[0], 1, "graph_unary", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_unary", &x)) return false;
    if (!jaiArgInt(args[2], 3, "graph_unary", &op)) return false;
    *out = INT_VAL(jaiGraphUnary(builder, (int)x, (int)op));
    return true;
}

static bool nGraphBinary(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t left, right, op;
    if (!requireBuilder(args[0], 1, "graph_binary", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_binary", &left)) return false;
    if (!jaiArgInt(args[2], 3, "graph_binary", &right)) return false;
    if (!jaiArgInt(args[3], 4, "graph_binary", &op)) return false;
    *out = INT_VAL(jaiGraphBinary(builder, (int)left, (int)right, (int)op));
    return true;
}

/* Nine parameters for a convolution and eight for a pooling, both narrowed
 * from the list the caller passes. A short list is a caller mistake rather
 * than a graph the device declines, so it raises. */
static bool narrow(Value v, int index, const char *fnName, int wanted, int32_t *slots) {
    int64_t *values;
    int count;
    if (!intsOf(v, index, fnName, &values, &count)) return false;
    if (count != wanted) {
        freeInts(values, count);
        return jaiThrow(vm.cValueError, "%s(): argument %d needs %d values, got %d",
                        fnName, index, wanted, count);
    }
    for (int i = 0; i < wanted; i++) slots[i] = (int32_t)values[i];
    freeInts(values, count);
    return true;
}

static bool nGraphConv(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, w, bias;
    int32_t params[9];
    if (!requireBuilder(args[0], 1, "graph_conv", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_conv", &x)) return false;
    if (!jaiArgInt(args[2], 3, "graph_conv", &w)) return false;
    if (!jaiArgInt(args[3], 4, "graph_conv", &bias)) return false;
    if (!narrow(args[4], 5, "graph_conv", 9, params)) return false;
    *out = INT_VAL(jaiGraphConv(builder, (int)x, (int)w, (int)bias, params));
    return true;
}

static bool nGraphPool(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, kind;
    int32_t params[8];
    if (!requireBuilder(args[0], 1, "graph_pool", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_pool", &x)) return false;
    if (!narrow(args[2], 3, "graph_pool", 8, params)) return false;
    if (!jaiArgInt(args[3], 4, "graph_pool", &kind)) return false;
    *out = INT_VAL(jaiGraphPool(builder, (int)x, params, (int)kind));
    return true;
}

static bool nGraphConcat(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t axis;
    int64_t *ids;
    int count;
    if (!requireBuilder(args[0], 1, "graph_concat", &builder)) return false;
    if (!intsOf(args[1], 2, "graph_concat", &ids, &count)) return false;
    if (!jaiArgInt(args[2], 3, "graph_concat", &axis)) {
        freeInts(ids, count);
        return false;
    }
    int *narrowed = count > 0 ? JAI_ALLOC(int, (size_t)count) : NULL;
    for (int i = 0; i < count; i++) narrowed[i] = (int)ids[i];
    *out = INT_VAL(jaiGraphConcat(builder, narrowed, count, (int)axis));
    if (narrowed != NULL) JAI_FREE_ARRAY(int, narrowed, (size_t)count);
    freeInts(ids, count);
    return true;
}

static bool nGraphReshape(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x;
    int64_t *dims;
    int rank;
    if (!requireBuilder(args[0], 1, "graph_reshape", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_reshape", &x)) return false;
    if (!intsOf(args[2], 3, "graph_reshape", &dims, &rank)) return false;
    *out = INT_VAL(jaiGraphReshape(builder, (int)x, dims, rank));
    freeInts(dims, rank);
    return true;
}

static bool nGraphTranspose(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x;
    int64_t *perm;
    int rank;
    if (!requireBuilder(args[0], 1, "graph_transpose", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_transpose", &x)) return false;
    if (!intsOf(args[2], 3, "graph_transpose", &perm, &rank)) return false;
    int32_t *order = rank > 0 ? JAI_ALLOC(int32_t, (size_t)rank) : NULL;
    for (int i = 0; i < rank; i++) order[i] = (int32_t)perm[i];
    *out = INT_VAL(jaiGraphTranspose(builder, (int)x, order, rank));
    if (order != NULL) JAI_FREE_ARRAY(int32_t, order, (size_t)rank);
    freeInts(perm, rank);
    return true;
}

static bool nGraphSlice(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x;
    int64_t *starts = NULL, *ends = NULL, *steps = NULL;
    int a = 0, b = 0, c = 0;
    if (!requireBuilder(args[0], 1, "graph_slice", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_slice", &x)) return false;
    if (!intsOf(args[2], 3, "graph_slice", &starts, &a)) return false;
    if (!intsOf(args[3], 4, "graph_slice", &ends, &b)) {
        freeInts(starts, a);
        return false;
    }
    if (!intsOf(args[4], 5, "graph_slice", &steps, &c)) {
        freeInts(starts, a);
        freeInts(ends, b);
        return false;
    }
    bool ok = a == b && b == c;
    int id = -1;
    if (ok && a > 0) {
        int32_t *s0 = JAI_ALLOC(int32_t, (size_t)a);
        int32_t *s1 = JAI_ALLOC(int32_t, (size_t)a);
        int32_t *s2 = JAI_ALLOC(int32_t, (size_t)a);
        for (int i = 0; i < a; i++) {
            s0[i] = (int32_t)starts[i];
            s1[i] = (int32_t)ends[i];
            s2[i] = (int32_t)steps[i];
        }
        id = jaiGraphSlice(builder, (int)x, s0, s1, s2, a);
        JAI_FREE_ARRAY(int32_t, s0, (size_t)a);
        JAI_FREE_ARRAY(int32_t, s1, (size_t)a);
        JAI_FREE_ARRAY(int32_t, s2, (size_t)a);
    }
    freeInts(starts, a);
    freeInts(ends, b);
    freeInts(steps, c);
    *out = INT_VAL(id);
    return true;
}

static bool nGraphSoftmax(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, axis;
    if (!requireBuilder(args[0], 1, "graph_softmax", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_softmax", &x)) return false;
    if (!jaiArgInt(args[2], 3, "graph_softmax", &axis)) return false;
    *out = INT_VAL(jaiGraphSoftmax(builder, (int)x, (int)axis));
    return true;
}

static bool nGraphResize(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, height, width;
    if (!requireBuilder(args[0], 1, "graph_resize_nearest", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_resize_nearest", &x)) return false;
    if (!jaiArgInt(args[2], 3, "graph_resize_nearest", &height)) return false;
    if (!jaiArgInt(args[3], 4, "graph_resize_nearest", &width)) return false;
    *out = INT_VAL(jaiGraphResizeNearest(builder, (int)x, (int)height, (int)width));
    return true;
}

static bool nGraphMatmul(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t left, right;
    if (!requireBuilder(args[0], 1, "graph_matmul", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_matmul", &left)) return false;
    if (!jaiArgInt(args[2], 3, "graph_matmul", &right)) return false;
    *out = INT_VAL(jaiGraphMatmul(builder, (int)left, (int)right));
    return true;
}

static bool nGraphReduce(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t x, kind;
    int64_t *axes;
    int count;
    if (!requireBuilder(args[0], 1, "graph_reduce", &builder)) return false;
    if (!jaiArgInt(args[1], 2, "graph_reduce", &x)) return false;
    if (!intsOf(args[2], 3, "graph_reduce", &axes, &count)) return false;
    if (!jaiArgInt(args[3], 4, "graph_reduce", &kind)) {
        freeInts(axes, count);
        return false;
    }
    int32_t *over = count > 0 ? JAI_ALLOC(int32_t, (size_t)count) : NULL;
    for (int i = 0; i < count; i++) over[i] = (int32_t)axes[i];
    *out = INT_VAL(jaiGraphReduce(builder, (int)x, over, count, (int)kind));
    if (over != NULL) JAI_FREE_ARRAY(int32_t, over, (size_t)count);
    freeInts(axes, count);
    return true;
}

static bool nGraphCompile(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphBuilder *builder;
    int64_t *inputs = NULL, *outputs = NULL;
    int inCount = 0, outCount = 0;
    if (!requireBuilder(args[0], 1, "graph_compile", &builder)) return false;
    if (!intsOf(args[1], 2, "graph_compile", &inputs, &inCount)) return false;
    if (!intsOf(args[2], 3, "graph_compile", &outputs, &outCount)) {
        freeInts(inputs, inCount);
        return false;
    }
    int *in = inCount > 0 ? JAI_ALLOC(int, (size_t)inCount) : NULL;
    int *outIds = outCount > 0 ? JAI_ALLOC(int, (size_t)outCount) : NULL;
    for (int i = 0; i < inCount; i++) in[i] = (int)inputs[i];
    for (int i = 0; i < outCount; i++) outIds[i] = (int)outputs[i];
    JaiGraphPlan *plan = jaiGraphCompile(builder, in, inCount, outIds, outCount);
    if (in != NULL) JAI_FREE_ARRAY(int, in, (size_t)inCount);
    if (outIds != NULL) JAI_FREE_ARRAY(int, outIds, (size_t)outCount);
    freeInts(inputs, inCount);
    freeInts(outputs, outCount);
    /* Null rather than an error: a graph the compiler will not take is a
     * reason to run the operators one at a time, not a reason to stop. */
    *out = plan == NULL ? NULL_VAL : INT_VAL(jaiHandleAdd(HANDLE_GRAPH_PLAN, plan));
    return true;
}

static bool nGraphPlanOutputShape(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphPlan *plan;
    int64_t index;
    if (!requirePlan(args[0], 1, "graph_plan_output_shape", &plan)) return false;
    if (!jaiArgInt(args[1], 2, "graph_plan_output_shape", &index)) return false;
    int rank = jaiGraphPlanOutputRank(plan, (int)index);
    if (rank < 0)
        return jaiThrow(vm.cValueError, "graph_plan_output_shape(): no output %lld",
                        (long long)index);
    ObjList *list = jaiListNew(rank);
    if (list == NULL) return false;
    if (rank > 0) {
        int64_t *dims = JAI_ALLOC(int64_t, (size_t)rank);
        bool ok = jaiGraphPlanOutputShape(plan, (int)index, dims, rank);
        if (ok) {
            jaiGCPushRoot(OBJ_VAL(list));
            jaiListReserve(list, rank);
            jaiGCPopRoot();
            if (list->capacity >= rank) {
                for (int i = 0; i < rank; i++) list->items[i] = INT_VAL(dims[i]);
                list->count = rank;
                list->version++;
            } else {
                ok = false;
            }
        }
        JAI_FREE_ARRAY(int64_t, dims, (size_t)rank);
        if (!ok)
            return jaiThrow(vm.cRuntimeError,
                            "graph_plan_output_shape(): output %lld has no fixed shape",
                            (long long)index);
    }
    *out = OBJ_VAL(list);
    return true;
}

/* Device buffers for one side of a run, in the order the plan expects. */
static bool buffersOf(Value v, int index, const char *fnName,
                      JaiGpuBuffer ***out, size_t **offsets, int *count) {
    ObjList *list;
    if (!jaiArgList(v, index, fnName, &list)) return false;
    *count = list->count;
    if (list->count == 0) {
        *out = NULL;
        *offsets = NULL;
        return true;
    }
    JaiGpuBuffer **buffers = JAI_ALLOC(JaiGpuBuffer *, (size_t)list->count);
    size_t *starts = JAI_ALLOC(size_t, (size_t)list->count);
    for (int i = 0; i < list->count; i++) {
        JaiGpuBuffer *native = NULL;
        int64_t origin = 0;
        if (!jaiGpuBufferOf(list->items[i], index, fnName, &native, &origin)) {
            JAI_FREE_ARRAY(JaiGpuBuffer *, buffers, (size_t)list->count);
            JAI_FREE_ARRAY(size_t, starts, (size_t)list->count);
            return false;
        }
        buffers[i] = native;
        starts[i] = (size_t)origin * sizeof(float);
    }
    *out = buffers;
    *offsets = starts;
    return true;
}

static bool nGraphRun(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphPlan *plan;
    if (!requirePlan(args[0], 1, "graph_run", &plan)) return false;
    JaiGpuBuffer **ins = NULL, **outs = NULL;
    size_t *inOffsets = NULL, *outOffsets = NULL;
    int inCount = 0, outCount = 0;
    if (!buffersOf(args[1], 2, "graph_run", &ins, &inOffsets, &inCount)) return false;
    if (!buffersOf(args[2], 3, "graph_run", &outs, &outOffsets, &outCount)) {
        if (ins != NULL) {
            JAI_FREE_ARRAY(JaiGpuBuffer *, ins, (size_t)inCount);
            JAI_FREE_ARRAY(size_t, inOffsets, (size_t)inCount);
        }
        return false;
    }
    bool ok = jaiGraphRun(plan, ins, inOffsets, outs, outOffsets);
    if (ins != NULL) {
        JAI_FREE_ARRAY(JaiGpuBuffer *, ins, (size_t)inCount);
        JAI_FREE_ARRAY(size_t, inOffsets, (size_t)inCount);
    }
    if (outs != NULL) {
        JAI_FREE_ARRAY(JaiGpuBuffer *, outs, (size_t)outCount);
        JAI_FREE_ARRAY(size_t, outOffsets, (size_t)outCount);
    }
    if (!ok) return jaiThrow(vm.cRuntimeError, "graph_run(): the plan did not run");
    *out = NULL_VAL;
    return true;
}

static bool nGraphPlanFree(int argc, Value *args, Value *out) {
    (void)argc;
    JaiGraphPlan *plan;
    if (!requirePlan(args[0], 1, "graph_plan_free", &plan)) return false;
    jaiHandleRelease(AS_INT(args[0]));
    jaiGraphPlanFree(plan);
    *out = NULL_VAL;
    return true;
}

void jaiRegisterGraphPrimitives(void) {
    jaiDefineNative("__prim__.graph_new",            nGraphNew,            0, 0);
    jaiDefineNative("__prim__.graph_free",           nGraphFree,           1, 1);
    jaiDefineNative("__prim__.graph_input",          nGraphInput,          2, 2);
    jaiDefineNative("__prim__.graph_constant",       nGraphConstant,       3, 3);
    jaiDefineNative("__prim__.graph_unary",          nGraphUnary,          3, 3);
    jaiDefineNative("__prim__.graph_binary",         nGraphBinary,         4, 4);
    jaiDefineNative("__prim__.graph_conv",           nGraphConv,           5, 5);
    jaiDefineNative("__prim__.graph_pool",           nGraphPool,           4, 4);
    jaiDefineNative("__prim__.graph_concat",         nGraphConcat,         3, 3);
    jaiDefineNative("__prim__.graph_reshape",        nGraphReshape,        3, 3);
    jaiDefineNative("__prim__.graph_transpose",      nGraphTranspose,      3, 3);
    jaiDefineNative("__prim__.graph_slice",          nGraphSlice,          5, 5);
    jaiDefineNative("__prim__.graph_softmax",        nGraphSoftmax,        3, 3);
    jaiDefineNative("__prim__.graph_resize_nearest", nGraphResize,         4, 4);
    jaiDefineNative("__prim__.graph_matmul",         nGraphMatmul,         3, 3);
    jaiDefineNative("__prim__.graph_reduce",         nGraphReduce,         4, 4);
    jaiDefineNative("__prim__.graph_compile",        nGraphCompile,        3, 3);
    jaiDefineNative("__prim__.graph_plan_output_shape", nGraphPlanOutputShape, 2, 2);
    jaiDefineNative("__prim__.graph_run",            nGraphRun,            3, 3);
    jaiDefineNative("__prim__.graph_plan_free",      nGraphPlanFree,       1, 1);
}
