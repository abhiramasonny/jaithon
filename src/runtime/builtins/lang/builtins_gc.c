/* builtins_gc.c — the collector's native surface: gc_collect, gc_stats,
 * gc_disable, gc_enable (spec Appendix C).
 *
 * Split out of builtins_io.c, which explains what else moved and why. This
 * group is four small, independent natives over `vm.gc`; the only thing it
 * shares with the rest of that file's old contents is jaiIODictPut
 * (builtins_io.h), for the same reason io_stat and os_spawn's result do.
 */

#include "runtime/runtime.h"
#include "runtime/builtins/io/builtins_io.h"

#include "vm/gc.h"

static size_t heapBytes(void) {
    return jaiAllocatedBytes();
}

static bool nGcCollect(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    size_t before = heapBytes();
    jaiGCCollect();
    size_t after = heapBytes();
    *out = INT_VAL(after < before ? (int64_t)(before - after) : 0);
    return true;
}

static bool nGcStats(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    GCState *gc = vm.gc;

    ObjDict *stats = jaiDictNew();
    jaiGCPushRoot(OBJ_VAL(stats));
    jaiIODictPut(stats, "bytes_allocated", INT_VAL((int64_t)heapBytes()));
    jaiIODictPut(stats, "next_gc", INT_VAL(gc != NULL ? (int64_t)gc->nextGC : 0));
    jaiIODictPut(stats, "min_heap", INT_VAL(gc != NULL ? (int64_t)gc->minHeap : 0));
    jaiIODictPut(stats, "grow_factor", FLOAT_VAL(gc != NULL ? gc->growFactor : 0.0));
    jaiIODictPut(stats, "collections", INT_VAL(gc != NULL ? (int64_t)gc->collections : 0));
    jaiIODictPut(stats, "total_freed", INT_VAL(gc != NULL ? (int64_t)gc->totalFreed : 0));
    jaiIODictPut(stats, "pause_seconds",
            FLOAT_VAL(gc != NULL ? gc->totalPauseSeconds : 0.0));
    jaiIODictPut(stats, "temp_roots", INT_VAL(gc != NULL ? gc->tempRootCount : 0));
    jaiIODictPut(stats, "enabled", BOOL_VAL(gc != NULL ? gc->enabled : false));
    jaiIODictPut(stats, "stress", BOOL_VAL(gc != NULL ? gc->stress : false));
    jaiGCPopRoot();

    *out = OBJ_VAL(stats);
    return true;
}

static bool nGcDisable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    jaiGCEnable(false);
    *out = NULL_VAL;
    return true;
}

static bool nGcEnable(int argc, Value *args, Value *out) {
    (void)argc;
    (void)args;
    jaiGCEnable(true);
    *out = NULL_VAL;
    return true;
}

void jaiRegisterGCPrimitives(void) {
    jaiDefineNative("__prim__.gc_collect",   nGcCollect,  0, 0);
    jaiDefineNative("__prim__.gc_stats",     nGcStats,    0, 0);
    jaiDefineNative("__prim__.gc_disable",   nGcDisable,  0, 0);
    jaiDefineNative("__prim__.gc_enable",    nGcEnable,   0, 0);
}
