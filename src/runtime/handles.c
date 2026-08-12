/* handles.c — the one handle table, shared by std.thread, std.gui and std.gpu.
 * A handle from one subsystem must never resolve inside another: the kind
 * check below is what makes passing a mutex to gpu_dispatch an error instead
 * of a crash. */

#include "handles.h"

#include "../common/common.h"
#include "../vm/vm.h"
#include "runtime.h"

typedef struct {
    HandleKind kind;
    void      *ptr;
} HandleSlot;

static HandleSlot *gHandles;
static int         gHandleCount;
static int         gHandleCapacity;

static const char *handleKindName(HandleKind kind) {
    switch (kind) {
    case HANDLE_THREAD:     return "thread";
    case HANDLE_MUTEX:      return "mutex";
    case HANDLE_COND:       return "condition";
    case HANDLE_ATOMIC:     return "atomic";
    case HANDLE_WINDOW:     return "window";
    case HANDLE_GPU_BUFFER: return "GPU buffer";
    case HANDLE_GPU_KERNEL: return "GPU kernel";
    case HANDLE_FREE:       break;
    }
    return "handle";
}

int64_t jaiHandleAdd(HandleKind kind, void *ptr) {
    for (int i = 0; i < gHandleCount; i++) {
        if (gHandles[i].kind != HANDLE_FREE) continue;
        gHandles[i].kind = kind;
        gHandles[i].ptr = ptr;
        return i + 1;
    }
    if (gHandleCount + 1 > gHandleCapacity) {
        int oldCapacity = gHandleCapacity;
        gHandleCapacity = JAI_GROW_CAP(oldCapacity);
        gHandles = JAI_GROW_ARRAY(HandleSlot, gHandles, oldCapacity,
                                  gHandleCapacity);
    }
    gHandles[gHandleCount].kind = kind;
    gHandles[gHandleCount].ptr = ptr;
    return ++gHandleCount;
}

bool jaiHandleGet(Value v, int index, HandleKind kind, const char *fnName,
                  void **out) {
    int64_t id;
    if (!jaiArgInt(v, index, fnName, &id)) return false;
    if (id < 1 || id > gHandleCount || gHandles[id - 1].kind != kind ||
        gHandles[id - 1].ptr == NULL)
        return jaiThrow(vm.cValueError, "%s(): %lld is not a live %s handle",
                        fnName, (long long)id, handleKindName(kind));
    *out = gHandles[id - 1].ptr;
    return true;
}

void jaiHandleRelease(int64_t id) {
    if (id < 1 || id > gHandleCount) return;
    gHandles[id - 1].kind = HANDLE_FREE;
    gHandles[id - 1].ptr = NULL;
}
