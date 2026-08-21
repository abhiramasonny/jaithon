/* handles.h — integer handles for native resources the GC cannot trace
 * (threads, mutexes, windows, device memory), reached through a small int
 * instead of an object pointer. 1-based indices into a table only the VM
 * thread touches (no lock needed); 0 is never a valid handle. */
#ifndef JAI_HANDLES_H
#define JAI_HANDLES_H

#include "vm/value.h"

typedef enum {
    HANDLE_FREE = 0,
    HANDLE_THREAD,
    HANDLE_MUTEX,
    HANDLE_COND,
    HANDLE_ATOMIC,
    HANDLE_WINDOW,
    HANDLE_GPU_BUFFER,
    HANDLE_GPU_KERNEL,
    HANDLE_GRAPH_BUILDER,
    HANDLE_GRAPH_PLAN,
    HANDLE_COREML_MODEL,
    HANDLE_COREML_TICKET,
} HandleKind;

int64_t jaiHandleAdd(HandleKind kind, void *ptr);

bool jaiHandleGet(Value v, int index, HandleKind kind, const char *fnName,
                  void **out);

void jaiHandleRelease(int64_t id);

#endif /* JAI_HANDLES_H */
