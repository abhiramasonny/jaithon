/* handles.h — integer handles for native resources the GC cannot trace
 * (threads, mutexes, windows, device memory), reached through a small int
 * instead of an object pointer. 1-based indices into a table only the VM
 * thread touches (no lock needed); 0 is never a valid handle. */
#ifndef JAI_HANDLES_H
#define JAI_HANDLES_H

#include "../vm/value.h"

typedef enum {
    HANDLE_FREE = 0,
    HANDLE_THREAD,
    HANDLE_MUTEX,
    HANDLE_COND,
    HANDLE_ATOMIC,
    HANDLE_WINDOW,
    HANDLE_GPU_BUFFER,
    HANDLE_GPU_KERNEL,
} HandleKind;

/* Reuses the lowest free slot, so open/close loops don't grow the table. */
int64_t jaiHandleAdd(HandleKind kind, void *ptr);

/* `index` is the 1-based argument position, as the caller wrote the call. */
bool jaiHandleGet(Value v, int index, HandleKind kind, const char *fnName,
                  void **out);

/* Release before destroying what the handle points at — a later use then gets
 * a clean "not live" error, not a use-after-free. */
void jaiHandleRelease(int64_t id);

#endif /* JAI_HANDLES_H */
