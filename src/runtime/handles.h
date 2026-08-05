/* handles.h — integer handles for native resources the GC cannot own.
 *
 * A thread, a mutex, a window and a block of device memory all live outside the
 * Jaithon heap. The collector cannot trace them and cannot know when the last
 * reference went away, so none of them may be reached through an object
 * pointer: they are reached through a small integer that the owning std class
 * keeps in a field and releases explicitly. That is why every one of those
 * classes documents a `close` or `free` and pairs with `defer` (spec §7.3).
 *
 * Handles are 1-based indices into a table only the VM thread ever touches, so
 * the table itself needs no lock. 0 is never a valid handle.
 */
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

/* Claim a handle for `ptr`. Reuses the lowest free slot, so a program that
 * opens and closes in a loop does not grow the table. */
int64_t jaiHandleAdd(HandleKind kind, void *ptr);

/* Resolve `v` to the pointer behind a live handle of `kind`. Raises ValueError
 * and returns false otherwise; `index` is the 1-based argument position, which
 * is what the caller counted when writing the call. */
bool jaiHandleGet(Value v, int index, HandleKind kind, const char *fnName,
                  void **out);

/* Retire a handle. A later use is then a clean "not a live handle" error rather
 * than a use-after-free, so release before destroying what it points at. */
void jaiHandleRelease(int64_t id);

#endif /* JAI_HANDLES_H */
