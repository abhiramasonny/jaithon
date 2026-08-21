/* parallel.c — see parallel.h.
 *
 * Grand Central Dispatch does the scheduling on Apple, which is the platform
 * this matters on and the one that knows how many of its cores are the fast
 * kind. Everywhere else the loop stays on the calling thread; the callers are
 * all correct either way, so there is nothing to fall back to. */
#include "runtime/parallel.h"

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>
#endif

/* More chunks than cores, so that a core which finishes early takes more
 * rather than idling -- the efficiency cores run slower than the performance
 * ones and an even split would wait for them. */
#define JAI_CHUNKS_PER_CORE 4

void jaiParallelChunks(size_t count, size_t leastPerChunk,
                       void (*body)(void *context, size_t start, size_t end),
                       void *context) {
    if (body == NULL || count == 0) return;
    if (leastPerChunk == 0) leastPerChunk = 1;

#ifdef __APPLE__
    if (count >= leastPerChunk * 2) {
        static long cores;
        if (cores == 0) {
            size_t width = sizeof(cores);
            if (sysctlbyname("hw.logicalcpu", &cores, &width, NULL, 0) != 0 || cores < 1) {
                cores = 1;
            }
        }
        size_t chunks = count / leastPerChunk;
        const size_t ceiling = (size_t)cores * JAI_CHUNKS_PER_CORE;
        if (chunks > ceiling) chunks = ceiling;
        if (chunks > 1) {
            const size_t per = (count + chunks - 1) / chunks;
            dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t index) {
                const size_t start = index * per;
                if (start >= count) return;
                size_t end = start + per;
                if (end > count) end = count;
                body(context, start, end);
            });
            return;
        }
    }
#endif

    body(context, 0, count);
}
