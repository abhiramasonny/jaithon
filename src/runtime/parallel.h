/* parallel.h — splitting one flat loop across the cores that are sitting idle.
 *
 * The runtime's heavy loops are all of one shape: walk a range, read from one
 * array, write to another, no allocation and nothing shared. Turning a frame's
 * pixels into list elements is three million iterations of exactly that, and
 * it runs on one core of twelve.
 *
 * `body` is called once per chunk with a half-open range, from several threads
 * at once. It must touch nothing but the memory those indices name -- no
 * allocation, no VM state, no garbage collector. Ranges never overlap, so a
 * body that writes only `[start, end)` of its output needs no locking at all.
 *
 * Below `leastPerChunk` iterations the whole thing runs on the calling thread,
 * because handing work to another core costs more than a short loop does. */
#ifndef JAI_PARALLEL_H
#define JAI_PARALLEL_H

#include <stddef.h>

void jaiParallelChunks(size_t count, size_t leastPerChunk,
                       void (*body)(void *context, size_t start, size_t end),
                       void *context);

#endif /* JAI_PARALLEL_H */
