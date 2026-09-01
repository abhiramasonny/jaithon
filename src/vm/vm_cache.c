/* vm_cache.c — the interpreter's cache maintenance: the builtin-receiver
 * inline-cache key, the shared megamorphic method table, the per-site record
 * of an INVOKE's result kind, and the shape-id table the compiled tier reads.
 */
#include <stdlib.h>

#include "vm/vm_internal.h"

uint32_t builtinShapeTag(Value v) {
    switch (jaiValueType(v)) {
    case VAL_INT:   return IC_BUILTIN_TAG | 1u;
    case VAL_FLOAT: return IC_BUILTIN_TAG | 2u;
    case VAL_OBJ:   break;
    default:        return 0;
    }
    switch (OBJ_TYPE(v)) {
    case OBJ_STRING: return IC_BUILTIN_TAG | 3u;
    case OBJ_LIST:   return IC_BUILTIN_TAG | 4u;
    case OBJ_DICT:   return IC_BUILTIN_TAG | 5u;
    case OBJ_SET:    return IC_BUILTIN_TAG | 6u;
    case OBJ_TUPLE:  return IC_BUILTIN_TAG | 7u;
    case OBJ_RANGE:  return IC_BUILTIN_TAG | 8u;
    case OBJ_BYTES:  return IC_BUILTIN_TAG | 9u;
    case OBJ_ITER:   return IC_BUILTIN_TAG | 10u;
    default:         return 0;
    }
}

MegaEntry sMegaCache[JAI_MEGA_WAYS];

/* JAITHON_MEGA_STRESS=1 collapses the table to a single entry, so every
 * megamorphic (class, name) pair collides with every other and the key is the
 * only thing left deciding what a call reaches. Without it a wrong key is
 * invisible: nine classes in five hundred slots simply never collide, so a
 * cache keyed on the name alone passes every test in the tree. Same idea as
 * JAITHON_JIT_DEOPT_STRESS -- make the rare path the only path. */
unsigned sMegaMask = JAI_MEGA_WAYS - 1u;

void jaiMethodCacheInit(void) {
    const char *v = getenv("JAITHON_MEGA_STRESS");
    sMegaMask = (v != NULL && v[0] != '\0') ? 0u : JAI_MEGA_WAYS - 1u;
}

void jaiMethodCacheRemoveWhite(void) {
    for (unsigned i = 0; i < JAI_MEGA_WAYS; i++) {
        MegaEntry *e = &sMegaCache[i];
        if (e->klass == NULL) continue;
        if (!((Obj *)e->klass)->isMarked ||
            !((Obj *)e->name)->isMarked ||
            (IS_OBJ(e->method) && !AS_OBJ(e->method)->isMarked)) {
            e->klass = NULL;
            e->name = NULL;
            e->method = NULL_VAL;
        }
    }
}

/* The key a receiver files its way under: a class's shape for an instance, a
 * type tag for a builtin. The two keyspaces are disjoint by IC_BUILTIN_TAG's
 * high bit, so one site holds both without either matching the other. */
static uint32_t invokeCacheKey(Value receiver) {
    if (IS_INSTANCE(receiver)) {
        ObjClass *klass = AS_INSTANCE(receiver)->klass;
        return klass != NULL ? klass->shapeId : 0u;
    }
    return builtinShapeTag(receiver);
}

uint8_t jaiInvokeResultFeedback(const Chunk *chunk, uint16_t cacheIdx,
                                Value receiver) {
    uint32_t tag = invokeCacheKey(receiver);
    if (tag == 0 || chunk->caches == NULL ||
        (int)cacheIdx >= chunk->cacheCount) {
        return JAI_FB_NONE;
    }
    const InlineCache *ic = &chunk->caches[cacheIdx];
    for (int w = 0; w < ic->count; w++) {
        if (ic->shapeId[w] == tag) return ic->resultKind[w];
    }
    return JAI_FB_NONE;
}

struct JaiResultSite sResultSite;

/* The class carrying `shape`, found by walking the classes the tier has
 * already seen. Only used at compile time. */
/* Direct-mapped, and 256 buckets rather than 64 because shape ids are handed
 * out by a monotonic counter and the self-hosted compiler's run past 108: any
 * two classes exactly 64 apart evicted each other for ever. Counted with a
 * temporary hook in one `check --no-cache` of compile/parser.jai: **27
 * evictions with 64 buckets and 0 with 256**, and every pair exactly 64 apart
 * -- 108 and 44 displacing each other twenty-five times between them, then 107
 * and 43, then 106 and 42. Each eviction costs its caller the callee's return
 * class for as long as it stands, and the invoke arm turns that into "return
 * kind not usable": 173 function-tier declines on that file become **168**,
 * exactly reproducible (1323 function-tier declines either way, 3 runs of 3).
 *
 * No clock claim, and none is expected -- five sites is five sites. It earns
 * the four characters because it silently un-does part of whatever any
 * return-kind work buys, so today's counts are an undercount of what the tier
 * could already resolve.
 *
 * The identity check below is the whole soundness argument and must stay: a
 * wrong bucket can only ever LOSE a compile, never hand back the wrong class,
 * so widening the table cannot turn into a miscompile. Costs 1.5KB of BSS.
 *
 * A two-way table, or the resolved class stored beside `obsReturnShape` on the
 * function record, removes the collision class outright and is the better
 * shape; this is the version that is four characters. */
static ObjClass *sShapeCache[256];
void jaiClassRememberShape(ObjClass *c) {
    if (c == NULL) return;
    sShapeCache[c->shapeId & 255u] = c;
}
bool jaiClassForShape(uint32_t shape, ObjClass **out) {
    ObjClass *c = sShapeCache[shape & 255u];
    if (c == NULL || c->shapeId != shape) return false;
    *out = c;
    return true;
}
