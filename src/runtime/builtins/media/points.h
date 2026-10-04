/* points.h — reading jaicv's points from C.
 *
 * A contour is a list of `Point` objects, and the primitives that work on one
 * read each point's `x` and `y` where the object holds them rather than having
 * Jaithon copy every coordinate into a list of ints first -- that copy was a
 * loop of its own, as long as the work it fed. A point is anything with integer
 * fields of those names; anything else is reported, so the caller can take its
 * own path rather than be refused. */
#ifndef JAI_BUILTINS_POINTS_H
#define JAI_BUILTINS_POINTS_H

#include <stdint.h>

#include "runtime/runtime.h"

typedef struct {
    ObjClass  *klass;
    int        slotX, slotY;
    ObjString *nameX, *nameY;
} JaiPointReader;

JAI_INLINE void jaiPointReaderInit(JaiPointReader *r) {
    r->klass = NULL;
    r->slotX = -1;
    r->slotY = -1;
    r->nameX = jaiStringInternC("x");
    r->nameY = jaiStringInternC("y");
}

/* The `x` and `y` of `v`. False when it is not an instance with integer
 * fields of those names. The slots are looked up once a class. */
JAI_INLINE bool jaiReadPoint(JaiPointReader *r, Value v, int64_t *x, int64_t *y) {
    if (!IS_INSTANCE(v)) return false;
    ObjInstance *point = AS_INSTANCE(v);
    if (point->klass != r->klass) {
        r->klass = point->klass;
        r->slotX = jaiClassFieldSlot(point->klass, r->nameX);
        r->slotY = jaiClassFieldSlot(point->klass, r->nameY);
    }
    if (r->slotX < 0 || r->slotY < 0) return false;
    if (r->slotX >= point->fieldCount || r->slotY >= point->fieldCount) return false;
    const Value vx = point->fields[r->slotX];
    const Value vy = point->fields[r->slotY];
    if (!IS_INT(vx) || !IS_INT(vy)) return false;
    *x = AS_INT(vx);
    *y = AS_INT(vy);
    return true;
}

#endif /* JAI_BUILTINS_POINTS_H */
