/* jit_exemplar.c -- predicting what a container holds from one live element,
 * for the arms that specialise on an element kind the model cannot see. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* The kind a container's elements may be held as, read off ONE live element.
 *
 * The sample specialises and the tag guard at the read site confirms: a
 * container that later holds something else deoptimises rather than being
 * answered wrongly. An instance carries its class too, since a tag check alone
 * cannot tell two shapes apart. */
bool exemplarKind(Value elem, SlotKind *kind, unsigned *tag,
                         ObjClass **cls, uint32_t *shape) {
    *cls = NULL;
    *shape = 0;
    if (IS_INT(elem))        { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem))      { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    if (IS_BOOL(elem))       { *kind = SLOT_BOOL;  *tag = VAL_BOOL;  return true; }
    if (IS_LIST(elem))       { *kind = SLOT_LIST;  *tag = VAL_OBJ;   return true; }
    if (rawObjValue(elem))   { *kind = SLOT_OBJ;   *tag = VAL_OBJ;   return true; }
    if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
        *kind  = SLOT_INST;
        *tag   = VAL_OBJ;
        *cls   = AS_INSTANCE(elem)->klass;
        *shape = (*cls)->shapeId;
        return true;
    }
    return false;
}

/* An exemplar for the elements a freshly-built list is about to hold.
 *
 * `[1, 2, 3]` has no live list to sample -- the list does not exist until the
 * compiled code runs -- but the model knows the kind of every entry that went
 * into it. For a scalar that is enough: a synthesised zero of the right kind
 * answers every question the iterate arm asks of a sample, and being an
 * immediate it has no lifetime to worry about. For an object the element's own
 * sample is reused, which was already being held on the operand stack.
 *
 * All `n` must agree, and an instance must agree on its class too, for the same
 * reason the dict walk insists on it: a mispredicted element kind deoptimises on
 * every read, which is worse than not compiling at all.
 *
 * Returns false when the list is empty (a comprehension's accumulator) or the
 * elements disagree -- in both cases there is simply nothing to predict. */
bool buildListExemplar(const Emit *e, unsigned first, unsigned n,
                              Value *out) {
    if (n == 0) return false;
    Value chosen = NULL_VAL;
    for (unsigned i = 0; i < n; i++) {
        unsigned idx = first + i;
        Value here;
        switch (e->stack[idx]) {
        case SLOT_INT:   here = INT_VAL(0);        break;
        case SLOT_FLOAT: here = FLOAT_VAL(0.0);    break;
        case SLOT_BOOL:  here = BOOL_VAL(false);   break;
        case SLOT_OBJ:
        case SLOT_LIST:
        case SLOT_INST:
            here = e->stackSeen[idx];
            if (!IS_OBJ(here) || AS_OBJ(here) == NULL) return false;
            break;
        default:
            return false;
        }
        if (i == 0) { chosen = here; continue; }
        if (jaiValueType(here) != jaiValueType(chosen)) return false;
        if (IS_OBJ(here) && OBJ_TYPE(here) != OBJ_TYPE(chosen)) return false;
        if (IS_INSTANCE(here) &&
            AS_INSTANCE(here)->klass != AS_INSTANCE(chosen)->klass) {
            return false;
        }
    }
    *out = chosen;
    return true;
}

/* One value out of a live dict, and only if every value in it agrees.
 *
 * A LIST is sampled at index 0 alone, because a list's elements are usually
 * built by one loop and a wrong guess costs a deopt per read. A dict is not:
 * `{"name": "x", "count": 3}` is an ordinary dict and its values disagree, so
 * predicting off the first entry would deoptimise every iteration -- measured
 * elsewhere at 5.7x worse than declining outright (chunk.h states the same for
 * an invoke's result). Refusing a mixed dict is the point of the walk.
 *
 * Capped, so compiling a body that indexes a large dict does not walk it. Past
 * the cap the guard still holds; only the prediction is made on a prefix. */
#define JIT_DICT_SAMPLE_MAX 256u

bool dictUniformValue(ObjDict *dict, Value *out) {
    const JaiTable *t = &dict->table;
    if (t->entries == NULL || t->count <= 0) return false;
    bool have = false;
    Value first = NULL_VAL;
    unsigned seen = 0;
    for (int i = 0; i < t->capacity && seen < JIT_DICT_SAMPLE_MAX; i++) {
        /* A live entry is one with a nonnegative order; empty and tombstoned
         * slots both carry a negative one (see table.c's entryIsLive). */
        if (t->entries[i].order < 0) continue;
        Value v = t->entries[i].value;
        seen++;
        if (!have) { first = v; have = true; continue; }
        if (jaiValueType(v) != jaiValueType(first)) return false;
        if (IS_OBJ(v) && OBJ_TYPE(v) != OBJ_TYPE(first)) return false;
        if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != AS_INSTANCE(first)->klass) {
            return false;
        }
    }
    if (!have) return false;
    *out = first;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
