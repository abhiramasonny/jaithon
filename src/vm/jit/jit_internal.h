/* jit_internal.h -- types, constants and cross-file declarations shared by the
 * whole-function JIT tier's translation units (jit_func.c and its siblings). */
#ifndef JAI_VM_JIT_INTERNAL_H
#define JAI_VM_JIT_INTERNAL_H

#include "vm/jit/jit.h"
#include "vm/vm.h"

#include <stdbool.h>
#include <stdint.h>

#if (defined(__aarch64__) || defined(__arm64__))

/* Two words in x0/x1 per AAPCS64 struct return (no store/load like the old global flag). A bailed
 * callee sends its caller straight to its own bail block, so recursion unwinds in one shot. */
typedef struct { int64_t value; int64_t bailed; } JitResult;

#define JIT_MAX_ARITY    8u   /* arguments arrive in x0..x7 */

/* A `self` entry (the callee of a recursive call) occupies no register; register numbers are
 * derived from the count of value entries below an entry, not from its depth. */
typedef enum {
    SLOT_INT,
    SLOT_FLOAT,
    SLOT_INST,
    /* Fixed class or null, held as the pointer or zero (`x == null` is then a compare against zero);
 * refusing this stopped six hundred stdlib bodies. Cost: materialising picks VAL_NULL/VAL_OBJ off the register at run time -- the tag isn't a static property of the kind. */
    SLOT_MAYBE_INST,
    SLOT_SELF,
    SLOT_OPAQUE,  /* present in a register, but nothing may be done with it */
    SLOT_CLOSURE,
    SLOT_CLASS,
    SLOT_FUNC,
    SLOT_NATIVE,
    SLOT_ITER,    /* ObjIter this body built, held raw; its index stays in memory (not a register) -- costs a load/store
                   * per iteration but means a deopt needs no write-back, since the stack's iterator is always current. */
    SLOT_BOOL,    /* 0 or 1 in a register -- a Value's boolean member is its low byte, so the same word serves both. */
    SLOT_NULL,    /* What `-> void` returns: a defined zero in a register (droppable, or written out by a deopt) whose
                   * tag is VAL_NULL rather than the VAL_OBJ every other kind chain in this file falls through to. */
    SLOT_OBJ,     /* Heap object of a type this tier doesn't model, held raw: may only be read, passed, stored and rooted. */
    SLOT_LIST     /* ObjList *, raw -- safe for the same reason an instance is: nothing moves, and a call spills it as a root first. */
} SlotKind;

#endif /* arm64 */

#endif /* JAI_VM_JIT_INTERNAL_H */
