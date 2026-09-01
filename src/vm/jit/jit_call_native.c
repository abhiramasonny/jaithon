/* jit_call_native.c -- calls to a builtin whose result kind is known before the
 * call is made, and the one-binary A/B switches that sit alongside them. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Builtins whose result kind is a property of the FUNCTION and not of its
 * arguments: `str(x)` is a string whatever x is, `len(x)` is an int, `bool(x)`
 * is a bool. That is the only thing the surrounding body needs to know, so the
 * call can be an ordinary call out and everything around it stays compiled.
 *
 * Worth having because the alternative was not a slower call but no compiled
 * body at all -- one `str()` in a loop declined the whole enclosing function.
 * A probe doing `str(i % 10_000)` per iteration ran 90,000,323 interpreted
 * instructions against 1,231 for the f-string spelling of the same thing.
 *
 * The kinds are read off the natives in builtins_core.c, and the returned tag
 * is guarded regardless: a wrong row here costs a deopt, never an answer. */
typedef struct {
    const char *name;
    unsigned    argc;
    SlotKind    kind;
    uint8_t     tag;
} NativeResult;

static const NativeResult kNativeResults[] = {
    { "str",        1, SLOT_OBJ,  VAL_OBJ  },
    { "repr",       1, SLOT_OBJ,  VAL_OBJ  },
    { "chr",        1, SLOT_OBJ,  VAL_OBJ  },
    { "type_of",    1, SLOT_OBJ,  VAL_OBJ  },
    { "len",        1, SLOT_INT,  VAL_INT  },
    { "hash",       1, SLOT_INT,  VAL_INT  },
    { "id",         1, SLOT_INT,  VAL_INT  },
    { "ord",        1, SLOT_INT,  VAL_INT  },
    { "int",        1, SLOT_INT,  VAL_INT  },
    { "int",        2, SLOT_INT,  VAL_INT  },
    { "bool",       1, SLOT_BOOL, VAL_BOOL },
    { "callable",   1, SLOT_BOOL, VAL_BOOL },
    { "isinstance", 2, SLOT_BOOL, VAL_BOOL },
    /* No `range` row. It would emit, but the loop that consumes the result
     * declines one instruction later ("iterating something other than a list
     * or range") because SLOT_OBJ does not say `range` -- so the body is
     * refused either way and the row only buys an allocation. */
};

/* `sum`, `min` and `max` over a LIST answer with the element's own kind: an int
 * list sums to an int and a float list to a float (checked against `type_of`,
 * not assumed). So unlike every row in kNativeResults their result is a
 * property of the ARGUMENT, and the table cannot state it.
 *
 * Worth the separate arm because they are cheap next to the loop around them,
 * which is the test a builtin row has to pass -- `sorted` is not, and its rows
 * were built and discarded for measuring zero
 * (docs/research/FALSIFIED-list-returning-builtins.md). A five-element `sum` is
 * five adds; `docs/probes/p20_sum_min_max.jai` ran 17,000,327 interpreted
 * instructions, i.e. the whole loop, on account of this one refusal.
 *
 * The exemplar comes from a live list if the model has one and otherwise from
 * `stackElem`, which is what OP_BUILD_LIST recorded -- the same two sources the
 * iterate arm reads, in the same order. */
/* JAITHON_JIT_LIST_SCALAR=0 turns the arm below off, so it can be A/B'd inside
 * ONE binary. Not a nicety: two-binary comparisons are where this tree's
 * measurements go wrong -- each switch invalidates __jaicache__, and three
 * people measuring one change tonight two-binary got 3.9x, 100x and 6%
 * SLOWER for what a switch settled in one command. */
/* JAITHON_JIT_NEGATE=0 turns off the OP_NEG arm, for a one-binary A/B. */
bool jitSoftField(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SOFT_FIELD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitMembership(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MEMBERSHIP");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitTuple(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_TUPLE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool jitNegate(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NEGATE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_LIST_RESULT=0 turns off the predicted result for a list method
 * that is neither a field read nor discarded, for a one-binary A/B. */
bool jitListResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_RESULT");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitListScalarResult(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_SCALAR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool listScalarResult(const Emit *e, const char *nm, unsigned argc,
                             SlotKind *kind, uint8_t *tag) {
    if (!jitListScalarResult()) return false;
    if (argc != 1 && !(argc == 2 && strcmp(nm, "sum") == 0)) return false;
    if (strcmp(nm, "sum") != 0 && strcmp(nm, "min") != 0 &&
        strcmp(nm, "max") != 0) {
        return false;
    }
    unsigned idx = e->depth - argc;
    if (e->stack[idx] != SLOT_LIST) return false;

    Value elem = NULL_VAL;
    Value seen = e->stackSeen[idx];
    if (IS_LIST(seen) && AS_LIST(seen)->count > 0) {
        elem = jaiListGet(AS_LIST(seen), 0);
    }
    if (IS_NULL(elem)) elem = e->stackElem[idx];

    if (IS_INT(elem))   { *kind = SLOT_INT;   *tag = VAL_INT;   return true; }
    if (IS_FLOAT(elem)) { *kind = SLOT_FLOAT; *tag = VAL_FLOAT; return true; }
    return false;
}

/* 1 emitted, 0 no row for this builtin, -1 the emit failed. */
int emitNativeResultCall(Emit *e, Value cv, const char *nm,
                                unsigned argc, uint32_t afterIp) {
    /* inlinableBody admits OP_CALL only for the two builtins the tier emits as
     * a single instruction, on the grounds that an inlined body cannot call.
     * Refusing here keeps that true even if a constant string reaches an
     * `int()` inside one. */
    if (e->inlining) return 0;

    NativeResult derived;
    const NativeResult *nr = NULL;
    for (size_t i = 0; i < sizeof kNativeResults / sizeof kNativeResults[0]; i++) {
        if (kNativeResults[i].argc == argc &&
            strcmp(kNativeResults[i].name, nm) == 0) {
            nr = &kNativeResults[i];
            break;
        }
    }
    if (nr == NULL) {
        SlotKind dk;
        uint8_t dtag;
        if (!listScalarResult(e, nm, argc, &dk, &dtag)) return 0;
        derived.name = nm;
        derived.argc = argc;
        derived.kind = dk;
        derived.tag  = dtag;
        nr = &derived;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return -1;
    }
    for (unsigned i = 0; i < argc; i++) {
        /* A class argument occupies no register, so there is nothing to pop --
         * only the entry to drop. */
        if (e->depth > 0 && !holdsRegister(e->stack[e->depth - 1])) {
            e->depth--;
            continue;
        }
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return -1; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_NATIVE) {
        e->whyNot = "callee was not where it should be";
        return -1;
    }
    e->depth--;
    if (!pushValue(e, nr->kind, 0, NULL)) return -1;

    unsigned at = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, at));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, nr->tag));
    /* Resumes AFTER the call, taking the result from the descriptor: the
     * native has already run and may have written, so re-running it is not on
     * offer. `lastFromDesc` is what hands the interpreter the Value the native
     * actually produced, whatever tag it turned out to have. */
    branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
    /* A bool is ONE byte of the Value union; reading eight would carry the
     * neighbouring bytes of the result slot into the register. */
    if (nr->kind == SLOT_BOOL) {
        emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, at + 8));
    } else {
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31, at + 8));
    }
    /* A call is an effect: no bail may follow it. */
    e->wroteHeap = true;
    return 1;
}

#endif /* __aarch64__ || __arm64__ */
