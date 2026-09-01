/* jit_call_desc.c -- the call descriptor: rooting the live values a call out
 * could collect, and building the descriptor a helper reads its arguments from. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* ownStatus: caller decodes the helper's return itself, skipping the built-in "nonzero means raised"
 * test. Written for the iterator step (0 yielded, 1 exhausted, 2 raised), whose call-out the list arm of
 * OP_FOR_ITER_BIND no longer makes -- the default test sent `exhausted` to the throw stub, which found no pending exception and died on "internal error: failed operation raised nothing". Kept because any helper with a three-way answer needs it, and because the lesson is not rediscoverable from the code. */
/* Root-fills the descriptor: shared by the descriptor path (a C helper pushes them) and the self-call
 * path (the emitted code links the descriptor onto the collector's frame chain instead, since a bare `bl` pushes nothing). */
/* JAITHON_JIT_ROOT_LIMIT=10 puts the root cap back where it was when it shared
 * the register budget, for a one-binary A/B. The ARRAY is always the wider one,
 * so only the refusal moves. */
static unsigned jitRootLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ROOT_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JIT_MAX_ROOTS;
        if (cached < 1 || cached > (int)JIT_MAX_ROOTS) cached = (int)JIT_MAX_ROOTS;
    }
    return (unsigned)cached;
}

bool emitRootFill(Emit *e, unsigned d, unsigned *nrootsOut) {
    unsigned nroots = 0;
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        if (e->localKind[slot] != SLOT_INST &&
            e->localKind[slot] != SLOT_LIST &&
            e->localKind[slot] != SLOT_OBJ &&
            e->localKind[slot] != SLOT_ITER &&
            e->localKind[slot] != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        unsigned rslot = localIn(e, slot, JIT_SCRATCH_C);
        emitTagFor(e, e->localKind[slot], rslot, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(rslot, 31, at + 8));
        nroots++;
    }

    /* Locals alone weren't enough once OP_GET_ITER started leaving an ObjIter live in a register across
 * a call that might collect -- only visible under --gc-stress, and only once such a loop could compile at all. */
    /* Counts register-holding entries from the bottom, not by assuming they're the top `valueDepth` --
 * a no-register entry (class/function/builtin/self) can sit in the middle of the stack, e.g. `join(f(a), f(b))` pushes a callee before its arguments. Subtracting valueDepth would name the wrong register above it and skip entries that still need rooting; the deopt stub has always counted this way. */
    unsigned seen = 0;
    for (unsigned idx = 0; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (!holdsRegister(k)) continue;
        unsigned reg = valueBankReg(e, seen);
        seen++;
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            continue;
        }
        if (nroots >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        unsigned at = d + (unsigned)offsetof(JitCallDesc, roots) +
                      nroots * (unsigned)sizeof(Value);
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
        nroots++;
    }

    *nrootsOut = nroots;
    return true;
}

bool emitDescriptorStatus(Emit *e, Value calleeVal, unsigned first,
                                 unsigned nargs, void *helper, bool ownStatus,
                                 int calleeReg) {
    if (nargs > JIT_MAX_ARGS_OUT) { e->whyNot = "call argc"; return false; }
    if (!e->callsOut) { e->whyNot = "callsOut off"; return false; }

    unsigned d = e->descOffset;

    /* Callee as a whole Value, from a register when only known at run time (a closure held in a local):
     * baking the compile-time-live closure would freeze its upvalues -- `closure_calls` builds a fresh closure over a different `step` every outer iteration. */
    if (calleeReg >= 0) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emit(e, jaiA64StrX((unsigned)calleeReg, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    } else {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, (unsigned)calleeVal.type, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)calleeVal.as.obj);
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           d + (unsigned)offsetof(JitCallDesc, callee) + 8));
    }

    /* The value index of the first argument. Counted rather than derived from
     * `depth - valueDepth`, which is the number of register-free entries below
     * `depth` ANYWHERE: a class argument is one of those, and every argument
     * above it would then be read out of the wrong register. */
    unsigned vidx = e->valueDepth;
    for (unsigned idx = first; idx < e->depth; idx++) {
        if (holdsRegister(e->stack[idx])) vidx--;
    }

    /* The arguments, which for an invoke begin with the receiver. */
    for (unsigned i = 0; i < nargs; i++) {
        unsigned idx = first + i;
        SlotKind k = e->stack[idx];
        unsigned at = d + (unsigned)offsetof(JitCallDesc, args) +
                      i * (unsigned)sizeof(Value);
        /* A class occupies no register -- it is a constant of the module, and
         * the class the interpreter would have found is the one the model
         * recorded. Baking it is the same trust the callee slot above already
         * takes, and it is what lets `isinstance(x, T)` be called at all. */
        if (k == SLOT_CLASS) {
            ObjClass *argCls = e->stackClass[idx];
            if (argCls == NULL) {
                e->whyNot = "a class argument the model did not pin";
                return false;
            }
            emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_A, 31, at));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)argCls);
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31, at + 8));
            continue;
        }
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            e->whyNot = "an argument kind this call cannot pass";
            return false;
        }
        unsigned reg = valueBankReg(e, vidx);
        vidx++;
        /* A maybe-instance's tag is not a property of its kind, and this Value
         * reaches jaiCallValue: writing VAL_OBJ over a zero payload would hand
         * the interpreter a null pointer dressed as an object. */
        emitTagFor(e, k, reg, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
        emit(e, jaiA64StrX(reg, 31, at + 8));
    }

    unsigned nroots = 0;
    if (!emitRootFill(e, d, &nroots)) return false;
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nargs, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, argc)));
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, nroots, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                       d + (unsigned)offsetof(JitCallDesc, nroots)));

    emit(e, jaiA64AddXImm(0, 31, d));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)helper);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));

    if (ownStatus) return true;

    /* Nonzero means the callee raised; the interpreter owns it from here. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    emit(e, jaiA64SubsXImm(31, 0, 0));
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_NE, 0));
    return true;
}

bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper) {
    return emitDescriptorStatus(e, calleeVal, first, nargs, helper, false, -1);
}

#endif /* __aarch64__ || __arm64__ */
