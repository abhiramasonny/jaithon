/* jit_call_desc.c -- the call descriptor: rooting the live values a call out
 * could collect, and building the descriptor a helper reads its arguments from. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

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

/* The raw layout (gJitRawRoots): each root is the bare payload, stored two at
 * a time. A register waits in `pend` for a partner; a pair goes down with one
 * `stp` while its offset is in stp's reach from sp and as two `str` past it. */
typedef struct {
    Emit    *e;
    unsigned at;      /* byte offset of pointer 0 */
    unsigned n;       /* pointers placed or pending */
    int      pend;    /* register waiting for a partner, or -1 */
} RawFill;

static void rawFlush(RawFill *f) {
    if (f->pend < 0) return;
    unsigned off = f->at + 8u * (f->n - 1u);
    emit(f->e, jaiA64StrX((unsigned)f->pend, 31, off));
    f->pend = -1;
}

static void rawPut(RawFill *f, unsigned reg) {
    if (f->pend < 0) {
        f->pend = (int)reg;
        f->n++;
        return;
    }
    unsigned off = f->at + 8u * (f->n - 1u);
    if (off + 8u <= 504u) {
        emit(f->e, jaiA64StpOff((unsigned)f->pend, reg, 31, (int32_t)off));
    } else {
        emit(f->e, jaiA64StrX((unsigned)f->pend, 31, off));
        emit(f->e, jaiA64StrX(reg, 31, off + 8u));
    }
    f->pend = -1;
    f->n++;
}

/* The scratch a local loads into must not be the one already waiting. */
static unsigned rawScratch(const RawFill *f) {
    return f->pend == (int)JIT_SCRATCH_C ? JIT_SCRATCH_B : JIT_SCRATCH_C;
}

static bool emitRootFillRaw(Emit *e, unsigned d, unsigned *nrootsOut) {
    RawFill f = { e, d + (unsigned)offsetof(JitCallDesc, roots), 0u, -1 };
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        /* An inline's home is dead outside the inline and never zeroed: as a
         * bare pointer it would hand the marker whatever the slot last held. */
        if (e->inlHomeLo != 0 && slot >= e->inlHomeLo) continue;
        SlotKind k = e->localKind[slot];
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            continue;
        }
        if (f.n >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        /* A dynamic local is guarded before it is read, and the guard settles
         * the operand stack: nothing of this fill may be left waiting in a
         * register across that. */
        if (e->dynamicLocal[slot]) rawFlush(&f);
        rawPut(&f, localIn(e, slot, rawScratch(&f)));
    }
    /* The operand stack, counted from the bottom exactly as the tagged fill
     * below counts it. */
    unsigned seen = 0;
    for (unsigned idx = 0; idx < e->depth; idx++) {
        SlotKind k = e->stack[idx];
        if (!holdsRegister(k)) continue;
        unsigned reg = valueBankReg(e, seen);
        seen++;
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST && k != SLOT_MAYBE_OBJ) {
            continue;
        }
        if (f.n >= jitRootLimit()) {
            e->whyNot = "too many roots"; return false;
        }
        rawPut(&f, reg);
    }
    rawFlush(&f);
    *nrootsOut = f.n;
    return true;
}

bool emitRootFill(Emit *e, unsigned d, unsigned *nrootsOut) {
    if (gJitRawRoots) return emitRootFillRaw(e, d, nrootsOut);
    unsigned nroots = 0;
    for (unsigned slot = e->base; slot < e->base + e->locals; slot++) {
        /* An inline's home is dead outside the inline, and nothing inside
         * one calls: never a root, and possibly stale. */
        if (e->inlHomeLo != 0 && slot >= e->inlHomeLo) continue;
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
        /* SLOT_MAYBE_OBJ belongs here for the reason the others do and one
         * more: an allow-list that SKIPS a live pointer leaves it unrooted
         * across the call, which is not a refusal but a collected object.
         * Its null case costs nothing -- emitTagFor writes VAL_NULL for a
         * zero payload, and a null root is inert. */
        if (k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST && k != SLOT_MAYBE_OBJ) {
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

/* JAITHON_JIT_LEAF_CALL_ROOTS (default on): a bare `bl` from the function
 * tier to a compiled body that cannot reach the collector except on its way
 * to raising fills no roots and links nothing.
 *
 * Why that is sound. Roots matter only while a collection can run, and the
 * only collection such a callee can start is the one its raise allocates in
 * (jitThrowOverflow). A raise answers the caller with verdict 2, and a
 * function-tier frame answers that by leaving through its epilogue, which
 * reads none of its object registers; so does every function-tier frame
 * above it, until the C entry or an OSR frame -- and an OSR frame, whose way
 * out does write its registers back into the interpreter's slots, always
 * roots. A callee that writes is excluded because its verdict 4 is finished
 * in the interpreter from this frame's stub, which collects with this frame
 * live; a callee that bails or deopts without writing has run nothing that
 * collects, and this frame's own deopt then reads its registers as they were.
 * A compiled body is recorded by its entry address, which the arena never
 * reuses, so the answer stays true of the code a caller bakes in. */
static bool jitLeafCallRoots(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_LEAF_CALL_ROOTS");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Entry addresses of compiled bodies that cannot collect, open addressing. */
static uintptr_t *gNoCollect;
static size_t     gNoCollectCap, gNoCollectCount;

static size_t noCollectSlot(uintptr_t key, size_t cap) {
    return (size_t)((key >> 4) * 0x9E3779B97F4A7C15ull) & (cap - 1u);
}

void jitNoCollectRecord(const uint8_t *code, bool noCollect) {
    if (code == NULL || !noCollect) return;
    if ((gNoCollectCount + 1u) * 2u > gNoCollectCap) {
        size_t cap = gNoCollectCap ? gNoCollectCap * 2u : 256u;
        uintptr_t *grown = calloc(cap, sizeof *grown);
        if (grown == NULL) return;
        for (size_t i = 0; i < gNoCollectCap; i++) {
            uintptr_t k = gNoCollect[i];
            if (k == 0) continue;
            size_t at = noCollectSlot(k, cap);
            while (grown[at] != 0) at = (at + 1u) & (cap - 1u);
            grown[at] = k;
        }
        free(gNoCollect);
        gNoCollect = grown;
        gNoCollectCap = cap;
    }
    uintptr_t key = (uintptr_t)code;
    size_t at = noCollectSlot(key, gNoCollectCap);
    while (gNoCollect[at] != 0) {
        if (gNoCollect[at] == key) return;
        at = (at + 1u) & (gNoCollectCap - 1u);
    }
    gNoCollect[at] = key;
    gNoCollectCount++;
}

static bool jitNoCollectKnown(const uint8_t *code) {
    if (code == NULL || gNoCollectCap == 0) return false;
    uintptr_t key = (uintptr_t)code;
    size_t at = noCollectSlot(key, gNoCollectCap);
    while (gNoCollect[at] != 0) {
        if (gNoCollect[at] == key) return true;
        at = (at + 1u) & (gNoCollectCap - 1u);
    }
    return false;
}

/* Whether a bare `bl` to `cfn`'s compiled entry may leave its roots out; the
 * caller then announces the call with callExempt so that this body stays
 * collect-free in turn. `hasSelfSlow`: the site finishes a deoptimised callee
 * in the interpreter, which collects. */
bool jitCallSkipsRoots(const Emit *e, const ObjFunction *cfn, bool hasSelfSlow) {
    /* Inside a `try` the raise is not this frame's way out but its handler's
     * way in, and the handler reads the registers this skip left unrooted.
     * raiseExitAllowed refuses such a call today; this keeps the skip sound
     * if a handler is ever resumed in the compiled frame. */
    if (!jitLeafCallRoots() || e->osr || hasSelfSlow || e->inProtected) {
        return false;
    }
    return jitNoCollectKnown(cfn->jitFunc);
}

_Static_assert(offsetof(JitCallDesc, link) == 0 &&
               offsetof(JitCallDesc, nroots) == 8,
               "emitChainLink stores link and nroots as one pair");

/* Links this frame's descriptor onto gJitFrames around a bare `bl`, which
 * pushes no roots, with `nroots` roots already filled: the old head and the
 * count go down as one pair. */
void emitChainLink(Emit *e, unsigned nroots) {
    unsigned d = e->descOffset;
    emitChainHeadAddr(e, JIT_SCRATCH_A);
    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64MovzX(JIT_SCRATCH_C, nroots, 0));
    if (d + 8u <= 504u) {
        emit(e, jaiA64StpOff(JIT_SCRATCH_B, JIT_SCRATCH_C, 31, (int32_t)d));
    } else {
        emit(e, jaiA64StrX(JIT_SCRATCH_B, 31, d));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, 31, d + 8u));
    }
    emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
    emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
}

/* Puts back the head emitChainLink saved. x0 and x1 carry the callee's
 * answer, so only the scratches are touched. */
void emitChainUnlink(Emit *e) {
    emit(e, jaiA64LdrX(JIT_SCRATCH_B, 31, e->descOffset));
    emitChainHeadAddr(e, JIT_SCRATCH_A);
    emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}

bool emitDescriptorStatus(Emit *e, Value calleeVal, unsigned first,
                                 unsigned nargs, void *helper, bool ownStatus,
                                 int calleeReg) {
    return emitDescriptorFull(e, calleeVal, first, nargs, helper, ownStatus,
                              calleeReg, false, 0, false);
}

/* `noRoots`: a LEAF helper that never collects (it declines instead), so the
 * root fill -- a store per live object the body holds -- is skipped and the
 * descriptor carries none. `protThrow` with `protOff`: the call sits in a
 * `try` in an OSR loop, so a raise unwinds to the handler through the
 * throw-ip trampoline instead of declining; `protOff` is the call's own
 * offset (the helper prefers the outer call when inlining). */
bool emitDescriptorFull(Emit *e, Value calleeVal, unsigned first,
                        unsigned nargs, void *helper, bool ownStatus,
                        int calleeReg, bool noRoots, uint32_t protOff,
                        bool protThrow) {
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
            k != SLOT_ITER && k != SLOT_MAYBE_INST && k != SLOT_MAYBE_OBJ) {
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
    if (!noRoots && !emitRootFill(e, d, &nroots)) return false;
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

    /* Nonzero means the callee raised; the interpreter owns it from here. A
     * descriptor call never re-executes -- the interpreter throws with the
     * effects intact -- so the trampoline is sound for any callee, however
     * much it writes. */
    if (!protThrow &&
        !raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    emit(e, jaiA64SubsXImm(31, 0, 0));
    if (protThrow) {
        if (!emitProtectedThrew(e, JAI_A64_EQ, protOff)) {
            e->failed = true;
            return false;
        }
    } else {
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
    }
    return true;
}

bool emitDescriptor(Emit *e, Value calleeVal, unsigned first,
                           unsigned nargs, void *helper) {
    return emitDescriptorStatus(e, calleeVal, first, nargs, helper, false, -1);
}

bool emitDescriptorAt(Emit *e, Value calleeVal, unsigned first,
                             unsigned nargs, void *helper, uint32_t callOff,
                             bool protThrow) {
    return emitDescriptorFull(e, calleeVal, first, nargs, helper, false, -1,
                              false, callOff, protThrow);
}

#endif /* __aarch64__ || __arm64__ */
