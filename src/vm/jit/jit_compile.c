/* jit_compile.c -- the compile driver: eligibility, seeding the model from a live
 * frame, register planning, and assembling the two passes into a code arena. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
#include "runtime/runtime.h"
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

#include <pthread.h>

/* ------------------------------------------------------------------ */
/* Run-time state shared with compiled code                             */
/* ------------------------------------------------------------------ */

/* Derived from the thread's real bounds, not the compiling frame's sp -- that would bail on
 * every entry for a function compiled near the top of stack and called later from deep in the interpreter. */
static uintptr_t stackLimit(void) {
    pthread_t self = pthread_self();
    void  *top  = pthread_get_stackaddr_np(self);
    size_t size = pthread_get_stacksize_np(self);
    if (top == NULL || size == 0) return 0;
    /* Margin covers the deepest compiled frame plus what the interpreter needs to unwind and report the error. */
    return (uintptr_t)top - size + (256u * 1024u);
}

/* ------------------------------------------------------------------ */
/* Assembly                                                             */
/* ------------------------------------------------------------------ */


/* The kinds of the parameters, read off the arguments this call was made with.
 * Everything downstream is specialised to them, and the entry guard re-checks
 * them on every later call. */
static bool seedLocals(Emit *e, Value *slotBase) {
    e->observed = slotBase;
    for (unsigned i = 0; i < e->base + e->locals; i++) {
        e->localKind[i]   = SLOT_INT;
        e->localShape[i]  = 0;
        e->localClass[i]  = NULL;
        e->localTyped[i]  = false;
    }
    for (unsigned i = e->base; i <= e->arity; i++) {
        Value v = slotBase[i];
        if (IS_INT(v)) {
            e->localKind[i] = SLOT_INT;
        } else if (IS_FLOAT(v)) {
            e->localKind[i] = SLOT_FLOAT;
        } else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            /* A slot an earlier attempt found sometimes-null holds the pointer or zero; the class still comes
         * from this call's argument -- a maybe-instance is a correct supertype, so widening can't make a field offset wrong. */
            e->localKind[i]  = e->nullableLocal[i] ? SLOT_MAYBE_INST
                                                   : SLOT_INST;
            e->localClass[i] = AS_INSTANCE(v)->klass;
            e->localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_LIST(v)) {
            e->localKind[i] = SLOT_LIST;
        } else if (IS_OBJ(v)) {
            e->localKind[i] = SLOT_OBJ;
        } else if (IS_BOOL(v)) {
            e->localKind[i] = SLOT_BOOL;
        } else if (IS_NULL(v)) {
            /* A null argument (a defaulted parameter, mostly): nothing can be done with it, but a body that
             * never reads it compiles, and the entry guard needs no check since an opaque slot is never read. Refusing outright stopped every stdlib function with a defaulted parameter -- most of them. */
            e->localKind[i] = SLOT_OPAQUE;
        } else if (i == 0) {
            /* A plain function's slot 0 is the closure being called. Nothing
             * can be done with it, but the body has no reason to read it. */
            e->localKind[i] = SLOT_OPAQUE;
        } else {
            e->whyNot = "a parameter of a kind the tier has no register for";
            return false;
        }
        e->localTyped[i] = true;
    }
    return true;
}

/* A local that is not a parameter takes the kind of the first thing bound to
 * it; after that it must keep it, because every read of it was compiled to one
 * instruction chosen from that kind. */
bool adoptLocalKindSeen(Emit *e, unsigned slot, SlotKind kind,
                               uint32_t shape, ObjClass *klass, Value seen) {
    if (!IS_NULL(seen)) e->localSeen[slot] = seen;
    if (!e->localTyped[slot]) {
        /* A slot an earlier attempt asked to be widened takes the wider kind
         * the FIRST time something is bound to it too, not only when it was
         * seeded from a live value. A local declared inside the loop holds
         * nothing at the moment the loop tier looks -- it seeds SLOT_OPAQUE --
         * so without this the retry seeded nothing and found the same clash
         * again, which is exactly what `var at = head` inside the outer loop
         * does. */
        e->localKind[slot]  = (kind == SLOT_INST && e->nullableLocal[slot])
                                  ? SLOT_MAYBE_INST : kind;
        e->localShape[slot] = shape;
        e->localClass[slot] = klass;
        e->localTyped[slot] = true;
        return true;
    }
    if (e->localKind[slot] == kind && e->localShape[slot] == shape) return true;

    /* Two kinds for one slot. Rather than give the function up, note that this
     * slot has to carry its tag and let the caller compile again with that
     * decided from the start -- every read of it then guards, so the two
     * kinds stop being a contradiction.
     *
     * An instance and a nullable instance of the SAME class are not really two
     * kinds, though: the maybe-instance is the supertype, holds the identical
     * pointer-or-zero, and every field offset resolved against the class stays
     * right. So that clash asks for the nullable seed first and only falls to
     * the dynamic one if a second attempt still disagrees -- which matters
     * because dynamic keeps the slot in memory with a run-time tag, and this
     * is the shape every list walk has: `var at = head` then
     * `at = at.next`. */
    /* An instance into a slot already typed maybe-instance is not a clash at
     * all: it is the subtype going into the supertype, the register holds the
     * same bare pointer, and the tag a materialisation computes off that
     * pointer is the right one. The slot keeps the wider kind. Only the other
     * direction has to ask for anything, because a null reaching a slot typed
     * as a plain instance is what lets a later field read dereference zero. */
    if (e->localKind[slot] == SLOT_MAYBE_INST && kind == SLOT_INST &&
        e->localShape[slot] == shape) {
        return true;
    }
    if (e->localKind[slot] == SLOT_INST && kind == SLOT_MAYBE_INST &&
        e->localShape[slot] == shape &&
        !e->nullableLocal[slot] && !e->dynamicLocal[slot]) {
        e->needNullable[slot] = true;
        e->clashKind = kind;
        if (e->measuring && jitCollectClashes()) {
            e->pendingRetry = true;
            e->localKind[slot] = SLOT_MAYBE_INST;
            return true;
        }
        return false;
    }
    if (!e->dynamicLocal[slot]) {
        e->needDynamic[slot] = true;
        e->clashKind = kind;
        if (e->measuring && jitCollectClashes()) {
            e->pendingRetry = true;
            e->localKind[slot]  = kind;
            e->localShape[slot] = shape;
            e->localClass[slot] = klass;
            return true;
        }
        return false;
    }
    e->localKind[slot]  = kind;
    e->localShape[slot] = shape;
    e->localClass[slot] = klass;
    return true;
}

bool adoptLocalKind(Emit *e, unsigned slot, SlotKind kind,
                           uint32_t shape, ObjClass *klass) {
    return adoptLocalKindSeen(e, slot, kind, shape, klass, NULL_VAL);
}

void jitFree(int *map, int *depths, int *chunkDepth, int count) {
    JAI_FREE_ARRAY(int, map, count);
    JAI_FREE_ARRAY(int, depths, count);
    JAI_FREE_ARRAY(int, chunkDepth, count);
}

/* The bytecode's own answer for every offset, or NULL if it cannot be had.
 * NULL is not a failure: modelAgreesWithChunk simply has nothing to check
 * against, which is the state the tier was in before this existed. */
int *chunkDepthTable(const ObjFunction *fn) {
    int *d = JAI_ALLOC(int, fn->chunk.count + 1);
    if (d == NULL) return NULL;
    if (!jaiChunkStackDepths(fn, d)) {
        JAI_FREE_ARRAY(int, d, fn->chunk.count + 1);
        return NULL;
    }
    return d;
}

static bool eligible(ObjFunction *fn) {
    const char *why = NULL;
    /* Arity 0 is allowed: a method whose only parameter is the receiver has
     * arity 0, and getters are the commonest shape there is. A plain function
     * with no arguments reaches here too and declines on its opcodes. */
    if (fn->arity > JIT_MAX_ARITY) why = "arity";
    else if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > JIT_MAX_SLOTS) why = "maxSlots";
    /* Defaults used to make a function categorically ineligible, which ruled
     * out most library code -- an API designed with optional arguments was
     * uncompilable by construction. It is safe because every entry already
     * requires the arguments to have been supplied in full: callClosure and
     * the one-argument path both test `argc == fn->arity`, and OP_TAIL_CALL
     * reuses its window only for a callee with no defaults at all. A call that
     * omits one therefore never reaches compiled code; it fills the defaults
     * and runs interpreted, exactly as before. */
    else if (fn->flags & (FN_VARIADIC | FN_KWREST)) why = "flags";

    else if (fn->module == NULL) why = "no module";
    else if (fn->chunk.count <= 0) why = "empty";
    if (why != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s ineligible: %s (arity=%d maxSlots=%d)\n",
                    fn->name ? fn->name->chars : "<anon>", why, (int)fn->arity,
                    (int)fn->maxSlots);
        }
        return false;
    }
    return true;
}


/* Greedy: most instructions-saved first, across BOTH register banks at once (an int slot read 4x in
 * a loop outranks a float slot read 2x, different pools, but comparing them in one order keeps that meaningful when only one pool runs out -- see noteSlotCost). Three exclusions: a zero-saving slot is skipped outright, not just ranked last (else a tie-break could pay a prologue write for nothing); a dynamic slot keeps its frame home, since only the frame has anywhere to put a run-time tag; a wrong guess costs an `fmov`, never a wrong answer, since only fmov/ldr/str ever touch a home. */
/* Both tiers plan through this, on the same ledgers (see noteSlotCost). What is
 * still per-tier is named here rather than duplicated:
 *
 *   `skip`     slots the caller has already ruled out for reasons the ledgers
 *              cannot see -- OSR passes the ones captured by reference by a
 *              closure, which must stay in memory whatever they save, and the
 *              ones its probe found to be dynamic. NULL for the function tier,
 *              whose dynamic slots are in e->dynamicLocal already.
 *   xBase      OSR's callee-saved window opens ABOVE the registers the loop
 *              head reserved for its iterator; the function tier's opens at
 *              x19. Matches regBase, which the operand stack is numbered from.
 *   the FP gate  the function tier lets the two ledgers pick the bank, because
 *              jitArgIn marshals every incoming argument and an FP home there
 *              only ever sees a clean value. OSR's homes are the interpreter's
 *              own slots and its prologue fills an FP home with a full-width
 *              `ldr d`; for a SLOT_BOOL that loads the seven bytes above a
 *              one-byte `strb`, which is the garbage-high-byte bug the OSR
 *              prologue's X arm narrows against. Only a float may take an FP
 *              home there.
 *   `strandedOut`  counted for the OSR census: slots that earned a register and
 *              found none left, which is what a wider bank would buy. */
void planSlotRegisters(Emit *e, const Emit *m, unsigned availX,
                              const bool *skip, unsigned *strandedOut) {
    unsigned availFp = JIT_FP_MAX_SAVED;
    unsigned xBase = e->osr ? osrReserved(e) : 0u;
    unsigned top = e->base + e->locals;
    if (top > JIT_MAX_SLOTS + 1u) top = JIT_MAX_SLOTS + 1u;
    while (availX > 0 || availFp > 0) {
        unsigned bestSlot = 0, bestGain = 0;
        bool bestFp = false;
        for (unsigned slot = e->base; slot < top; slot++) {
            if (e->slotXReg[slot] != 0 || e->slotFpReg[slot] != 0) continue;
            if (e->dynamicLocal[slot]) continue;
            if (skip != NULL && skip[slot]) continue;
            bool fpOk = !e->osr || m->localKind[slot] == SLOT_FLOAT;
            if (availFp > 0 && fpOk && m->slotSaveFp[slot] > bestGain) {
                bestGain = m->slotSaveFp[slot];
                bestSlot = slot;
                bestFp   = true;
            }
            if (availX > 0 && m->slotSaveX[slot] > bestGain) {
                bestGain = m->slotSaveX[slot];
                bestSlot = slot;
                bestFp   = false;
            }
        }
        if (bestGain == 0) break;
        if (bestFp) {
            e->slotFpReg[bestSlot] =
                (uint8_t)(JIT_FP_FIRST_SAVED + e->fpLocals++);
            availFp--;
        } else {
            e->slotXReg[bestSlot] =
                (uint8_t)(JIT_FIRST_SAVED + xBase + e->xLocals++);
            availX--;
        }
    }
    if (strandedOut == NULL) return;
    for (unsigned slot = e->base; slot < top; slot++) {
        if (e->slotXReg[slot] != 0 || e->slotFpReg[slot] != 0) continue;
        if (e->dynamicLocal[slot]) continue;
        if (skip != NULL && skip[slot]) continue;
        if (m->slotSaveX[slot] == 0 && m->slotSaveFp[slot] == 0) continue;
        (*strandedOut)++;
    }
}

/* Writes a body into the arena and seals it again, or seals and reports
 * failure.
 *
 * The sealing is the point. `jaiCodeArenaUnseal` turns the WHOLE arena back to
 * read-write, which takes the execute bit off every function already compiled
 * into it -- so an early return between the unseal and the seal leaves the
 * tier's entire back catalogue unexecutable, and the next call into any of it
 * dies with KERN_PROTECTION_FAILURE inside `callClosure`. That is not
 * hypothetical: the write fails exactly when the 1 MB arena is full, which a
 * long enough run reaches, and it produced an intermittent SIGBUS in the test
 * suite that moved around as unrelated changes altered how much got compiled.
 * Every exit from here re-seals. */
uint8_t *arenaEmit(JaiCodeArena *arena, const uint32_t *code,
                          unsigned count) {
    while ((arena->used & 31u) != 0) {
        uint32_t pad = jaiA64Nop();
        if (jaiCodeArenaWrite(arena, &pad, sizeof pad) == NULL) {
            jaiCodeArenaSeal(arena);
            return NULL;
        }
    }
    uint8_t *entry = jaiCodeArenaWrite(arena, code, count * sizeof code[0]);
    if (entry == NULL) {
        jaiCodeArenaSeal(arena);
        return NULL;
    }
    if (!jaiCodeArenaSeal(arena)) return NULL;
    return entry;
}

static bool compileFuncOnce(ObjClosure *closure, Value *slotBase,
                            const bool *dynamic, bool *needDynamic,
                            const bool *nullable, bool *needNullable,
                            bool noInline);

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    ObjFunction *fn = closure->fn;
    if (!eligible(fn)) return false;

    /* Up to a few attempts: each one may discover another slot that two paths
     * disagree about, and the next begins knowing it. */
    bool dynamic[JIT_MAX_SLOTS + 1];
    bool need[JIT_MAX_SLOTS + 1];
    bool nullable[JIT_MAX_SLOTS + 1];
    bool needNull[JIT_MAX_SLOTS + 1];
    memset(dynamic, 0, sizeof dynamic);
    memset(nullable, 0, sizeof nullable);
    for (int attempt = 0; attempt < 4; attempt++) {
        memset(need, 0, sizeof need);
        memset(needNull, 0, sizeof needNull);
        if (compileFuncOnce(closure, slotBase, dynamic, need, nullable,
                            needNull, false)) {
            return true;
        }
        /* An inlined body that could not be emitted is not a decline: the
         * same call through the descriptor still compiles, and a compiled
         * form with a real call in it beats none at all. */
        if (gInlineFailed &&
            compileFuncOnce(closure, slotBase, dynamic, need, nullable,
                            needNull, true)) {
            return true;
        }
        bool grew = false;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (need[i] && !dynamic[i]) { dynamic[i] = true; grew = true; }
            /* Dynamic wins: it is the more general representation, and a slot
             * that wants both is one the nullable form cannot describe. */
            if (needNull[i] && !dynamic[i] && !nullable[i]) {
                nullable[i] = true; grew = true;
            }
        }
        if (!grew) return false;
    }
    return false;
}

static bool compileFuncOnce(ObjClosure *closure, Value *slotBase,
                            const bool *dynamic, bool *needDynamic,
                            const bool *nullable, bool *needNullable,
                            bool noInline) {
    ObjFunction *fn = closure->fn;
    gInlineFailed = false;

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] considering %s\n",
                fn->name ? fn->name->chars : "<anon>");
    }

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int *depths = JAI_ALLOC(int, fn->chunk.count + 1);
    int *chunkDepth = chunkDepthTable(fn);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }


    /* Static, not automatic: at this size two of them would be a large stack
     * frame, and compilation is not reentrant -- nothing it calls compiles
     * anything. */
    static Emit e;
    memset(&e, 0, sizeof e);
    memcpy(e.dynamicLocal, dynamic, sizeof e.dynamicLocal);
    memcpy(e.nullableLocal, nullable, sizeof e.nullableLocal);
    e.arity        = fn->arity;
    e.noInline     = noInline;
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
    e.limitLiteral = -1;
    e.bailBlock    = -1;

    /* The prologue can't be emitted first: its save set depends on how deep the operand stack gets,
     * which only the body knows. So the body goes into the buffer at a fixed offset and the prologue is written in front of it afterwards, with every instruction index shifted by the same amount. */
    static Emit body;
    memset(&body, 0, sizeof body);
    memcpy(body.dynamicLocal, dynamic, sizeof body.dynamicLocal);
    memcpy(body.nullableLocal, nullable, sizeof body.nullableLocal);
    body.arity        = fn->arity;
    body.noInline     = noInline;
    /* The measuring pass runs with slot 0 available, purely to find out
     * whether the body reads it; the real pass then drops it if not. */
    body.base         = 0;
    body.locals       = (unsigned)fn->maxSlots;
    body.usesUpvalues = fn->upvalueCount > 0;
    body.callsOut     = true;      /* the measuring pass may emit one */
    body.measuring    = true;
    body.descOffset   = 16u;
    body.offsetToInst = map;
    body.offsetToDepth = depths;
    body.chunkDepth = chunkDepth;
    body.chunkDepthCount = fn->chunk.count + 1;
    /* So the per-slot charges the accessors record are weighted by how deeply
     * nested the site is. Without it every site in the function counts the
     * same and `n`, read once to set the loop up, outranks nothing. */
    body.loopDepth = loopDepthFor(&fn->chunk, &body.loopDepthCount);
    if (!seedLocals(&body, slotBase)) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    body.whyNot ? body.whyNot : "its arguments");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* A first pass with a provisional frame, only to learn maxValue. The
     * emitted words are thrown away: frameBytes appears in the epilogue, so
     * they would be wrong. */
    body.savedCount = JIT_MAX_SAVED;
    body.frameBytes = 16 + 8 * JIT_MAX_SAVED + 8;   /* 16-aligned below */
    body.frameBytes = (body.frameBytes + 15u) & ~15u;
    /* Snapshot before the walk, so the chain diagnostic can re-run it. */
    static Emit chainProto;
    if (jitChainOn()) memcpy(&chainProto, &body, sizeof body);
    if (!compileBody(&body, closure) || body.pendingRetry) {
        memcpy(needDynamic, body.needDynamic, sizeof body.needDynamic);
        memcpy(needNullable, body.needNullable, sizeof body.needNullable);
        if (getenv("JAI_JIT_WHY")) {
            /* A RETRY IS NOT A REFUSAL, and printing it as one cost a whole
             * session of analysis. This exit is `!compileBody(...) ||
             * body.pendingRetry`: on the retry arm the walk SUCCEEDED and is
             * only asking to run again with a clashing local widened, so
             * nothing ever set a reason -- and declineReason then falls back
             * to the name of whatever opcode happened to be last. `_fuse_at`,
             * the largest single body on the compiler workload, was read as
             * "declining at OP_RETURN" for exactly that reason. It declines
             * nowhere near there. See Emit::pendingRetry. */
            if (body.pendingRetry) {
                fprintf(stderr,
                        "[jit] %s asked to retry (measuring): a local wants a "
                        "wider kind\n",
                        fn->name ? fn->name->chars : "<anon>");
            } else {
                fprintf(stderr, "[jit] %s stopped (measuring): %s\n",
                        fn->name ? fn->name->chars : "<anon>",
                        declineReason(&body));
            }
        }
        /* Chasing a chain from a retry walks a body that has not refused. */
        if (jitChainOn() && !body.pendingRetry) {
            reportChain(&chainProto, &body, closure, fn);
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* A self-call cannot reproduce slot 0: no register holds the callee. A
     * body that both recurses and reads slot 0 is not compiled. */
    if (body.usesSlot0 && body.hasSelfCall) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: a body that both recurses and reads slot 0\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }
    e.base         = body.usesSlot0 ? 0u : 1u;
    /* Only as far as the body reaches, never the whole window. */
    unsigned highest = body.maxSlotUsed;
    if (highest < fn->arity) highest = fn->arity;
    e.locals       = highest + 1u - e.base;
    e.usesUpvalues = fn->upvalueCount > 0;

    unsigned argCount = fn->arity + 1u - e.base;
    unsigned closureArg = argCount;
    if (e.usesUpvalues) argCount++;
    if (argCount > JIT_MAX_ARITY) {
        e.whyNot = "more incoming arguments than argument registers";
    }

    unsigned extra = e.usesUpvalues ? 1u : 0u;
    unsigned saved = e.locals + extra + body.maxValue;
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
        /* A tag only has somewhere to live in the frame. One dynamic slot used
         * to send every local to the frame with it; now it sends only itself,
         * because the plan below is per slot. */
        if (e.dynamicLocal[i]) { saved = JIT_MAX_SAVED + 1; break; }
    }
    if (saved > JIT_MAX_SAVED) {
        /* Too many to keep in registers, so the operand stack takes the
         * registers first -- that is expression depth, not the number of
         * variables a function happens to declare -- and whatever is left over
         * goes to the slots that earn it. What does not earn one lives in the
         * frame. */
        e.spilled = true;
        saved = extra + body.maxValue;
        if (saved > JIT_MAX_SAVED) {
            e.whyNot = "the operand stack alone exceeds the registers";
        } else {
            planSlotRegisters(&e, &body, JIT_MAX_SAVED - saved, NULL, NULL);
            saved += e.xLocals;
        }
    }
    if (e.whyNot != NULL) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s (%u locals, %u stack)\n",
                    fn->name ? fn->name->chars : "<anon>", e.whyNot, e.locals,
                    body.maxValue);
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    e.savedCount = saved;
    e.callsOut   = body.callsOut;
    unsigned frame = 16u + 8u * saved;
    if (e.spilled) {
        e.localsFrameOffset = frame;
        frame += 16u * e.locals;
        frame = (frame + 15u) & ~15u;
    }
    if (e.callsOut) {
        e.descOffset = frame;
        frame += (unsigned)sizeof(JitCallDesc);
    }
    if (e.fpLocals > 0) {
        /* v8..v15 are callee-saved in their low 64 bits, which is exactly a
         * double, so str d / ldr d is the whole protocol -- but this tier is
         * itself a callee, so the ones it takes have to be put back. */
        e.fpSaveOffset = (frame + 7u) & ~7u;
        frame = e.fpSaveOffset + 8u * e.fpLocals;
    }
    e.frameBytes = (frame + 15u) & ~15u;
    if (e.frameBytes > 4095u) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: frame of %u bytes\n",
                    fn->name ? fn->name->chars : "<anon>", e.frameBytes);
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* Prologue. */
    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    emitFpSaveRestore(&e, true);
    /* Real arguments land in local registers in order; the closure (if any) lives just past the locals,
     * where closureReg expects it. Placing it by argument index instead once put it three registers low, and the first upvalue read dereferenced whatever was there. */
    unsigned realArgs = e.usesUpvalues ? argCount - 1u : argCount;
    for (unsigned i = 0; i < realArgs; i++) {
        unsigned slot = e.base + i;
        if (!e.spilled) {
            emit(&e, jaiA64MovX(JIT_FIRST_SAVED + i, i));
        } else if (e.slotXReg[slot] != 0) {
            /* Arguments arrive in x0..x7 and homes start at x19, so no
             * destination here can be a later argument's source. */
            emit(&e, jaiA64MovX(e.slotXReg[slot], i));
        } else if (e.slotFpReg[slot] != 0) {
            emit(&e, jaiA64FmovDX(e.slotFpReg[slot], i));
        } else {
            /* Spilled local is a whole Value: tag first, payload 8 bytes on. Writing the payload at the tag's
             * offset instead leaves the payload word untouched -- the first read then gets whatever garbage the frame held, often a small int that reads as a pointer, crashing somewhere else entirely. */
            /* From the measuring pass: `e`'s own kinds are seeded after this
             * point, so reading them here would take whatever the struct was
             * zeroed to. */
            if (localTagInFrame(&e, slot)) {
                unsigned tag = localTagFor(&body, slot);
                /* Payload-dependent for every object-ish kind, exactly as
                 * localOut writes one: a null argument arrives as a zero
                 * payload (jitArgIn's SLOT_MAYBE_INST arm), and a constant
                 * VAL_OBJ over it would leave the frame claiming an object at
                 * address zero -- which localIn's guard would then trust as
                 * far as `Obj.type`, and which a deopt copies out verbatim as
                 * OBJ_VAL(NULL) for the interpreter to trip over. The other
                 * VAL_OBJ kinds cannot be null (jitArgIn refuses), so for
                 * them the csel below only ever picks the same VAL_OBJ. */
                if (tag == VAL_OBJ) {
                    emitTagFor(&e, SLOT_MAYBE_INST, i, JIT_SCRATCH_D,
                               JIT_SCRATCH_C);
                } else {
                    emit(&e, jaiA64MovzX(JIT_SCRATCH_D, tag, 0));
                }
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(&e, slot)));
            }
            emit(&e, jaiA64StrX(i, 31, localFrameOff(&e, slot) + 8));
        }
    }
    if (e.usesUpvalues) {
        emit(&e, jaiA64MovX(closureReg(&e), realArgs));
    }
    /* A local the interpreter would have left as NULL_VAL starts at zero here.
     * The checker guarantees definite assignment before any read, so this is
     * belt and braces -- but a register holding the last call's value would be
     * a bug that only shows up under recursion. */
    for (unsigned i = realArgs; i < e.locals; i++) {
        unsigned slot = e.base + i;
        if (!e.spilled) {
            emit(&e, jaiA64MovzX(JIT_FIRST_SAVED + i, 0, 0));
        } else if (e.slotXReg[slot] != 0) {
            emit(&e, jaiA64MovzX(e.slotXReg[slot], 0, 0));
        } else if (e.slotFpReg[slot] != 0) {
            emit(&e, jaiA64MovzX(JIT_SCRATCH_C, 0, 0));
            emit(&e, jaiA64FmovDX(e.slotFpReg[slot], JIT_SCRATCH_C));
        } else {
            emit(&e, jaiA64MovzX(JIT_SCRATCH_C, 0, 0));
            if (localTagInFrame(&e, slot)) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_D, VAL_NULL, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(&e, slot)));
            }
            emit(&e, jaiA64StrX(JIT_SCRATCH_C, 31, localFrameOff(&e, slot) + 8));
        }
    }
    /* Stack guard: bail rather than run off the end of the thread's stack,
     * so that runaway recursion still becomes a RecursionError. */
    int guardLoad = (int)e.count;
    emit(&e, jaiA64LdrLit(JIT_SCRATCH_A, 0));         /* patched below */
    emit(&e, jaiA64AddXImm(JIT_SCRATCH_B, 31, 0));    /* mov x10, sp */
    emit(&e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
    unsigned guardBranch = e.count;
    emit(&e, jaiA64BCond(JAI_A64_LO, 0));             /* patched below */

    unsigned prologue = e.count;

    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    e.offsetToInst  = map;
    e.offsetToDepth = depths;
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
    if (!seedLocals(&e, slotBase)) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: its locals could not be seeded on the real pass\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }
    /* declineReason, not e.whyNot: an arm that noted only a whySub used to
     * print "an unsupported operand form", which is how math.sqrt's own body
     * managed to stop on a named refusal and report nothing. */
    if (!compileBody(&e, closure) && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n", fn->name ? fn->name->chars : "<anon>",
                declineReason(&e));
    }
    if (e.failed || e.whyNot != NULL)
        { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
    if (e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    e.whyNot ? e.whyNot : "the emitter ran out of room");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }
    (void)prologue;

    /* The bail block: say so, return anything, and let the caller throw the
     * whole computation away. */
    e.bailBlock = (int)e.count;
    emit(&e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(&e, 1);

    /* The callee raised. The interpreter owns the exception and must not run
     * this call again, so this is a third answer, not a bail. */
    e.exceptionExit = (int)e.count;
    emit(&e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(&e, 2);

    emitSelfSlowStubs(&e, closure);

    emitGrowStubs(&e);

    /* One stub per guard, out of line. Each writes the record the interpreter
     * resumes from: the locals, the operand stack as it stood at that
     * instruction, and the offset of the instruction itself. */
    for (unsigned k = 0; k < e.deoptCount; k++) {
        e.deopt[k].stub = (int)e.count;
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);

        uint64_t skipLocals = 0;
        for (unsigned i = 0; i < (e.osr ? 0u : e.locals); i++) {
            unsigned slot = e.base + i;
            SlotKind kind = e.localKind[slot];
            /* An opaque slot is one the compiled body never reads, so the tier
             * never learned what is in it -- and jitArgIn passed a raw 0 for
             * it. The interpreter this record hands over to DOES read it, and
             * bindCallArgs has already put the caller's real argument in the
             * frame, so the record must say "not mine" rather than null.
             * See JitDeoptRecord::skipLocals. */
            if (kind == SLOT_OPAQUE) {
                skipLocals |= (uint64_t)1 << i;
                continue;
            }
            /* Only an opaque slot is null; everything else non-scalar is an object. Listing object kinds
             * explicitly instead once wrote a SLOT_OBJ local (string/dict/closure) out as null on every deopt -- invisible until a body holding one could compile and then actually deopt. */
            unsigned tag = kind == SLOT_INT    ? VAL_INT
                         : kind == SLOT_FLOAT  ? VAL_FLOAT
                         : kind == SLOT_BOOL   ? VAL_BOOL
                         : kind == SLOT_OPAQUE ? VAL_NULL
                         : kind == SLOT_NULL   ? VAL_NULL
                                               : VAL_OBJ;
            unsigned at = (unsigned)offsetof(JitDeoptRecord, locals) +
                          i * (unsigned)sizeof(Value);

            /* Nothing to read: the payload is a defined zero either way. */
            if (tag == VAL_NULL) {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }

            /* Must describe every local correctly whichever of the three homes it came from -- the one place a
             * register-plan mistake becomes a wrong answer, not a crash. A dynamic slot is the only one whose tag isn't a compile-time fact and the only one the plan never gives a register, so it's the only case that copies a tag through. */
            if (e.spilled && e.slotXReg[slot] == 0 && e.slotFpReg[slot] == 0 &&
                localTagInFrame(&e, slot)) {
                emit(&e, jaiA64LdrW(JIT_SCRATCH_C, 31, localFrameOff(&e, slot)));
                emit(&e, jaiA64StrW(JIT_SCRATCH_C, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot) + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, at + 8));
                continue;
            }

            unsigned pr;
            if (!e.spilled) {
                pr = JIT_FIRST_SAVED + i;
            } else if (e.slotXReg[slot] != 0) {
                pr = e.slotXReg[slot];
            } else if (e.slotFpReg[slot] != 0) {
                emit(&e, jaiA64FmovXD(JIT_SCRATCH_C, e.slotFpReg[slot]));
                pr = JIT_SCRATCH_C;
            } else {
                emit(&e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                                    localFrameOff(&e, slot) + 8));
                pr = JIT_SCRATCH_C;
            }
            if (kind == SLOT_MAYBE_INST) {
                /* Not JIT_SCRATCH_C when C is holding the payload, which it is
                 * whenever the slot came from the frame or an FP home. */
                unsigned spare = pr == JIT_SCRATCH_C ? JIT_SCRATCH_D
                                                     : JIT_SCRATCH_C;
                emitTagFor(&e, kind, pr, JIT_SCRATCH_B, spare);
            } else {
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            }
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            emit(&e, jaiA64StrX(pr, JIT_SCRATCH_A, at + 8));
        }
        /* Written unconditionally, including the zero: gDeopt is one global and
         * a mask left over from another function's stub would silently drop a
         * local this one does describe. */
        emitConst64(&e, JIT_SCRATCH_B, (int64_t)skipLocals);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, skipLocals)));

        unsigned valueSeen = 0;
        for (unsigned i = 0; i < e.deopt[k].depth; i++) {
            SlotKind kind = e.deopt[k].kinds[i];
            unsigned at = (unsigned)offsetof(JitDeoptRecord, stack) +
                          i * (unsigned)sizeof(Value);
            if (kind == SLOT_CLASS || kind == SLOT_SELF ||
                kind == SLOT_FUNC || kind == SLOT_NATIVE) {
                /* Neither holds a register; both are compile-time constants. */
                uintptr_t p = kind != SLOT_SELF
                                  ? (uintptr_t)e.deopt[k].classes[i]
                                  : (uintptr_t)closure;
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emitConst64(&e, JIT_SCRATCH_B, (int64_t)p);
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }
            bool fromDesc = e.deopt[k].lastFromDesc &&
                            i + 1 == e.deopt[k].depth;
            if (fromDesc) {
                /* The result of a call that has already happened: it is in the
                 * descriptor, not in a register, and its tag is whatever the
                 * callee actually returned. */
                unsigned rat = e.descOffset +
                               (unsigned)offsetof(JitCallDesc, result);
                emit(&e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, 31, rat + 8));
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            unsigned reg0 = valueBankReg(&e, valueSeen);
            /* The stub is reached by a branch from the guard, so the FP bank
             * still holds whatever it held there. Moving it here rather than
             * before the guard is what keeps the hot path free of it. */
            if (e.deopt[k].fpLive & (1u << valueSeen)) {
                emit(&e, jaiA64FmovXD(reg0, fpRegAt(&e, valueSeen)));
            }
            if (kind == SLOT_MAYBE_INST) {
                emitTagFor(&e, kind, reg0, JIT_SCRATCH_B, JIT_SCRATCH_C);
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64StrX(reg0, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            /* SLOT_NULL is a void call's result entry: register payload zero, tag NOT VAL_OBJ. Reaching this
             * chain through the default arm once wrote a null pointer out tagged as an object. */
            unsigned tag = kind == SLOT_INT   ? VAL_INT
                         : kind == SLOT_FLOAT ? VAL_FLOAT
                         : kind == SLOT_BOOL  ? VAL_BOOL
                         : kind == SLOT_NULL  ? VAL_NULL
                                              : VAL_OBJ;
            unsigned reg = valueBankReg(&e, valueSeen);
            valueSeen++;
            emit(&e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            emit(&e, jaiA64StrX(reg, JIT_SCRATCH_A, at + 8));
        }

        emitConst64(&e, JIT_SCRATCH_B, (int64_t)e.deopt[k].ip);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, ip)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.base, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, base)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.locals, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nlocals)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.deopt[k].depth, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nstack)));

        emit(&e, jaiA64MovzX(0, 0, 0));
        emitEpilogue(&e, 4);
    }

    /* One throwing stub per operator, out of line: the hot path keeps the same
     * single not-taken b.vs it always had. */
    for (unsigned i = 0; i < 3; i++) {
        if (!e.overflowUsed[i]) { e.overflowStub[i] = -1; continue; }
        e.overflowStub[i] = (int)e.count;
        emit(&e, jaiA64MovzX(0, i, 0));
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jitThrowOverflow);
        emit(&e, jaiA64Blr(JIT_SCRATCH_A));
        emit(&e, jaiA64MovzX(0, 0, 0));
        emitEpilogue(&e, 2);
    }

    /* Literal pool, 8-byte aligned so the 64-bit loads are aligned. */
    if ((e.count & 1u) != 0) emit(&e, jaiA64Nop());
    e.limitLiteral = (int)e.count;
    uintptr_t limit = stackLimit();
    if (limit == 0) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: no stack bound available\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }
    emit(&e, (uint32_t)(uint64_t)limit);
    emit(&e, (uint32_t)((uint64_t)limit >> 32));

    if (e.failed) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>",
                    e.whyNot ? e.whyNot : "the emitter ran out of room");
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    e.code[guardLoad] = jaiA64LdrLit(JIT_SCRATCH_A, e.limitLiteral - guardLoad);
    e.code[guardBranch] =
        jaiA64BCond(JAI_A64_LO, e.bailBlock - (int)guardBranch);

    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) {
            target = e.bailBlock;
        } else if (f->targetOffset == FIXUP_THREW) {
            target = e.exceptionExit;
        } else if (f->targetOffset <= FIXUP_EXIT &&
                   f->targetOffset > FIXUP_EXIT - JIT_MAX_EXIT) {
            target = e.exitStub[FIXUP_EXIT - f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_SELFSLOW &&
                   f->targetOffset > FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW) {
            target = e.selfSlow[FIXUP_SELFSLOW - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a self-call block was "
                                    "never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_DEOPT &&
                   f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT) {
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_GROW &&
                   f->targetOffset > FIXUP_GROW - JIT_MAX_GROW) {
            target = e.grow[FIXUP_GROW - f->targetOffset].stub;
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a list-growth block was "
                                    "never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset <= FIXUP_OVF &&
                   f->targetOffset >= FIXUP_OVF - 2u) {
            target = e.overflowStub[FIXUP_OVF - f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a stub it branches to "
                                    "was never emitted\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        } else if (f->targetOffset == FIXUP_ENTRY) {
            target = 0;
        } else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: a branch target past the end of the chunk\n",
                            fn->name ? fn->name->chars : "<anon>");
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
            target = map[f->targetOffset];
            if (target < 0) {
                if (getenv("JAI_JIT_WHY")) {
                    if (e.unarmedOp != 0) {
                        fprintf(stderr, "[jit] %s stopped: %s at %u ended the "
                                "walk, so the branch to offset %u (%s) has "
                                "nowhere to land\n",
                                fn->name ? fn->name->chars : "<anon>",
                                jaiOpName((OpCode)e.unarmedOp), e.unarmedAt,
                                f->targetOffset,
                                f->targetOffset < (uint32_t)fn->chunk.count
                                    ? jaiOpName((OpCode)fn->chunk.code[f->targetOffset])
                                    : "past the end");
                    } else {
                        fprintf(stderr, "[jit] %s stopped: a branch to offset "
                                "%u, which this walk never emitted (%s)\n",
                                fn->name ? fn->name->chars : "<anon>",
                                f->targetOffset,
                                f->targetOffset < (uint32_t)fn->chunk.count
                                    ? jaiOpName((OpCode)fn->chunk.code[f->targetOffset])
                                    : "past the end");
                    }
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
            /* Registers are assigned from the operand-stack depth at each point, so a join reached at two
             * different depths would read a value out of a register holding something else. The linear bytecode walk can't see that, so it's checked here and declined rather than mis-compiled. */
            if (f->depth >= 0 && depths[f->targetOffset] != f->depth) {
                if (getenv("JAI_JIT_WHY")) {
                    fprintf(stderr, "[jit] %s stopped: offset %u is reached "
                                    "with two different operand stacks\n",
                            fn->name ? fn->name->chars : "<anon>",
                            f->targetOffset);
                }
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) {
            e.code[f->instIndex] = jaiA64Bl(rel);
        } else if (f->conditional) {
            e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        } else {
            e.code[f->instIndex] = jaiA64B(rel);
        }
    }
    /* JAI_JIT_DUMP=<function> writes that function's words to jit_<function>.bin and prints the
     * bytecode-offset-to-instruction map, so the code can be read back with `llvm-mc --disassemble --triple=aarch64` on the file's bytes. Reading the code is how the register plan gets checked at all -- three of this tier's bugs were found no other way. */
    {
        const char *dump = getenv("JAI_JIT_DUMP");
        if (dump != NULL && fn->name != NULL &&
            strcmp(dump, fn->name->chars) == 0) {
            char path[256];
            snprintf(path, sizeof path, "jit_%s.bin", fn->name->chars);
            FILE *fp = fopen(path, "wb");
            if (fp != NULL) {
                fwrite(e.code, sizeof e.code[0], e.count, fp);
                fclose(fp);
                fprintf(stderr, "[jit] %u words to %s\n", e.count, path);
                for (unsigned i = 0; i <= (unsigned)fn->chunk.count; i++) {
                    if (map[i] >= 0) {
                        fprintf(stderr, "[jit] bc %u is inst %d\n", i, map[i]);
                    }
                }
            }
        }
    }
    jitFree(map, depths, chunkDepth, fn->chunk.count + 1);

    /* A `bl` at instruction i must reach instruction 0 of this function, so
     * the recursive-call fixups above are relative to the function's own
     * start, which is where the arena is about to place it. */
    if (!jaiCodeArenaUnseal(arena)) return false;
    /* The entry is 32-aligned, which is two things at once.
     *
     * The literal pool's alignment is what it looks, as the 8-align this
     * replaced already gave. And every body now starts at a fixed offset
     * modulo the fetch block, so a size change ANYWHERE upstream stops
     * relabelling where every later body lands. That relabelling was the
     * suite's largest source of false A/B results: sweeping 0-7 padding nops
     * moved bitops +/-13% and loop_sum +/-7% with byte-identical loop bodies,
     * and matrix_mul +/-18.9% and list_ops +/-15.4% were both observed between
     * binaries with identical dynamic instruction counts. The cost is at most
     * 28 wasted bytes per compiled body in a 1 MB arena. */
    uint8_t *entry = arenaEmit(arena, e.code, e.count);
    if (entry == NULL) {
        /* The one silent decline a body could reach AFTER `[jit] considering`:
         * everything upstream names its reason, so a full arena looked exactly
         * like a walk that stopped for no reason at all. It is also the only
         * decline that says nothing about the body -- a one-instruction
         * `fn init(self) {}` reaches it the same as a thousand-instruction
         * loop, and which one you get depends only on how much code the
         * process has already compiled.
         *
         * Worth naming twice over, because the decline is not the end of it:
         * jaiJitEnter falls through to compileReturnNull and compileAccessor,
         * which write into a DIFFERENT arena and so still succeed. A full
         * arena therefore does not stop compiling, it silently changes which
         * tier compiles -- and that is what exposed compileReturnNull
         * returning null from an initializer (see jit.c). */
        e.whyNot = "the code arena is full";
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: %s\n",
                    fn->name ? fn->name->chars : "<anon>", e.whyNot);
        }
        return false;
    }

    if (e.whyNot != NULL && getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] %s stopped: %s\n",
                fn->name ? fn->name->chars : "<anon>", e.whyNot);
    }
    if (e.assumedIntReturn && e.returnKind != SLOT_INT) {
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] %s stopped: a self-call had to guess the "
                            "return kind and guessed wrong\n",
                    fn->name ? fn->name->chars : "<anon>");
        }
        /* A self-call before the first return had to guess, and guessed
         * wrong. */
        return false;
    }
    for (unsigned i = 0; i < argCount; i++) {
        if (e.usesUpvalues && i == closureArg) {
            fn->jitParamKind[i]  = (uint8_t)SLOT_CLOSURE;
            fn->jitParamShape[i] = 0;
            continue;
        }
        fn->jitParamKind[i]  = (uint8_t)e.localKind[i + e.base];
        fn->jitParamShape[i] = e.localShape[i + e.base];
    }
    fn->jitReturnKind = (uint8_t)e.returnKind;
    fn->jitReturnKnown = e.sawReturn;
    fn->jitReturnShape = e.returnShape;
    if (e.returnKind == SLOT_INST && e.returnShape != 0) {
        ObjClass *rc = NULL;
        for (unsigned i = 0; i < e.base + e.locals; i++) {
            if (e.localClass[i] != NULL &&
                e.localClass[i]->shapeId == e.returnShape) {
                rc = e.localClass[i];
                break;
            }
        }
        jaiClassRememberShape(rc);
    }
    fn->jitArgBase    = (uint8_t)e.base;
    fn->jitArgCount   = (uint8_t)argCount;
    fn->jitFuncNoWrite = !e.wroteHeap;

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr,
                "[jit] compiled %s  arity=%u locals=%u insts=%u saved=%u "
                "spill=%d xloc=%u fploc=%u fix=%u deopt=%u maxval=%u base=%u\n",
                fn->name ? fn->name->chars : "<anon>", e.arity, e.locals,
                e.count, e.savedCount, (int)e.spilled, e.xLocals, e.fpLocals,
                e.fixupCount, e.deoptCount, body.maxValue, e.base);
        /* A partial compile used to be entirely silent: emitUnarmedDeopt
         * interprets from an unsupported opcode ONWARD rather than declining
         * the body, so "compiled" was printed for a body of which two
         * instructions were compiled and forty were not. That is how OP_NEG
         * stayed hidden long enough to cost 9x, and it was only ever visible
         * when a forward branch happened to dangle past the stopping point.
         *
         * Naming it here turns "which opcode should I arm next" from a manual
         * audit into a grep. */
        if (e.unarmedOp != 0) {
            fprintf(stderr,
                    "[jit] %s walked only to %s%s at %u -- the rest of the body "
                    "is interpreted\n",
                    fn->name ? fn->name->chars : "<anon>",
                    jaiOpName((OpCode)e.unarmedOp),
                    unarmedDetail(fn, e.unarmedOp, e.unarmedAt), e.unarmedAt);
        }
    }
    fn->jitFunc = entry;
    return true;
}

#else

bool jaiJitCompileFunc(ObjClosure *closure, Value *slotBase) {
    (void)closure; (void)slotBase; return false;
}

#endif
