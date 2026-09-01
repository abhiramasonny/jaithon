/* jit_func.c -- whole-function JIT tier: compiles self-recursive, integer-only bodies to native arm64, bailing to the interpreter on overflow, deep recursion, or an unsupported shape. */
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

/* Off puts the refusal back, so the declared-element route can be measured
 * against the sample-only one in the same binary. */
bool elemDeclOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("JAITHON_JIT_ELEM_DECL");
        on = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* A SlotKind's name, for JAI_JIT_WHY. Several refusals used to print the raw
 * enumerator ("the container is kind 13, not a list"), which names the one
 * thing a reader of the message cannot look up. */
const char *slotKindName(SlotKind k) {
    switch (k) {
    case SLOT_INT:        return "int";
    case SLOT_FLOAT:      return "float";
    case SLOT_INST:       return "instance";
    case SLOT_MAYBE_INST: return "instance-or-null";
    case SLOT_SELF:       return "self";
    case SLOT_OPAQUE:     return "opaque";
    case SLOT_CLOSURE:    return "closure";
    case SLOT_CLASS:      return "class";
    case SLOT_FUNC:       return "function";
    case SLOT_NATIVE:     return "builtin";
    case SLOT_ITER:       return "iterator";
    case SLOT_BOOL:       return "bool";
    case SLOT_NULL:       return "null";
    case SLOT_OBJ:        return "object";
    case SLOT_LIST:       return "list";
    }
    return "an unnamed kind";
}

bool holdsRegister(SlotKind k) {
    return k != SLOT_SELF && k != SLOT_CLASS && k != SLOT_FUNC &&
           k != SLOT_NATIVE;
}

void emit(Emit *e, uint32_t word) {
    if (e->count >= JIT_MAX_INSTS) {
        /* Named, because it said nothing at all: with no reason set,
         * declineReason falls back to the bare name of the last opcode, so a
         * body that outgrew the buffer was reported as refusing at whatever
         * instruction it happened to be on. */
        e->whyNot = "the body needs more instructions than the buffer holds";
        e->failed = true;
        return;
    }
    e->code[e->count++] = word;
}

/* OSR reserves only the slots pointer plus (for a range loop) the iterator's index and limit --
 * not the ObjIter or the start, which are folded in via a prologue bias. Bias is sound only while start+limit fits int64; jaiJitEnterOsr refuses entry otherwise since that's a property of the iterator, not the code. */
unsigned osrReserved(const Emit *e) {
    if (!e->hasIter) return 1u;
    if (e->iterKind == 1) return 3u;
    if (e->iterKind == 3) return 2u;   /* dict items: the ObjIter, nothing else */
    return 5u;
}

static unsigned regBase(const Emit *e) {
    if (e->osr) return osrReserved(e) + e->xLocals;
    return e->spilled ? e->xLocals : e->locals;
}

static unsigned localReg(const Emit *e, unsigned slot) {
    if (e->osr || e->spilled) return e->slotXReg[slot];
    return JIT_FIRST_SAVED + (slot - e->base);
}

/* Sixteen bytes per slot, not eight: the tag travels with the value so a local whose kind varies can
 * be read behind a guard. Every slot keeps a frame home even with a register too -- dense layout costs one multiply instead of a table, and the wasted words are nothing against the 4095-byte frame limit. */
unsigned localFrameOff(const Emit *e, unsigned slot) {
    return e->localsFrameOffset + (slot - e->base) * 16u;
}

/* A frame home only needs a stored tag when something will READ it: a fixed-kind slot's tag is
 * rebuildable by the deopt stub from the kind, cutting a spilled write from three instructions to one. Only a dynamic slot (kind disagrees across paths) has a runtime tag, and only its reads check it. */
bool localTagInFrame(const Emit *e, unsigned slot) {
    return e->dynamicLocal[slot];
}

/* Per-access instruction savings from giving `slot` a register instead of a frame home, accumulated
 * by the accessors during the measuring pass and weighted by loop nesting. Zero-saving slots are excluded outright, not just ranked last -- otherwise a tie-break could hand a register to a slot the loop never touches, paying a prologue write for nothing.
 *
 * Both tiers feed this, but with their OWN instruction counts, which is why the
 * numbers at the call sites differ rather than the mechanism. A function-tier
 * frame home is a bare payload word for every slot the plan will consider (only
 * a dynamic slot stores a tag, and a dynamic slot never gets a register), so a
 * write costs one `str` there and makes a weak case for a register -- weak
 * enough that a float write makes none at all, see localOutFp. An OSR home is
 * the INTERPRETER's own Value slot, so the same write stores tag and payload
 * both and costs three; in OSR a write is worth as much as two reads. Ranking
 * OSR on the function tier's numbers would therefore under-rank exactly the
 * write-heavy loop variables OSR exists to speed up. */
static void noteSlotCost(Emit *e, unsigned slot, unsigned saveX,
                         unsigned saveFp) {
    if (!e->measuring || e->inlining) return;
    if (slot > JIT_MAX_SLOTS) return;
    unsigned w = 1u;
    if (e->loopDepth != NULL && e->curOffset < e->loopDepthCount) {
        unsigned d = e->loopDepth[e->curOffset];
        if (d > 6u) d = 6u;
        w = 1u << (2u * d);
    }
    e->slotSaveX[slot]  += w * saveX;
    e->slotSaveFp[slot] += w * saveFp;
}


/* Every kind but SLOT_MAYBE_INST has a tag fixed at compile time; that one reads it off the payload (null vs non-null). */
void emitTagFor(Emit *e, SlotKind kind, unsigned payloadReg,
                       unsigned tagReg, unsigned spare) {
    if (kind != SLOT_MAYBE_INST) {
        unsigned tag = kind == SLOT_INT    ? VAL_INT
                     : kind == SLOT_FLOAT  ? VAL_FLOAT
                     : kind == SLOT_BOOL   ? VAL_BOOL
                     : kind == SLOT_OPAQUE ? VAL_NULL
                     : kind == SLOT_NULL   ? VAL_NULL
                                           : VAL_OBJ;
        emit(e, jaiA64MovzX(tagReg, tag, 0));
        return;
    }
    emit(e, jaiA64SubsXImm(31, payloadReg, 0));
    emit(e, jaiA64MovzX(tagReg, VAL_OBJ, 0));
    emit(e, jaiA64MovzX(spare, VAL_NULL, 0));
    emit(e, jaiA64CselX(tagReg, spare, tagReg, JAI_A64_EQ));
}

unsigned localTagFor(const Emit *e, unsigned slot) {
    SlotKind k = e->localKind[slot];
    return k == SLOT_INT    ? VAL_INT
         : k == SLOT_FLOAT  ? VAL_FLOAT
         : k == SLOT_BOOL   ? VAL_BOOL
         : k == SLOT_OPAQUE ? VAL_NULL
         : k == SLOT_NULL   ? VAL_NULL
                            : VAL_OBJ;
}

/* Register mode: the local's own register, `scratch` unused. Memory mode: loaded into `scratch`.
 * Only the payload moves -- a local's kind is fixed for the whole function, so the tag is never restored. */
unsigned localIn(Emit *e, unsigned slot, unsigned scratch) {
    if (e->osr) {
        /* An X home answers with no instruction at all, an FP home costs the
         * `fmov` below and memory costs the `ldr` -- so an X register saves
         * one here and an FP register saves nothing, exactly as in the
         * function tier's arm. */
        noteSlotCost(e, slot, 1u, 0u);
        if (e->slotXReg[slot] != 0) return e->slotXReg[slot];
        if (e->slotFpReg[slot] != 0) {
            emit(e, jaiA64FmovXD(scratch, e->slotFpReg[slot]));
            return scratch;
        }
        /* One byte for a bool. The slot is the interpreter's, and BOOL_VAL is a
         * `strb`, so the seven bytes above it are whatever the slot held
         * before. See the OSR prologue for the same load and what it cost. */
        if (e->localKind[slot] == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        } else {
            emit(e, jaiA64LdrX(scratch, JIT_SLOTS_REG, slot * 16u + 8u));
        }
        return scratch;
    }
    noteSlotCost(e, slot, 1u, 0u);
    if (!e->spilled) return localReg(e, slot);
    if (e->slotXReg[slot] != 0) return e->slotXReg[slot];
    if (e->slotFpReg[slot] != 0) {
        emit(e, jaiA64FmovXD(scratch, e->slotFpReg[slot]));
        return scratch;
    }
    if (e->dynamicLocal[slot]) {
        /* Two paths reached here disagreeing about this slot, so what it holds
         * is a runtime fact: check it against what this read was compiled for
         * and hand the instruction back otherwise. */
        /* Into `scratch`, not JIT_SCRATCH_A, since the caller may already be holding an operand there: this
         * exact mistake once loaded the tag over the constant on a dynamic slot, so `for j in i + 1..n` silently ran from i+2 and every nested loop was one iteration short. */
        emit(e, jaiA64LdrW(scratch, 31, localFrameOff(e, slot)));
        emit(e, jaiA64SubsXImm(31, scratch, localTagFor(e, slot)));
        branchOnDeopt(e, JAI_A64_NE);

        /* VAL_OBJ is shared by SLOT_LIST, SLOT_OBJ and SLOT_INST alike (see
         * localTagFor), so the tag check above cannot tell a list from a dict
         * a sibling write left in this slot -- confirmed the same way
         * OP_GET_INDEX's own SLOT_LIST arm does, once, before a consumer
         * trusts it with no check of its own. Chained through `scratch`
         * alone (no second register): the payload is reloaded fresh into it,
         * then `Obj.type` is loaded from that address back into the same
         * register -- valid on this encoder elsewhere (e.g. the
         * OP_GET_INDEX/OP_SET_INDEX list arms chain JIT_SCRATCH_C the same
         * way), and it never needs the pointer again afterward, since the
         * unconditional reload below re-reads it from the frame regardless. */
        if (e->localKind[slot] == SLOT_LIST) {
            emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
            emit(e, jaiA64LdrW(scratch, scratch, (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, scratch, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (e->localKind[slot] == SLOT_INST ||
                   e->localKind[slot] == SLOT_MAYBE_INST) {
            /* Same hazard, for a slot that took two different classes: a tag
             * of VAL_OBJ says "an instance", not "an instance of THIS class",
             * so the object type and its class shape are both confirmed here
             * before a consumer reads a field at an offset only this class
             * has. JIT_SCRATCH_D is free at every call site that can carry an
             * instance-kinded dynamic local through this helper (audited:
             * OP_GET_LOCAL, OP_GET_LOCAL2, OP_GET_FIELD_LOCAL's two reads of
             * its receiver, the init-returns-self arms of
             * OP_RETURN_NULL/OP_POP_RETURN_NULL, emitRootFill's root loop --
             * every other site names a scalar kind and cannot reach here).
             * `scratch` keeps the instance pointer as the base throughout --
             * loading FROM it doesn't clobber it -- until it is chained into
             * the class pointer and then the shapeId, since nothing after
             * this needs the original pointer back (the unconditional reload
             * below restores it for the return regardless).
             *
             * SLOT_MAYBE_INST shares the arm rather than going unchecked: it
             * is a kind a dynamic slot really does take, both from the
             * nullable-parameter seed and from any OP_BIND of a nullable
             * instance field, and its tag is the same VAL_OBJ, so without
             * this a `Bird` left in the slot by a sibling write was read at
             * `Dog`'s field offsets. What it does NOT share is the pointer
             * being known non-null. A null cannot be chased and cannot be
             * jumped over either -- every branch this file emits leaves the
             * block for a stub -- so it deopts, and the interpreter finishes
             * the instruction. That costs one deopt on a value the tier could
             * in principle have carried, which is the price of not letting a
             * guard load off address zero. (The prologue's own tag write is
             * payload-dependent for the same reason, so a null argument is
             * usually already stopped by the tag check above.) */
            emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
            if (e->localKind[slot] == SLOT_MAYBE_INST) {
                emit(e, jaiA64SubsXImm(31, scratch, 0));
                branchOnDeopt(e, JAI_A64_EQ);
            }
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, scratch, (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(scratch, scratch, (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(scratch, scratch, (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_D, (int64_t)e->localShape[slot]);
            emit(e, jaiA64SubsXReg(31, scratch, JIT_SCRATCH_D));
            branchOnDeopt(e, JAI_A64_NE);
        }
    }
    emit(e, jaiA64LdrX(scratch, 31, localFrameOff(e, slot) + 8));
    return scratch;
}

/* Only register-resident locals can be written in place. In OSR mode `localReg` would name a register
 * already holding something else (the loop counter) -- once let a float add land in x21, turning the induction variable into a bit pattern, so the loop finished early or never finished. */
unsigned localDest(const Emit *e, unsigned slot) {
    if (e->osr) {
        return e->slotXReg[slot] != 0 ? e->slotXReg[slot] : JIT_SCRATCH_C;
    }
    if (e->spilled) {
        return e->slotXReg[slot] != 0 ? e->slotXReg[slot] : JIT_SCRATCH_C;
    }
    return localReg(e, slot);
}

/* A local's X register is about to be written, so nothing may still be borrowing it. Can't just copy
 * the borrow out here (same reason as deoptRecordAt: the write can sit inside a span an earlier branch skips) -- declines instead. Can't fire from the whitelist as it stands: no whitelisted opcode writes a local. */
static void xHomeWritten(Emit *e, unsigned reg) {
    if (e->xBorrow == 0 || reg == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->xBorrow & (1u << i)) == 0) continue;
        if (e->xBorrowReg[i] != reg) continue;
        e->whyNot = "a local was written under a borrow of its register";
        e->failed = true;
        return;
    }
}

/* Retires every field-kind memo (Emit::known, written by recordFieldStore and
 * read by knownFieldKind). An entry says "a store THIS body emitted put kind K
 * in that field, and nothing since has put another kind there" -- which holds
 * only while this walk is the only thing that can have run, and only along the
 * fall-through edge that made it. Two things end that, and neither is a store
 * the walk can see:
 *
 *   * a call. The callee reaches the same object through the argument it was
 *     handed, through a global or upvalue, or as the receiver it was invoked
 *     on, and `o.v = "s"` in there leaves the memo claiming SLOT_INT -- so the
 *     read after the call skips its tag guard and hands a consumer an
 *     ObjString pointer as an integer, or an integer as a pointer to
 *     dereference. Hooked in noteScratchClobber, which every call out passes
 *     through.
 *
 *   * an offset something other than fall-through can reach. The store may sit
 *     on the arm a branch skipped (`if c { o.v = 7 }` then a read), and a back
 *     edge re-enters above stores further down the body (a read at a loop top
 *     whose second iteration meets the kind the bottom of the body stored).
 *     Hooked in the walk, on the same offsetIsBranchTarget test the ASCII
 *     proofs above it use, and for the same reason.
 *
 * Clearing all of them rather than one is deliberate: a call can write any
 * field of any object it can reach, so there is nothing narrower to say. */
void forgetFieldKinds(Emit *e) { e->knownCount = 0; }

/* The narrower one: an entry names a LOCAL, so writing that local retires it.
 * The slot may now hold a different object entirely -- `b.v = 7; b = c; b.v`
 * read c's field at b's recorded kind before this existed. */
static void forgetFieldKindsOfLocal(Emit *e, unsigned slot) {
    unsigned out = 0;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].local == (int)slot) continue;
        e->known[out++] = e->known[i];
    }
    e->knownCount = out;
}

/* Every write to a local goes through localOut or localOutFp, so recording it
 * in those two places is what makes "this slot does not change inside that
 * loop" a fact about the emitter rather than a re-reading of the bytecode. */
static void noteSlotWrite(Emit *e, unsigned slot) {
    /* Above the range check on purpose: the memo is keyed on the same slot
     * numbers, so a slot too high to record is still one to retire. */
    if (e->knownCount != 0) forgetFieldKindsOfLocal(e, slot);
    if (slot > JIT_MAX_SLOTS) return;
    if (!e->measuring) {
        /* The real pass writing a slot the measuring pass said this loop never
         * touches means the two walks disagreed, and a hoisted header would
         * then be stale. It cannot happen -- both walk the same bytecode with
         * the same inlining -- so this is a ratchet, not a path: it costs a
         * compile, never a wrong answer. */
        for (unsigned i = 0; i < e->hoistCount; i++) {
            if (e->hoist[i].slot != (uint8_t)slot) continue;
            if (e->curOffset < e->hoist[i].top) continue;
            if (e->curOffset >= e->hoist[i].end) continue;
            e->whyNot = "a hoisted list header's local was written after all";
            e->failed = true;
        }
        return;
    }
    if (e->inlining) return;
    if (e->curOffset < e->slotWriteLo[slot]) e->slotWriteLo[slot] = e->curOffset;
    if (e->curOffset > e->slotWriteHi[slot]) e->slotWriteHi[slot] = e->curOffset;
}

/* Twin of noteSlotWrite for the other half of the question: where a local is
 * used as the base of a subscript, and how hot those sites are. Weighted by
 * loop nesting for the same reason slotUse is -- a header read once per row
 * must not outrank one read every iteration. */
/* Records that `slot` was subscripted at the loop counter plus `off`, or --
 * when `shaped` is false -- that one of its subscripts is not a shape the loop
 * head can guard. One unshaped site does not spoil the others: they still skip
 * their checks and it still emits its own. What it DOES spoil is nothing, so
 * `spanOk` exists only to record that a slot was seen at all. */
void noteIndexSpan(Emit *e, int slot, bool shaped, int32_t off,
                          uint8_t base) {
    if (!e->measuring || e->inlining) return;
    if (slot < 0 || slot > (int)JIT_MAX_SLOTS) return;
    if (!shaped) return;
    if (!e->spanSeen[slot]) {
        e->spanSeen[slot] = true;
        e->spanBase[slot] = base;
    } else if (e->spanBase[slot] != base) {
        e->spanOk[slot] = false;
        return;
    }
    if (off < e->spanLo[slot]) e->spanLo[slot] = off;
    if (off > e->spanHi[slot]) e->spanHi[slot] = off;
}

void noteSlotIndexed(Emit *e, int slot) {
    if (!e->measuring || e->inlining || slot < 0 || slot > (int)JIT_MAX_SLOTS) {
        return;
    }
    unsigned w = 1u;
    if (e->loopDepth != NULL && e->curOffset < e->loopDepthCount) {
        unsigned d = e->loopDepth[e->curOffset];
        if (d > 6u) d = 6u;
        w = 1u << (2u * d);
    }
    e->slotIndexUse[slot] += w;
    if (e->curOffset < e->slotIndexLo[slot]) e->slotIndexLo[slot] = e->curOffset;
    if (e->curOffset > e->slotIndexHi[slot]) e->slotIndexHi[slot] = e->curOffset;
}

void localOut(Emit *e, unsigned slot, unsigned src) {
    noteSlotWrite(e, slot);
    xHomeWritten(e, e->osr ? e->slotXReg[slot]
                           : (e->spilled ? 0u : localReg(e, slot)));
    if (e->osr) {
        /* The memory arm below is three instructions -- the tag built, the tag
         * stored, the payload stored -- against one `mov` into an X home or one
         * `fmov` into an FP home. Two saved either way, so a write votes for a
         * register without voting for a bank; the reads pick the bank. */
        noteSlotCost(e, slot, 2u, 2u);
        if (e->slotFpReg[slot] != 0) fpReleaseHome(e, e->slotFpReg[slot]);
        if (e->slotXReg[slot] != 0) {
            if (src != e->slotXReg[slot]) {
                emit(e, jaiA64MovX(e->slotXReg[slot], src));
            }
            return;
        }
        if (e->slotFpReg[slot] != 0) {
            emit(e, jaiA64FmovDX(e->slotFpReg[slot], src));
            return;
        }
        /* Tag as well as payload, written straight through -- what makes a deopt here free; the kind is fixed for the whole compile. */
        SlotKind k = e->localKind[slot];
        emitTagFor(e, k, src, JIT_SCRATCH_D, JIT_SCRATCH_C);
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrX(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    noteSlotCost(e, slot, 1u, 0u);
    if (!e->spilled) {
        if (src != localReg(e, slot)) emit(e, jaiA64MovX(localReg(e, slot), src));
        return;
    }
    if (e->slotFpReg[slot] != 0) fpReleaseHome(e, e->slotFpReg[slot]);
    if (e->slotXReg[slot] != 0) {
        if (src != e->slotXReg[slot]) emit(e, jaiA64MovX(e->slotXReg[slot], src));
        return;
    }
    if (e->slotFpReg[slot] != 0) {
        emit(e, jaiA64FmovDX(e->slotFpReg[slot], src));
        return;
    }
    if (localTagInFrame(e, slot)) {
        emitTagFor(e, e->localKind[slot], src, JIT_SCRATCH_D, JIT_SCRATCH_C);
        emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    }
    emit(e, jaiA64StrX(src, 31, localFrameOff(e, slot) + 8));
}

/* Float half of localIn/localOut: an FP-bank local never visits an X register at all (memory mode
 * collapses `ldr x; fmov d,x` to one `ldr d`, and the store side likewise) -- only for a fixed-kind local; a dynamic one has a tag to check and goes the ordinary (X) way. */
void localInFp(Emit *e, unsigned slot, unsigned dst) {
    if (e->osr) {
        /* Mirrors the function tier's charge for the same read: the FP home is
         * the one that pays, and an X home is worth no more than the load. */
        noteSlotCost(e, slot, 0u, 1u);
        if (e->slotFpReg[slot] != 0) {
            /* Both banks, so the numbers never collide: a local's home is
             * v8..v15 and the operand stack's is v16 up. */
            if (dst != e->slotFpReg[slot]) {
                emit(e, jaiA64FmovDD(dst, e->slotFpReg[slot]));
            }
            return;
        }
        if (e->slotXReg[slot] != 0) {
            emit(e, jaiA64FmovDX(dst, e->slotXReg[slot]));
            return;
        }
        emit(e, jaiA64LdrD(dst, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    noteSlotCost(e, slot, 0u, 1u);
    if (!e->spilled) {
        emit(e, jaiA64FmovDX(dst, localReg(e, slot)));
        return;
    }
    if (e->slotFpReg[slot] != 0) {
        if (dst != e->slotFpReg[slot]) {
            emit(e, jaiA64FmovDD(dst, e->slotFpReg[slot]));
        }
        return;
    }
    if (e->slotXReg[slot] != 0) {
        emit(e, jaiA64FmovDX(dst, e->slotXReg[slot]));
        return;
    }
    emit(e, jaiA64LdrD(dst, 31, localFrameOff(e, slot) + 8));
}

void localOutFp(Emit *e, unsigned slot, unsigned src) {
    noteSlotWrite(e, slot);
    if (e->osr) {
        /* Unlike the function tier's float write just below -- which is one
         * instruction into any of the three homes and so charges nothing --
         * the OSR memory arm here writes the tag as well, so this write really
         * does buy two, the same as localOut's. */
        noteSlotCost(e, slot, 2u, 2u);
        if (e->slotFpReg[slot] != 0) {
            fpReleaseHome(e, e->slotFpReg[slot]);
            if (src != e->slotFpReg[slot]) {
                emit(e, jaiA64FmovDD(e->slotFpReg[slot], src));
            }
            return;
        }
        if (e->slotXReg[slot] != 0) {
            emit(e, jaiA64FmovXD(e->slotXReg[slot], src));
            return;
        }
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, VAL_FLOAT, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, slot * 16u));
        emit(e, jaiA64StrD(src, JIT_SLOTS_REG, slot * 16u + 8u));
        return;
    }
    /* No noteSlotCost: a float write is one instruction whichever home it has
     * (`fmov d,d`, `fmov x,d` or `str d`), so it makes no case for a register.
     * The reads are what pay. */
    if (!e->spilled) {
        emit(e, jaiA64FmovXD(localReg(e, slot), src));
        return;
    }
    if (e->slotFpReg[slot] != 0) {
        fpReleaseHome(e, e->slotFpReg[slot]);
        if (src != e->slotFpReg[slot]) {
            emit(e, jaiA64FmovDD(e->slotFpReg[slot], src));
        }
        return;
    }
    if (e->slotXReg[slot] != 0) {
        emit(e, jaiA64FmovXD(e->slotXReg[slot], src));
        return;
    }
    if (localTagInFrame(e, slot)) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, VAL_FLOAT, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, 31, localFrameOff(e, slot)));
    }
    emit(e, jaiA64StrD(src, 31, localFrameOff(e, slot) + 8));
}

bool localInRange(Emit *e, unsigned slot) {
    if (e->osr) {
        if (slot >= e->locals) return false;
        if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
        /* Nested-loop sites are worth more: flat counting would give `width`/`height` (read once per row)
         * the same claim as the recurrence variable read every iteration. */
        if (e->measuring && !e->inlining && e->loopDepth != NULL &&
            e->curOffset < e->loopDepthCount && slot <= JIT_MAX_SLOTS) {
            unsigned d = e->loopDepth[e->curOffset];
            if (d > 6u) d = 6u;
            e->slotUse[slot] += 1u << (2u * d);
        }
        return true;
    }
    if (slot < e->base || slot >= e->base + e->locals) return false;
    if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
    return true;
}

/* The ListStore the frame's slot is backed by, or BOXED for anything else.
 *
 * Read one slot at a time, and only once the walk has decided the slot holds a
 * LIST. Sweeping them all up front is what the first version did, and it
 * segfaulted the COMPILER: an OSR frame's slots run to fn->maxSlots, and the
 * ones the program has not reached yet hold whatever the last frame at that
 * depth left behind -- including a VAL_OBJ tag over a null pointer. IS_LIST
 * dereferences without checking, so the sweep read `Obj::type` off address
 * zero. Once in eight runs of tests/bench/jaiframe/frameops, and never in
 * 3041 tests, because it needs a slot that is both stale and untouched. */
uint8_t localStgOf(const Emit *e, unsigned slot) {
    if (slot > JIT_MAX_SLOTS || e->observed == NULL) return LIST_STORE_BOXED;
    if (e->localKind[slot] != SLOT_LIST) return LIST_STORE_BOXED;
    Value v = e->observed[slot];
    if (!IS_LIST(v)) return LIST_STORE_BOXED;
    return AS_LIST(v)->stg;
}

/* Whether `e->observed[slot]` may be READ, which is not the same question as
 * whether the slot is in the observed window -- there may be no observed frame
 * at all. Eight call sites spell `localObserved(e, slot) ? e->observed[slot]
 * : e->localSeen[slot]`, and every one of them dereferenced a NULL
 * `observed` whenever the walk ran without a live frame.
 *
 * It crashed the COMPILER, not compiled code: EXC_BAD_ACCESS at address 0 in
 * compileBody, reading `e->observed[slot]` for the interned-string compare at
 * OP_JUMP_IF_CMP_LOCAL_K. Reproduced under JAITHON_JIT_DEOPT_STRESS with 32+
 * files in one run; the stress only makes it likely, it is not the cause.
 * `localStgOf` above has always guarded `e->observed == NULL` for the same
 * reason, with its own note about the segfault that taught it.
 *
 * The window test keeps its `maxSlotUsed` side effect: register planning reads
 * it, and a missing observed frame says nothing about which slots the body
 * touches. */
static bool localObserved(Emit *e, unsigned slot) {
    if (e->osr) return slot < e->locals && e->observed != NULL;
    if (slot < e->base || slot > e->arity) return false;
    if (slot > e->maxSlotUsed) e->maxSlotUsed = slot;
    return e->observed != NULL;
}


/* The live value for a slot, or NULL_VAL when there is none.
 *
 * NEVER returns a VAL_OBJ over a null pointer. An OSR frame's slots run to
 * fn->maxSlots and the ones the program has not reached yet hold whatever the
 * last frame at that depth left behind -- a stale tag with a dead pointer
 * among them -- and every IS_STRING / IS_LIST / IS_INSTANCE macro dereferences
 * `Obj::type` to answer. Reading one crashed the COMPILER: EXC_BAD_ACCESS at
 * address 0 inside compileBody, reproducible under JAITHON_JIT_DEOPT_STRESS.
 *
 * Eight sites spelled this ternary out by hand and none of them checked. They
 * all go through here now, so a ninth cannot be written without the check.
 * localStgOf has guarded the same hazard since a sweep read `Obj::type` off
 * address zero once in eight runs. */
Value seenLocal(Emit *e, unsigned slot) {
    Value v = localObserved(e, slot) ? e->observed[slot] : e->localSeen[slot];
    if (IS_OBJ(v) && AS_OBJ(v) == NULL) return NULL_VAL;
    return v;
}

unsigned closureReg(const Emit *e) {
    return JIT_FIRST_SAVED + regBase(e);
}

/* Where entry 0 of the ordinary operand stack sits. Two answers, one rule: it
 * is x0 when the body proved nothing can clobber a caller-saved register while
 * it runs (see Emit::scratchValues), and otherwise the register just past the
 * locals, where it has always been. Every site that used to spell this sum out
 * by hand goes through here, so the two banks cannot disagree. */
static unsigned valueBankBase(const Emit *e) {
    if (e->scratchValues) return JIT_INL_BANK;
    return JIT_FIRST_SAVED + regBase(e) + (e->usesUpvalues ? 1u : 0u);
}

/* How many entries the operand stack may hold. The scratch bank is a fixed
 * nine; the callee-saved one is whatever the locals and the reserved registers
 * left behind. */
static unsigned valueBankRoom(const Emit *e) {
    if (e->scratchValues) return e->scratchRoom;
    unsigned taken = regBase(e) + (e->usesUpvalues ? 1u : 0u);
    unsigned saved = taken < JIT_MAX_SAVED ? JIT_MAX_SAVED - taken : 0u;
    /* Split: `splitAt` entries in what is left of the callee-saved bank and
     * nine more above them. If the callee-saved half does not itself fit --
     * which the plan sized it not to, but `usesUpvalues` is decided later --
     * the answer is that smaller number, so the push declines rather than
     * running off the end of the bank. */
    if (e->splitAt != 0) {
        return saved < e->splitAt ? saved
                                  : e->splitAt + JIT_SCRATCH_BANK_COUNT;
    }
    return saved;
}

/* Something about to be emitted can destroy x0..x8 with the body still live:
 * a call out, or one of the two stubs that call and then branch back in. The
 * measuring pass records it so the real pass can choose a bank; the real pass
 * declines if it happens anyway, so a missed site costs a decline and never a
 * value read out of a register a helper overwrote. */
void noteScratchClobber(Emit *e) {
    e->clobbersScratch = true;
    /* The one place every call out passes through, which makes it the one
     * place a field-kind memo can be retired at all of them; see
     * forgetFieldKinds. The sites that reach here without running user code
     * (a list grow, an instance allocation) only over-retire, which costs the
     * tag guard the read would have emitted anyway. */
    forgetFieldKinds(e);
    if (e->measuring) {
        /* An inlined body's offsets are the CALLEE's, so they say nothing
         * about where in the caller this sits; `inlIp` is the caller's own
         * OP_CALL, which is the offset every range in this file is measured
         * in. An inlined body is not supposed to reach here at all (see
         * inlineGlobalCall) -- but "not supposed to" is not "cannot", and a
         * callee offset landing inside a caller loop by coincidence would
         * over-report, not under-report, which is the safe direction. */
        uint32_t at = e->inlining ? e->inlIp : e->curOffset;
        if (e->clobberCount < JIT_MAX_CLOBBER) {
            e->clobberOff[e->clobberCount++] = at;
        } else {
            e->clobberSpill = true;
        }
        if (e->valueDepth > e->clobberDepth) e->clobberDepth = e->valueDepth;
    }
    if (e->scratchValues) {
        e->whyNot = "a call reached a body whose values are in scratch";
        e->failed = true;
    }
    /* Same ratchet one granularity down. A hoisted header sits in a
     * caller-saved register for the length of a loop the measuring pass called
     * call-free; a call turning up inside that loop in the real pass means the
     * two walks disagreed, and the header would be read out of a register the
     * helper had overwritten. It cannot happen -- both walk the same bytecode
     * with the same inlining -- so this costs a compile, never an answer. */
    if (!e->measuring && e->hoistCount > 0) {
        uint32_t at = e->inlining ? e->inlIp : e->curOffset;
        for (unsigned i = 0; i < e->hoistCount; i++) {
            if (at < e->hoist[i].top || at >= e->hoist[i].end) continue;
            e->whyNot = "a call reached a loop a header was hoisted out of";
            e->failed = true;
            return;
        }
    }
    /* And the ratchet the split bank rests on. `splitAt` was chosen as the
     * deepest the measuring pass ever saw the stack at one of these, so a real
     * pass standing deeper means the two walks disagreed and an entry in
     * x0..x8 is about to be run over by the helper. Costs a compile. */
    if (!e->measuring && e->splitAt != 0 && e->valueDepth > e->splitAt) {
        e->whyNot = "a call stood deeper than the split bank allows";
        e->failed = true;
    }
}

/* An inlined body needs its own bank only when the caller's is somewhere else.
 * Once the caller's operand stack is already x0..x8 the two are the SAME bank,
 * and restarting at x0 would overwrite the entries the call site is standing
 * on -- so the inlined entries simply continue the caller's numbering, which
 * `scratchValues` has already proved fits (probe.maxValueAll <= the bank). */
static bool inlineOwnBank(const Emit *e) {
    return e->inlining && !e->scratchValues;
}

/* Where entry `idx` of the ordinary (non-inlined) operand stack lives. One run
 * of registers unless the bank is split, in which case the entries at and above
 * `splitAt` continue in x0.. instead. Every site that used to write
 * `valueBankBase(e) + idx` says this now, so the split cannot be half-applied:
 * with splitAt == 0 the two are the same expression. */
unsigned valueBankReg(const Emit *e, unsigned idx) {
    if (e->splitAt != 0 && idx >= e->splitAt) {
        return JIT_INL_BANK + (idx - e->splitAt);
    }
    return valueBankBase(e) + idx;
}

unsigned valueXReg(const Emit *e, unsigned idx) {
    if (inlineOwnBank(e) && idx >= e->inlValueBase) {
        return JIT_INL_BANK + (idx - e->inlValueBase);
    }
    return valueBankReg(e, idx);
}

/* One past the top entry's register. Expressing it this way (not from the bottom) is what keeps
 * `pushReg(e) - 1` == "the entry just pushed" true across both register banks.
 *
 * Subtracting TWO or more from it is not safe under a split bank -- the entry
 * below the top may be in the other half -- so those sites name the entry they
 * mean, as `valueXReg(e, e->valueDepth - n)`. */
unsigned pushReg(const Emit *e) {
    if (e->valueDepth == 0) return valueXReg(e, 0);
    return valueXReg(e, e->valueDepth - 1) + 1;
}

unsigned fpRegAt(const Emit *e, unsigned idx) {
    if (inlineOwnBank(e) && idx >= e->inlValueBase) {
        return JIT_INL_FP_BANK + (idx - e->inlValueBase);
    }
    return JIT_FP_BANK + idx;
}

/* The d register entry `idx` is actually in: its own, or a borrowed local's. Every READ of a live FP
 * entry goes through this; writes keep naming fpRegAt, which is what a borrow releases back into. */
unsigned fpHeldIn(const Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    return fpRegAt(e, idx);
}

void fpSyncOne(Emit *e, unsigned idx) {
    if (!(e->fpLive & (1u << idx))) return;
    unsigned d = fpHeldIn(e, idx);
    e->fpLive   &= ~(1u << idx);
    e->fpBorrow &= ~(1u << idx);
    emit(e, jaiA64FmovXD(valueXReg(e, idx), d));
}

/* A local's d register is about to be written, so every entry borrowing it
 * takes a copy of its own first. */
void fpReleaseHome(Emit *e, unsigned reg) {
    if (e->fpBorrow == 0 || reg == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->fpBorrow & (1u << i)) == 0) continue;
        if (e->fpBorrowReg[i] != reg) continue;
        e->fpBorrow &= ~(1u << i);
        emit(e, jaiA64FmovDD(fpRegAt(e, i), reg));
    }
}

/* ...and the same for all of them, before anything records where the
 * interpreter should resume: the stub writes entries out of fpRegAt. */
void fpReleaseAll(Emit *e) {
    if (e->fpBorrow == 0) return;
    for (unsigned i = 0; i < 32u; i++) {
        if ((e->fpBorrow & (1u << i)) == 0) continue;
        e->fpBorrow &= ~(1u << i);
        emit(e, jaiA64FmovDD(fpRegAt(e, i), e->fpBorrowReg[i]));
    }
}

void fpSyncAll(Emit *e) {
    if (e->fpLive == 0) return;
    for (unsigned i = 0; i < 32u; i++) fpSyncOne(e, i);
}


unsigned fpOperand(Emit *e, unsigned idx) {
    if (e->fpBorrow & (1u << idx)) return e->fpBorrowReg[idx];
    unsigned d = fpRegAt(e, idx);
    if (!(e->fpLive & (1u << idx))) {
        /* xHeldIn, not valueXReg: a float entry that's a plain read of an X-resident local lives in THAT
         * register; reading its own would read whatever the bank last held. */
        emit(e, jaiA64FmovDX(d, xHeldIn(e, idx)));
    }
    return d;
}

/* Entry `idx` has just been computed into v(16 + idx); its X register is now
 * stale until something asks for it. */
void fpClaim(Emit *e, unsigned idx) {
    e->fpLive |= 1u << idx;
    e->fpBorrow &= ~(1u << idx);
}

void fpBorrowLocal(Emit *e, unsigned idx, unsigned reg) {
    e->fpLive |= 1u << idx;
    e->fpBorrow |= 1u << idx;
    e->fpBorrowReg[idx] = (uint8_t)reg;
}

/* Float op writing straight to a local's home instead of the bank + fmov (saves the trailing `fmov`
 * on every `*_BIND`). Safe only because the borrow release happens HERE, before the operator, not after in localOutFp -- a borrower wants the pre-operator value, and since arm64 reads sources before writing its destination, `sum += x` naming the home among its own sources is fine once the borrow is already released. No FP register on the local: bank returned unchanged, so callers see one shape either way. */
unsigned fpBindDest(Emit *e, unsigned slot, unsigned bank) {
    if (!e->osr || e->slotFpReg[slot] == 0) return bank;
    fpReleaseHome(e, e->slotFpReg[slot]);
    return e->slotFpReg[slot];
}

/* Fuses an OP_ADD/OP_SUB immediately followed by OP_BIND to a float local, by aiming the operator's
 * result straight at the local's home register instead of the bank + fmov. Only safe when nothing between them can deopt/branch into the gap (tracked via homeEarly, checked against fixups post-walk) and fpBindDest already gave the local sole ownership of the register. Returns 0 (ordinary path) otherwise. */
unsigned fpBindLookahead(Emit *e, const uint8_t *code, int next,
                                int stop, const ObjFunction *fn,
                                uint32_t *bindOffOut) {
    if (!e->osr || e->fpOff || e->inlining) return 0;
    if (e->homeEarlyCount >= 32) return 0;
    if (next < stop && code[next] == OP_TYPE_GUARD) {
        /* Only the settled form: a guard that widens an int emits `scvtf` (making the entry an int, so this
         * arm is never reached); a guard naming any other type declines below. */
        if (next + 4 > stop) return 0;
        uint32_t idx = jaiReadU24(code + next + 1);
        if (idx >= (uint32_t)fn->chunk.constants.count) return 0;
        Value t = fn->chunk.constants.data[idx];
        if (!IS_STRING(t) || strcmp(AS_STRING(t)->chars, "float") != 0) return 0;
        next += 4;
    }
    if (next + 3 > stop || code[next] != OP_BIND) return 0;
    unsigned slot = jaiReadU16(code + next + 1);
    /* Deliberately NOT localInRange: that one counts a use for the register
     * allocator, and a slot merely peeked at has not been used. The bound it
     * would apply is applied here instead. */
    if (slot > JIT_MAX_SLOTS || slot >= e->locals) return 0;
    if (e->dynamicLocal[slot] || e->slotFpReg[slot] == 0) return 0;
    *bindOffOut = (uint32_t)next;
    return e->slotFpReg[slot];
}

/* ------------------------------------------------------------------ */
/* Deferred X entries: pending constants and borrowed locals            */
/* ------------------------------------------------------------------ */


bool anyDeferred(const Emit *e) {
    return (e->kPend | e->xBorrow) != 0;
}

/* Puts entry `idx` in its own register. Both forms are one instruction and neither touches NZCV
 * (movz/movk, or `orr xd,xzr,xs`), so this stays safe between a compare and the branch reading its flags. */
static void settleEntry(Emit *e, unsigned idx) {
    if (e->kPend & (1u << idx)) {
        e->kPend &= ~(1u << idx);
        emitConst64(e, valueXReg(e, idx), e->kPendVal[idx]);
        return;
    }
    if (e->xBorrow & (1u << idx)) {
        unsigned src = e->xBorrowReg[idx];
        e->xBorrow &= ~(1u << idx);
        emit(e, jaiA64MovX(valueXReg(e, idx), src));
    }
}

void settleAll(Emit *e) {
    if (!anyDeferred(e)) return;
    for (unsigned i = 0; i < 32u; i++) settleEntry(e, i);
}

/* Register entry `idx` may be READ from. A borrow answers with the local's own register for free; a
 * pending constant must first be materialised -- exactly the instruction deferral was avoiding -- so an arm that can fold checks kPend before calling this. */
unsigned xHeldIn(Emit *e, unsigned idx) {
    if (e->xBorrow & (1u << idx)) return e->xBorrowReg[idx];
    if (e->kPend & (1u << idx)) settleEntry(e, idx);
    return valueXReg(e, idx);
}

/* X register a local permanently lives in, or 0 when it lives nowhere a stack entry could borrow
 * (a spilled frame slot, or an OSR slot the register plan left in memory). */
unsigned localHomeX(const Emit *e, unsigned slot) {
    if (e->osr) return e->slotXReg[slot];
    if (e->spilled) return 0u;
    return localReg(e, slot);
}

void xBorrowLocal(Emit *e, unsigned idx, unsigned reg) {
    e->xBorrow |= 1u << idx;
    e->xBorrowReg[idx] = (uint8_t)reg;
}

void kPendLocal(Emit *e, unsigned idx, int64_t k) {
    e->kPend |= 1u << idx;
    e->kPendVal[idx] = k;
}

/* The pending literal on top, when it fits the imm12 both `adds` and `subs`
 * take. Reports it without consuming it: the caller folds only once it knows
 * the rest of the shape allows it. */
bool pendingImm12(const Emit *e, unsigned idx, int64_t *out) {
    if ((e->kPend & (1u << idx)) == 0) return false;
    int64_t k = e->kPendVal[idx];
    if (k < -4095 || k > 4095) return false;
    *out = k;
    return true;
}

/* `rn - k`, flags only, in one instruction. A negative `k` becomes `cmn`,
 * which subtracts it just as exactly -- the same trick OP_INC_LOCAL uses for a
 * negative step, and for the same reason: both forms set V for the operation
 * actually performed. */
void emitCmpImm(Emit *e, unsigned rn, int64_t k) {
    if (k >= 0) emit(e, jaiA64SubsXImm(31, rn, (unsigned)k));
    else        emit(e, jaiA64AddsXImm(31, rn, (unsigned)(-k)));
}

/* `rd = rn + k` for OP_ADD, `rd = rn - k` for OP_SUB, with V set for the sum
 * or difference that was actually asked for. */
void emitAddSubImm(Emit *e, unsigned rd, unsigned rn, int64_t k,
                          bool subtract) {
    bool down = subtract ? (k >= 0) : (k < 0);
    unsigned m = (unsigned)(k >= 0 ? k : -k);
    if (down) emit(e, jaiA64SubsXImm(rd, rn, m));
    else      emit(e, jaiA64AddsXImm(rd, rn, m));
}

bool pushValue3(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass,
                       Value seen, int fromLocal) {
    if (e->depth >= JIT_MAX_STACK) {
        e->whyNot = "the operand stack is deeper than the model allows";
        return false;
    }
    if (!e->measuring && inlineOwnBank(e) &&
        e->valueDepth + 1u - e->inlValueBase > JIT_SCRATCH_BANK_COUNT) {
        e->whyNot = "an inlined body wants more registers than a call leaves free";
        return false;
    }
    if (!e->measuring && !inlineOwnBank(e) &&
        e->valueDepth + 1 > valueBankRoom(e)) {
        e->whyNot = "more live values than there are callee-saved registers";
        return false;
    }
    e->stackShape[e->depth] = shape;
    e->stackClass[e->depth] = klass;
    e->stackSeen[e->depth]  = seen;
    e->stackLocal[e->depth] = fromLocal;
    e->stackAscii[e->depth] = false;
    e->stackNullLit[e->depth] = false;
    e->stackUnit[e->depth]  = false;
    e->stackObjType[e->depth] = 0;
    e->stackElem[e->depth] = NULL_VAL;
    e->stackElemDecl[e->depth] =
        (fromLocal >= 0 && fromLocal <= (int)JIT_MAX_SLOTS)
            ? e->localElemDecl[fromLocal] : 0;
    e->stackObjType[e->depth] =
        (fromLocal >= 0 && fromLocal <= (int)JIT_MAX_SLOTS)
            ? e->localObjType[fromLocal] : 0;
    e->stack[e->depth++] = kind;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    e->idxKnown &= ~(1u << e->valueDepth);
    e->valueDepth++;
    /* An inlined body's entries are not in the caller's bank, so they do not
     * widen its save set -- which is the whole reason they fit. */
    if (e->valueDepth > e->maxValue &&
        !(e->inlining && e->valueDepth > e->inlValueBase)) {
        e->maxValue = e->valueDepth;
    }
    /* The same number counted the other way: how wide the stack gets when the
     * inlined entries are NOT given a bank of their own. That is the question
     * "may this body's values live in x0..x8" asks, and maxValue cannot answer
     * it -- it deliberately stops counting at the inline boundary. */
    if (e->valueDepth > e->maxValueAll) e->maxValueAll = e->valueDepth;
    return true;
}

bool pushValue(Emit *e, SlotKind kind, uint32_t shape, ObjClass *klass) {
    return pushValue3(e, kind, shape, klass, NULL_VAL, -1);
}

/* The kind a field is known to hold, or SLOT_SELF for "not known". Trusting an
 * answer here is skipping a tag guard, so what makes it safe is not this
 * lookup but the four things that retire an entry: recordFieldStore below,
 * plus forgetFieldKinds (a call, a branch target) and forgetFieldKindsOfLocal
 * (a write to the local named). */
SlotKind knownFieldKind(const Emit *e, int local, uint16_t field) {
    if (local < 0) return SLOT_SELF;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].local == local && e->known[i].field == field) {
            return e->known[i].kind;
        }
    }
    return SLOT_SELF;
}

/* Records a store and drops what any OTHER receiver claimed about the same field: two locals can
 * alias the same object, so a store through one must retire the other's knowledge of it. */
void recordFieldStore(Emit *e, int local, uint16_t field, SlotKind kind) {
    unsigned out = 0;
    for (unsigned i = 0; i < e->knownCount; i++) {
        if (e->known[i].field == field) continue;
        e->known[out++] = e->known[i];
    }
    e->knownCount = out;
    if (local < 0) return;
    if (e->knownCount >= 16) return;
    e->known[e->knownCount].local = local;
    e->known[e->knownCount].field = field;
    e->known[e->knownCount].kind  = kind;
    e->knownCount++;
}

bool pushSelf(Emit *e) {
    if (e->depth >= JIT_MAX_STACK) return false;
    e->stackAscii[e->depth] = false;
    e->stackNullLit[e->depth] = false;
    e->stackUnit[e->depth]  = false;
    e->stackObjType[e->depth] = 0;
    e->stackElem[e->depth] = NULL_VAL;
    e->stackElemDecl[e->depth] = 0;
    e->stack[e->depth++] = SLOT_SELF;
    return true;
}

/* Retire every proof about an operand-stack entry -- "came out of the ASCII
 * table", "is a variant's shared unit value" and "is the null literal" alike.
 * See Emit::stackAscii. */
void clearStackProofs(Emit *e) {
    for (unsigned i = 0; i < JIT_MAX_STACK; i++) {
        e->stackAscii[i]   = false;
        e->stackUnit[i]    = false;
        e->stackNullLit[i] = false;
    }
}

bool anyStackProof(const Emit *e) {
    for (unsigned i = 0; i < e->depth; i++) {
        if (e->stackAscii[i] || e->stackUnit[i] || e->stackNullLit[i]) {
            return true;
        }
    }
    return false;
}

/* Pop an entry that has already been read out of its FP register, or that was
 * never in one. Only the float paths may call this; everything else goes
 * through popValue, which materialises first. */
bool popValueRaw(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    e->depth--;
    e->valueDepth--;
    e->fpLive   &= ~(1u << e->valueDepth);
    e->fpBorrow &= ~(1u << e->valueDepth);
    e->kPend    &= ~(1u << e->valueDepth);
    e->kKnown   &= ~(1u << e->valueDepth);
    e->xBorrow  &= ~(1u << e->valueDepth);
    e->idxKnown &= ~(1u << e->valueDepth);
    if (kind != NULL) *kind = e->stack[e->depth];
    *reg = valueXReg(e, e->valueDepth);
    return true;
}

/* Pops, reporting the register the value is ACTUALLY in -- a borrowed entry reports the local's own
 * register, removing the copy, since every arm reads one register and writes `pushReg(e)-1`. No arm writes back into a popped register; the whitelist is what keeps that true. */
bool popValue(Emit *e, unsigned *reg, SlotKind *kind) {
    if (e->depth == 0 || !holdsRegister(e->stack[e->depth - 1])) return false;
    fpSyncOne(e, e->valueDepth - 1);
    unsigned held = xHeldIn(e, e->valueDepth - 1);
    if (!popValueRaw(e, reg, kind)) return false;
    *reg = held;
    return true;
}

/* Removes the callee entry a builtin leaves under its result. That entry holds
 * no register, so copying the result's entry down over it and shortening the
 * stack leaves the result on top, still in the register it was computed into. */
void dropCalleeEntry(Emit *e) {
    e->stack[e->depth - 2]      = e->stack[e->depth - 1];
    e->stackShape[e->depth - 2] = 0;
    e->stackClass[e->depth - 2] = NULL;
    e->stackSeen[e->depth - 2]  = NULL_VAL;
    e->stackLocal[e->depth - 2] = -1;
    /* The proofs travel down with the entry they are about. Leaving them
     * behind would let a claim made about the callee slot be read as one
     * about the result that replaced it. */
    e->stackAscii[e->depth - 2] = e->stackAscii[e->depth - 1];
    e->stackUnit[e->depth - 2]  = e->stackUnit[e->depth - 1];
    e->stackObjType[e->depth - 2] = e->stackObjType[e->depth - 1];
    e->stackElem[e->depth - 2] = e->stackElem[e->depth - 1];
    e->stackElemDecl[e->depth - 2] = e->stackElemDecl[e->depth - 1];
    e->depth--;
}

/* Depth and the kind of every entry, in one word. Registers are assigned from
 * the depth and instructions are chosen from the kinds, so a join reached with
 * either one different is a join this tier cannot compile. */
uint32_t stackSignatureAt(const Emit *e, unsigned depth) {
    uint32_t sig = depth & 0xfu;
    for (unsigned i = 0; i < depth && i < 9; i++) {
        sig |= ((uint32_t)e->stack[i] & 3u) << (4 + 2 * i);
    }
    return sig;
}

uint32_t stackSignature(const Emit *e) {
    return stackSignatureAt(e, e->depth);
}

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */

void emitConst64(Emit *e, unsigned rd, int64_t value) {
    if (value >= 0 && value <= 0xffff) {
        emit(e, jaiA64MovzX(rd, (unsigned)value, 0));
        return;
    }
    if (value < 0 && value >= -0x10000) {
        emit(e, jaiA64MovnX(rd, (unsigned)(~(uint64_t)value & 0xffffu)));
        return;
    }
    /* MOVZ where the first non-zero chunk is, not always at chunk 0: it zeroes the other three either way,
     * so a `movz rd,#0` under a single `movk` was one instruction spent writing nothing. Every float constant has this shape -- IEEE-754 puts sign, exponent and the leading mantissa bits in the TOP chunk -- so `1.0` was two instructions and is now one, everywhere a float literal reaches a register. */
    uint64_t bits = (uint64_t)value;
    unsigned first = 0;
    while (first < 3 && ((bits >> (16 * first)) & 0xffffu) == 0) first++;
    emit(e, jaiA64MovzX(rd, (unsigned)((bits >> (16 * first)) & 0xffffu), first));
    for (unsigned shift = first + 1; shift < 4; shift++) {
        unsigned part = (unsigned)((bits >> (16 * shift)) & 0xffffu);
        if (part != 0) emit(e, jaiA64MovkX(rd, part, shift));
    }
}

/* ------------------------------------------------------------------ */
/* Prologue and epilogue                                                */
/* ------------------------------------------------------------------ */

void emitSaveRestore(Emit *e, bool save) {
    for (unsigned i = 0; i < e->savedCount; i += 2) {
        unsigned r1 = JIT_FIRST_SAVED + i;
        int32_t at = (int32_t)(16 + 8 * i);
        if (i + 1 < e->savedCount) {
            emit(e, save ? jaiA64StpOff(r1, r1 + 1, 31, at)
                         : jaiA64LdpOff(r1, r1 + 1, 31, at));
        } else {
            emit(e, save ? jaiA64StrX(r1, 31, (unsigned)at)
                         : jaiA64LdrX(r1, 31, (unsigned)at));
        }
    }
}

/* STP's pre-index immediate is a signed 7-bit field scaled by 8: reaches -512 going in but only +504
 * coming out (imm/8=64 read back as a signed 7-bit field is -64), so an exactly-512-byte frame passed entry and silently truncated on exit -- `ldp x29,x30,[sp],#-512` moves SP a kilobyte the WRONG way, corrupting a caller's frame instead of crashing at the fault site. framePairFits()'s <=504 bound is what both ends agree on. */
static bool framePairFits(const Emit *e) { return e->frameBytes <= 504u; }

void emitFrameEnter(Emit *e) {
    if (framePairFits(e)) {
        emit(e, jaiA64StpPre(29, 30, 31, -(int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64SubXImm(31, 31, e->frameBytes));
    emit(e, jaiA64StpOff(29, 30, 31, 0));
}

static void emitFrameLeave(Emit *e) {
    if (framePairFits(e)) {
        emit(e, jaiA64LdpPost(29, 30, 31, (int32_t)e->frameBytes));
        return;
    }
    emit(e, jaiA64LdpOff(29, 30, 31, 0));
    emit(e, jaiA64AddXImm(31, 31, e->frameBytes));
}

/* v8..v15 are callee-saved only in their low 64 bits -- exactly a double -- so `str d`/`ldr d` is the
 * whole protocol; no FP STP in this encoder, but it only runs once per entry/exit, never in a loop. */
void emitFpSaveRestore(Emit *e, bool save) {
    for (unsigned i = 0; i < e->fpLocals; i++) {
        unsigned r = JIT_FP_FIRST_SAVED + i;
        unsigned at = e->fpSaveOffset + 8u * i;
        emit(e, save ? jaiA64StrD(r, 31, at) : jaiA64LdrD(r, 31, at));
    }
}

void emitEpilogue(Emit *e, unsigned bailed) {
    emit(e, jaiA64MovzX(1, bailed, 0));
    emitFpSaveRestore(e, false);
    emitSaveRestore(e, false);
    emitFrameLeave(e);
    emit(e, jaiA64Ret());
}

/* ------------------------------------------------------------------ */
/* The body                                                            */
/* ------------------------------------------------------------------ */

/* In OSR mode a jump out of the compiled range leaves the loop: it becomes a
 * stub that reports the offset the interpreter should carry on from. */
static uint32_t exitTargetFor(Emit *e, uint32_t target) {
    for (unsigned i = 0; i < e->exitCount; i++) {
        if (e->exitOffset[i] == target) return FIXUP_EXIT - i;
    }
    if (e->exitCount >= JIT_MAX_EXIT) { e->whyNot = "too many ways out of the loop"; e->failed = true; return FIXUP_EXIT; }
    e->exitOffset[e->exitCount] = target;
    return FIXUP_EXIT - e->exitCount++;
}

void branchTo(Emit *e, uint32_t targetOffset, bool conditional,
                     unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    /* A join must agree where every value is: nothing may cross a branch in an FP register or deferred.
     * Settling here stays safe after a compare, since neither fmov, mov, nor movz touches NZCV. */
    fpSyncAll(e);
    settleAll(e);
    if (e->osr && targetOffset < UINT32_MAX - 64u &&
        (targetOffset < e->osrTop || targetOffset >= e->osrEnd)) {
        targetOffset = exitTargetFor(e, targetOffset);
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = targetOffset;
    e->fixups[e->fixupCount].conditional  = conditional;
    e->fixups[e->fixupCount].depth        = (int)stackSignature(e);
    e->fixupCount++;
    emit(e, conditional ? jaiA64BCond(cond, 0) : jaiA64B(0));
}

/* A conditional branch whose target is reached with a different operand stack
 * than the branch leaves from -- the exhausted arm of a for-loop, where the
 * interpreter drops the iterator. */
void branchToDepth(Emit *e, uint32_t targetOffset, unsigned cond,
                          int depthOverride) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    settleAll(e);   /* see branchTo: a join agrees about where every value is */
    if (e->osr && targetOffset < UINT32_MAX - 64u &&
        (targetOffset < e->osrTop || targetOffset >= e->osrEnd)) {
        targetOffset = exitTargetFor(e, targetOffset);
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = targetOffset;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = depthOverride;
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

/* A guard failed: not a bail (unsound once the body has written anything, and the guards that matter
 * guard field reads inside loops that write) -- records where the interpreter resumes and what it holds; the stub is emitted after the body so the hot path keeps one not-taken branch. */
static bool jitDeoptStress(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_DEOPT_STRESS");
        cached = (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

/* JAITHON_JIT_MODULE_CALLS=0 turns off the module-member call arm at OP_INVOKE,
 * so the two shapes can be compared inside ONE binary -- alternating two builds
 * cannot be trusted here, since each switch invalidates __jaicache__ and every
 * sample then pays a stdlib recompile.
 *
 * Default ON. Read once: a body compiled with the arm and a body compiled
 * without it must not coexist in one run. */
bool jitModuleCalls(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MODULE_CALLS");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_CLASS_CALLS=0 turns off the static-member call arm at OP_INVOKE,
 * for the same reason jitModuleCalls exists: the two shapes have to be
 * comparable inside ONE binary. Alternating two builds is not an A/B here --
 * every switch invalidates __jaicache__ and each sample then pays a stdlib
 * recompile, which is larger than the effect being measured.
 *
 * Default ON. Read once, so a body compiled with the arm and a body compiled
 * without it cannot coexist in one run. */
bool jitClassCalls(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CLASS_CALLS");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_MODULE_NATIVE=0 turns off BOTH halves of the `__prim__.f64_sqrt`
 * arm at once -- OP_GET_GLOBAL's globalNamespace resolution and OP_INVOKE's
 * emitModuleNativeCall -- so the two shapes compare inside ONE binary, same
 * reason jitModuleCalls and jitClassCalls exist as their own switches:
 * alternating two builds is not an A/B here, each rebuild invalidates
 * __jaicache__ and the stdlib recompile it pays swamps the effect being
 * measured. One switch for both halves because neither compiles anything
 * useful alone -- OP_GET_GLOBAL pushing the namespace with nothing at
 * OP_INVOKE able to consume it just moves the refusal one instruction later.
 *
 * Default ON. Read once, so a body compiled with the arm and a body compiled
 * without it cannot coexist in one run. */
bool jitModuleNativeCalls(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MODULE_NATIVE");
        cached = (v != NULL && strcmp(v, "0") == 0) ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_SPLIT_STRESS=1 puts the split bank's boundary into every OSR body
 * that can take one, instead of only the ones that pay for it -- the same idea
 * as JAITHON_JIT_DEOPT_STRESS, for the same reason.
 *
 * A split makes the operand stack two runs of registers instead of one, and the
 * failure mode is a site that adds an index to a base and lands one past the
 * end of the first run. That is silent: the value is written to a register
 * nothing reads. It shipped once already -- OP_GET_GLOBAL wrote through
 * `pushReg`, one past the CURRENT top rather than the register the next push
 * lands in, and bitops printed 68720029766 for 999625 on the runs where its
 * loop compiled. It was found by the benchmark differential because bitops
 * happened to be split-eligible AND to read a global at exactly the boundary.
 * Under this flag it would have been found by any of them. */
bool jitSplitStress(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SPLIT_STRESS");
        cached = (v != NULL && v[0] != '\0' && strcmp(v, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

/* Neither an inline's state nor mid-instruction state is the model's actual current state. Inside an
 * inline the interpreter hasn't made the call yet, so it resumes at OP_CALL holding just callee+args, not the inlined body's locals/temporaries. Outside one, a guard mid-instruction (OP_GET_LOCAL2 pushes then guards) leaves the model deeper than the interpreter's stack there -- handing over those extra entries strands them, and a loop can read its own iterator as its loop variable. `instDepth` is the model at the instruction's START, matching the interpreter; only entries THIS instruction pushed (the topmost) may be trimmed back to it -- an instruction that already popped something the interpreter still holds cannot be repaired and is refused. */
static bool deoptSite(Emit *e, uint32_t ip, uint32_t *ipOut,
                      unsigned *depthOut, unsigned *valueDepthOut) {
    if (e->inlining) {
        *ipOut = e->inlIp;
        *depthOut = e->inlDepth;
        unsigned seen = 0;
        for (unsigned i = 0; i < e->inlDepth; i++) {
            if (holdsRegister(e->stack[i])) seen++;
        }
        *valueDepthOut = seen;
        return true;
    }
    *ipOut = ip;
    *depthOut = e->depth;
    *valueDepthOut = e->valueDepth;
    /* Only for a guard that resumes at the instruction being compiled. A guard
     * that names a later offset -- the one after a call, where the result is
     * already on the stack -- is describing a point this walk has not reached
     * and `instDepth` says nothing about it. */
    if (ip != e->curOffset) return true;
    /* OP_BUILD_RANGE emits nothing (deferred, folded into the following OP_GET_ITER), so a deopt record
     * taken at this offset would hand the interpreter two ints where it expects the range object -- it resumed, ran OP_GET_ITER, and reported 'int' object is not iterable. Fix: resume one instruction EARLIER, at the OP_BUILD_RANGE the model still describes (nothing between them has run). Reachable once a guard can fire inside the header itself, e.g. via root-filling a dynamic-local iterator descriptor; needs no spilling and no floats to reproduce. */
    if (e->pendingRange) {
        *ipOut = e->rangeBuildIp;
        *depthOut = e->instDepth;
        unsigned nseen = 0;
        for (unsigned i = 0; i < e->instDepth; i++) {
            if (holdsRegister(e->stack[i])) nseen++;
        }
        *valueDepthOut = nseen;
        return true;
    }
    if (e->depth < e->instDepth) {
        e->whyNot = "a guard resumes an instruction whose operands it has "
                    "already consumed";
        return false;
    }
    if (e->depth == e->instDepth) return true;
    unsigned seen = 0;
    for (unsigned i = 0; i < e->instDepth; i++) {
        if (holdsRegister(e->stack[i])) seen++;
    }
    *depthOut = e->instDepth;
    *valueDepthOut = seen;
    return true;
}

/* Does the operand model still say what the BYTECODE says at this offset?
 *
 * The model is maintained by ~140 hand-written arms plus two inliners, and
 * until this check nothing asked it to agree with anything: a wrong depth still
 * emits a self-consistent body, because every register is derived from an index
 * and the indices all shift together. What it corrupts is the deopt record --
 * the one part of the model another component reads -- which then hands the
 * interpreter operand entries it has not got. That is not a crash, it is a
 * wrong answer, and the two it produced were `'float' object has no method
 * 'push'` out of lib/std/gui/path.jai (inlineMethod's dry walk left two entries
 * behind) and the same class from a walk resuming past an unarmed deopt with a
 * stale model (see emitUnarmedDeopt). Neither was visible in the generated
 * code; both are one comparison away here.
 *
 * jaiChunkStackDepths is the oracle, and this file did not write it: it is the
 * verifier's own pass, the one the interpreter's stack discipline is defined
 * by. -1 is "no answer" -- an offset no path reaches, one an unmodelled opcode
 * stopped the walk at, or one whose depth came from an imprecise handler seed
 * -- and is never a failure.
 *
 * Asked only of the function tier's own walk. An OSR body's model starts at the
 * loop head rather than at offset 0, so its depth is relative and disagrees by
 * a constant; an inlined body's offsets are the callee's and mean nothing in
 * the caller's table.
 *
 * Declines, so an arm that drifts in future costs coverage and not an answer. */
bool modelAgreesWithChunk(const Emit *e, uint32_t off) {
    if (e->osr || e->inlining || e->chunkDepth == NULL) return true;
    if (off >= (uint32_t)e->chunkDepthCount) return true;
    int want = e->chunkDepth[off];
    return want < 0 || want == (int)e->depth;
}

/* Names the opcode whose arm let the borrow through. Which arm it was is the
 * whole question when this fires, and without the name the message only says
 * where the loop started. */
static const char *borrowWhyFor(const Emit *e) {
    static char why[80];
    snprintf(why, sizeof why, "a float borrow reached %s's guard",
             jaiOpName((OpCode)e->lastOp));
    return why;
}

/* Take the record without emitting the branch to it. The model is what it is
 * at this moment, so a site whose *code* is emitted later -- a self-call's
 * cold block, which lives with the stubs -- still has to record here. */
bool deoptRecordAt(Emit *e, uint32_t ip, bool lastFromDesc,
                          unsigned *out) {
    /* Assertion, not the fix: a deopt stub writes every entry out of fpRegAt, so nothing may still be
     * borrowing a local's register here. Releasing HERE (rather than at the top of the instruction) was tried and is wrong -- a guard can sit inside a span an earlier branch skips (emitBoundsNormalise's does), so the fmov landed on a not-taken path and matrix_mul read `sum` from a register nothing had written. Declines rather than miscompiles if fpBorrowSurvives let something through it shouldn't have. */
    if (e->fpBorrow != 0) {
        /* Names the opcode: which arm let the borrow through is the whole
         * question, and without it the message only says where the loop
         * started. */
        static char borrowWhy[80];
        snprintf(borrowWhy, sizeof borrowWhy, "a float borrow reached %s's guard",
                 jaiOpName((OpCode)e->lastOp));
        e->whyNot = borrowWhy;
        e->failed = true;
        return false;
    }
    /* Same reasoning as above, for a pending constant or X-register borrow: an assertion that the
     * whitelist held, declining rather than miscompiling if it didn't. */
    if (anyDeferred(e)) {
        e->whyNot = "a deferred value reached a guard";
        e->failed = true;
        return false;
    }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return false;
    }
    unsigned k = e->deoptCount++;
    if (!deoptSite(e, ip, &e->deopt[k].ip, &e->deopt[k].depth,
                   &e->deopt[k].valueDepth)) {
        e->failed = true;
        return false;
    }
    if (lastFromDesc && !e->inlining && e->deopt[k].depth != e->depth) {
        /* The from-descriptor entry is the top of the record, so a record that
         * was trimmed is no longer describing it. No site does both today. */
        e->whyNot = "a call's result guard resumes before the call";
        e->failed = true;
        return false;
    }
    e->deopt[k].lastFromDesc = lastFromDesc;
    e->deopt[k].fpLive       = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    *out = k;
    return true;
}

void branchOnDeoptAt(Emit *e, unsigned cond, uint32_t ip,
                            bool lastFromDesc) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    unsigned k;
    if (!deoptRecordAt(e, ip, lastFromDesc, &k)) return;
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

void branchOnDeopt(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->fpBorrow != 0) {          /* see deoptRecordAt */
        e->whyNot = borrowWhyFor(e);
        e->failed = true;
        return;
    }
    if (anyDeferred(e)) {            /* see deoptRecordAt */
        e->whyNot = "a deferred value reached a guard";
        e->failed = true;
        return;
    }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    if (!deoptSite(e, e->curOffset, &e->deopt[k].ip, &e->deopt[k].depth,
                   &e->deopt[k].valueDepth)) {
        e->failed = true;
        return;
    }
    e->deopt[k].lastFromDesc = false;
    e->deopt[k].fpLive     = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    /* JAITHON_JIT_DEOPT_STRESS makes every guard fail, so the whole test suite exercises the resume path
     * -- otherwise reached only when a program changes a field's type, which almost none do. */
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

/* A guard that resumes at the START of the instruction being compiled, whatever
 * that instruction's arm has already popped.
 *
 * deoptRecordAt refuses this case ("a guard resumes an instruction whose
 * operands it has already consumed") because the entries below `depth` are the
 * only ones it will describe. They are still readable here: popValue moves
 * `depth` and `valueDepth` and nothing else, so entry i's kind is still in
 * stack[i] and its register is still valueXReg(i) -- the mapping is positional.
 * What is NOT guaranteed is that the arm has left those registers alone, which
 * is why this is not a general facility: its one caller is the overflow guard
 * inside a `try`, and the arms that reach it compute into a scratch (ovfDest)
 * so nothing an entry lives in, and no local, has been written when it fires.
 *
 * fpLive is read as it stands rather than as it was: popValue calls fpSyncOne
 * first, so an entry whose bit this instruction cleared has its X register
 * current, which is exactly what the stub then writes out. */
void branchOnDeoptInstStart(Emit *e, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    if (e->fpBorrow != 0) {          /* see deoptRecordAt */
        e->whyNot = borrowWhyFor(e);
        e->failed = true;
        return;
    }
    if (anyDeferred(e)) {            /* see deoptRecordAt */
        e->whyNot = "a deferred value reached a guard";
        e->failed = true;
        return;
    }
    if (e->deoptCount >= JIT_MAX_DEOPT) {
        e->whyNot = "too many guards to record";
        e->failed = true;
        return;
    }
    /* The record below describes entries this instruction may already have
     * popped, and popValue clears the bit that said an entry was only ever a
     * borrow of a local's register. So the question has to be asked of the
     * instruction's START, where the walk recorded it. */
    if (!e->instClean && !e->inlining) {
        e->whyNot = "a raise resumes an instruction whose operands were borrowed";
        e->failed = true;
        return;
    }
    unsigned k = e->deoptCount++;
    if (e->inlining) {
        /* Inside an inline the interpreter has not made the call yet, so the
         * only resume point is the caller's OP_CALL -- which deoptSite already
         * answers, and which is already "the start of an instruction". */
        if (!deoptSite(e, e->curOffset, &e->deopt[k].ip, &e->deopt[k].depth,
                       &e->deopt[k].valueDepth)) {
            e->failed = true;
            return;
        }
    } else {
        e->deopt[k].ip         = e->curOffset;
        e->deopt[k].depth      = e->instDepth;
        e->deopt[k].valueDepth = e->instValueDepth;
    }
    e->deopt[k].lastFromDesc = false;
    e->deopt[k].fpLive       = e->fpLive;
    for (unsigned i = 0; i < e->deopt[k].depth; i++) {
        e->deopt[k].kinds[i]   = e->stack[i];
        e->deopt[k].classes[i] = e->stackClass[i];
    }
    bool always = jitDeoptStress();
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_DEOPT - k;
    e->fixups[e->fixupCount].conditional  = !always;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, always ? jaiA64B(0) : jaiA64BCond(cond, 0));
}

/* Branch to the bail block on `cond`. The block's index is not known yet, so
 * it is patched with the rest. */
/* NaN comparison is a TypeError here, not false, matching the interpreter's isnan check: fcmp sets V
 * on an unordered result, and that's routed to a deopt so the interpreter raises exactly what it would have. */
void nanToDeopt(Emit *e) { branchOnDeopt(e, JAI_A64_VS); }

/* `c < "0"` and its three siblings, both operands a string one byte long.
 *
 * Ordering on strings had no arm at all, so one `character >= "0"` declined
 * the whole body around it: json_parse's `integer` is 50% of that benchmark's
 * interpreted instructions and the self-hosted lexer's `_is_digit`/`_is_alpha`
 * 3.5% of a compile's. Only the one-byte shape is compiled -- compareStrings
 * on anything longer is a memcmp, which is a call, and a character-class test
 * is the shape that actually occurs. The sample says one byte and the emitted
 * code GUARDS it, so a longer string arriving later deoptimises rather than
 * being answered wrongly.
 *
 * The bytes go in through LDRB, which zero-extends, so both are 0..255 and the
 * signed conditions the integer arm already computed give memcmp's unsigned
 * answer unchanged -- no separate condition table. */
bool isOrdering(uint8_t op) {
    return op == OP_LT || op == OP_LE || op == OP_GT || op == OP_GE;
}

bool stringOperand(const Emit *e, unsigned at) {
    return e->stackAscii[at] || IS_STRING(e->stackSeen[at]);
}

/* Statically one byte: the ASCII table's own singleton, or a sample that is a
 * one-character string -- which for the literal side of a character-class test
 * is the constant-pool entry itself and so cannot be anything else. */
static bool knownOneByte(const Emit *e, unsigned at) {
    Value v = e->stackSeen[at];
    return e->stackAscii[at] || (IS_STRING(v) && AS_STRING(v)->length == 1);
}

/* One side has to be known one byte before this is worth compiling. The other
 * is only guarded, and a sample can lie: OP_GET_INDEX hands its result the
 * RECEIVER as a sample (only the type is read from it), so `s[i]`'s sample is
 * the whole subject string. Without the literal side to anchor on, a general
 * `a < b` over long strings would compile and then deopt every iteration,
 * which is slower than never compiling the body at all. */
bool oneBytePair(const Emit *e, unsigned a, unsigned b) {
    return stringOperand(e, a) && stringOperand(e, b) &&
           (knownOneByte(e, a) || knownOneByte(e, b));
}

void emitOneByteString(Emit *e, unsigned at, unsigned reg, unsigned dst) {
    /* What the ASCII table produced is a one-byte interned string by
     * construction, so it needs neither guard -- see the same skip in OP_EQ. */
    if (!e->stackAscii[at]) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, reg, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, reg,
                           (unsigned)offsetof(ObjString, length)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 1));
        branchOnDeopt(e, JAI_A64_NE);
    }
    /* `chars` is a pointer, not an inline array: a string may address bytes it
     * does not own. Two loads, and the second is the byte itself. */
    emit(e, jaiA64LdrX(dst, reg, (unsigned)offsetof(ObjString, chars)));
    emit(e, jaiA64LdrByte(dst, dst, 0));
}

/* A/B switch, default on, and not optional: the machine runs several agents at
 * once, so before and after have to be the same binary minutes apart rather
 * than two binaries. JAITHON_JIT_STRCMP=0 puts the arm below back to the
 * decline it was, with nothing else changed. */
bool jitStrCmpOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STRCMP");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* PROTOTYPE, second switch: send `==`/`!=` to the leaf call when the operands
 * are not KNOWN interned, instead of letting the pointer arm compile a guard
 * that deoptimises on every iteration. Default on; JAITHON_JIT_STRCMP_EQ=0
 * restores the arm exactly as proposed. */
bool jitStrCmpEqOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STRCMP_EQ");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* A sample that is interned says the site is an interned-string site, which is
 * a prediction the pointer arm's own subFlag guard already checks. What it is
 * used for here is only which arm to emit. */
static bool knownInternedString(const Emit *e, unsigned at) {
    if (e->stackAscii[at]) return true;
    Value v = e->stackSeen[at];
    return IS_STRING(v) && JAI_STR_INTERNED(AS_STRING(v));
}

/* True when the pointer arm should stand aside for the leaf call at this
 * equality site: the leaf call has to be available (same switch, same
 * inlining rule) and at least one operand must not be known interned. */
bool preferLeafEquality(const Emit *e, unsigned da, unsigned db) {
    if (!jitStrCmpEqOn() || !jitStrCmpOn() || e->inlining) return false;
    return !(knownInternedString(e, da) && knownInternedString(e, db));
}

/* `a <=> b` for two strings whose lengths nothing knows, as a guarded LEAF
 * call. Leaves NZCV set from `cmp x0, #0`, which is the shape every other arm
 * in these switches leaves behind, so the `cset` or the branch after it is
 * unchanged and all six operators come out of the one sequence: the order is
 * -1, 0 or 1, and `a OP b` is `order OP 0` for every one of them.
 *
 * WHY THIS IS NOT THE PREDICTION THE ONE-BYTE ARM REFUSED. That comment says a
 * sample can lie -- OP_GET_INDEX hands its result the RECEIVER as a sample --
 * so a general compare compiled on a guess about LENGTH would deopt every
 * iteration, which is worse than never compiling the body. Nothing here
 * predicts a length. `Obj.type == OBJ_STRING` is EXACT: every string passes it,
 * so there is no deopt loop, and the sample is used only to decide that the
 * arm is worth emitting at all.
 *
 * WHY A LEAF AND NOT A DESCRIPTOR. jaiStringOrder allocates nothing, roots
 * nothing and cannot re-enter the interpreter, so it needs none of the dozen
 * stores, the root fill or the collector-chain link a descriptor call pays --
 * far more than an eleven-byte memcmp costs, and it would have measured zero
 * or worse. Modelled on the jitInstanceAlloc call in emitCallOut, which is a
 * leaf for the same reason.
 *
 * WHAT IT CLOBBERS, and how each is accounted for:
 *   - x0..x17 and x30, per AAPCS64. The operand stack is in x0..x8 whenever
 *     the body is otherwise call-free (Emit::scratchValues) or split
 *     (Emit::splitAt), so this goes through noteScratchClobber like every
 *     other call out: the measuring pass records the site, which turns
 *     scratchValues off for the whole body and puts the split boundary at or
 *     above this depth, and the real pass fails the compile if it reaches here
 *     with values in scratch anyway. It also retires the field-kind memos,
 *     which over-retires (this callee writes no field) and costs a tag guard.
 *   - v16.. -- the FP half of the operand bank is caller-saved on purpose.
 *     fpSyncAll below writes every live float entry back to its X home first.
 *     Float LOCALS are in v8..v15, which the ABI preserves.
 *   - x13..x17, which planHoists spends on loop-invariant list headers. The
 *     same noteScratchClobber recorded the offset, so regionCalls reports this
 *     loop as calling and no header is hoisted out of it; the ratchet in
 *     noteScratchClobber fails the compile if the two passes ever disagree.
 *   - x30, saved by emitFrameEnter at entry on every path.
 * The body is NOT call-free for register planning afterwards, and that is the
 * real price of this arm: a body whose only call is this one loses x0..x8 for
 * its operand stack and can decline for want of registers where it used to
 * compile. Measured anyway -- see docs/agents/string-compare.md. */
void emitStringOrder(Emit *e) {
    unsigned da = e->depth - 2, db = e->depth - 1;
    /* Settled before anything is read: the guards record deopts, which cannot
     * describe a deferred entry, and the call would destroy a borrowed one. */
    settleAll(e);
    fpSyncAll(e);
    unsigned ra = valueXReg(e, e->valueDepth - 2);
    unsigned rb = valueXReg(e, e->valueDepth - 1);

    /* Both operands, before either is consumed, so a miss resumes at this
     * instruction with the pair still on the interpreter's stack -- and the
     * interpreter then raises whatever the operator raises for the pair it
     * actually has. */
    for (unsigned side = 0; side < 2; side++) {
        /* What the ASCII table produced is a string by construction; see the
         * same skip in OP_EQ. */
        if (e->stackAscii[side == 0 ? da : db]) continue;
        unsigned r = side == 0 ? ra : rb;
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, r, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
        branchOnDeopt(e, JAI_A64_NE);
    }

    /* Into x0/x1 without assuming where the operands live. The split bank puts
     * entries in x0..x8, and although this call's own depth is what the
     * boundary is chosen from, a `mov` that reads a register it has already
     * written is a miscompile rather than a decline -- so the aliasing is
     * handled instead of argued away. */
    if (rb == 0) {
        emit(e, jaiA64MovX(JIT_SCRATCH_C, rb));
        rb = JIT_SCRATCH_C;
    }
    if (ra != 0) emit(e, jaiA64MovX(0, ra));
    if (rb != 1) emit(e, jaiA64MovX(1, rb));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jaiStringOrder);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    /* The whole of x0: jaiStringOrder returns int64_t precisely so this does
     * not have to trust the top half of a 32-bit return. */
    emit(e, jaiA64SubsXImm(31, 0, 0));
}


/* One past the LAST back edge to `top`, or 0 if nothing branches back there.
 * Not findLoopEnd, which stops at the first: `continue` is a second back edge
 * to the same head, and stopping at it would call the rest of the body
 * "outside the loop" -- which is exactly the half a hoist must not believe. */
static uint32_t loopBodyEnd(const Chunk *c, uint32_t top) {
    uint32_t last = 0;
    for (int off = (int)top; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return 0;
        if (c->code[off] == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            if ((uint32_t)((int32_t)(off + 3) + jump) == top) {
                last = (uint32_t)(off + 3);
            }
        }
        off += len;
    }
    return last;
}

/* Proves the list in `rList` is still the boxed Value[] every element access
 * below is emitted against. An unboxed `list[int]` packs its elements eight
 * bytes apart with no tag, so a body compiled for the boxed layout would read
 * two of them as one tagged pair -- and the storage is a property of the LIST,
 * not of the slot, so nothing the walk knows can rule it out.
 *
 * Two instructions, on the boxed path as well. The loop tier pins the storage
 * per slot instead and jaiJitEnterOsr proves it once at entry (JaiOsrForm::
 * kinds), so the sites that can name their slot skip this entirely; this is
 * for the ones that cannot -- a nested iterator's source, a list reached
 * through a field, anything the function tier subscripts. */
void emitListBoxedGuard(Emit *e, unsigned rList, unsigned scratch) {
    emit(e, jaiA64LdrByte(scratch, rList, (unsigned)offsetof(ObjList, stg)));
    emit(e, jaiA64SubsXImm(31, scratch, LIST_STORE_BOXED));
    branchOnDeopt(e, JAI_A64_NE);
}


/* The ListStore a site may emit its element accesses against, and whatever
 * guard makes that true.
 *
 * A slot the loop tier pinned needs no runtime check at all -- jaiJitEnterOsr
 * proved it at entry, the same way it proves an instance slot's class, so the
 * stride below is right before the body starts. Anything the walk cannot name
 * a slot for -- a list reached through a field, a temporary, anything the
 * function tier subscripts -- is proved here instead, and BOXED is the only
 * answer those two instructions accept. */
/* The one unboxed storage a list holding a `vk` could have. BOXED means there
 * is no other possibility: a list of instances, strings or lists is boxed and
 * nothing else, so those sites need no second arm. */
uint8_t listAltFor(SlotKind vk) {
    switch (vk) {
    case SLOT_INT:   return (uint8_t)LIST_STORE_I64;
    case SLOT_FLOAT: return (uint8_t)LIST_STORE_F64;
    case SLOT_BOOL:  return (uint8_t)LIST_STORE_U8;
    default:         return (uint8_t)LIST_STORE_BOXED;
    }
}

/* Decides between the three ways a site can know its storage, and emits
 * whatever guard the chosen one owes.
 *
 * PINNED. The loop tier sampled the storage from the live frame and
 * jaiJitEnterOsr proves it at entry, the way it proves an instance slot's
 * class. One arm, no test, no tag check: this is the whole point of the
 * unboxing, and it is where the 1.3-1.5x lives.
 *
 * PROVED. Nothing pinned it, and no unboxed storage could hold a `vk` anyway
 * (a list of instances). Two instructions say BOXED or deoptimise, and BOXED
 * is a fact that stays true -- `stg` only ever moves towards boxed.
 *
 * DISPATCHED. Nothing pinned it and two storages are possible. Emitting the
 * boxed arm behind a deopt guard is what the first version did, and it is
 * ruinous rather than merely slower: the guard fails on EVERY access, and each
 * failure leaves the compiled loop and re-enters it. `var f: list[bool] = []`
 * at the top of a loop body is enough to arrange it -- the slot is written, so
 * nothing pins it -- and building a 300k sieve thirty times went from 80ms to
 * 300ms. Two arms and one compare cost four instructions and never leave. */
ListAccess listAccessFor(Emit *e, unsigned rList, int slot,
                                SlotKind vk, unsigned scratch) {
    ListAccess a;
    if (e->osr && slot >= 0 && slot <= (int)JIT_MAX_SLOTS &&
        e->localStgPin[slot]) {
        a.stg = localStgOf(e, slot);
        a.alt = a.stg;
        a.dynamic = false;
        return a;
    }
    a.alt = listAltFor(vk);
    a.stg = (uint8_t)LIST_STORE_BOXED;
    a.dynamic = a.alt != LIST_STORE_BOXED;
    if (!a.dynamic) emitListBoxedGuard(e, rList, scratch);
    return a;
}

/* Opens the runtime test. Returns the index of the placeholder branch, or -1
 * when the access is static and no second arm follows.
 *
 * Tests for `alt` EXACTLY, not merely for "not boxed". `alt` is the storage a
 * list holding a `vk` would have -- it is a prediction, not a reading, and the
 * list is free to have a third one. `xs[i] = 3` on a `list[bool]` predicts I64
 * from the value's kind while the array is one byte per element, and an
 * eight-byte store at `i << 3` then runs off the end of it. So the third case
 * deoptimises: the interpreter's jaiListPut de-specialises or raises, which is
 * an answer this cannot emit inline. */
int listDispatchBegin(Emit *e, const ListAccess *a, unsigned rList,
                             unsigned scratch) {
    if (!a->dynamic) return -1;
    emit(e, jaiA64LdrByte(scratch, rList, (unsigned)offsetof(ObjList, stg)));
    emit(e, jaiA64SubsXImm(31, scratch, a->alt));
    int skip = (int)e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64SubsXImm(31, scratch, LIST_STORE_BOXED));
    branchOnDeopt(e, JAI_A64_NE);
    return skip;
}

/* Closes the first arm and opens the second. Returns the join placeholder. */
int listDispatchElse(Emit *e, int skip) {
    int join = (int)e->count;
    emit(e, jaiA64B(0));
    e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)((int)e->count - skip));
    return join;
}

void listDispatchEnd(Emit *e, int join) {
    e->code[join] = jaiA64B((int32_t)((int)e->count - join));
}


/* log2 of the element stride: a boxed Value is sixteen bytes, an int or a
 * double eight, a bool one. */
unsigned listStgShift(uint8_t stg) {
    switch ((ListStore)stg) {
    case LIST_STORE_I64:
    case LIST_STORE_F64: return 3;
    case LIST_STORE_U8:  return 0;
    case LIST_STORE_BOXED: break;
    }
    return 4;
}

/* The SlotKind every element of an unboxed store has. The storage IS the type,
 * which is most of what the unboxing buys: the boxed path spends a load, a
 * compare and a branch per element proving what this knows statically. */
SlotKind listStgKind(uint8_t stg) {
    switch ((ListStore)stg) {
    case LIST_STORE_I64: return SLOT_INT;
    case LIST_STORE_F64: return SLOT_FLOAT;
    case LIST_STORE_U8:  return SLOT_BOOL;
    case LIST_STORE_BOXED: break;
    }
    return SLOT_OPAQUE;
}

/* The same store with the base and index named, for the subscript arm, whose
 * items pointer may be a hoisted register rather than JIT_SCRATCH_C. */
void emitElemStoreAt(Emit *e, uint8_t stg, unsigned rItems,
                            unsigned rIdx, unsigned vtag, unsigned rVal) {
    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, rItems, rIdx, listStgShift(stg)));
    if (stg == LIST_STORE_BOXED) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, vtag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
    } else if (stg == LIST_STORE_U8) {
        emit(e, jaiA64StrByte(rVal, JIT_SCRATCH_C, 0));
    } else {
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 0));
    }
}

/* One element store, at whatever width `stg` says, from JIT_SCRATCH_C (items)
 * and JIT_SCRATCH_A (index). Leaves JIT_SCRATCH_C pointing at the element. */
static void emitListElemStore(Emit *e, uint8_t stg, unsigned vtag,
                              unsigned rVal) {
    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_A,
                          listStgShift(stg)));
    if (stg == LIST_STORE_BOXED) {
        emit(e, jaiA64MovzX(JIT_SCRATCH_D, vtag, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 8));
    } else if (stg == LIST_STORE_U8) {
        emit(e, jaiA64StrByte(rVal, JIT_SCRATCH_C, 0));
    } else {
        /* A double's bits and an int's are both the whole register: the tier
         * keeps a SLOT_FLOAT payload in an X register exactly as the boxed
         * store did. */
        emit(e, jaiA64StrX(rVal, JIT_SCRATCH_C, 0));
    }
}


/* items+count via one ldp: rCount comes back as `count | capacity << 32` (the two int32s share the
 * pair's second doubleword), so every reader must use the uxtw forms. Layout is asserted below, not assumed -- a field reordered in object.h would load a pointer as a count and index off the end of the array. */
void emitListHeader(Emit *e, unsigned rList, unsigned rItems,
                           unsigned rCount) {
    _Static_assert(offsetof(ObjList, count) == offsetof(ObjList, items) + 8,
                   "ObjList.count must follow items for the ldp");
    _Static_assert(offsetof(ObjList, capacity) == offsetof(ObjList, count) + 4,
                   "ObjList.capacity must share the count's doubleword");
    emit(e, jaiA64LdpOff(rItems, rCount, rList,
                         (int32_t)offsetof(ObjList, items)));
}

/* Can anything in [lo, hi) destroy a caller-saved register? The measuring pass
 * recorded every site that can (noteScratchClobber), so this is a lookup and
 * not a re-derivation -- which matters, because the set of things that clobber
 * is not the set of things that look like calls: the list-grow stub is an
 * OP_LIST_APPEND, and the self-call slow path is an OP_CALL that never leaves
 * the body. Whatever reaches noteScratchClobber is in here by construction. */
bool regionCalls(const Emit *e, uint32_t lo, uint32_t hi) {
    if (e->clobberSpill) return true;
    for (unsigned i = 0; i < e->clobberCount; i++) {
        if (e->clobberOff[i] >= lo && e->clobberOff[i] < hi) return true;
    }
    return false;
}

/* The hoisted header for a subscript whose base is a plain read of local
 * `slot`, or -1. `curOffset` is checked against the loop the entry was made
 * for, so an entry the walk has already left cannot be picked up again by a
 * later loop that happens to name the same slot. */
int hoistFor(const Emit *e, int slot) {
    if (slot < 0 || e->inlining) return -1;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].slot != (uint8_t)slot) continue;
        if (e->curOffset < e->hoist[i].top) continue;
        if (e->curOffset >= e->hoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* Nothing outside [top, end) may branch INTO it. The hoisted loads sit just
 * above the head, so any path that reaches the body without passing through
 * them would run it against registers nobody loaded. Written as "into the
 * range", not "to the head", because the head is only the entrance a
 * structured loop is supposed to have -- this is what makes that a checked
 * fact rather than an assumption about the emitter. */
static bool onlyBackEdgesEnter(const Chunk *c, uint32_t top, uint32_t end) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return false;
        if ((uint32_t)at >= top && (uint32_t)at < end) { at += len; continue; }
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int32_t to = (int32_t)(at + len) +
                         jaiReadI16(c->code + at + 1 + rel);
            if (to >= (int32_t)top && to < (int32_t)end) return false;
        }
        at += len;
    }
    return true;
}

/* Loop-invariant list headers, hoisted above the loop head.
 *
 * Every `xs[i]` reloads `items` and `count` off the ObjList, and in a stencil
 * (five neighbours, four of them from rows the loop is not walking) that is
 * five loads an iteration of something no iteration changes. The reload is
 * also what makes the read SOUND without a version guard, so hoisting it needs
 * the guard back -- unless the stretch of code the load is hoisted over can be
 * shown to contain nothing that could move a list at all. Every way a list is
 * resized (push, insert, remove, clear, slice, anything through a descriptor)
 * is a call out, and so is every collection; an in-place `xs[i] = v` moves
 * neither `items` nor `count`. So across a call-free stretch a header is
 * invariant for as long as the LOCAL is, and that is a fact the measuring pass
 * already recorded (noteSlotWrite).
 *
 * That stretch is the CANDIDATE LOOP, not the body. `bodyCalls` -- the whole
 * body -- is what this used to ask, and it is both sound and far too strong:
 * matrix_mul's `k` loop calls nothing, but the `i` loop it sits in pushes a row
 * per iteration, so the body answered "calls" and the innermost loop in the
 * benchmark got nothing. A hoist out of a call-free loop is sound for exactly
 * the same two reasons it was before -- nothing between the load and its uses
 * can move the list, and nothing between them can overwrite a caller-saved
 * register -- because both are statements about the code the value is live
 * across, and that is the loop.
 *
 * Registers come from x13..x17, which the tier otherwise never names, plus
 * whatever the operand stack left unused at the top of the scratch bank. Both
 * are caller-saved, which is what makes them free and also what confines a
 * hoist to a call-free region: a call outside the loop may destroy them, and is
 * welcome to -- by then the hoisted header is dead, and re-entering the loop
 * re-runs the load that sits above its head. The scratch-bank half is offered
 * only under `scratchValues`, which is still a whole-body claim, since those
 * registers are the operand stack's everywhere else in the body. */
/* Chooses the loop each candidate is hoisted out of and pays for the registers
 * busiest first, ONCE, before a line of the body is emitted. Doing it at the
 * loop heads as the walk reaches them spends the pool in program order, which
 * means the outermost loop -- the one whose header load happens least often --
 * takes the registers the innermost one wanted. Weight is the measuring pass's
 * own loop-depth-weighted count of subscript sites, the same currency the
 * local allocator ranks slots in.
 *
 * The loop chosen is the OUTERMOST one the slot is invariant across, so the
 * load runs as rarely as the proof allows. */
void planHoists(Emit *e, ObjFunction *fn) {
    if (e->measuring || !e->osr) return;
    const Chunk *c = &fn->chunk;

    struct { uint32_t top, end, use; uint8_t slot; } cand[JIT_MAX_SLOTS + 1];
    unsigned ncand = 0;

    for (unsigned s = 0; s < e->locals && s <= JIT_MAX_SLOTS; s++) {
        if (e->localKind[s] != SLOT_LIST) continue;
        if (e->slotXReg[s] == 0) continue;   /* no register to load from */
        if (e->slotIndexUse[s] == 0) continue;
        uint32_t bestTop = 0, bestEnd = 0;
        for (int at = (int)e->osrTop; at < (int)e->osrEnd;) {
            int len = instructionLength(c, at);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)at;
            uint32_t le = loopBodyEnd(c, lt);
            at += len;
            if (le == 0 || le <= lt || le > e->osrEnd) continue;
            /* Every subscript of this slot inside the loop... */
            if (e->slotIndexLo[s] < lt || e->slotIndexHi[s] >= le) continue;
            /* ...and no write to it anywhere in the loop. */
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            /* ...and nothing in the loop that could resize the list or take
             * back the registers the header is being put in. */
            if (regionCalls(e, lt, le)) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        cand[ncand].top  = bestTop;
        cand[ncand].end  = bestEnd;
        cand[ncand].use  = e->slotIndexUse[s];
        cand[ncand].slot = (uint8_t)s;
        ncand++;
    }

    while (e->hoistCount < JIT_MAX_HOIST &&
           e->hoistPoolCount - e->hoistTaken >= 2u) {
        unsigned pick = ncand, bestUse = 0;
        for (unsigned i = 0; i < ncand; i++) {
            if (cand[i].use > bestUse) { bestUse = cand[i].use; pick = i; }
        }
        if (pick == ncand) break;
        cand[pick].use = 0;                  /* taken */
        unsigned rI = e->hoistPool[e->hoistTaken++];
        unsigned rC = e->hoistPool[e->hoistTaken++];
        if (rI < e->scratchRoom) e->scratchRoom = rI;
        if (rC < e->scratchRoom) e->scratchRoom = rC;
        e->hoist[e->hoistCount].top      = cand[pick].top;
        e->hoist[e->hoistCount].end      = cand[pick].end;
        e->hoist[e->hoistCount].slot     = cand[pick].slot;
        e->hoist[e->hoistCount].itemsReg = (uint8_t)rI;
        e->hoist[e->hoistCount].countReg = (uint8_t)rC;
        e->hoist[e->hoistCount].rangeOk  = false;
        uint32_t ht = cand[pick].top;
        if (ht + 9u <= (uint32_t)c->count && c->code[ht] == OP_FOR_RANGE_BIND) {
            e->hoist[e->hoistCount].rangeOk = true;
            e->hoist[e->hoistCount].rVar = jaiReadU16(c->code + ht + 3);
            e->hoist[e->hoistCount].rCur = jaiReadU16(c->code + ht + 5);
            e->hoist[e->hoistCount].rEnd = jaiReadU16(c->code + ht + 7);
        }
        e->hoistCount++;
    }
}

/* Whether the loop head's guard already covers this subscript, so the compare
 * and branch here can go.
 *
 * planHoists proves two of the three things that need to be true: it hoists a
 * slot's header only over a region with no write to the slot and no call that
 * could resize the list, which is exactly what makes `count` a loop invariant
 * worth checking once. The third is that the index IS the loop counter plus a
 * constant (Emit::idxKnown), and that the counter is written nowhere but the
 * loop's own head -- `for j in ...` that assigns to `j` inside the body would
 * otherwise index with a value the head's guard never saw.
 *
 * Restricted to a hoist over the OSR loop ITSELF, because the guard reads
 * JIT_IDX_REG and JIT_LIM_REG: an inner loop nested inside the compiled one
 * has its own counter and those registers describe the outer. */
bool boundsCoveredAtHead(const Emit *e, int slot, unsigned vidx,
                                int32_t *offOut, uint8_t *baseOut) {
    if (!e->osr) return false;
    if (slot < 0 || slot > (int)JIT_MAX_SLOTS) return false;
    if ((e->idxKnown & (1u << vidx)) == 0) return false;
    unsigned base = e->idxBase[vidx];
    if (base > JIT_MAX_SLOTS) return false;
    /* The index has to be a loop VARIABLE, written by its head's bind and by
     * nothing else: a body that assigns to `j` indexes with a value no head
     * ever bounded. One write site, and it is a range head. */
    if (e->slotWriteLo[base] != e->slotWriteHi[base]) return false;
    uint32_t bindAt = e->slotWriteLo[base];
    if (bindAt >= (uint32_t)e->chunkDepthCount) return false;
    *offOut  = e->idxOff[vidx];
    *baseOut = (uint8_t)base;
    /* The measuring pass answers "is this index a shape", not "is it covered".
     * Two reasons it cannot answer the second: the span it is being asked
     * about is the one it is still building, and hoists do not exist yet --
     * planHoists runs BETWEEN the passes. Both made an earlier version record
     * nothing at all and measure exactly 1.000x. */
    if (e->measuring) return true;
    if (!e->spanOk[slot] || e->spanLo[slot] > e->spanHi[slot]) return false;
    int h = hoistFor(e, slot);
    if (h < 0 || !e->hoist[h].rangeOk) return false;
    /* The guard is emitted at THIS hoist's loop head, so it is that loop's
     * variable the index has to be measured against. */
    if (e->hoist[h].rVar != base) return false;
    if (*offOut < e->spanLo[slot] || *offOut > e->spanHi[slot]) return false;
    return true;
}

/* The loads themselves, emitted just above the head of the loop they were
 * planned out of -- which is where the walk is when it reaches that offset. */
void emitHoistsAt(Emit *e, uint32_t off) {
    /* `off` is also the resume point for the bounds guard below: the hoists are
     * emitted ABOVE the offset map, so e->curOffset still names the PREVIOUS
     * instruction and a guard taken against it resumes somewhere whose operand
     * model has already been consumed. */
    if (e->inlining) return;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].top != off) continue;
        /* Outside the loop, so the function tier pays its two instructions
         * once per entry rather than per element. */
        if (!e->osr) {
            emitListBoxedGuard(e, e->slotXReg[e->hoist[i].slot], JIT_SCRATCH_A);
        }
        emitListHeader(e, e->slotXReg[e->hoist[i].slot],
                       e->hoist[i].itemsReg, e->hoist[i].countReg);

        /* And, once, the bounds every subscript of this slot inside the loop
         * would otherwise check for itself. The counter runs [IDX, LIM), so
         * the indices reached are [IDX + spanLo, LIM - 1 + spanHi]; proving
         * both ends here lets each site drop a compare and a branch, which is
         * 10 of the ~39 instructions a five-point stencil executes per cell.
         *
         * Skipped when the loop will not run at all: LIM <= IDX makes the
         * lower end of that interval meaningless, and deoptimising an empty
         * loop would hand the whole rest of the function to the interpreter
         * for no reason. */
        unsigned sl = e->hoist[i].slot;
        if (!e->osr || !e->hoist[i].rangeOk) continue;
        if (!e->spanOk[sl] || e->spanLo[sl] > e->spanHi[sl]) continue;
        if (!e->spanSeen[sl] || e->spanBase[sl] != e->hoist[i].rVar) continue;
        /* localIn, not localHomeX: the counter and the end are temporaries the
         * emitter hands out per loop, and they often miss out on a register
         * entirely. A load apiece is nothing here -- this runs once per entry
         * to the compiled loop, not once per iteration -- and requiring homes
         * made the guard decline on every stencil that has one. */
        unsigned rCur = localIn(e, e->hoist[i].rCur, JIT_SCRATCH_B);
        unsigned rEnd = localIn(e, e->hoist[i].rEnd, JIT_SCRATCH_C);

        /* The counter has not necessarily started at the range's own start --
         * OSR is entered on a back edge, so the loop may be half run -- but
         * that only narrows the interval, and narrowing is sound. Skipped
         * entirely when nothing is left to run: deoptimising an empty loop
         * would hand the rest of the function to the interpreter for nothing. */
        emit(e, jaiA64SubsXReg(31, rCur, rEnd));
        unsigned empty = e->count;
        emit(e, jaiA64BCond(JAI_A64_GE, 0));

        /* emitAddSubImm carries the sign itself, so the span goes in as it
         * stands: cur + spanLo is the first index the loop still reaches, and
         * end - 1 + spanHi the last. */
        emitAddSubImm(e, JIT_SCRATCH_A, rCur, (int64_t)e->spanLo[sl], false);
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
        branchOnDeoptAt(e, JAI_A64_LT, off, false);

        emitAddSubImm(e, JIT_SCRATCH_A, rEnd, (int64_t)e->spanHi[sl] - 1,
                      false);
        emit(e, jaiA64SubsXUxtw(31, JIT_SCRATCH_A, e->hoist[i].countReg));
        branchOnDeoptAt(e, JAI_A64_HS, off, false);

        if (empty < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[empty] = jaiA64BCond(JAI_A64_GE,
                                         (int32_t)(e->count - empty));
        }
    }
}

/* Normalises the index and bounds-checks it in one unsigned compare: a negative index is a huge
 * unsigned value and fails the same test as one past the end. `countW` means rCount's low half only, as an ObjList header `ldp` leaves it (`count | capacity << 32`) -- every read goes through uxtw; an ObjString length (no capacity above it) passes false and uses the plain register form instead. */
void emitBoundsNormalise(Emit *e, unsigned rIdx, unsigned rCount,
                                unsigned rOut, bool countW) {
    emit(e, jaiA64MovX(rOut, rIdx));
    emit(e, countW ? jaiA64SubsXUxtw(31, rOut, rCount)
                   : jaiA64SubsXReg(31, rOut, rCount));
    /* Skip length used to be hand-counted (four instructions) -- branchOnDeopt may itself emit an FP-
     * borrow release, and one extra instruction inside the span turned the skip into a jump onto the bail branch (same matrix_mul `sum`-from-nothing bug as deoptRecordAt). Measured with `e->count`, so it can't rot. */
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_LO, 0));
    emit(e, countW ? jaiA64AddXUxtw(rOut, rOut, rCount)
                   : jaiA64AddX(rOut, rOut, rCount));
    emit(e, countW ? jaiA64SubsXUxtw(31, rOut, rCount)
                   : jaiA64SubsXReg(31, rOut, rCount));
    branchOnDeopt(e, JAI_A64_HS);
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_LO, (int32_t)(e->count - skip));
    }
}

/* An overflow stub reads no operand-stack entry -- it raises -- so the entries
 * do NOT have to be in their X homes to branch to one. The fpSyncAll it used
 * to do was pure hot-path cost: in a float expression carrying an int guard
 * through it (a stencil's `mid[j-1]`), each one wrote every live float out and
 * the next use read it back, four cross-bank fmovs sitting in the middle of the
 * accumulate chain. A join (branchTo) and a deopt record are different and
 * still settle -- the first because the other edge must agree, the second
 * because the record is read.
 *
 * There used to be a `branchOnCondition` here that took a body to its BAIL
 * block on a condition, and one arm used it: an indirect call whose callee
 * came back with a non-zero verdict. That arm takes a deopt at the call offset
 * now (see it for why a bail was unsound in the OSR tier), which leaves the
 * entry stack-limit guard as the only thing that can reach the bail block --
 * and that fires before a single body instruction runs. So a bail can no
 * longer follow a write of any kind, and the `bailAfterWrite` decline that
 * guarded against it went with the function. */

/* `cond` isn't always VS: adds/subs set the overflow flag, but the multiply test compares the
 * product's high half against the low half's replicated sign, so its answer is NE -- routing multiply through VS meant its overflow was never detected (4 * 2^62 silently came back as 0). */
void branchOnOverflow(Emit *e, unsigned which, unsigned cond) {
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return; }
    /* Inside a `try` the overflow stub's raise would unwind past the region
     * that should have caught it (see Emit::inProtected). Resume at this
     * instruction instead: the interpreter re-executes it, overflows too, and
     * raises with a frame and an ip that name the right handler. The deopt
     * record IS read, so that path settles the FP bank; the overflow stub is
     * not, which is why the unconditional fpSyncAll above it is gone. */
    if (e->inProtected) {
        fpSyncAll(e);
        branchOnDeoptInstStart(e, cond);
        return;
    }
    e->overflowUsed[which] = true;
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_OVF - which;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(cond, 0));
}

/* Where an integer arm that can overflow computes its result.
 *
 * Inside a `try` the overflow guard resumes at the instruction (see
 * branchOnOverflow), so the result must not have reached its home yet -- the
 * interpreter would otherwise apply the operation a second time on top of the
 * wrapped value, and `total += x` inside a caught `try` would come back wrong.
 * A scratch keeps every canonical register untouched until the guard is past;
 * the copy out is a `mov`, which §5 of the roadmap prices at zero on this core,
 * and it only appears inside a protected region at all. */
unsigned ovfDest(const Emit *e, unsigned home) {
    return e->inProtected ? JIT_SCRATCH_B : home;
}

/* Whether a raise that leaves compiled code with the exception pending can be
 * emitted here. It cannot inside a `try`: the effects already happened, so the
 * site cannot resume at its instruction the way an overflow can, and the
 * unwinder would consult an offset outside the protected region. Declines --
 * which is exactly what the whole function did before OP_GET_EXC had a deopt,
 * so no shape that used to compile stops. */
bool raiseExitAllowed(Emit *e, const char *what) {
    if (!e->inProtected) return true;
    e->whyNot = what;
    e->failed = true;
    return false;
}

/* The condition to branch on when the comparison is FALSE: the opcode jumps
 * over the taken side, `if (!taken) ip += offset`. */
bool negatedCondition(uint8_t cmp, unsigned *out) {
    switch (cmp) {
    case OP_EQ: *out = JAI_A64_NE; return true;
    case OP_NE: *out = JAI_A64_EQ; return true;
    case OP_LT: *out = JAI_A64_GE; return true;
    case OP_LE: *out = JAI_A64_GT; return true;
    case OP_GT: *out = JAI_A64_LE; return true;
    case OP_GE: *out = JAI_A64_LT; return true;
    default: return false;
    }
}

ObjClass *globalClass(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    return IS_CLASS(bound) ? AS_CLASS(bound) : NULL;
}

/* The first entry a dict walk would yield, for the component kinds the compiled
 * pair head specialises on -- the dict equivalent of taking items[0] off a list.
 * False for a dict with nothing live in it, which declines rather than guessing.
 * Reads the order array exactly as jaiTableNext does, so "first" here and
 * "first" at run time are the same entry. */
bool firstLiveEntry(const JaiTable *t, Value *key, Value *value) {
    if (t->entries == NULL) return false;
    for (int i = 0; i < t->orderCount; i++) {
        const int32_t slot = t->order[i];
        if (slot < 0) continue;
        *key   = t->entries[slot].key;
        *value = t->entries[slot].value;
        return true;
    }
    return false;
}

ObjFunction *globalFunction(ObjClosure *closure, uint32_t nameIdx,
                                   Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return NULL;
    if (!IS_CLOSURE(bound)) return NULL;
    *out = bound;
    return AS_CLOSURE(bound)->fn;
}

/* Resolved the way the interpreter resolves a builtin: the module first, `vm.builtins` only when the
 * module has no such name -- a module-level binding is never mistaken for it, and if one appears later the module's version retires this compiled form. */
ObjNative *globalNative(ObjClosure *closure, uint32_t nameIdx,
                               Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL || vm.builtins == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value shadow;
    if (jaiModuleGet(fn->module, AS_STRING(name), &shadow)) return NULL;
    Value bound;
    if (!jaiModuleGet(vm.builtins, AS_STRING(name), &bound)) return NULL;
    if (!IS_NATIVE(bound)) return NULL;
    *out = bound;
    return AS_NATIVE(bound);
}

/* `__prim__`, resolved the same way globalNative resolves a bare builtin --
 * the module first, so a real user binding is never mistaken for it -- except
 * what sits at `vm.builtins`'s name is an ObjModule (a native namespace:
 * jaiDefineNative's dotted names build one the first time a "ns.leaf" name
 * registers, in namespaceFor/makeNamespace, builtins.c), not an ObjNative.
 * `math.sqrt`'s own body is `__prim__.f64_sqrt(x)` -- lib/std/math.jai:218 --
 * so this is not a hypothetical name, it is the one OP_GET_GLOBAL's own
 * refusal message already names as "not a compiled global function" on every
 * `__prim__.f64_*` body in the tree.
 *
 * Resolved BY VALUE, same as globalNative: nothing here is re-checked at run
 * time beyond fn->module->version at entry (guards a later shadow), so this
 * carries exactly the soundness globalNative already carries for a bare
 * builtin -- no better, no worse. What DOES need a per-call guard is the
 * MEMBER read through this namespace afterward, because `mod.attr = v` is
 * real syntax for any module receiver (jaiSetProperty's IS_MODULE arm) and
 * `__prim__` is reachable as a bare identifier -- see emitModuleNativeCall's
 * m->version check for that half. */
ObjModule *globalNamespace(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL || vm.builtins == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    Value shadow;
    if (jaiModuleGet(fn->module, AS_STRING(name), &shadow)) return NULL;
    Value bound;
    if (!jaiModuleGet(vm.builtins, AS_STRING(name), &bound)) return NULL;
    if (!IS_MODULE(bound)) return NULL;
    return AS_MODULE(bound);
}

bool globalIsSelf(ObjClosure *closure, uint32_t nameIdx) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return false;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return false;

    Value bound;
    if (!jaiModuleGet(fn->module, AS_STRING(name), &bound)) return false;
    return IS_CLOSURE(bound) && AS_CLOSURE(bound)->fn == fn;
}

/* ------------------------------------------------------------------ */
/* Module globals                                                       */
/* ------------------------------------------------------------------ */

/* A JaiEntry's address is stable as long as the table doesn't rehash/delete/clear -- overwriting an
 * existing global never moves it (ensureRoom only runs for a NEW key) -- which is what lets a compiled load be one `ldr` from a baked pointer. `keyVersion` counts every event that breaks this; emitGlobalsGuard checks it. */
JaiEntry *globalSlot(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                            Value *out) {
    ObjFunction *fn = closure->fn;
    if (fn->module == NULL) return NULL;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return NULL;
    Value name = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(name)) return NULL;
    JaiTable *t = &fn->module->globals;
    JaiEntry *slot = jaiTableFindEntryInterned(t, AS_STRING(name));
    if (slot == NULL) return NULL;
    /* The whole body reads one table, so one guard covers every slot. */
    if (e->globalsTable == NULL) {
        e->globalsTable = t;
        e->globalsKeyVersion = t->keyVersion;
    } else if (e->globalsTable != t) {
        return NULL;
    }
    if (out != NULL) *out = slot->value;
    return slot;
}

/* Emitted before EVERY access, not hoisted: hoisting is sound only given a control-flow claim (no
 * call-out between a guard and a later access on a back edge) -- exactly the kind of reasoning this file has been bitten by before. Costs four instructions on a predictable branch. */
void emitGlobalsGuard(Emit *e) {
    uint32_t at = e->globalsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->globalsTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* JAITHON_JIT_STATIC_FIELD=0 turns the SLOT_CLASS arm of OP_GET_FIELD off, so
 * the same binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jitPicEnabled's JAITHON_JIT_PIC. Read once: OP_GET_FIELD is hot enough that
 * an uncached getenv on every static access would be its own cost. */
bool jitStaticFieldEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_STATIC_FIELD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* `klass->statics` is a JaiTable exactly like a module's globals table (same
 * struct, same keyVersion), so a static field is read the same way a module
 * global is read BY ADDRESS above: bake the JaiEntry*, not the value. A
 * static is reassignable at runtime -- jaiSetProperty's IS_CLASS arm (vm.c)
 * takes any value with no isLet check, and the checker's own
 * _check_field_assign skips its immutability error whenever `static_access`
 * is true, so `let` buys no promise here that the interpreter or the checker
 * actually keeps. Nothing below may bake the VALUE, only its address, behind
 * the same two guards a module global stands on. One table per body, same
 * plan as globalsTable -- see that field's comment on the struct. */
JaiEntry *staticFieldSlot(Emit *e, ObjClass *klass, ObjString *name) {
    JaiTable *t = &klass->statics;
    JaiEntry *slot = jaiTableFindEntryInterned(t, name);
    if (slot == NULL) return NULL;
    if (e->staticsTable == NULL) {
        e->staticsTable = t;
        e->staticsKeyVersion = t->keyVersion;
    } else if (e->staticsTable != t) {
        return NULL;
    }
    return slot;
}

/* A member of an imported module, resolved to the entry's address. The VALUE
 * is never baked, only the address, behind the same keyVersion guard a global
 * stands on -- a module global is assignable, so the tag is re-checked at every
 * read and a rebind deoptimises. */
JaiEntry *moduleMemberSlot(Emit *e, ObjModule *m, ObjString *name) {
    JaiTable *t = &m->globals;
    JaiEntry *slot = jaiTableFindEntryInterned(t, name);
    if (slot == NULL) return NULL;
    if (e->modTable == NULL) {
        e->modTable = t;
        e->modKeyVersion = t->keyVersion;
    } else if (e->modTable != t) {
        return NULL;
    }
    return slot;
}

void emitModuleGuard(Emit *e) {
    uint32_t at = e->modKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->modTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* Off leaves an observed list return as SLOT_OBJ, so the promotion can be
 * measured apart from the probe that precedes it. */
static bool retListKindOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RET_LIST");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off refuses a predicted-list receiver again, so the probe and the plumbing
 * that feeds it can be measured apart. */
bool listProbeOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_LIST_PROBE");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off drops the callee's observed object type again, so the difference is
 * measurable in one binary. */
static bool retObjTypeOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_RET_OBJTYPE");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Off puts the refusal back, so a body reading `math.PI` can be measured both
 * ways in one binary. */
bool moduleFieldOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_MODULE_FIELD");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* emitGlobalsGuard's counterpart for e->staticsTable: not hoisted, for the
 * same reason. */
void emitStaticsGuard(Emit *e) {
    uint32_t at = e->staticsKeyVersion;
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&e->staticsTable->keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    if (at <= 0xfffu) {
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, at));
    } else {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)at);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    }
    branchOnDeopt(e, JAI_A64_NE);
}

/* `m->version` retires every cache that resolved a NAME to a heap object or presence (interpreter's
 * global inline cache, OP_FORMAT's builtin-str check, this tier's baked classes/closures/natives) -- all of which required the bound value to BE a heap object. A store replacing a non-object with a non-object changes none of them and may skip the bump; getting this wrong made `total = total + 1` at module scope retire every compiled function in the module. */
void emitVersionBump(Emit *e, ObjModule *m) {
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}

/* Predicted stack kind + guard tag for a recorded OP_INVOKE result; false when the tier has no use
 * for the byte. SLOT_INST is deliberately excluded: it carries a class shape this byte can't encode, and admitting it would silently lose the shape every field offset was resolved against. Class/closure/native excluded too -- they have register-free stack kinds of their own. */
/* What an InlineCache::resultKind byte says, for JAI_JIT_WHY. A site the tier
 * refuses for want of a result kind is refused for one of three quite different
 * reasons, and the fix differs for each. */
const char *jaiFeedbackName(uint8_t fb) {
    if (fb == JAI_FB_NONE) return "never observed";
    if (fb == JAI_FB_MIXED) return "mixed";
    if (fb >= JAI_FB_OBJ && fb < JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return jaiObjTypeName((ObjType)(fb - JAI_FB_OBJ));
    }
    switch ((ValueType)(fb - 1u)) {
    case VAL_NULL:  return "null";
    case VAL_BOOL:  return "bool";
    case VAL_INT:   return "int";
    case VAL_FLOAT: return "float";
    default: break;
    }
    return "something the tier does not name";
}

/* `objType` reports the ObjType the feedback named, as ObjType + 1, or 0 when
 * the result is not an object. See Emit::stackObjType for what it is for. */
bool feedbackSlotKind(uint8_t fb, SlotKind *k, unsigned *tag,
                             uint8_t *objType) {
    *objType = 0;
    switch (fb) {
    case 1u + VAL_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case 1u + VAL_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case 1u + VAL_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    default: break;
    }
    if (fb < JAI_FB_OBJ || fb >= JAI_FB_OBJ + (unsigned)OBJ_TYPE_COUNT) {
        return false;
    }
    switch ((ObjType)(fb - JAI_FB_OBJ)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    /* Anything else on the heap can be loaded, passed and stored and nothing
     * else, which is exactly SLOT_OBJ. The tag guard below is the whole of
     * what makes that sound: whatever object comes back, it is an object. */
    *k = SLOT_OBJ; *tag = VAL_OBJ;
    *objType = (uint8_t)(fb - JAI_FB_OBJ + 1u);
    return true;
}

/* The store half of `list.push` and of a comprehension's append: they differ
 * only in where the list sits on the stack and in what is left behind, so the
 * bounds check, the grow fixup and the two stores live here. Returns false
 * with `whyNot` set when the value's kind has no tag to store.
 *
 * Appending is a bounds check and two stores -- a descriptor+native round trip
 * costs far more than the work itself (list_ops spent all its time on the
 * call). A full list goes out to the `grow` stubs' realloc helper and comes
 * straight back; see there for why this used to be a deopt and what it cost. */
bool emitListStore(Emit *e, SlotKind vk, unsigned rList, unsigned rVal,
                          int slot) {
    unsigned vtag = vk == SLOT_INT   ? VAL_INT
                  : vk == SLOT_FLOAT ? VAL_FLOAT
                  : vk == SLOT_BOOL  ? VAL_BOOL
                  : (vk == SLOT_INST || vk == SLOT_LIST ||
                     vk == SLOT_OBJ)  ? VAL_OBJ
                                      : 0xffffffffu;
    if (vtag == 0xffffffffu) {
        e->whyNot = "pushing a kind the tier cannot store";
        return false;
    }

    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    /* jitListGrow only reserves, and jaiListReserve is width-aware, so the
     * growth half of this needs nothing; it is the store below that has to
     * know how wide an element is. */
    ListAccess pAcc = listAccessFor(e, rList, slot, vk, JIT_SCRATCH_A);
    if (!pAcc.dynamic && pAcc.stg != LIST_STORE_BOXED &&
        vk != listStgKind(pAcc.stg)) {
        return subWhy(e, "pushing kind %d onto storage %u", (int)vk, pAcc.stg);
    }

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_B, rList,
                       (unsigned)offsetof(ObjList, capacity)));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    if (e->growCount >= JIT_MAX_GROW) {
        e->whyNot = "more list pushes than the tier tracks";
        return false;
    }
    /* jitListGrow can raise, and the stub routes that to the exception exit
     * (see emitGrowStubs). */
    if (!raiseExitAllowed(e, "a list growth inside a try")) return false;

    noteScratchClobber(e);
    unsigned gi = e->growCount++;
    e->grow[gi].listReg  = rList;
    e->grow[gi].valReg   = rVal;
    e->grow[gi].tag      = vtag;
    e->grow[gi].countReg = JIT_SCRATCH_A;
    e->grow[gi].stub     = -1;
    if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
    e->fixups[e->fixupCount].instIndex    = (int)e->count;
    e->fixups[e->fixupCount].targetOffset = FIXUP_GROW - gi;
    e->fixups[e->fixupCount].conditional  = true;
    e->fixups[e->fixupCount].depth        = -1;
    e->fixupCount++;
    emit(e, jaiA64BCond(JAI_A64_GE, 0));
    e->grow[gi].returnTo = (int)e->count;

    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rList,
                       (unsigned)offsetof(ObjList, items)));
    /* JIT_SCRATCH_C is the items pointer and JIT_SCRATCH_A the index; every
     * arm below starts from those two, so the test costs a load, a compare and
     * two branches and touches nothing else. */
    int pSkip = listDispatchBegin(e, &pAcc, rList, JIT_SCRATCH_D);
    emitListElemStore(e, pAcc.stg, vtag, rVal);
    if (pSkip >= 0) {
        int pJoin = listDispatchElse(e, pSkip);
        emitListElemStore(e, pAcc.alt, vtag, rVal);
        listDispatchEnd(e, pJoin);
    }
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, count)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
    emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                       (unsigned)offsetof(ObjList, version)));
    e->wroteHeap = true;
    return true;
}

/* What an OP_INVOKE site has been observed to return, merged over every way its
 * inline cache holds.
 *
 * For a site whose receiver class the model cannot pin there is no callee to
 * ask -- but the interpreter watched the same site run, and eight
 * implementations of one trait method that all return an int agree on that
 * much. Ways that recorded nothing are skipped rather than merged: NONE
 * against a real kind is MIXED, which would throw away the evidence the other
 * ways did gather.
 *
 * A PREDICTION. The tag guard the caller emits after the call is the whole of
 * what makes it sound; a way that later returns something else deoptimises. */
bool siteInvokeResultKind(const Chunk *chunk, uint16_t cacheIdx,
                                 SlotKind *k, unsigned *tag) {
    if (chunk->caches == NULL || (int)cacheIdx >= chunk->cacheCount) {
        return false;
    }
    const InlineCache *ic = &chunk->caches[cacheIdx];
    uint8_t merged = JAI_FB_NONE;
    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        if (ic->resultKind[w] == JAI_FB_NONE) continue;
        merged = jaiFeedbackMerge(merged, ic->resultKind[w]);
    }
    if (merged == JAI_FB_NONE || merged == JAI_FB_MIXED) return false;
    uint8_t objType;
    return feedbackSlotKind(merged, k, tag, &objType);
}

/* The same test feedbackSlotKind applies to a call's result, as a predicate on a Value: is this a heap
 * object with no stack kind of its own, so SLOT_OBJ's "read, pass, store, root and nothing else" describes it exactly? Excludes the kinds a later arm resolves to a register-free entry (class/closure/native/...), which SLOT_OBJ would silently outrank. */
bool rawObjValue(Value v) {
    if (!IS_OBJ(v) || AS_OBJ(v) == NULL) return false;
    switch (OBJ_TYPE(v)) {
    case OBJ_INSTANCE: case OBJ_CLASS: case OBJ_TRAIT:
    case OBJ_CLOSURE:  case OBJ_FUNCTION: case OBJ_NATIVE:
    case OBJ_BOUND:    case OBJ_ENUM: case OBJ_ENUM_CTOR:
        return false;
    default: break;
    }
    return true;
}

/* Predicted stack kind for a callee that has not compiled, from what it has been observed to return
 * (ObjFunction::obsReturnKind). Same contract as feedbackSlotKind's: a prediction the caller must guard.
 * SLOT_INST is admissible here where it is not there, because a per-callee record can carry the class
 * shape a one-byte-per-way cache cannot -- and the shape is guarded after the call like the tag. */
/* `objType` may be NULL. It is ObjType + 1 when the record says the callee
 * returns a particular heap object, and 0 otherwise.
 *
 * It used to be computed and dropped on the floor -- feedbackSlotKind filled a
 * local that nothing read. So a callee OBSERVED to return a list handed its
 * caller SLOT_OBJ and NO type, and `w.len()` on the result then refused with
 * "an object with no sample and no known type" even though the record said
 * plainly that it was a list. That refusal was 18.4% of the interpreted work
 * on the self-hosted compiler, the largest by a factor of seven. */
bool observedReturnKind(const ObjFunction *cfn, SlotKind *k,
                               uint32_t *shape, uint8_t *objType) {
    uint8_t fb = cfn->obsReturnKind;
    *shape = 0;
    if (objType != NULL) *objType = 0;
    if (fb == 1u + (unsigned)VAL_NULL) { *k = SLOT_NULL; return true; }
    if (fb == JAI_FB_OBJ + (unsigned)OBJ_INSTANCE) {
        if (cfn->obsReturnShape == 0) return false;
        *k = SLOT_INST;
        *shape = cfn->obsReturnShape;
        return true;
    }
    unsigned tag;
    uint8_t seenType = 0;
    if (!feedbackSlotKind(fb, k, &tag, &seenType)) return false;
    if (objType != NULL) *objType = seenType;
    /* A list earns the stronger kind here, where a per-way cache's byte cannot:
     * this is the per-callee record, the same reason SLOT_INST is admissible
     * above. It costs nothing to guard -- emitCallOutResult already emits the
     * OBJ_LIST check for SLOT_LIST -- and it is the difference between `w[2]`
     * compiling and refusing, since a subscript wants a list and SLOT_OBJ is
     * only "some object".
     *
     * `_fuse_at` is `let w = window_of(...)` then `w.len()` and `w[1]`,
     * `w[2]`, `w[3]`. Answering the method lookup alone left every subscript
     * still refusing; this is the other half of that pair. */
    if (retListKindOn() && *k == SLOT_OBJ &&
        seenType == (uint8_t)(OBJ_LIST + 1)) {
        *k = SLOT_LIST;
    }
    return true;
}

bool globalKind(Value v, SlotKind *k, uint32_t *shape, ObjClass **kls) {
    *shape = 0; *kls = NULL;
    if (IS_INT(v))      { *k = SLOT_INT;   return true; }
    if (IS_FLOAT(v))    { *k = SLOT_FLOAT; return true; }
    if (IS_BOOL(v))     { *k = SLOT_BOOL;  return true; }
    if (IS_LIST(v))     { *k = SLOT_LIST;  return true; }
    if (IS_INSTANCE(v)) {
        ObjInstance *inst = AS_INSTANCE(v);
        if (inst->klass == NULL) return false;
        *k = SLOT_INST; *kls = inst->klass; *shape = inst->klass->shapeId;
        return true;
    }
    /* Anything else on the heap (dict, string, closure) can be loaded, passed and stored, nothing else.
 * A class, function or native never reach here: OP_GET_GLOBAL resolves those to their own register-free stack kinds first. */
    if (IS_OBJ(v) && AS_OBJ(v) != NULL && !IS_CLASS(v) && !IS_CLOSURE(v) &&
        !IS_NATIVE(v)) {
        *k = SLOT_OBJ; return true;
    }
    return false;
}

/* A scalar field's DECLARED kind (FieldInfo::typeId, OP_FIELD_DEF's bits 4-7,
 * spec Sec3.7), for a receiver with a pinned class but no sample Value to read
 * `inst->fields[slot]` off -- OP_GET_FIELD's twin of OP_ELEM_KIND's own use of
 * the same bits (see that case) rather than a sampled container.
 *
 * Not a new promise: OP_SET_FIELD's guard (jaiKindAccepts, vm.c) already
 * refuses any store that disagrees with this field's declared kind, so a
 * caller here is only reading a fact the runtime enforces on every write, and
 * the tag is checked again at the load below regardless -- a stale or wrong
 * record still deopts rather than answers.
 *
 * INT/FLOAT/BOOL only. FIELD_KIND_LIST names the box but not the element, so
 * admitting it here would still leave a consumer that iterates the field with
 * nothing to look at (the "iterating a list with nothing to look at" refusal,
 * unresolved either way); FIELD_KIND_INSTANCE names no specific class to guard
 * against; FIELD_KIND_ANY and FIELD_KIND_STR/DICT promise nothing scalar. All
 * four are left to the sampled path, unchanged. */
bool declaredScalarFieldKind(uint32_t typeId, SlotKind *k, unsigned *tag) {
    switch (typeId) {
    case FIELD_KIND_INT:   *k = SLOT_INT;   *tag = VAL_INT;   return true;
    case FIELD_KIND_FLOAT: *k = SLOT_FLOAT; *tag = VAL_FLOAT; return true;
    case FIELD_KIND_BOOL:  *k = SLOT_BOOL;  *tag = VAL_BOOL;  return true;
    /* A declared `list[T]` field. VAL_OBJ is every heap object, so the caller
     * must ALSO prove OBJ_LIST before anything reads ObjList's header off it --
     * the same hazard that segfaulted the VM through the dict-index arm. It is
     * separated from the three scalars above because it is the only kind here
     * that needs a second guard.
     *
     * Worth predicting because a list field is where a chain bottoms out:
     * `code.data[i]` is a field read the tier could not classify, and behind
     * that one refusal sat eleven functions and 14.27% of one file's
     * interpreted work. */
    /* Predicts that the field IS a list, and cannot predict what is IN it: the
     * index arm below wants an element exemplar and a prediction has no value
     * to take one from. So `d.items` compiles and `d.items[i]` still declines,
     * which is why clearing this link alone did not free `_fuse_at`. */
    case FIELD_KIND_LIST:  *k = SLOT_LIST;  *tag = VAL_OBJ;   return true;
    /* A declared `str`. Same two-guard shape as the list above, and worth its
     * own row because a string entry carrying a SAMPLE unlocks the arms below
     * it -- `.len()`, `==` on interned pointers, the ordering leaf call -- all
     * of which ask `stringOperand`, which asks the sample and not the kind. */
    case FIELD_KIND_STR:   *k = SLOT_OBJ;   *tag = VAL_OBJ;   return true;
    /* A user type name. VAL_OBJ is exactly what it promises and exactly what
     * the tag guard proves, so no second guard is wanted -- unlike the list and
     * str rows above, this one does not claim a particular ObjType.
     *
     * A field holding NULL deopts here, which is the honest cost: a field
     * declared `Foo` is null only before its init assigns it, and a field never
     * assigned at all would deopt on every read rather than answer wrongly. */
    case FIELD_KIND_DECLARED: *k = SLOT_OBJ; *tag = VAL_OBJ;   return true;
    default: return false;
    }
}

/* JAITHON_JIT_FIELD_DECL_KIND=0 turns declaredScalarFieldKind's OP_GET_FIELD
 * arm off, reproducing the pre-fix decline for an A/B inside one binary --
 * same cached-getenv idiom as jitDeoptStress above, but default ON since this
 * is a fix, not a stress knob. */

/* JAI_JIT_CHAIN=1: print the whole chain of refusals a body would hit, not just
 * the first one.
 *
 * "What would this body stop at NEXT?" is the question that decides whether an
 * arm is worth building, and until now it was answered by BUILDING the arm and
 * re-running -- a day per link, and how three separate changes came to measure
 * exactly zero after clearing one link of a longer chain.
 *
 * The mechanism is deliberately dumb: recompile the body with the offending
 * offset forced onto the unarmed path, and see what it says next. That reuses a
 * path the tier already exercises constantly, rather than continuing a walk
 * whose model has gone inconsistent -- which was tried, and segfaults.
 *
 * Diagnostic only. Each link costs one extra compile of one body, and nothing
 * here runs unless the env var is set. */
bool jitChainOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_CHAIN");
        cached = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

void reportChain(const Emit *proto, Emit *first, ObjClosure *closure,
                        ObjFunction *fn) {
    static Emit probe;
    uint32_t skips[JIT_MAX_CHAIN];
    unsigned n = 0;
    const char *name = fn->name != NULL ? fn->name->chars : "<anon>";

    fprintf(stderr, "[jit] chain %s:\n", name);
    /* Link 1 is a fact. Everything below it is a PROBE, and the probe is not
     * the same thing as a fix.
     *
     * Stepping over an instruction takes the unarmed path, which abandons the
     * rest of that straight-line block and can only resume at a later
     * independently-reachable jump target. If a bind lived in the abandoned
     * part, the walk reaches the next link with that local never bound at all
     * -- and then reports a refusal ("local N has kind int, not instance")
     * that a genuinely fixed link 1 would never have produced. One night's
     * `.len()` chain read that way and the link 2 it named was an artefact.
     *
     * So: chase link 1. Treat the rest as a hint about where to look next,
     * never as a list of things that must all be cleared. */
    fprintf(stderr, "[jit]   (link 1 is measured; the links below are probed "
                    "by forcing it unarmed,\n[jit]    which skips the rest of "
                    "its block -- treat them as hints, not facts)\n");
    fprintf(stderr, "[jit]   1. %s  (at %u)\n", declineReason(first),
            first->curOffset);
    skips[n++] = first->curOffset;

    for (unsigned link = 2; link <= JIT_MAX_CHAIN; link++) {
        memcpy(&probe, proto, sizeof probe);
        memcpy(probe.chainSkip, skips, n * sizeof skips[0]);
        probe.chainSkipCount = n;
        if (compileBody(&probe, closure)) {
            fprintf(stderr, "[jit]   %u. compiles, once the %u above %s "
                            "cleared\n", link, n, n == 1 ? "is" : "are");
            return;
        }
        /* Refusing again at the SAME offset means the unarmed path cannot step
         * over that instruction: deoptSite has nowhere to resume, which is a
         * real property of the instruction and not an artefact of this probe.
         * Say so rather than numbering it as the next link, because it is not
         * one -- it is where the walk stops being able to look. */
        if (probe.curOffset == skips[n - 1]) {
            fprintf(stderr,
                    "[jit]   ... cannot look past link %u: stepping over it "
                    "gives \"%s\"\n", n, declineReason(&probe));
            return;
        }
        fprintf(stderr, "[jit]   %u. %s  (at %u)\n", link,
                declineReason(&probe), probe.curOffset);
        if (n >= JIT_MAX_CHAIN) {
            fprintf(stderr, "[jit]   ... and the chain runs longer than %u\n",
                    JIT_MAX_CHAIN);
            return;
        }
        skips[n++] = probe.curOffset;
    }
}

bool jitDeclaredFieldKindEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FIELD_DECL_KIND");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A callee that is not this function: only a class, whose result is an
 * instance of a shape known here. Anything else would need a guard on a return
 * value nothing can predict. */
/* Recognises an initializer that does nothing but store its arguments into fields in order
 * (`GET_LOCAL2 0 k; SET_FIELD f` repeated, then RETURN_NULL) -- anything else (a default, a computed field, a call, a branch) goes the long way. Lets `Point(a, b)` become an allocation and two stores instead of a descriptor + jaiCallValue + invokeCallable's type switch + a compiled init. */
static bool simpleInitFields(ObjClass *cls, unsigned argc, uint16_t *slots) {
    Value initv;
    if (cls == NULL) return false;
    if (!jaiClassFindMethod(cls, vm.strInit, &initv)) return false;
    if (!IS_CLOSURE(initv)) return false;
    ObjFunction *ifn = AS_CLOSURE(initv)->fn;
    if (ifn->arity != argc || ifn->defaultCount != 0) return false;
    if (ifn->flags & (FN_VARIADIC | FN_KWREST)) return false;
    if (ifn->upvalueCount != 0) return false;

    const uint8_t *c = ifn->chunk.code;
    int n = ifn->chunk.count;
    int off = 0;
    for (unsigned i = 0; i < argc; i++) {
        if (off + 5 > n || c[off] != OP_GET_LOCAL2) return false;
        if (jaiReadU16(c + off + 1) != 0) return false;
        if (jaiReadU16(c + off + 3) != i + 1) return false;
        off += 5;
        if (off + 6 > n || c[off] != OP_SET_FIELD) return false;
        uint32_t nameIdx = jaiReadU24(c + off + 1);
        if (nameIdx >= (uint32_t)ifn->chunk.constants.count) return false;
        Value nm = ifn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nm)) return false;
        const FieldInfo *fi = jaiClassFieldInfo(cls, AS_STRING(nm));
        if (fi == NULL || fi->isStatic) return false;
        slots[i] = fi->slot;
        off += 6;
    }
    return off < n && c[off] == OP_RETURN_NULL;
}

bool isClassCallee(const Emit *e, unsigned argc) {
    return e->depth >= argc + 1u &&
           e->stack[e->depth - argc - 1] == SLOT_CLASS;
}

/* Which local the tier could not settle on one kind for.
 *
 * The reason on its own says a body has such a local but not which, and a
 * body with twenty of them then has to be read line by line to find it. The
 * slot number is what the disassembly labels its locals with, so the two can
 * be put side by side.
 */
/* Record which unnamed refusal an arm took, and return false so the call sites
 * read as `return subWhy(e, "...")`. See Emit::whySub. */
bool subWhy(Emit *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->whySub, sizeof e->whySub, fmt, ap);
    va_end(ap);
    return false;
}

/* What `JAI_JIT_WHY` prints: the named reason when there is one, otherwise the
 * opcode with whatever the arm noted about it. */
const char *declineReason(Emit *e) {
    if (e->whyNot != NULL) return e->whyNot;
    const char *name = jaiOpName((OpCode)e->lastOp);
    if (e->whySub[0] == '\0') return name;
    snprintf(e->whyBuf, sizeof e->whyBuf, "%s: %s", name, e->whySub);
    return e->whyBuf;
}

bool jitCollectClashes(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_COLLECT_CLASHES");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

const char *kindClash(Emit *e, unsigned slot) {
    /* WHICH two kinds, not just which slot. The slot number says where to look
     * and the pair says what to do about it: an int meeting a float is a
     * widening the tier could learn, an instance meeting a list is a genuinely
     * polymorphic local and nothing will help it. Without the pair, ~290 of
     * these across four compiler files were one undifferentiated heap. */
    snprintf(e->whyBuf, sizeof e->whyBuf,
             "local %u was given two kinds, %s and %s", slot,
             slotKindName(e->localKind[slot]), slotKindName(e->clashKind));
    return e->whyBuf;
}


/* ownStatus: caller decodes the helper's return itself, skipping the built-in "nonzero means raised"
 * test. Written for the iterator step (0 yielded, 1 exhausted, 2 raised), whose call-out the list arm of
 * OP_FOR_ITER_BIND no longer makes -- the default test sent `exhausted` to the throw stub, which found no pending exception and died on "internal error: failed operation raised nothing". Kept because any helper with a three-way answer needs it, and because the lesson is not rediscoverable from the code. */
/* Root-fills the descriptor: shared by the descriptor path (a C helper pushes them) and the self-call
 * path (the emitted code links the descriptor onto the collector's frame chain instead, since a bare `bl` pushes nothing). */
/* JAITHON_JIT_SHAPE_LIMIT=8 puts the OSR instance-shape cap back where it was,
 * for a one-binary A/B. The array in JaiOsrForm is always the wider one, so
 * only the refusal moves. */
unsigned jitShapeLimit(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_SHAPE_LIMIT");
        cached = (v != NULL) ? atoi(v) : (int)JAI_OSR_SHAPES;
        if (cached < 1 || cached > (int)JAI_OSR_SHAPES) cached = (int)JAI_OSR_SHAPES;
    }
    return (unsigned)cached;
}

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

/* Is the pair on top of the stack `str + str`?
 *
 * A string is SLOT_OBJ here, which pins nothing, so the sample is what makes
 * this worth emitting and the guards below are what make it sound. Only ONE
 * side needs a string sample: OP_FORMAT pushes its result without one (there
 * is no Value to carry at compile time), and `text = text + f"..."` -- the
 * shape word_freq's whole hot loop is -- has exactly that on the right. */
bool concatOperands(const Emit *e, Value *sample) {
    /* Not inside an inlined body. OP_ADD is on inlinableBody's whitelist
     * because every arm it had emitted straight-line code; this one calls, and
     * an inlined body that calls breaks two things at once. Its entries live in
     * x0..x8 when the caller's bank is callee-saved (inlineOwnBank), which is
     * the register file the call destroys, and emitDescriptor reads its
     * arguments through valueBankReg -- which does not know about that bank, so
     * the descriptor is filled from callee-saved registers holding something
     * else entirely. `fn cat(a: any) -> any { return a + "x" }` in a loop
     * segfaults without this line. Declining costs nothing: the inline fails,
     * and both tiers retry the whole body with inlining off, where this arm
     * fires normally. */
    if (e->inlining) return false;
    if (e->depth < 2) return false;
    if (e->stack[e->depth - 1] != SLOT_OBJ) return false;
    if (e->stack[e->depth - 2] != SLOT_OBJ) return false;
    Value sa = e->stackSeen[e->depth - 2], sb = e->stackSeen[e->depth - 1];
    if (!IS_STRING(sa) && !IS_STRING(sb)) return false;
    *sample = IS_STRING(sa) ? sa : sb;
    return true;
}

/* `a + b` out to jaiStringConcat. Concatenation allocates, so there is nothing
 * to inline; the point is that the rest of the loop body stops being given up.
 * word_freq declined its whole `main` loop on the ADD_BIND this replaces -- one
 * refusal costing an LCG step, an f-string and the append around it.
 *
 * Measured (best of five, alternating builds, under the GPU lock): a 3M-step
 * `text = text + f"w{n} "` loop 204.4ms -> 128.4ms, 1.59x; word_freq at 3M
 * words 392.8ms -> 313.9ms end to end, 1.25x; word_freq as shipped 35.2ms ->
 * 28.2ms, 1.85x once the ~20ms startup floor is taken off. Ruled out: a
 * self-hosted `check` of compile/parser.jai does NOT move (775ms -> 767ms,
 * noise) even though OP_ADD is a few percent of its interpreted work -- the
 * bodies holding it decline for other reasons anyway, so freeing this one
 * changes nothing there.
 *
 * Both operands are guarded, not just the sample-less one: SLOT_OBJ says
 * "heap object" and no more, and a guard here resumes at this instruction with
 * both operands still on the interpreter's stack, so a miss costs nothing --
 * a loop alternating `str + str` with `list + list` agrees with the
 * interpreter under both --gc-stress and JAITHON_JIT_DEOPT_STRESS.
 *
 * Deliberately NOT the general `arithmetic()` fallback: that answers int, float
 * and string alike, so the result would need a tag test after the call to know
 * whether the register holds a pointer. Guarding the two operands instead
 * settles the result kind before the call is made, which is the same argument
 * OP_GET_SLICE makes for guarding its container. */
bool emitStringConcat(Emit *e, Value sample) {
    settleAll(e);                    /* this path guards */
    unsigned ra = valueXReg(e, e->valueDepth - 2);
    unsigned rb = valueXReg(e, e->valueDepth - 1);

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, ra, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, rb, (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, NULL_VAL, e->depth - 2, 2,
                        (void *)&jitStringConcat)) {
        return false;
    }
    unsigned drop;
    if (!popValue(e, &drop, NULL)) return false;
    if (!popValue(e, &drop, NULL)) return false;
    /* Carry a sample so the next instruction still knows this is a string --
     * the same reason the string-index arm carries its receiver's. */
    if (!pushValue3(e, SLOT_OBJ, 0, NULL, sample, -1)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset +
                           (unsigned)offsetof(JitCallDesc, result) + 8));
    e->wroteHeap = true;
    return true;
}

/* Put a local on the model stack, the way OP_GET_LOCAL does, for an arm that
 * needs stack operands but was handed slot numbers. Only the general X path --
 * a float local wants OP_GET_LOCAL's FP handling and no caller here has one. */
/* `x == null` where the model calls x an object -- a string, a list, a dict.
 *
 * The arm already mixes SLOT_INST with SLOT_MAYBE_INST because both are a
 * pointer or a zero in a register. A SLOT_OBJ is the same shape, and stronger:
 * emitTagFor gives it VAL_OBJ unconditionally, so its register never holds a
 * zero and the answer is always "not null" -- which is exactly what comparing
 * it against the null literal's zero register produces.
 *
 * EQUALITY ONLY. `x < null` is a TypeError in the interpreter, and compiling it
 * as a register compare would answer where it should raise.
 *
 * SLOT_OBJ ONLY, and that is the load-bearing half. A SLOT_INT is also "never
 * null", but its register holds a NUMBER -- and zero is a perfectly good int,
 * so `0 == null` would compare equal and answer true. Only a kind whose
 * register holds a pointer may be compared against the null literal's zero.
 *
 * 108 declines across four compiler files, in `_parse_postfix`,
 * `_parse_decorators`, `lookup_type_name` and their kin -- the shape is
 * `if tok == null` on something the model did not name more precisely. */
static bool jitNullPair(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_NULL_PAIR");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool nullLiteralPair(const Emit *e, uint8_t op, SlotKind ka, SlotKind kb) {
    if (!jitNullPair()) return false;
    if (op != OP_EQ && op != OP_NE) return false;
    if (e->depth < 2) return false;
    if (ka == SLOT_OBJ && kb == SLOT_MAYBE_INST &&
        e->stackNullLit[e->depth - 1]) {
        return true;
    }
    return kb == SLOT_OBJ && ka == SLOT_MAYBE_INST &&
           e->stackNullLit[e->depth - 2];
}

/* Does the instruction after a call throw its result away?
 *
 * `OP_POP` is the obvious spelling and the only one this used to test for. But
 * `lib/jaithon/compile/opt/peephole.jai` fuses `Pop; ReturnNull` into
 * OP_POP_RETURN_NULL at the default -O2, so a `-> void` method called as the
 * LAST STATEMENT of a branch never matched -- and that is the commonest place
 * such a call appears. The escape hatch was written against un-optimised
 * bytecode.
 *
 * It cost more than any other single miss in the tier. `_scan_token`'s
 * `self._line_continuation()` -- a `-> void` method on a backslash branch no
 * source file in this tree even takes -- declined the whole lexer entry point:
 * 2,607,687 interpreted instructions, 7.6% of one file, from five compile
 * attempts before the retry budget ran out.
 *
 * `discardedAfter` reports WHICH, because the fused form has to emit the
 * return half itself rather than simply stepping over the pop. */
static bool jitFusedDiscard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FUSED_DISCARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

DiscardKind discardedAfter(const uint8_t *code, int at, int count) {
    if (at >= count) return DISCARD_NO;
    if (code[at] == OP_POP) return DISCARD_POP;
    if (code[at] == OP_POP_RETURN_NULL && jitFusedDiscard()) {
        return DISCARD_POP_RETURN;
    }
    return DISCARD_NO;
}

/* The RETURN_NULL half of an OP_POP_RETURN_NULL that followed a call whose
 * result nothing observes.
 *
 * The fused opcode carries the function's return with it, so it cannot be
 * stepped over the way a bare OP_POP is -- doing that would drop the return
 * entirely. This is OP_RETURN_NULL's arm, unchanged, factored out because the
 * same escape hatch appears at four sites in the invoke family and duplicating
 * it four times is how three of them came to be missing the fix in the first
 * place.
 *
 * Returns false having set whyNot; the caller returns false. On true the caller
 * must advance by 8 and set `afterUncond`, because an invoke falls through and
 * a return does not. */
bool emitFusedReturnNull(Emit *e, ObjFunction *fn) {
    if (e->osr) {
        e->whyNot = "a return inside an OSR loop";
        return false;
    }
    if ((fn->flags & FN_INIT) != 0) {
        /* An initialiser's return yields the object, not null, and wants
         * slot 0 -- a different arm entirely. */
        e->whyNot = "a discarded call fused with an initialiser's return";
        return false;
    }
    if (e->sawReturn && e->returnKind != SLOT_NULL) {
        e->whyNot = "two different return kinds";
        return false;
    }
    e->sawReturn  = true;
    e->returnKind = SLOT_NULL;
    emit(e, jaiA64MovzX(0, 0, 0));
    emitEpilogue(e, 0);
    return true;
}

bool jitStrIter(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_STR_ITER");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* What an unarmed opcode was ABOUT, when the opcode alone does not say.
 *
 * `walked only to OP_GET_GLOBAL at 53` names the instruction and not the
 * question. The question is WHICH global, because that is what decides whether
 * the stop is the deliberate one on a cold `throw` path or a callee that could
 * have compiled: `sqrt` and `floor` each stop at one, and together they are
 * 4.2% of the jaicv benchmark's interpreted work. */
const char *unarmedDetail(const ObjFunction *fn, uint8_t op,
                                 uint32_t at) {
    if (op != OP_GET_GLOBAL) return "";
    if ((size_t)at + 4 > (size_t)fn->chunk.count) return "";
    uint32_t idx = jaiReadU24(fn->chunk.code + at + 1);
    if (idx >= (uint32_t)fn->chunk.constants.count) return "";
    Value name = fn->chunk.constants.data[idx];
    if (!IS_STRING(name)) return "";
    static char buf[96];
    snprintf(buf, sizeof buf, " (`%s`)", AS_STRING(name)->chars);
    return buf;
}

bool pushLocalAsValue(Emit *e, unsigned slot) {
    if (!pushValue3(e, e->localKind[slot], e->localShape[slot],
                    e->localClass[slot],
                    seenLocal(e, slot),
                    (int)slot)) {
        return false;
    }
    unsigned home = localHomeX(e, slot);
    if (home != 0) {
        xBorrowLocal(e, e->valueDepth - 1, home);
    } else {
        unsigned dst = pushReg(e) - 1;
        unsigned src = localIn(e, slot, dst);
        if (src != dst) emit(e, jaiA64MovX(dst, src));
    }
    return true;
}

bool jitConcatLocals(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CONCAT_LOCALS");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A builtin whose whole body is a load from its receiver (jit_field_read.h).
 * The caller has already guarded that the receiver is `fr->type`, so the load
 * is the entire call: no callee Value, no argument Value, no root fill, no
 * status test, no result tag test. `k.len()` was 39 instructions and a `blr`
 * into jitInvokeNative for a 32-bit field.
 *
 * A lazily-computed field (ObjString::scalars) keeps the descriptor call as a
 * slow path *inline*, reached only when the memo is empty, so the native fills
 * it and every later call takes the load. Deopting there instead would be
 * wrong: a loop over freshly built strings would leave the compiled body on
 * every iteration. Both paths land on the same register, and the span over the
 * slow path is measured rather than counted -- emitDescriptor's length moves
 * with the number of roots the body holds. */
bool emitFieldRead(Emit *e, const JaiJitFieldRead *fr, Value nativeVal,
                          unsigned ridx, unsigned argc, uint32_t afterIp) {
    if (fr->tag != VAL_INT) {
        e->whyNot = "a field-reading builtin whose result is not an int";
        return false;
    }
    unsigned rRecv   = pushReg(e) - argc - 1;
    unsigned skipSlow = 0;

    if (fr->width == 4) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rRecv, fr->offset));
    } else {
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, rRecv, fr->offset));
    }

    if (fr->lazy) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint64_t)fr->sentinel);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        skipSlow = e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));      /* patched below */
        if (!emitDescriptor(e, nativeVal, ridx, argc + 1,
                            (void *)&jitInvokeNative)) {
            return false;
        }
    }

    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, SLOT_INT, 0, NULL)) return false;

    if (fr->lazy) {
        unsigned rat = e->descOffset +
                       (unsigned)offsetof(JitCallDesc, result);
        emit(e, jaiA64LdrW(JIT_SCRATCH_B, 31, rat));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, VAL_INT));
        branchOnDeoptAt(e, JAI_A64_NE, afterIp, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, 31, rat + 8));
        if (skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        /* The slow path calls out, and a native may write. */
        e->wroteHeap = true;
    }
    emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
    return true;
}

/* Verdict 4: the callee deoptimised part-way and may have written, so the call can't be re-executed
 * or recorded over (gDeopt is a single global) -- instead the callee is FINISHED in the interpreter from its own record, and the value handed back here, consuming the record at the innermost frame that sees it. Makes a recursive body that writes compilable, and (since `callee` may name someone else) a direct call to a writing method too. OSR form: fall-through continues the loop, so nothing syncs here -- every branch out goes to a stub that syncs itself. */
void emitSelfSlowStubs(Emit *e, ObjClosure *closure) {
    for (unsigned si = 0; si < e->selfSlowCount; si++) {
        e->selfSlow[si].stub = (int)e->count;
        unsigned d = e->descOffset;
        unsigned resultAt = d + (unsigned)offsetof(JitCallDesc, result);

        emit(e, jaiA64SubsXImm(31, 1, 2));             /* the callee raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        emit(e, jaiA64SubsXImm(31, 1, 4));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptBail;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* jaiJitFinishDeopt runs interpreted code, which allocates, so the
         * roots this frame filled before the `bl` go back on the chain. */
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
        }
        emitConst64(e, 0, (int64_t)(uintptr_t)(e->selfSlow[si].callee != NULL
                                                   ? e->selfSlow[si].callee
                                                   : closure));
        emit(e, jaiA64AddXImm(1, 31, resultAt));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jaiJitFinishDeopt);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        if (e->selfSlow[si].roots > 0) {
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, d));
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                               (unsigned)offsetof(JitCallDesc, link)));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        }
        emit(e, jaiA64SubsXImm(31, 0, 0));             /* false: it raised */
        emit(e, jaiA64BCond(JAI_A64_EQ,
                            (int32_t)(e->exceptionExit - (int)e->count)));

        /* The interpreted continuation may return a kind this body was not
         * compiled for. It is in the descriptor with whatever tag it really
         * has, which is exactly what a `lastFromDesc` record writes out. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, resultAt));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, e->selfSlow[si].tag));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset =
            FIXUP_DEOPT - e->selfSlow[si].deoptKind;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
        /* One byte for a bool, as every other descriptor-result site loads
         * one. BOOL_VAL writes only the union's one-byte member, so a slot the
         * interpreter last used for a large int or a pointer keeps its upper
         * seven bytes -- verified on this toolchain at -O2 -flto: a slot
         * holding INT_VAL(0x1122334455667788) then assigned BOOL_VAL(false)
         * reads back 0x1122334455667700. Every SLOT_BOOL consumer branches on
         * the whole word, so that `false` reads as `true`.
         *
         * The pinned arm already reached this stub, so the widening predates
         * the polymorphic one -- but the PIC routes a whole new class of site
         * through here, and those sites previously always took the one-byte
         * load. */
        if (e->selfSlow[si].tag == VAL_BOOL) {
            emit(e, jaiA64LdrByte(e->selfSlow[si].resultReg, 31, resultAt + 8));
        } else {
            emit(e, jaiA64LdrX(e->selfSlow[si].resultReg, 31, resultAt + 8));
        }

        /* VAL_OBJ is every heap object -- reading `klass` off the wrong type (e.g. an ObjString) doesn't
         * fault, it answers wrongly, so the type is checked before the shape. */
        if (e->selfSlow[si].retType >= 0) {
            unsigned rr = e->selfSlow[si].resultReg;
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A,
                                   (unsigned)e->selfSlow[si].retType));
            if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset =
                FIXUP_DEOPT - e->selfSlow[si].deoptKind;
            e->fixups[e->fixupCount].conditional  = true;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            if (e->selfSlow[si].retShape != 0) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rr,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_B,
                            (int64_t)e->selfSlow[si].retShape);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; break; }
                e->fixups[e->fixupCount].instIndex    = (int)e->count;
                e->fixups[e->fixupCount].targetOffset =
                    FIXUP_DEOPT - e->selfSlow[si].deoptKind;
                e->fixups[e->fixupCount].conditional  = true;
                e->fixups[e->fixupCount].depth        = -1;
                e->fixupCount++;
                emit(e, jaiA64BCond(JAI_A64_NE, 0));
            }
        }
        emit(e, jaiA64B((int32_t)(e->selfSlow[si].returnTo - (int)e->count)));
    }
}

/* Cold half of `xs.push(v)`: reserve, refill the count the fast path already loaded, branch back in.
 * No descriptor/roots (see jitListGrow). This is a continuation, not an exit -- an OSR form must NOT sync its iterator or locals here, since the loop carries on with them where they are. */
void emitGrowStubs(Emit *e) {
    for (unsigned gi = 0; gi < e->growCount; gi++) {
        e->grow[gi].stub = (int)e->count;
        emit(e, jaiA64MovX(0, e->grow[gi].listReg));
        emit(e, jaiA64MovzX(1, e->grow[gi].tag, 0));
        emit(e, jaiA64MovX(2, e->grow[gi].valReg));
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&jitListGrow);
        emit(e, jaiA64Blr(JIT_SCRATCH_D));
        emit(e, jaiA64SubsXImm(31, 0, 0));
        emit(e, jaiA64BCond(JAI_A64_NE,
                            (int32_t)(e->exceptionExit - (int)e->count)));
        emit(e, jaiA64LdrW(e->grow[gi].countReg, e->grow[gi].listReg,
                           (unsigned)offsetof(ObjList, count)));
        emit(e, jaiA64B((int32_t)(e->grow[gi].returnTo - (int)e->count)));
    }
}

/* Direct-branch arguments arrive as a raw payload, so the caller's kind must match what the callee
 * was specialised for -- and for an instance, the same class shape, since every field offset was resolved against it. jaiJitEnterFunc is the only check standing between a float's bits and a body that treats them as a pointer. `firstIdx`: the entry above the callee for a plain call, or the receiver (callee's slot 0) for a method. */
bool directCallArgsMatch(Emit *e, const ObjFunction *cfn,
                                unsigned firstIdx, unsigned argc) {
    for (unsigned i = 0; i < argc; i++) {
        unsigned idx = firstIdx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) {
            e->whyNot = "a direct call argument that is not in a register";
            return false;
        }
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) {
                e->whyNot = "a direct call argument is not the parameter's kind";
                return false;
            }
        } else if (have != want) {
            e->whyNot = "a direct call argument is not the parameter's kind";
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            e->whyNot = "a direct call passing a different class";
            return false;
        }
    }
    return true;
}

/* Branches straight to a compiled callee's entry, skipping the descriptor/jaiCallValue/interpreter-
 * frame path. Convention: raw payloads in x0.., closure in the last arg register if the callee reads an upvalue, x0/x1 = value/verdict on return -- the same one jaiJitEnterFunc checks and a self-call already uses, so skipping that entry means answering its checks here instead: module version (why the callee must live in the caller's module), every parameter's kind+shape (by the caller's model), and the verdict (below). Nonzero verdict: a callee that writes NOTHING can have the whole call abandoned and re-executed from the pre-call stack (two compares, no stub); a callee that WRITES cannot be re-run -- verdict 4 means it deoptimised part-way and is FINISHED in the interpreter from its own record, sharing the `selfSlow` machinery a recursive self-call already uses. A raised exception goes to the throw exit instead, since its effects already happened. `calleeReg`: the ObjClosure register, or -1 if baked in. `cidx`: operand-stack index of the callee entry (the RECEIVER for a method, i.e. its slot 0). `after`: offset of the fall-through instruction. */
/* A `-> T?` result read back out of a call descriptor.
 *
 * Two tags are acceptable where every other kind has exactly one, so the single
 * compare the other arms use cannot serve. VAL_NULL is zero, which is what lets
 * "object or null" be two csels and a compare rather than a branch: afterwards
 * D holds the payload or a defined zero, and B is zero exactly when the tag was
 * one of the two.
 *
 * The class checks apply only to a non-null. The field arm at
 * OP_GET_FIELD_LOCAL keeps its equivalents branch-free by redirecting the loads
 * at its live receiver; a call result has no such pointer to borrow, so they
 * sit behind a forward branch instead. A deopt inside that span is sound for
 * the reason emitBoundsNormalise's is: the record is written from the
 * descriptor, which holds the true Value whichever way the branch went.
 *
 * Worth having because after this the refusal it removes was the tier's single
 * largest on the self-hosted compiler: one `check --no-cache` of
 * compile/parser.jai stopped 64 bodies at "callee's return kind not usable"
 * with the kind being exactly SLOT_MAYBE_INST, and every one of them had a
 * resolvable class. `-> Node?` is what a parser's methods return. */
void emitMaybeInstResult(Emit *e, unsigned dst, unsigned rat,
                                uint32_t rshape, uint32_t deoptIp) {
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64LdrX(JIT_SCRATCH_D, 31, rat + 8));
    emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
    emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D, JIT_SCRATCH_B,
                        JAI_A64_EQ));
    emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B, JIT_SCRATCH_A,
                        JAI_A64_EQ));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);

    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    unsigned skip = e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(Obj, type)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
    branchOnDeoptAt(e, JAI_A64_NE, deoptIp, true);
    /* Nothing to jump over means the arena filled mid-sequence; leaving the
     * branch unpatched would run the class loads on a null pointer. */
    if (skip < e->count && e->count <= JIT_MAX_INSTS) {
        e->code[skip] = jaiA64BCond(JAI_A64_EQ, (int32_t)(e->count - skip));
    } else {
        e->failed = true;
    }
    emit(e, jaiA64MovX(dst, JIT_SCRATCH_D));
}

bool jitAnyGuard(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ANY_GUARD");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

static bool jitReturnKnownOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_RETURN_KNOWN");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool emitDirectCall(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                           Value calleeVal, int calleeReg, unsigned cidx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           bool method) {
    /* Either verdict path below can come back with an exception pending. */
    if (!raiseExitAllowed(e, "a call that can raise inside a try")) return false;
    if (cfn->module != caller->module) {
        e->whyNot = "a direct callee from another module";
        return false;
    }
    /* Slot 0 is the closure for a plain function and the receiver for a
     * method, so which of the two this is decides where the arguments start
     * and whether the receiver is one of them. */
    if (cfn->jitArgBase != (method ? 0u : 1u)) {
        e->whyNot = method
            ? "a direct method that does not take its receiver in slot 0"
            : "a direct callee that is not a plain function";
        return false;
    }
    /* The callee's baked classes/closures/natives are pinned by ITS jitFuncModuleVersion, checked at the
     * entry this call skips -- valid only if the caller's own check agrees NOW: a global rebound after the callee compiled would leave jitFuncModuleVersion stale but still reachable via a direct call, and a caller compiling afterwards would silently pin the newer version. */
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        e->whyNot = "a direct callee compiled against an older module";
        return false;
    }
    /* A callee that writes is finished in the interpreter on verdict 4, and
     * that needs its ObjClosure -- which only a callee baked in at compile
     * time provides. */
    bool writes = !cfn->jitFuncNoWrite;
    if (writes && (calleeReg >= 0 || !IS_CLOSURE(calleeVal))) {
        e->whyNot = "a direct callee that writes and is not known here";
        return false;
    }
    /* `nargs` is how many registers the branch fills. A method's receiver is
     * one of them; a plain call's callee entry holds no register at all. */
    unsigned nargs = method ? argc + 1u : argc;
    unsigned firstIdx = method ? cidx : cidx + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) {
        e->whyNot = "a direct callee with a different arity";
        return false;
    }
    if (calleeArgs > JIT_MAX_ARITY) {
        e->whyNot = "a direct callee with too many arguments";
        return false;
    }

    if (!directCallArgsMatch(e, cfn, firstIdx, nargs)) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        e->whyNot = "a direct callee whose trailing argument is not its closure";
        return false;
    }
    if (wantsClosure && calleeReg < 0 && !IS_CLOSURE(calleeVal)) {
        e->whyNot = "a direct callee that wants a closure it has not got";
        return false;
    }
    /* SLOT_NULL: a `-> void` function. Epilogue leaves x0 zero, so the pushed entry has a fixed tag and
     * zero payload -- same treatment a self-call to a void function gets. Refusing it declined every caller of a procedure, which in nbody is the whole of `main`. */
    /* A callee whose walk never reached an OP_RETURN has no return kind to be
     * the contract, only the SLOT_INT of a zeroed Emit. `_is_ident_start` in
     * the lexer walks only to OP_GET_GLOBAL at offset 0 -- a cold `throw` on
     * its first instruction -- and still claimed to return an int, which made
     * `_is_ident_start(c) or _is_digit(c)` decline on "a branch on a int, not
     * a bool" and left `_ident_run_end`'s OSR loop retrying it eighty times.
     *
     * Declining here is not a coverage loss: the caller falls back to the
     * guarded emitGlobalCall path, which asks the interpreter's own
     * observation first and gets the right answer. */
    if (!cfn->jitReturnKnown && jitReturnKnownOn()) {
        e->whyNot = "a direct callee whose walk never reached a return";
        return false;
    }
    SlotKind rk = (SlotKind)cfn->jitReturnKind;
    ObjClass *rcls = NULL;
    /* SLOT_MAYBE_INST rides with SLOT_INST here and needs no guard of its own:
     * a direct branch takes the callee's raw payload, and for a nullable
     * instance that payload IS the pointer or a zero -- the same
     * representation this caller will hold. The callee's declared return kind
     * is the contract, exactly as it is for every other kind at this arm. */
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (cfn->jitReturnShape == 0 ||
         !jaiClassForShape(cfn->jitReturnShape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    /* Asked here rather than where the slot is taken: below this point the
     * root fill has been emitted and the descriptor linked onto the collector's
     * chain, so there is no falling back to the descriptor path any more. */
    if (writes && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        e->whyNot = "more slow call sites than the tier tracks";
        return false;
    }

    /* Past here everything is settled and this call is happening: a failure
     * below is the emitter running out of room, not a decision, so it stops
     * the compile rather than falling back to the descriptor path onto a
     * half-written instruction stream. */

    /* Roots before the branch: a `bl` pushes none, and the callee may
     * allocate -- OP_GET_SLICE builds a fresh list without ever counting as a
     * heap write. */
    unsigned callRoots = 0;
    if (!emitRootFill(e, e->descOffset, &callRoots)) { e->failed = true; return false; }
    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, callRoots, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           dd + (unsigned)offsetof(JitCallDesc, nroots)));
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, JIT_SCRATCH_A, 0));
    }

    unsigned firstArg = firstIdx - (e->depth - e->valueDepth);
    for (unsigned i = 0; i < nargs; i++) {
        emit(e, jaiA64MovX(i, valueXReg(e, firstArg + i)));
    }
    if (wantsClosure) {
        if (calleeReg >= 0) emit(e, jaiA64MovX(nargs, (unsigned)calleeReg));
        else emitConst64(e, nargs, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    }
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)cfn->jitFunc);
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_D));

    if (callRoots > 0) {
        unsigned dd = e->descOffset;
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gJitFrames);
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, 31, dd));
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                           (unsigned)offsetof(JitCallDesc, link)));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    }

    unsigned si = 0;
    if (writes) {
        /* One compare and one not-taken branch on the fast path; every other
         * answer is the shared cold block. The record has to be taken here,
         * where the model still holds the receiver and the arguments, even
         * though the block is emitted with the stubs. */
        if (e->selfSlowCount >= JIT_MAX_SELF_SLOW) { e->failed = true; return false; }
        si = e->selfSlowCount++;
        e->selfSlow[si].roots    = callRoots;
        e->selfSlow[si].stub     = -1;
        e->selfSlow[si].callee   = AS_CLOSURE(calleeVal);
        e->selfSlow[si].retShape = rk == SLOT_INST ? cfn->jitReturnShape : 0;
        e->selfSlow[si].retType  = rk == SLOT_INST ? (int)OBJ_INSTANCE
                                 : rk == SLOT_LIST ? (int)OBJ_LIST
                                                   : -1;
        if (!deoptRecordAt(e, callOff, false, &e->selfSlow[si].deoptBail)) {
            e->failed = true;
            return false;
        }
        emit(e, jaiA64SubsXImm(31, 1, 0));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_SELFSLOW - si;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));
    } else {
        /* Verdict 2 is a pending exception: the interpreter owns it and this
         * call must not run again. */
        emit(e, jaiA64SubsXImm(31, 1, 2));
        if (e->fixupCount >= JIT_MAX_FIXUPS) { e->failed = true; return false; }
        e->fixups[e->fixupCount].instIndex    = (int)e->count;
        e->fixups[e->fixupCount].targetOffset = FIXUP_THREW;
        e->fixups[e->fixupCount].conditional  = true;
        e->fixups[e->fixupCount].depth        = -1;
        e->fixupCount++;
        emit(e, jaiA64BCond(JAI_A64_EQ, 0));
        /* Anything else -- a bail, or a guard that failed inside the callee --
         * hands the whole call back. The record is taken with the callee and
         * its arguments still on the model's stack, which is what the
         * interpreter expects to find at this offset. */
        emit(e, jaiA64SubsXImm(31, 1, 0));
        branchOnDeoptAt(e, JAI_A64_NE, callOff, false);
    }

    for (unsigned i = 0; i < nargs; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    if (!method) {
        if (e->depth == 0) { e->failed = true; return false; }
        e->depth--;
    }
    if (!pushValue(e, rk, cfn->jitReturnShape, rcls)) { e->failed = true; return false; }
    emit(e, jaiA64MovX(pushReg(e) - 1, 0));
    if (writes) {
        e->selfSlow[si].resultReg = pushReg(e) - 1;
        e->selfSlow[si].returnTo  = (int)e->count;
        e->selfSlow[si].tag = rk == SLOT_INT   ? VAL_INT
                            : rk == SLOT_FLOAT ? VAL_FLOAT
                            : rk == SLOT_BOOL  ? VAL_BOOL
                                               : VAL_OBJ;
        /* The interpreted continuation is typed by nothing this compiled for,
         * so what it hands back is checked and a surprise resumes AFTER the
         * call -- which has happened and must not happen twice. */
        if (!deoptRecordAt(e, after, true, &e->selfSlow[si].deoptKind)) {
            e->failed = true;
            return false;
        }
        /* A call that writes is an effect, so no bail may follow it -- the
         * same rule the descriptor path lives under. */
        e->wroteHeap = true;
    }
    /* Deliberately NOT wroteHeap for a non-writing callee, unlike the descriptor path: it stores and
     * calls nothing, so a re-run repeats no visible effect. It may still allocate (OP_GET_SLICE does) -- a fresh object isn't an observable effect. */
    return true;
}

/* JAITHON_JIT_PIC=0 turns the one-way inline cache below off, so the same
 * binary can be A/B'd around it without a rebuild -- same idiom as
 * jaiListUnboxOn's JAITHON_LIST_UNBOX (object_collection.c) and
 * jaiJitEnabled's JAITHON_NO_JIT. Read once: this sits on every unpinned
 * OP_INVOKE, and an uncached getenv there is its own cost (see
 * jitReconTrace above). */
bool jitPicEnabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_PIC");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Mirrors every decision emitDirectCall makes before it commits to emitting,
 * for the one way this cache is about to speculate on. By the time the shape
 * compare below is in the instruction stream a decline has nowhere to fall
 * back to but e->failed, so this runs FIRST and answers instead of finding
 * out the hard way. */
static bool jitPic1Admissible(Emit *e, ObjFunction *caller, ObjFunction *cfn,
                              unsigned ridx, unsigned argc, uint32_t shape) {
    if (cfn->module != caller->module) return false;
    if (caller->module == NULL ||
        cfn->jitFuncModuleVersion != caller->module->version) {
        return false;
    }
    if (cfn->jitArgBase != 0u) return false;
    unsigned nargs = argc + 1u;
    unsigned calleeArgs = (unsigned)cfn->jitArgCount;
    bool wantsClosure = calleeArgs == nargs + 1u;
    if (!wantsClosure && calleeArgs != nargs) return false;
    if (calleeArgs > JIT_MAX_ARITY) return false;
    if (wantsClosure &&
        (SlotKind)cfn->jitParamKind[nargs] != SLOT_CLOSURE) {
        return false;
    }
    /* The receiver is the callee's slot 0, and the branch about to be
     * emitted is the proof of its class -- so the parameter the callee was
     * specialised for has to be that same class, not merely some instance. */
    if ((SlotKind)cfn->jitParamKind[0] != SLOT_INST) return false;
    if (cfn->jitParamShape[0] != shape) return false;
    for (unsigned i = 1; i < nargs; i++) {
        unsigned idx = ridx + i;
        SlotKind have = e->stack[idx];
        SlotKind want = (SlotKind)cfn->jitParamKind[i];
        if (!holdsRegister(have)) return false;
        if (want == SLOT_OPAQUE) continue;   /* never read; see seedLocals */
        if (want == SLOT_MAYBE_INST) {
            if (have != SLOT_INST && have != SLOT_MAYBE_INST) return false;
        } else if (have != want) {
            return false;
        }
        if ((want == SLOT_INST || want == SLOT_MAYBE_INST) &&
            e->stackShape[idx] != cfn->jitParamShape[i]) {
            return false;
        }
    }
    /* A callee that writes is finished in the interpreter from a selfSlow
     * record, and this arm takes one of its own -- the same budget a pinned
     * direct call draws from. */
    if (!cfn->jitFuncNoWrite && e->selfSlowCount >= JIT_MAX_SELF_SLOW) {
        return false;
    }
    return true;
}

/* The compile-time half of a ONE-WAY inline cache: what the model could not
 * pin about a receiver, read off the site's own InlineCache instead.
 *
 * One way only, and only the FIRST shape this site ever recorded --
 * ic->cached[0] / ic->shapeId[0] -- not the whole walk over every way a
 * polymorphic cache holds. The receiver's class is loaded once and compared
 * against that one shape; a hit branches straight into the recorded callee's
 * compiled entry exactly as a pinned receiver already does (emitDirectCall).
 * A miss -- every OTHER class at a polymorphic site, and the common case at a
 * megamorphic one -- falls through to jitInvokeByName, which is exactly what
 * the site emits today and is unchanged by any of this.
 *
 * The compare is not a guard and a miss is not a deopt: both sides of it run
 * BEFORE the call, so nothing has happened yet that must not happen twice.
 * That is the whole reason this needs no deopt record of its own, unlike the
 * call inside it (emitDirectCall still takes one, for what the CALL can do
 * after it starts).
 *
 * `havePrediction`/`rkind` is the fall-through's own prediction
 * (siteInvokeResultKind), which the one way must also return: only a
 * register-only kind (SLOT_INT/FLOAT/BOOL) is admitted, because those are the
 * only ones that carry no class shape for the two paths to disagree about --
 * see feedbackSlotKind's exclusion of SLOT_INST. `*toEnd` comes back holding
 * the arm's own branch to the merge point, for the caller to patch once it
 * has emitted the fall-through after it.
 *
 * Returns false having emitted nothing whenever there was something to
 * cleanly decline, so the caller can still take the descriptor path alone --
 * UNLESS e->failed, which means the shape compare is already in the stream
 * and there is nowhere left to fall back to (mirrors emitDirectCall's own
 * rule, since this arm ends by calling into it). */
bool emitInvokePic1(Emit *e, ObjFunction *fn, unsigned ridx,
                           unsigned argc, uint32_t callOff, uint32_t after,
                           int siteCache, bool havePrediction, SlotKind rkind,
                           int *toEnd) {
    if (!havePrediction) {
        return subWhy(e, "no predicted result kind to join the arm on");
    }
    if (rkind != SLOT_INT && rkind != SLOT_FLOAT && rkind != SLOT_BOOL) {
        return subWhy(e, "an unpinned receiver returning something with a shape");
    }
    if (siteCache < 0 || fn->chunk.caches == NULL ||
        siteCache >= fn->chunk.cacheCount) {
        return subWhy(e, "no inline cache recorded for this site");
    }
    const InlineCache *ic = &fn->chunk.caches[siteCache];
    /* IC_MEGA is admitted, and it is the case that matters. A site that runs
     * out of ways stops caching ALTOGETHER and re-resolves every call through
     * sMegaCache -- see the comment on that table -- which is exactly the
     * shape a trait with eight implementations makes, and exactly the shape
     * this arm exists for. The ways it recorded before it gave up are still
     * there and still true: a way is (shapeId, method), shapeIds come from a
     * monotonic counter that would need four billion classes to repeat, and
     * every way is re-checked below for a live class and a compiled callee.
     * Four ways against eight classes is a partial cache, not a complete one,
     * and a partial cache is the whole point -- the misses cost one compare
     * each and then do exactly what the site does today. */
    if (ic->state != IC_MONO && ic->state != IC_POLY &&
        ic->state != IC_MEGA) {
        return subWhy(e, "the site's cache is empty");
    }
    if (ic->count == 0) return subWhy(e, "the site's cache has no way filled");
    if (ridx + argc + 1u > JIT_MAX_STACK) {
        return subWhy(e, "past the stack depth the model can describe");
    }
    /* Every way the cache holds is tried, in the order it recorded them, so
     * a site that warmed up on its second-most-common class no longer
     * speculates on the wrong one. The compares chain: way w's compare falls
     * through to way w+1's, and the last falls through to the descriptor the
     * site emits today. A miss therefore costs one compare per way and then
     * does exactly what it did before. */
    /* Collect the ways this compile can actually take. A way is dropped, not
     * fatal: the site keeps its remaining arms and the dropped class simply
     * goes round the descriptor as it does today. */
    unsigned    wayShape[JAI_IC_WAYS];
    Value       wayVal  [JAI_IC_WAYS];
    ObjFunction *wayFn  [JAI_IC_WAYS];
    ObjClass    *wayCls [JAI_IC_WAYS];
    unsigned    ways = 0;

    for (int w = 0; w < ic->count && w < JAI_IC_WAYS; w++) {
        /* What the cache settles is which method a shape resolves to, not
         * whether THIS caller may call it -- one site can present as two
         * classes at different visibilities, so the interpreter re-decides
         * that on every hit and nothing emitted here can. */
        if (ic->payload[w] != 0) continue;
        Value cv = ic->cached[w];
        if (!IS_CLOSURE(cv)) continue;
        ObjFunction *cf = AS_CLOSURE(cv)->fn;
        if (cf->jitFunc == NULL) continue;
        if ((SlotKind)cf->jitReturnKind != rkind || cf->jitReturnShape != 0) {
            continue;
        }
        ObjClass *cc = NULL;
        if (!jaiClassForShape(ic->shapeId[w], &cc) || cc == NULL) continue;
        if (!jitPic1Admissible(e, fn, cf, ridx, argc, ic->shapeId[w])) continue;
        wayShape[ways] = ic->shapeId[w];
        wayVal[ways]   = cv;
        wayFn[ways]    = cf;
        wayCls[ways]   = cc;
        ways++;
    }
    if (ways == 0) return subWhy(e, "no way of this site's cache is usable");

    settleAll(e);
    fpReleaseAll(e);
    if (e->fpLive != 0) {
        return subWhy(e, "an unpinned receiver with a value in the float bank");
    }

    /* Past here the shape compare is in the stream and the site is
     * committed: a failure below stops the compile rather than falling
     * back. */
    unsigned rreg = valueXReg(e, ridx - (e->depth - e->valueDepth));
    /* The receiver's shape is loaded ONCE and every way compares against it. */
    emit(e, jaiA64LdrX(JIT_SCRATCH_A, rreg,
                       (unsigned)offsetof(ObjInstance, klass)));
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                       (unsigned)offsetof(ObjClass, shapeId)));

    /* The model as the fall-through must find it again: emitDirectCall
     * consumes the receiver and the arguments and pushes a result, and the
     * fall-through's own descriptor call has to see the SAME depth and
     * valueDepth it would have without any of this, or the two paths
     * disagree about where the result lands. */
    unsigned  saveDepth      = e->depth;
    unsigned  saveValueDepth = e->valueDepth;
    SlotKind  saveKind [JIT_MAX_STACK];
    uint32_t  saveShape[JIT_MAX_STACK];
    ObjClass *saveClass[JIT_MAX_STACK];
    Value     saveSeen [JIT_MAX_STACK];
    memcpy(saveKind,  e->stack,      sizeof saveKind);
    memcpy(saveShape, e->stackShape, sizeof saveShape);
    memcpy(saveClass, e->stackClass, sizeof saveClass);
    memcpy(saveSeen,  e->stackSeen,  sizeof saveSeen);

    /* One arm a way. Each is: prove the shape, call directly, jump to the
     * merge. A way that does not match falls into the next way's compare, and
     * the last falls into the descriptor path the caller emits -- which is
     * what this site did for every receiver before any of this. */
    int armMiss[JAI_IC_WAYS];
    for (unsigned w = 0; w < ways; w++) {
        emitConst64(e, JIT_SCRATCH_B, (int64_t)wayShape[w]);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        armMiss[w] = (int)e->count;
        emit(e, jaiA64BCond(JAI_A64_NE, 0));

        /* True on this arm alone, which is why it is also what the CALL's own
         * deopt records (inside emitDirectCall, for what happens after the
         * call starts -- not for this compare) should say: the branch just
         * above proved it. */
        e->depth      = saveDepth;
        e->valueDepth = saveValueDepth;
        memcpy(e->stack,      saveKind,  sizeof saveKind);
        memcpy(e->stackShape, saveShape, sizeof saveShape);
        memcpy(e->stackClass, saveClass, sizeof saveClass);
        memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
        e->stack[ridx]      = SLOT_INST;
        e->stackShape[ridx] = wayShape[w];
        e->stackClass[ridx] = wayCls[w];

        if (!emitDirectCall(e, fn, wayFn[w], wayVal[w], -1, ridx, argc,
                            callOff, after, true)) {
            /* jitPic1Admissible said it would take this and it did not: the
             * branch into it is already emitted, so there is nowhere left to
             * fall back to. */
            e->failed = true;
            return false;
        }
        if (e->picExitCount >= JIT_MAX_PIC_EXITS) {
            e->failed = true;
            return false;
        }
        e->picExits[e->picExitCount++] = (int)e->count;
        emit(e, jaiA64B(0));

        /* Same condition as the placeholder (NE: skip the call on a shape
         * that doesn't match), now with the real offset -- flipping it to EQ
         * would call this way's callee on every receiver whose shape did NOT
         * match, reading its fields at the wrong class's layout. That is what
         * test_mixed_list_runs_every_class caught: a wrong answer, not a
         * crash, because the read lands inside the instance's own allocation.
         *
         * Guarded because emit() silently DROPS the word once e->count reaches
         * JIT_MAX_INSTS and only sets e->failed -- so the slot may never have
         * been written, and Emit::code is immediately followed by `count` with
         * no padding between them. */
        if (armMiss[w] < (int)e->count && e->count <= JIT_MAX_INSTS) {
            e->code[armMiss[w]] =
                jaiA64BCond(JAI_A64_NE, (int32_t)((int)e->count - armMiss[w]));
        }
    }

    /* The model the caller's descriptor path must find, restored exactly. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);

    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] pic %u-way (of %u recorded, state %d) at %u\n",
                ways, (unsigned)ic->count, (int)ic->state, callOff);
    }

    /* The fall-through runs with the receiver and the arguments untouched,
     * so the caller emits its descriptor path against the model as it was
     * before any of this. */
    e->depth      = saveDepth;
    e->valueDepth = saveValueDepth;
    memcpy(e->stack,      saveKind,  sizeof saveKind);
    memcpy(e->stackShape, saveShape, sizeof saveShape);
    memcpy(e->stackClass, saveClass, sizeof saveClass);
    memcpy(e->stackSeen,  saveSeen,  sizeof saveSeen);
    return true;
}


/* ---- literal operands -------------------------------------------------- */

/* Whether anything but fall-through can reach `off`: reading the previous instruction's literal
 * without this check is a miscompile, not a decline -- `x // (if c {2} else {4})` puts OP_INT right before OP_FLOORDIV *and* a jump from the other arm onto it. Scans the WHOLE chunk, not just fixups already emitted (a back edge compiles after its target is walked, so the fixup list would miss loop tops) -- and handler/finally addresses too, since the unwinder can resume there with a stack this walk never saw. */
bool offsetIsBranchTarget(const Chunk *c, uint32_t off) {
    for (int at = 0; at < c->count;) {
        int len = instructionLength(c, at);
        if (len <= 0) return true;      /* undecodable: assume the worst */
        int rel = jaiOpBranchOperandAt(c->code[at]);
        if (rel >= 0) {
            int16_t jump = jaiReadI16(c->code + at + 1 + rel);
            /* Every branch operand is measured from the end of the
             * instruction, which is what `at + len` is. */
            if ((int32_t)(at + len) + jump == (int32_t)off) return true;
        }
        at += len;
    }
    return false;
}

/* OP_INT carries its value inline, OP_CONST names a pool entry -- the only two ways a literal reaches
 * the stack. The adjacency check is belt-and-braces (the walk is linear) but a real bug: OP_FORMAT once advanced `off` by nine instead of ten, and this is what would have caught it. */
bool literalIntOperand(const ObjFunction *fn, int prevOff, int off,
                              int64_t *out) {
    if (prevOff < 0 || prevOff >= off) return false;
    const Chunk *c = &fn->chunk;
    if (prevOff + instructionLength(c, prevOff) != off) return false;
    uint8_t prev = c->code[prevOff];
    if (prev == OP_INT) {
        *out = jaiReadI16(c->code + prevOff + 1);
    } else if (prev == OP_CONST) {
        uint32_t idx = jaiReadU24(c->code + prevOff + 1);
        if (idx >= (uint32_t)c->constants.count) return false;
        Value k = c->constants.data[idx];
        if (!IS_INT(k)) return false;
        *out = AS_INT(k);
    } else {
        return false;
    }
    return !offsetIsBranchTarget(c, (uint32_t)off);
}

/* `k` is 2^shift, for a shift this can name. Positive only: floor division by
 * a negative power of two is not a shift, and `k` is at most 2^62 because 2^63
 * does not fit in a positive int64. */
/* Collapses the general "add divisor back if remainder is non-zero and signs differ" (7 instructions)
 * to 2 when the divisor's sign is known at compile time: msub leaves |r| < |d| with r's sign following the dividend's, so for a positive divisor the whole test is "r < 0" (one bit), and for a negative one "r > 0". `r + d` cannot overflow since |r| < |d| puts the sum strictly between -|d| and |d|. */
void emitFloorFixup(Emit *e, unsigned rrem, unsigned rd,
                           bool signKnown, int64_t divisor, uint32_t fixup) {
    if (signKnown && divisor > 0) {
        emit(e, jaiA64Tbz(rrem, 63u, 2));
        emit(e, fixup);
        return;
    }
    if (signKnown) {
        emit(e, jaiA64SubsXImm(31, rrem, 0));
        emit(e, jaiA64BCond(JAI_A64_LE, 2));
        emit(e, fixup);
        return;
    }
    emit(e, jaiA64SubsXImm(31, rrem, 0));
    emit(e, jaiA64BCond(JAI_A64_EQ, 5));
    emit(e, jaiA64EorX(JIT_SCRATCH_D, rrem, rd));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
    emit(e, jaiA64BCond(JAI_A64_GE, 2));
    emit(e, fixup);
}

bool powerOfTwoShift(int64_t k, unsigned *shift) {
    if (k <= 0) return false;
    uint64_t u = (uint64_t)k;
    if ((u & (u - 1u)) != 0u) return false;
    unsigned s = 0;
    while ((u >> s) != 1u) s++;
    *shift = s;
    return true;
}

/* Something inside an inlined body could not be emitted, so the whole compile
 * is worth retrying with inlining off rather than declining: the same call
 * through the descriptor still compiles, and a compiled form with a real call
 * in it beats none at all. A file static for the same reason the Emit buffers
 * are -- compilation is not reentrant, nothing it calls compiles anything. */
bool gInlineFailed;

/* Structural check, answered before anything is emitted (a half-inlined body can't be taken back):
 * no branches (no offset map, no join, no fixup naming a callee offset in the caller's table); exactly one RETURN, last; locals only via the four opcodes the inline frame understands, and only slots this callee actually has; globals only for the two builtins the tier emits inline (else a global VALUE load would bake a JaiEntry from the callee's own table, needing its own guard); nothing that stores (a guard inside re-executes the WHOLE call, so an earlier store would run twice). What's left is straight-line register arithmetic -- the main walker already speaks it, so no second emitter is needed. `evalA` in spectral is fifteen instructions of exactly this shape. */
static bool inlinableBody(ObjClosure *callee, unsigned argc,
                          unsigned *maxSlotOut, bool *readsUpvalueOut) {
    ObjFunction *cfn = callee->fn;
    const Chunk *c = &cfn->chunk;
    if (cfn->arity != argc || cfn->defaultCount != 0) return false;
    if (cfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (c->count <= 0 || c->count > 128) return false;

    unsigned maxSlot = argc;
    bool sawReturn = false;
    bool readsUpvalue = false;
    for (int off = 0; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        unsigned slot = 0, slot2 = 0;
        switch (op) {
        /* The local frame the caller builds understands exactly these. Any
         * other opcode naming a slot -- a field read off one, a compare
         * against one, an in-place update -- would read the CALLER's local of
         * that number, which is a different variable entirely. */
        case OP_GET_LOCAL:
        case OP_BIND:
            slot = jaiReadU16(c->code + off + 1);
            if (slot > maxSlot) maxSlot = slot;
            break;
        case OP_GET_LOCAL2:
        case OP_ADD_LOCALS:
            slot  = jaiReadU16(c->code + off + 1);
            slot2 = jaiReadU16(c->code + off + 3);
            if (slot > maxSlot) maxSlot = slot;
            if (slot2 > maxSlot) maxSlot = slot2;
            break;
        /* An upvalue is reached through the closure that is actually being
         * called, which is a register the call site has to supply -- so this
         * is only inlinable where that register exists. OP_SET_UPVALUE is not
         * here and falls to `default`: a store would have to be undone if a
         * later guard in the same body deoptimised to the call. */
        case OP_GET_UPVALUE:
            if ((unsigned)c->code[off + 1] >= (unsigned)cfn->upvalueCount) {
                return false;
            }
            readsUpvalue = true;
            break;
        case OP_GET_GLOBAL: {
            uint32_t nameIdx = jaiReadU24(c->code + off + 1);
            Value nv;
            if (globalNative(callee, nameIdx, &nv) == NULL) return false;
            ObjNative *nat = AS_NATIVE(nv);
            const char *nm = nat->name != NULL ? nat->name->chars : "";
            if (strcmp(nm, "float") != 0 && strcmp(nm, "int") != 0) return false;
            break;
        }
        case OP_CALL:
            /* The only callee that can be on the stack here is one of the two
             * builtins above, and the tier emits those as one instruction. */
            if (c->code[off + 1] != 1) return false;
            break;
        case OP_CONST: case OP_INT: case OP_TRUE: case OP_FALSE:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV:
        case OP_FLOORDIV: case OP_MOD: case OP_POW: case OP_NEG:
        case OP_BAND: case OP_BOR: case OP_BXOR:
        case OP_SHL: case OP_SHR: case OP_BNOT:
        case OP_TYPE_GUARD:
            break;
        case OP_RETURN:
            if (off + len != c->count) return false;
            sawReturn = true;
            break;
        default:
            return false;
        }
        off += len;
    }
    if (!sawReturn) return false;
    if (maxSlot > JIT_MAX_SLOTS) return false;
    *maxSlotOut = maxSlot;
    *readsUpvalueOut = readsUpvalue;
    return true;
}

/* Inlines the callee's body: slot 1+i IS entry cidx+1+i already on the stack, so nothing is copied in;
 * a bound slot pins one more entry underneath what's pushed after it, sound only because the body is straight-line. Callee's module must be the caller's, and its baked builtins are retired by the CALLER's own module-version check (the callee's is never run). `calleeReg`: needed only if the body reads an upvalue, since `callee` is a SAMPLE closure at an indirect site -- one ObjFunction, many closures (`|x| x + step`), so its captured cells aren't necessarily the next call's. Constants/globals are safe from the sample since they belong to the function/module, not the closure. */
bool inlineGlobalCall(Emit *e, ObjFunction *caller, ObjClosure *callee,
                             unsigned argc, uint32_t callOff, int calleeReg) {
    if (e->noInline) return false;
    /* An inlined body's entries want x0..x8 (inlineOwnBank) and a split bank
     * is already using them, so the plan withholds the split from a body the
     * measuring pass saw inline. Refusing here as well is what makes that a
     * fact rather than an agreement between two passes: the worst this can do
     * is decline an inline the probe never took. */
    if (e->splitAt != 0) return false;
    ObjFunction *cfn = callee->fn;
    if (cfn->module != caller->module) return false;
    if (e->inlining) return false;             /* one level, no recursion */
    /* The inlined body's offsets are the callee's, so `inProtected` describes
     * the CALLER's regions throughout -- a `try` of the callee's own would go
     * unseen. inlinableBody's whitelist already refuses every opcode a handler
     * needs; this says so rather than relying on it. */
    if (cfn->exceptionCount > 0) return false;
    unsigned cidx = e->depth - argc - 1;
    unsigned maxSlot = 0;
    bool readsUpvalue = false;
    if (!inlinableBody(callee, argc, &maxSlot, &readsUpvalue)) return false;
    if (readsUpvalue && calleeReg < 0) return false;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] inlining %s\n",
                cfn->name ? cfn->name->chars : "<anon>");
    }

    /* Every argument has to be in a register, since that is where the body
     * will read its parameters from. */
    for (unsigned i = 0; i < argc; i++) {
        if (!holdsRegister(e->stack[cidx + 1u + i])) return false;
    }

    int savedSlot[JIT_MAX_SLOTS + 1];
    memcpy(savedSlot, e->inlSlot, sizeof savedSlot);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) e->inlSlot[i] = -1;
    for (unsigned i = 0; i < argc; i++) e->inlSlot[1u + i] = (int)(cidx + 1u + i);

    /* No noteScratchClobber here. An inlined body cannot call -- inlinableBody
     * admits nothing that does -- so it destroys x0..x8 only by USING them,
     * which is not a clobber but an allocation: when the caller already owns
     * that bank the two share one numbering (see inlineOwnBank), and when it
     * does not, the inlined entries have x0..x8 to themselves as before.
     * Anything inside that really does call still reaches noteScratchClobber
     * on its own, and under scratchValues that declines the compile. */
    e->inlining     = true;
    e->inlDepth     = cidx + 1u + argc;
    e->inlPinned    = 0;
    e->inlValueBase = e->valueDepth;
    e->inlIp        = callOff;
    e->inlClosureReg = calleeReg;

    /* The callee's own offset map, so its offsets cannot land in the
     * caller's. Nothing reads it back -- there are no branches -- but
     * compileBody writes one entry per instruction either way. */
    int cmap[129], cdepths[129];
    for (int i = 0; i <= cfn->chunk.count; i++) { cmap[i] = -1; cdepths[i] = -1; }
    int *savedMap = e->offsetToInst, *savedDepths = e->offsetToDepth;
    unsigned savedCarry = e->fpCarryCount;
    uint32_t savedCurOffset = e->curOffset;
    unsigned savedInstDepth = e->instDepth;
    unsigned savedInstValue = e->instValueDepth;
    e->offsetToInst = cmap;
    e->offsetToDepth = cdepths;

    bool ok = compileBody(e, callee);

    e->offsetToInst = savedMap;
    e->offsetToDepth = savedDepths;
    e->fpCarryCount = savedCarry;
    e->curOffset = savedCurOffset;
    e->instDepth = savedInstDepth;
    e->instValueDepth = savedInstValue;

    if (!ok || e->failed) {
        /* Instructions have been written; there is no taking them back. The
         * whole compile is retried with inlining off, which is the same answer
         * the register budget already gets. */
        e->inlining = false;
        memcpy(e->inlSlot, savedSlot, sizeof savedSlot);
        gInlineFailed = true;
        e->failed = true;
        return false;
    }

    /* OP_RETURN left the result on top and everything the body pinned beneath
     * it. Both are read while `inlining` is still set, because that is what
     * says which bank they are in; only the result's new home belongs to the
     * caller. */
    unsigned rres;
    SlotKind kres;
    uint32_t rshape;
    ObjClass *rcls;
    if (e->depth <= cidx) { e->failed = true; return false; }
    rshape = e->stackShape[e->depth - 1];
    rcls   = e->stackClass[e->depth - 1];
    /* Read while `inlining` is still set, so this names the inlined bank's d
     * register; the caller's own is taken after it is cleared. */
    bool rfp = (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
    unsigned rfpReg = rfp ? fpHeldIn(e, e->valueDepth - 1) : 0;
    if (rfp) {
        if (!popValueRaw(e, &rres, &kres)) { e->failed = true; return false; }
    } else if (!popValue(e, &rres, &kres)) { e->failed = true; return false; }
    /* Raw, because nothing reads these again: the body is over and its pinned
     * locals go with it, so materialising one costs an instruction whose
     * destination is dead. */
    while (e->depth > cidx) {
        if (holdsRegister(e->stack[e->depth - 1])) {
            unsigned r;
            if (!popValueRaw(e, &r, NULL)) { e->failed = true; return false; }
        } else {
            e->depth--;
        }
    }
    e->inlining = false;
    memcpy(e->inlSlot, savedSlot, sizeof savedSlot);

    if (!pushValue(e, kres, rshape, rcls)) { e->failed = true; return false; }
    if (rfp) {
        unsigned dd = fpRegAt(e, e->valueDepth - 1);
        if (dd != rfpReg) emit(e, jaiA64FmovDD(dd, rfpReg));
        fpClaim(e, e->valueDepth - 1);
    } else {
        unsigned dst = pushReg(e) - 1;
        if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    }
    e->inlined = true;
    return true;
}

/* What a descriptor call does with its result, once the arguments are consumed:
 * the predicted kind types the entry pushed for it, the tag that actually comes
 * back is checked, and a surprise deopts to the instruction AFTER the call --
 * which has happened and must not happen twice.
 *
 * Shared by the global-call and module-call arms below. They differ only in
 * what they consume before it and in how the callee was resolved; from the
 * descriptor's `result` onwards there is nothing to tell them apart. */
static bool emitCallOutResult(Emit *e, SlotKind rk, uint32_t rshape,
                              ObjClass *rcls, uint32_t after, uint8_t robj) {
    if (!pushValue(e, rk, rshape, rcls)) return false;
    /* A PREDICTION recorded on the entry, not a guard: every consumer of a
     * SLOT_OBJ checks Obj.type for itself before it reads anything, so this
     * only ever chooses which guard to emit and never deletes one. */
    if (retObjTypeOn() && rk == SLOT_OBJ && e->depth > 0) {
        e->stackObjType[e->depth - 1] = robj;
    }

    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    if (rk == SLOT_MAYBE_INST) {
        emitMaybeInstResult(e, pushReg(e) - 1, rat, rshape, after);
        return true;
    }
    unsigned wantTag = rk == SLOT_INT   ? VAL_INT
                     : rk == SLOT_FLOAT ? VAL_FLOAT
                     : rk == SLOT_BOOL  ? VAL_BOOL
                     : rk == SLOT_NULL  ? VAL_NULL
                                        : VAL_OBJ;
    emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, rat));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, wantTag));
    branchOnDeoptAt(e, JAI_A64_NE, after, true);
    /* A null carries no payload worth loading, but the register still stands
     * for the entry and a deopt materialises it, so it gets a defined zero
     * rather than whatever the descriptor happened to leave behind. */
    if (rk == SLOT_NULL) emit(e, jaiA64MovzX(pushReg(e) - 1, 0, 0));
    /* A byte, not a word: BOOL_VAL writes the union's `bool` member and leaves
     * the other seven bytes of the payload indeterminate, so a 64-bit load
     * brings back whatever the slot held before. The register stands for a
     * bool from here on and everything downstream tests it against zero, so
     * those bytes read as true -- `values.map(|v| is_nan(v))` came back all
     * true over a list with no NaN in it. Every other descriptor return site
     * already splits the two; this one did not. */
    else if (rk == SLOT_BOOL) emit(e, jaiA64LdrByte(pushReg(e) - 1, 31, rat + 8));
    else emit(e, jaiA64LdrX(pushReg(e) - 1, 31, rat + 8));
    if (rk == SLOT_INST) {
        /* The tag says "an object", which is not "an instance of this class",
         * and every field offset resolved against the entry below assumes it
         * is. The object type is checked before `klass` is read for the same
         * reason it is at the invoke arm: VAL_OBJ covers every heap object and
         * a returned string's header is shorter than an instance's. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(ObjInstance, klass)));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                           (unsigned)offsetof(ObjClass, shapeId)));
        emitConst64(e, JIT_SCRATCH_B, (int64_t)rshape);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    } else if (rk == SLOT_LIST) {
        /* Same hazard as SLOT_INST above: a callee entered with another
         * specialisation runs interpreted and may return any type, so
         * VAL_OBJ alone does not prove the payload is a list before
         * downstream code reads ObjList's fields off it unguarded. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeoptAt(e, JAI_A64_NE, after, true);
    }
    e->wroteHeap = true;
    return true;
}

/* A call to a global function that has itself compiled. Its return kind types
 * the result; the tag that actually comes back is checked, and a surprise
 * deopts to the instruction after the call, since the call has happened. */
bool emitGlobalCall(Emit *e, ObjFunction *caller, unsigned argc,
                           uint32_t callOff, uint32_t after) {
    unsigned cidx = e->depth - argc - 1;
    Value cv = e->stackSeen[cidx];
    if (!IS_CLOSURE(cv)) { e->whyNot = "callee vanished"; return false; }
    ObjFunction *cfn = AS_CLOSURE(cv)->fn;

    /* Straight to the callee's entry when everything jaiJitEnterFunc would
     * have checked can be checked here instead. Falling back rather than
     * declining matters: the descriptor path speaks a much wider language --
     * any argument kind, any module, a callee that writes -- and a call
     * through jaiCallValue still beats no compiled loop at all. */
    if (cfn->jitFunc != NULL) {
        const char *saved = e->whyNot;
        if (emitDirectCall(e, caller, cfn, cv, -1, cidx, argc, callOff,
                           after, false)) {
            return true;
        }
        if (e->failed) return false;   /* it had started emitting */
        e->whyNot = saved;
    }

    /* A callee with no compiled form of its own still knows what it has been
     * returning; see ObjFunction::obsReturnKind and the twin case at OP_INVOKE.
     * A recursive function is the ordinary way to reach this -- the loop being
     * compiled is inside the very function the call names, so there is nothing
     * for `jitReturnKind` to have been written by yet. */
    /* Observed first, compiled kind second -- the order the sibling site at
     * emitGlobalCall spells out, and which this one did not have.
     *
     * `jitFunc != NULL` does not make `jitReturnKind` a fact; it is stored
     * unconditionally at the end of a compile, so a body that compiled a
     * PREFIX and took the unarmed path before any OP_RETURN advertises the
     * SLOT_INT a zeroed Emit starts on. That is not hypothetical here:
     * `_is_ident_start` in the lexer walks only to OP_GET_GLOBAL at offset 0
     * -- a cold `throw` on its first instruction -- so it compiles nothing and
     * still claims to return an int. `_is_ident_cont`, which is
     * `_is_ident_start(c) or _is_digit(c)`, then declined on "a branch on a
     * int, not a bool", and `_ident_run_end`'s OSR loop retried it eighty
     * times waiting for a function that could never compile.
     *
     * A refusal is a chain, and this was three links of one. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    bool haveKind = observedReturnKind(cfn, &rk, &rshape, &robj);
    if (!haveKind && cfn->jitFunc != NULL) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
        haveKind = true;
    }
    if (haveKind && (rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        e->whyNot = "callee's return class not on record";
        return false;
    }
    if (!haveKind ||
        (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
         rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
         rk != SLOT_OBJ && rk != SLOT_NULL)) {
        e->whyNot = "callee's return kind not usable";
        return false;
    }

    if (!emitDescriptor(e, cv, e->depth - argc, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_FUNC) return false;
    e->depth--;
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* `math.sqrt(x)` is not a method call. It is a global call whose callee is
 * resolved through ANOTHER module, so what was missing was never the call
 * machinery -- it is the pinning.
 *
 * Two guards, in this order, both before anything is consumed so a miss resumes
 * at the invoke with the receiver and the arguments untouched:
 *
 *   THIS module. OP_GET_GLOBAL loads `math` by address behind a VAL_OBJ tag
 *   guard, and the arm's own object-type guard would only prove "some module",
 *   so the receiver is compared against the ObjModule this compiled against.
 *   Everything below -- the version word's address, the resolved closure --
 *   belongs to that one module.
 *
 *   STILL this binding. ObjModule::version is the counter for a memoised
 *   global VALUE or a resolved callee, and it is the one that moves when
 *   `math.sqrt = f` overwrites the member (jaiModuleSet bumps it because a
 *   closure is not inert). `globals.keyVersion` is the WRONG counter here and
 *   would be silent: overwriting an existing key never moves its entry, which
 *   is the whole reason keyVersion exists. Guarding it per call rather than
 *   at entry also covers a rebinding from inside the loop, which the entry
 *   check by itself does not -- see the note at OP_SET_GLOBAL.
 *
 * The callee must have compiled. The standing warning at OP_GET_GLOBAL says
 * admitting a callee whose jitFunc is NULL MISCOMPILES for a reason nobody has
 * written down; this arm does not cross it, and pays nothing for that --
 * `math.sqrt` compiles long before any loop calling it does.
 *
 * The receiver is dropped rather than passed: a module is not an argument. */
bool emitModuleCall(Emit *e, ObjModule *m, Value calleeVal,
                           unsigned ridx, unsigned argc, uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a module member that has not compiled");
    }
    /* WHICH RECORD SAYS WHAT COMES BACK, and it is not the obvious one.
     * `jitFunc != NULL` does NOT make `jitReturnKind` a fact: it is stored
     * unconditionally at the end of a compile, so a body whose walk never
     * reached an OP_RETURN -- one that compiled a prefix and bails -- leaves it
     * at the SLOT_INT a zeroed Emit starts on. math.sqrt and math.sin are both
     * exactly that (`sawReturn=0`, `obsReturnKind=float`), so trusting the
     * compiled kind here predicted int, the tag guard below failed on the first
     * call, and the loop left compiled code every iteration: p27 did not move at
     * all. The interpreter's own per-callee record is a measured fact and is
     * asked first; the compiled kind stands in only when there is none.
     *
     * Either way it is a prediction, and the tag guard after the call is what
     * makes it sound -- the same contract emitGlobalCall states. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    if (!observedReturnKind(cfn, &rk, &rshape, &robj)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a module member's return kind (%d)", (int)rk);
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a module member's return class is not on record");
    }
    /* Every entry this consumes, receiver included, has to be one the
     * descriptor can pass -- asked HERE rather than left to emitDescriptor,
     * which refuses only after the guards below have been emitted and would
     * turn a graceful decline into a whole-body one. It also settles the
     * register arithmetic: the receiver is named by counting back from the top
     * of the value bank, which is the same thing as counting back from the top
     * of the model only while every entry between them holds a register, and
     * every kind admitted here does. */
    for (unsigned i = 0; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a module call over an entry of kind %d", (int)k);
        }
    }
    /* Past here the guards are emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    unsigned rRecv = valueXReg(e, e->valueDepth - argc - 1);
    emitConst64(e, JIT_SCRATCH_C, (int64_t)(uintptr_t)m);
    emit(e, jaiA64SubsXReg(31, rRecv, JIT_SCRATCH_C));
    branchOnDeopt(e, JAI_A64_NE);

    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)m->version);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* What `__prim__.f64_sqrt` and its kin answer, since unlike a compiled
 * closure a native carries no jitReturnKind/observedReturnKind to ask --
 * emitModuleCall's whole "which record says what comes back" problem does not
 * apply, because there is no record. Scoped to natives reached ONLY through a
 * MODULE receiver (emitModuleNativeCall's one caller): every row here is one
 * of lib/std/math.jai's own `__prim__.f64_*` callees, registered in
 * builtins_math.c, none of which allocates or can throw past a domain check
 * the CALLER (math.jai) already made before reaching the primitive -- see
 * emitModuleNativeCall's own comment for how `sqrt`'s negative-argument raise
 * stays correct despite that.
 *
 * A row missing here is a decline, not a wrong answer -- the tag guard
 * downstream only matters for a row that IS present, same contract
 * kNativeResults states above. frexp/modf are deliberately absent: both
 * return `tuple[float, int]`, a kind this arm has no row shape for. */
typedef struct {
    const char *name;
    unsigned    argc;
    SlotKind    kind;
} PrimNativeResult;

static const PrimNativeResult kPrimNativeResults[] = {
    { "f64_sqrt",     1, SLOT_FLOAT },
    { "f64_exp",      1, SLOT_FLOAT },
    { "f64_log",      1, SLOT_FLOAT },
    { "f64_log2",     1, SLOT_FLOAT },
    { "f64_log10",    1, SLOT_FLOAT },
    { "f64_sin",      1, SLOT_FLOAT },
    { "f64_cos",      1, SLOT_FLOAT },
    { "f64_tan",      1, SLOT_FLOAT },
    { "f64_asin",     1, SLOT_FLOAT },
    { "f64_acos",     1, SLOT_FLOAT },
    { "f64_atan",     1, SLOT_FLOAT },
    { "f64_atan2",    2, SLOT_FLOAT },
    { "f64_sinh",     1, SLOT_FLOAT },
    { "f64_cosh",     1, SLOT_FLOAT },
    { "f64_tanh",     1, SLOT_FLOAT },
    { "f64_asinh",    1, SLOT_FLOAT },
    { "f64_acosh",    1, SLOT_FLOAT },
    { "f64_atanh",    1, SLOT_FLOAT },
    { "f64_floor",    1, SLOT_FLOAT },
    { "f64_ceil",     1, SLOT_FLOAT },
    { "f64_trunc",    1, SLOT_FLOAT },
    { "f64_round",    1, SLOT_FLOAT },
    { "f64_fmod",     2, SLOT_FLOAT },
    { "f64_pow",      2, SLOT_FLOAT },
    { "f64_hypot",    2, SLOT_FLOAT },
    { "f64_copysign", 2, SLOT_FLOAT },
    { "f64_ldexp",    2, SLOT_FLOAT },
    { "f64_erf",      1, SLOT_FLOAT },
    { "f64_gamma",    1, SLOT_FLOAT },
    { "f64_lgamma",   1, SLOT_FLOAT },
    { "f64_is_nan",    1, SLOT_BOOL },
    { "f64_is_inf",    1, SLOT_BOOL },
    { "f64_is_finite", 1, SLOT_BOOL },
};

static bool primNativeResultKind(const char *nm, unsigned argc, SlotKind *k) {
    for (size_t i = 0; i < sizeof kPrimNativeResults / sizeof kPrimNativeResults[0]; i++) {
        if (kPrimNativeResults[i].argc == argc &&
            strcmp(kPrimNativeResults[i].name, nm) == 0) {
            *k = kPrimNativeResults[i].kind;
            return true;
        }
    }
    return false;
}

/* `__prim__.f64_sqrt(x)` and its kin -- a NATIVE reached through a MODULE
 * receiver, the other half of what emitModuleCall does for a Jaithon-written
 * one (`math.sqrt`, wrapping this very call). Same shape, same reason: a
 * member of another namespace, resolved at compile time, called with the
 * receiver DROPPED -- resolveInvokeTarget's IS_MODULE arm (vm.c) hands back
 * jaiBuiltinMethod's raw result with `isMethod` left false, so invokeCallable
 * runs it as a PLAIN call, not a method call. jaiModuleMethod (module_methods.c)
 * returns the ObjNative straight out of `m->globals` with no bound wrapper --
 * `moduleExposes` succeeds on the first check because `__prim__`'s
 * `exports.count` is 0 (nothing ever declares an export list for a namespace
 * jaiDefineNative built, so every member reads as exposed) -- so `onative` at
 * the call site already IS the plain native, never a bound one to unwrap.
 *
 * jitCallOut's jaiCallValue -> invokeCallable dispatches on OBJ_NATIVE with
 * `args[0]` as the first REAL argument and no receiver slot at all (vm.c's
 * `case OBJ_NATIVE:` in invokeCallable) -- exactly the descriptor
 * emitDescriptor already builds from `ridx + 1, argc` for emitModuleCall's
 * closures, so the same call-out helper reaches a native correctly with no
 * changes of its own. The only work here is specific to a NATIVE: what it
 * returns (kPrimNativeResults, since there is no jitReturnKind to ask) and the
 * guards, which are NOT the same two as emitModuleCall's.
 *
 * THE RECEIVER IDENTITY GUARD emitModuleCall emits is *provably* redundant
 * here and is skipped: `math`'s receiver register is loaded from a JaiEntry
 * (globalSlot's "value case" arm), a genuinely runtime-variable location an
 * import could rebind, so comparing it against the ObjModule compiled against
 * is live work. `__prim__`'s register is instead loaded by
 * `emitConst64(e, dst, ...)` directly in OP_GET_GLOBAL's globalNamespace
 * branch -- a compile-time CONSTANT baked into the instruction stream, which
 * cannot hold anything else at run time by construction, so re-checking it
 * against itself would prove nothing a bug in the emitter could not also get
 * wrong in the omitted check.
 *
 * `m->version` IS kept, and unlike the identity check it is NOT redundant:
 * `mod.attr = v` is real syntax for any module receiver (jaiSetProperty's
 * IS_MODULE arm, vm.c) and `__prim__` is reachable as a bare identifier --
 * lib/std/math.jai names it in the open -- so `__prim__.f64_sqrt = something`
 * is something a running program could actually do. jaiModuleSet bumps
 * `m->version` on exactly that kind of write (ObjNative is not
 * jaiValueIsInertGlobal), which is what retires this compiled form if it
 * happens after the bake.
 *
 * `sqrt`'s own domain check (`if x < 0.0 { throw ValueError(...) }`,
 * lib/std/math.jai:217) is ordinary Jaithon ahead of this call and compiles
 * on its own merits -- ints/floats/branches/throw are all arms this tier
 * already has -- so it is not this arm's problem to solve; declining THIS
 * call alone (a too-wide argc, an unknown name) still leaves the raise
 * compiled, and only the call after it falls back to the interpreter via
 * emitUnarmedDeopt. `f64_sqrt` raising its OWN domain error for a negative
 * input it should never see is still reachable and still correct either way:
 * callNativeAt raises through the ordinary exception path jitCallOut's
 * `raiseExitAllowed` branch already handles, message and all -- nothing about
 * going through a descriptor changes what the native itself decides to
 * raise. */
/* Whether the `__prim__` read at `off` is paired with an OP_INVOKE this tier
 * can actually compile. If it is not, the namespace is left unresolved so the
 * read falls to the unarmed path exactly as it did before this arm existed --
 * which for a call the tier cannot emit is strictly better than resolving it.
 *
 * WHY THIS EXISTS. `span` ends in `__prim__.fill_span(...)` with ten arguments
 * and `fill_convex` with eleven. While `__prim__` had no arm the walk skipped
 * receiver, pushes and invoke as ONE unarmed block and span's prologue
 * compiled; resolving the namespace made the walk continue into those pushes
 * and hit a wall it cannot pass -- the argc cap, then the result-kind
 * whitelist, then the register budget, each a hard whole-body decline. span's
 * interpreted work DOUBLED, 5,821,736 to 10,867,344, from arming a
 * neighbouring opcode.
 *
 * PAIRING IS THE WHOLE DIFFICULTY, and an earlier attempt got it wrong: it
 * took the FIRST OP_INVOKE after the read, which is not the paired one
 * whenever an argument expression contains its own method call. When that
 * inner invoke happened to be whitelisted the lookahead said yes and the OUTER
 * one declined the body -- the very thing it was written to prevent.
 *
 * `chunkDepth` settles it exactly. It is the operand-stack depth BEFORE each
 * instruction, so the invoke paired with this read is the one that consumes
 * back to the depth the read started from: `chunkDepth[p] - (argc + 1) ==
 * chunkDepth[off]`. A nested invoke inside an argument sits deeper and cannot
 * match. Where the table is unavailable -- OSR, inlining -- the answer is no,
 * which keeps the old unarmed behaviour rather than guessing. */
bool primInvokePairFits(const Emit *e, const Chunk *chunk, uint32_t off) {
    if (e->chunkDepth == NULL || e->osr || e->inlining) return false;
    if (off >= (uint32_t)e->chunkDepthCount) return false;
    const int base = e->chunkDepth[off];
    const uint8_t *code = chunk->code;
    uint32_t p = off + 6;
    while (p < (uint32_t)chunk->count) {
        if (p >= (uint32_t)e->chunkDepthCount) return false;
        if (code[p] == OP_INVOKE) {
            unsigned argc = code[p + 4];
            if (e->chunkDepth[p] - (int)(argc + 1u) == base) {
                if (argc + 1u > (unsigned)JIT_MAX_ARGS_OUT) return false;
                uint32_t nameIdx = jaiReadU24(code + p + 1);
                if (nameIdx >= (uint32_t)chunk->constants.count) return false;
                Value nm = chunk->constants.data[nameIdx];
                if (!IS_STRING(nm)) return false;
                SlotKind rk;
                return primNativeResultKind(AS_STRING(nm)->chars, argc, &rk);
            }
        }
        unsigned len = instructionLength(chunk, p);
        if (len == 0) return false;
        p += len;
    }
    return false;
}

bool emitModuleNativeCall(Emit *e, ObjModule *m, Value calleeVal,
                                 unsigned ridx, unsigned argc, uint32_t after) {
    ObjNative *nat = AS_NATIVE(calleeVal);
    SlotKind rk;
    if (!primNativeResultKind(nat->name != NULL ? nat->name->chars : "",
                              argc, &rk)) {
        return subWhy(e, "`%s.%s`'s result kind is not on record",
                      m->name != NULL ? m->name->chars : "?",
                      nat->name != NULL ? nat->name->chars : "?");
    }
    for (unsigned i = 0; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a module call over an entry of kind %d", (int)k);
        }
    }
    /* Past here the guard is emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)&m->version);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)m->version);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    /* A __prim__ native's result kind comes from the whitelist, not from a
     * callee record, so there is no observed object type to carry. */
    return emitCallOutResult(e, rk, 0, NULL, after, 0);
}

/* `Klass.static_method(args)` -- an OP_INVOKE whose receiver is a CLASS. The
 * same job emitModuleCall does for `math.sqrt(x)`, and for the same reason: a
 * member of another namespace, resolved at compile time, called with the
 * receiver DROPPED. The interpreter drops it too. resolveInvokeTarget (vm.c)
 * leaves the class in slot 0 and reports `isMethod == false`, so the call goes
 * through invokeCallable and the closure's first parameter is the first
 * ARGUMENT, not the class -- which is exactly what jitCallOut does with a
 * descriptor holding argc arguments and no receiver.
 *
 * Three things differ from the module case, and only the third is machinery.
 *
 *   THE RECEIVER NEEDS NO GUARD. A module receiver is loaded from a global by
 *   address behind a bare VAL_OBJ tag check, so emitModuleCall has to compare
 *   it against the one ObjModule it compiled against. A class receiver is not
 *   in a register at all: SLOT_CLASS holds nothing (holdsRegister says so) and
 *   the ObjClass came from OP_GET_GLOBAL's `globalClass` arm, which resolves it
 *   BY VALUE and is retired wholesale by the module version check at entry if
 *   the name is rebound. There is nothing here that could be a different class
 *   at run time than it was at compile time.
 *
 *   THE MEMBER IS RESOLVED OUT OF `klass->statics`, DIRECTLY. Not through
 *   jaiBuiltinMethod, and not through anything that can hand back a BoundMethod
 *   to unwrap: unwrapping one and then dropping the receiver as this arm does
 *   loses both halves and calls an unbound closure with the first real argument
 *   sitting where `self` belongs. Only an IS_CLOSURE value straight out of the
 *   table is admitted. `klass->methods` is deliberately NOT consulted -- see
 *   the call site, which declines an instance method named through the class.
 *
 *   WHAT RETIRES THE BAKED CALLEE is the BINDING itself, re-read from the
 *   statics entry on every call and compared against the closure this site was
 *   compiled against. `Klass.name = v` reaches jaiSetProperty's IS_CLASS arm,
 *   which requires the key to be present already and then overwrites in place,
 *   so the entry never moves and reading it back asks the direct question: is
 *   this still what I compiled for.
 *
 *   It was a COUNTER first -- `klass->statics.version`, which tableSetHashed
 *   bumps on every value write -- and that was unsound. The field is uint32_t
 *   (table.h) and nothing filters the bump the way jaiModuleSet filters
 *   ObjModule::version through jaiValueIsInertGlobal, so an ordinary
 *   `Klass.counter = n` loop drives it at 51M writes/sec, measured. Land the
 *   count exactly 2^32 on from the bake and the guard reads the value it baked
 *   while the binding has changed: `Box.make` rebound after 2^32 writes
 *   answered 400008, against 2000000 from the interpreter, from the arm
 *   switched off, and from a control one write short. That is about 84 seconds
 *   of a loop any program might contain, not an unreachable corner.
 *
 *   Reading the binding is also strictly less trigger-happy than the counter,
 *   which retired the callee whenever any OTHER static of the same class was
 *   written. It costs 1.8% of the win (0.551s -> 0.561s on a 32M-call probe
 *   against 1.130s with the arm off).
 *
 * `keyVersion` is still needed, for a different job: it makes the entry ADDRESS
 * trustworthy. It moves on rehash, delete and clear -- never on an overwrite
 * (table.c: insertAt bumps it only for a NEW key) -- so unlike `version` a
 * running program cannot drive it. It is the guard the static-FIELD arm stands
 * on, baked per site here rather than through e->staticsTable so that a body
 * naming two classes still compiles both. */
bool emitClassCall(Emit *e, ObjClass *klass, JaiEntry *slot,
                          Value calleeVal, unsigned ridx, unsigned argc,
                          uint32_t after) {
    ObjFunction *cfn = AS_CLOSURE(calleeVal)->fn;
    if (cfn->jitFunc == NULL) {
        return subWhy(e, "a static that has not compiled");
    }
    /* Which record says what comes back: emitModuleCall's finding, and it is
     * not the obvious one. `jitFunc != NULL` does NOT make `jitReturnKind` a
     * fact -- it is stored unconditionally at the end of a compile, so a body
     * whose walk never reached an OP_RETURN leaves it at the SLOT_INT a zeroed
     * Emit starts on. The interpreter's own per-callee record is a measured
     * fact and is asked first; the compiled kind stands in only when there is
     * none. Either way it is a prediction, and emitCallOutResult's tag guard
     * after the call is what makes it sound. */
    SlotKind rk = SLOT_NULL;
    uint32_t rshape = 0;
    ObjClass *rcls = NULL;
    uint8_t robj = 0;
    if (!observedReturnKind(cfn, &rk, &rshape, &robj)) {
        rk = (SlotKind)cfn->jitReturnKind;
        rshape = cfn->jitReturnShape;
    }
    if (rk != SLOT_INT && rk != SLOT_FLOAT && rk != SLOT_BOOL &&
        rk != SLOT_INST && rk != SLOT_MAYBE_INST && rk != SLOT_LIST &&
        rk != SLOT_OBJ && rk != SLOT_NULL) {
        return subWhy(e, "a static's return kind (%s)", slotKindName(rk));
    }
    if ((rk == SLOT_INST || rk == SLOT_MAYBE_INST) &&
        (rshape == 0 || !jaiClassForShape(rshape, &rcls) || rcls == NULL)) {
        return subWhy(e, "a static's return class is not on record");
    }
    /* The ARGUMENTS only. The receiver is skipped where emitModuleCall checks
     * it, because a class entry holds no register and emitDescriptorStatus
     * already knows how to bake one -- but it is never passed here, so even
     * that does not arise. Asked HERE rather than left to emitDescriptor, which
     * refuses only after the guard below has been emitted and would turn a
     * graceful decline into a whole-body one. */
    for (unsigned i = 1; i <= argc; i++) {
        SlotKind k = e->stack[ridx + i];
        if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL &&
            k != SLOT_INST && k != SLOT_LIST && k != SLOT_OBJ &&
            k != SLOT_ITER && k != SLOT_MAYBE_INST) {
            return subWhy(e, "a static call over an argument of kind %s",
                          slotKindName(k));
        }
    }
    /* Past here the guard is emitted, so a later refusal stops the compile
     * rather than falling back onto a half-written instruction stream. */

    settleAll(e);
    emitConst64(e, JIT_SCRATCH_D,
                (int64_t)(uintptr_t)&klass->statics.keyVersion);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uint32_t)klass->statics.keyVersion);
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    /* The tag before the pointer, so a static rebound to an int whose payload
     * happened to equal the closure's address is not called as if it were the
     * closure.
     *
     * Comparing against the LIVE binding is also what makes address recycling
     * harmless rather than dangerous. If the old closure were collected and a
     * new object took its address, the object bound NOW is the one at that
     * address, and emitDescriptor's callee slot goes through jaiCallValue,
     * which dispatches on the value dynamically -- including raising, if what
     * is bound there is not callable. */
    emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)slot);
    emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value)));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, VAL_OBJ));
    branchOnDeopt(e, JAI_A64_NE);
    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                       (unsigned)offsetof(JaiEntry, value) + 8u));
    emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)AS_OBJ(calleeVal));
    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
    branchOnDeopt(e, JAI_A64_NE);

    if (!emitDescriptor(e, calleeVal, ridx + 1, argc, (void *)&jitCallOut)) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) return false;
    }
    /* The receiver held no register, so dropping it is the whole of popping it
     * -- popValue would refuse it via holdsRegister. Same as the static-field
     * arm at OP_GET_FIELD. */
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
    e->depth--;
    return emitCallOutResult(e, rk, rshape, rcls, after, robj);
}

/* Emit a method's body directly, when that body is one expression.
 *
 * Deliberately narrow: no jumps, no stores, no calls, only field reads of its
 * own parameters and int or float arithmetic. Those restrictions are what make
 * a second walker over the callee's bytecode safe to write -- with no branches
 * there is no offset map to keep, and with no stores there is nothing to undo
 * if a guard inside it deoptimises to the call site.
 *
 * Reached through inlineMethod, which is what puts the model back when this
 * declines -- see there. */
static bool inlineMethodWalk(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                             unsigned argc, int callOff) {
    if (e->noInline) return false;
    if (e->splitAt != 0) return false;   /* see inlineGlobalCall */
    unsigned ridx = e->depth - argc - 1;
    ObjClass *rcls = e->stackClass[ridx];
    if (rcls == NULL) return false;
    ObjFunction *cfn = closure->fn;
    if (nameIdx >= (uint32_t)cfn->chunk.constants.count) return false;
    Value mname = cfn->chunk.constants.data[nameIdx];
    if (!IS_STRING(mname)) return false;
    Value method;
    if (!jaiClassFindMethod(rcls, AS_STRING(mname), &method)) return false;
    if (!IS_CLOSURE(method)) return false;
    ObjFunction *mfn = AS_CLOSURE(method)->fn;
    if (mfn->exceptionCount > 0) return false;   /* see inlineGlobalCall */
    if (mfn->arity != argc || mfn->defaultCount != 0) return false;
    if (mfn->flags & (FN_VARIADIC | FN_KWREST | FN_INIT)) return false;
    if (mfn->upvalueCount != 0) return false;
    if (mfn->chunk.count > 96) return false;

    unsigned inReg[JIT_MAX_ARGS_OUT + 1];
    Value    inSeen[JIT_MAX_ARGS_OUT + 1];
    ObjClass *inCls[JIT_MAX_ARGS_OUT + 1];
    for (unsigned i = 0; i <= argc; i++) {
        unsigned idx = ridx + i;
        if (!holdsRegister(e->stack[idx])) return false;
        inReg[i]  = valueBankReg(e, idx - (e->depth - e->valueDepth));
        inSeen[i] = e->stackSeen[idx];
        inCls[i]  = e->stackClass[idx];
    }

    /* A dry walk first: nothing is emitted until the whole body is known to
     * be expressible, because a half-inlined body cannot be taken back. */
    const uint8_t *c = mfn->chunk.code;
    int n = mfn->chunk.count;
    for (int pass = 0; pass < 2; pass++) {
        int depth0 = (int)e->depth;
        for (int o = 0; o < n;) {
            uint8_t op = c[o];
            if (op == OP_GET_FIELD_LOCAL) {
                unsigned slot = jaiReadU16(c + o + 1);
                uint32_t nidx = jaiReadU24(c + o + 3);
                if (slot > argc) return false;
                if (e->stack[ridx + slot] != SLOT_INST) return false;
                if (nidx >= (uint32_t)mfn->chunk.constants.count) return false;
                Value fname = mfn->chunk.constants.data[nidx];
                if (!IS_STRING(fname)) return false;
                const FieldInfo *fi =
                    jaiClassFieldInfo(inCls[slot], AS_STRING(fname));
                if (fi == NULL || fi->isStatic) return false;
                if (!IS_INSTANCE(inSeen[slot])) return false;
                ObjInstance *si = AS_INSTANCE(inSeen[slot]);
                if (fi->slot >= si->fieldCount) return false;
                Value fv = si->fields[fi->slot];
                SlotKind fk; unsigned ftag;
                if (IS_INT(fv))        { fk = SLOT_INT;   ftag = VAL_INT; }
                else if (IS_FLOAT(fv)) { fk = SLOT_FLOAT; ftag = VAL_FLOAT; }
                else return false;
                unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                                 (unsigned)fi->slot * (unsigned)sizeof(Value);
                if (pass == 1) {
                    emit(e, jaiA64LdrW(JIT_SCRATCH_A, inReg[slot], fbase));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ftag));
                    branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)callOff, false);
                }
                if (!pushValue(e, fk, 0, NULL)) return false;
                if (pass == 1) {
                    emit(e, jaiA64LdrX(pushReg(e) - 1, inReg[slot], fbase + 8));
                }
                o += 8;
                continue;
            }
            if (op == OP_ADD || op == OP_SUB || op == OP_MUL) {
                unsigned rb, ra; SlotKind kb, ka;
                if (!popValue(e, &rb, &kb)) return false;
                if (!popValue(e, &ra, &ka)) return false;
                if (ka != kb) return false;
                if (ka != SLOT_INT && ka != SLOT_FLOAT) return false;
                if (!pushValue(e, ka, 0, NULL)) return false;
                unsigned rd = pushReg(e) - 1;
                if (pass == 1) {
                    if (ka == SLOT_FLOAT) {
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_A, ra));
                        emit(e, jaiA64FmovDX(JIT_FSCRATCH_B, rb));
                        emit(e, op == OP_ADD
                                 ? jaiA64FaddD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                             : op == OP_SUB
                                 ? jaiA64FsubD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B)
                                 : jaiA64FmulD(JIT_FSCRATCH_A, JIT_FSCRATCH_A, JIT_FSCRATCH_B));
                        emit(e, jaiA64FmovXD(rd, JIT_FSCRATCH_A));
                    } else if (op == OP_MUL) {
                        emit(e, jaiA64SmulhX(JIT_SCRATCH_A, ra, rb));
                        emit(e, jaiA64MulX(rd, ra, rb));
                        emit(e, jaiA64SubsXAsr(31, JIT_SCRATCH_A, rd, 63));
                        branchOnOverflow(e, 2u, JAI_A64_NE);
                    } else {
                        emit(e, op == OP_ADD ? jaiA64AddsX(rd, ra, rb)
                                             : jaiA64SubsXReg(rd, ra, rb));
                        branchOnOverflow(e, op == OP_ADD ? 0u : 1u, JAI_A64_VS);
                    }
                }
                o += 1;
                continue;
            }
            if (op == OP_RETURN) {
                if ((int)e->depth != depth0 + 1) return false;
                o += 1;
                if (o != n) return false;
                break;
            }
            return false;
        }
        if (pass == 0) {
            while ((int)e->depth > depth0) {
                unsigned r; if (!popValue(e, &r, NULL)) return false;
            }
        }
    }

    unsigned rres;
    SlotKind kres;
    if (!popValue(e, &rres, &kres)) return false;
    for (unsigned i = 0; i <= argc; i++) {
        unsigned r; if (!popValue(e, &r, NULL)) return false;
    }
    if (!pushValue(e, kres, 0, NULL)) return false;
    unsigned dst = pushReg(e) - 1;
    if (dst != rres) emit(e, jaiA64MovX(dst, rres));
    e->inlined = true;
    return true;
}

/* The model must be exactly where it was if the inline did not happen.
 *
 * inlineMethodWalk's dry pass pushes and pops as it reads the callee, and every
 * one of its two dozen refusals returns from the middle of that -- so on its
 * own it leaves the model as deep as the walk got. Its caller does NOT decline
 * when it declines: the OP_INVOKE arm falls through to the descriptor path,
 * which then names every later entry's register from an index that is too high
 * and, far worse, writes deopt records describing an operand stack the
 * interpreter does not have. `_crossings` in lib/std/gui/path.jai is the shape
 * that found this: `edge.crossing(y)` gets three instructions into `crossing`
 * before an OP_BIND stops the walk, so every deopt after it handed the
 * interpreter the receiver and the argument a second time and the next
 * instruction read a float where a list belonged.
 *
 * Unwinding here rather than at each `return false` is deliberate: there are
 * far too many of them to keep right by hand, and the dry pass's own tail
 * already pops back to its starting depth in exactly this way.
 *
 * A failure that has already emitted cannot be unwound at all -- the caller
 * would stack a second call sequence on top of half of this one -- so that
 * declines the compile instead. */
bool inlineMethod(Emit *e, ObjClosure *closure, uint32_t nameIdx,
                         unsigned argc, int callOff) {
    unsigned depth0 = e->depth;
    unsigned count0 = e->count;
    if (inlineMethodWalk(e, closure, nameIdx, argc, callOff)) return true;
    if (e->failed) return false;
    if (e->count != count0) { e->failed = true; return false; }
    while (e->depth > depth0) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->failed = true; return false; }
    }
    /* Below where it started is not something an unwind can repair: the
     * entries are the caller's and their registers are gone. */
    if (e->depth != depth0) { e->failed = true; return false; }
    return false;
}

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

bool emitCallOut(Emit *e, unsigned argc) {
    ObjClass *cls = e->stackClass[e->depth - argc - 1];
    if (cls == NULL) { e->whyNot = "callee class"; return false; }

    uint16_t fslots[JIT_MAX_ARGS_OUT];
    if (argc <= JIT_MAX_ARGS_OUT && simpleInitFields(cls, argc, fslots)) {
        unsigned first = e->depth - argc;
        SlotKind kinds[JIT_MAX_ARGS_OUT];
        unsigned regs[JIT_MAX_ARGS_OUT];
        for (unsigned i = 0; i < argc; i++) {
            kinds[i] = e->stack[first + i];
            if (kinds[i] != SLOT_INT && kinds[i] != SLOT_FLOAT &&
                kinds[i] != SLOT_BOOL && kinds[i] != SLOT_INST &&
                kinds[i] != SLOT_LIST && kinds[i] != SLOT_OBJ &&
                kinds[i] != SLOT_MAYBE_INST) {
                e->whyNot = "an argument kind a field cannot take";
                return false;
            }
            regs[i] = valueBankReg(e, first + i - (e->depth - e->valueDepth));
        }
        /* Fast path (jitInstanceAlloc) is a leaf that cannot collect -- it declines whenever jaiGCWanted(),
     * exactly when jaiInstanceNew would have collected -- so needs no descriptor/roots, versus the descriptor's dozen stores plus a root push/pop just to allocate 64 bytes. NULL means it did nothing, so falling into the descriptor path is always correct; both paths land at the load below with the instance in SCRATCH_C. Skipped for a class the small-object bins can't serve. */
        const size_t instBytes =
            sizeof(ObjInstance) + sizeof(Value) * (size_t)cls->fieldCount;
        unsigned skipSlow = 0;
        bool haveFast = jaiSmallServes(instBytes);
        if (haveFast) {
            emitConst64(e, 0, (int64_t)(uintptr_t)cls);
            emitConst64(e, JIT_SCRATCH_A,
                        (int64_t)(uintptr_t)&jitInstanceAlloc);
            noteScratchClobber(e);
            emit(e, jaiA64Blr(JIT_SCRATCH_A));
            emit(e, jaiA64MovX(JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, 0));
            skipSlow = e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));   /* patched below */
        }
        if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), first, 0,
                            (void *)&jitNewInstance)) {
            return false;
        }
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* The span is measured rather than counted: emitDescriptor's length
         * moves with the number of roots this body holds and with how many
         * halfwords the class pointer needs. */
        if (haveFast && skipSlow < e->count && e->count <= JIT_MAX_INSTS) {
            e->code[skipSlow] =
                jaiA64BCond(JAI_A64_NE, (int32_t)(e->count - skipSlow));
        }
        for (unsigned i = 0; i < argc; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) return false;
        e->depth--;
        if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
        /* Instance held in a scratch (SCRATCH_C) until every field is stored, since the result reuses the
     * first argument's register: loading it into its final register first overwrote the argument about to be stored into it -- alloc_churn came back in 5ms with a wrong answer. */
        unsigned rinst = pushReg(e) - 1;
        for (unsigned i = 0; i < argc; i++) {
            unsigned at = (unsigned)offsetof(ObjInstance, fields) +
                          (unsigned)fslots[i] * (unsigned)sizeof(Value);
            /* SCRATCH_C holds the instance, so the dynamic tag needs two
             * other scratches. */
            emitTagFor(e, kinds[i], regs[i], JIT_SCRATCH_A, JIT_SCRATCH_B);
            emit(e, jaiA64StrW(JIT_SCRATCH_A, JIT_SCRATCH_C, at));
            emit(e, jaiA64StrX(regs[i], JIT_SCRATCH_C, at + 8));
        }
        emit(e, jaiA64MovX(rinst, JIT_SCRATCH_C));
        e->wroteHeap = true;
        return true;
    }

    if (!emitDescriptor(e, OBJ_VAL((Obj *)cls), e->depth - argc, argc,
                        (void *)&jitCallOut)) {
        return false;
    }

    for (unsigned i = 0; i < argc; i++) {
        unsigned r;
        if (!popValue(e, &r, NULL)) { e->whyNot = "call argument"; return false; }
    }
    if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_CLASS) {
        e->whyNot = "callee was not where it should be";
        return false;
    }
    e->depth--;

    if (!pushValue(e, SLOT_INST, cls->shapeId, cls)) return false;
    emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                       e->descOffset + (unsigned)offsetof(JitCallDesc, result) + 8));
    /* A call is an effect: no bail may follow it, for the same reason no bail
     * may follow a store. */
    e->wroteHeap = true;
    return true;
}


#else

#endif
