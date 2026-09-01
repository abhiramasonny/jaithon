/* jit_func.c -- whole-function JIT tier: compiles self-recursive, integer-only bodies to native arm64, bailing to the interpreter on overflow, deep recursion, or an unsupported shape. */

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */

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

/* The read side of the dynamic-local contract stated at localTagInFrame: a
 * slot two paths disagreed about carries a run-time tag, and the kind this
 * read was compiled for is a speculation until that tag confirms it. Both
 * tiers guard; only the address of the frame home differs, so it is passed
 * in rather than recomputed here. */
static void localGuardDynamic(Emit *e, unsigned slot, unsigned scratch,
                              unsigned base, unsigned off) {
    /* Every exit below is a deopt, and a deopt record describes each stack entry
     * by its own register -- so nothing may still be a pending constant or a
     * borrow of a local's, and settling is what makes the guard legal at all.
     * What settling must not do is park an entry in the register the tag load
     * below is about to clobber; that one case keeps the older answer and hands
     * the body back, which is what every guard did here before it settled. */
    for (unsigned i = 0; i < 32u; i++) {
        if (((e->kPend | e->xBorrow) & (1u << i)) == 0) continue;
        if (valueXReg(e, i) != scratch) continue;
        e->whyNot = "a deferred value is in the register a local's guard needs";
        e->failed = true;
        return;
    }
    settleAll(e);
    /* Into `scratch`, not JIT_SCRATCH_A, since the caller may already be holding an operand there: this
     * exact mistake once loaded the tag over the constant on a dynamic slot, so `for j in i + 1..n` silently ran from i+2 and every nested loop was one iteration short. */
    emit(e, jaiA64LdrW(scratch, base, off));
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
        emit(e, jaiA64LdrX(scratch, base, off + 8));
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
        emit(e, jaiA64LdrX(scratch, base, off + 8));
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

/* Register mode: the local's own register, `scratch` unused. Memory mode: loaded into `scratch`.
 * Only the payload moves -- a fixed-kind local's tag is rebuildable from the kind, so it is never
 * restored. A dynamic local's kind is a speculation instead, so both tiers guard before the load. */
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
        /* The interpreter's own Value slot is the frame home here, so the tag
         * to check is the one already sitting beside the payload. */
        if (e->dynamicLocal[slot]) {
            localGuardDynamic(e, slot, scratch, JIT_SLOTS_REG, slot * 16u);
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
        localGuardDynamic(e, slot, scratch, 31, localFrameOff(e, slot));
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
unsigned valueBankRoom(const Emit *e) {
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
int64_t stackSignatureAt(const Emit *e, unsigned depth) {
    unsigned mode = jitJoinMode();
    if (mode == 0) {
        int64_t sig = (int64_t)(depth & 0xfu);
        for (unsigned i = 0; i < depth && i < 9; i++) {
            sig |= (int64_t)((uint32_t)e->stack[i] & 3u) << (4 + 2 * i);
        }
        return sig;
    }
    /* depth 5 bits, valueDepth 5 bits, then the WHOLE kind of each of the
     * first thirteen entries in 4 bits each -- 62 bits, and positive. */
    int64_t sig = (int64_t)(depth & 0x1fu);
    if (mode >= 2) {
        unsigned seen = 0;
        for (unsigned i = 0; i < depth; i++) {
            if (holdsRegister(e->stack[i])) seen++;
        }
        sig |= (int64_t)(seen & 0x1fu) << 5;
    }
    for (unsigned i = 0; i < depth && i < 13; i++) {
        sig |= (int64_t)((uint32_t)e->stack[i] & 0xfu) << (10 + 4 * i);
    }
    return sig;
}

int64_t stackSignature(const Emit *e) {
    return stackSignatureAt(e, e->depth);
}

#else

#endif
