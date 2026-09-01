/* jit_list.c -- list element access: storage dispatch, the loop-invariant hoist
 * planner, bounds normalisation and the overflow exits. */

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/bytecode/verify.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

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
void emitListElemStore(Emit *e, uint8_t stg, unsigned vtag,
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

#else

#endif
