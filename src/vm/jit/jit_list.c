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
 * (a list of instances). Two instructions say BOXED or deoptimise. BOXED is
 * nearly a fact that stays true -- `stg` moves towards boxed, except for an
 * untyped list of one scalar kind growing past its first eight
 * (jaiListShapeOnGrow), and a list that long being handed an object is rare
 * enough to leave to the deopt.
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
    /* PINNED AT THE HOIST. Not proved at entry, but at the head of the loop
     * this access sits in, by emitHoistsAt -- see jitHoistPinOn. */
    {
        int h = hoistFor(e, slot);
        if (h >= 0 && e->hoist[h].stgPin) {
            a.stg = e->hoist[h].stg;
            a.alt = a.stg;
            a.dynamic = false;
            return a;
        }
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

/* JAITHON_JIT_HOIST_LEAN: a hoist takes one register for `items`, a list
 * stored into takes one for its bumped version, and counts get the rest. See
 * Emit::hoist. */
static bool jitHoistLean(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_HOIST_LEAN");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The hoisted header for a subscript whose base is a plain read of local
 * `slot`, or -1. `curOffset` is checked against the loop the entry was made
 * for, so an entry the walk has already left cannot be picked up again by a
 * later loop that happens to name the same slot. */
int hoistFor(const Emit *e, int slot) {
    if (slot < 0 || e->inlining) return -1;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].str) continue;
        if (e->hoist[i].slot != (uint8_t)slot) continue;
        if (e->curOffset < e->hoist[i].top) continue;
        if (e->curOffset >= e->hoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* The same lookup for a hoisted STRING header. Separate so that no list arm
 * can ever pick up a `chars`/`length` pair and read it as `items`/`count`. */
int hoistForStr(const Emit *e, int slot) {
    if (slot < 0 || e->inlining) return -1;
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (!e->hoist[i].str) continue;
        if (e->hoist[i].slot != (uint8_t)slot) continue;
        if (e->curOffset < e->hoist[i].top) continue;
        if (e->curOffset >= e->hoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* JAITHON_JIT_STR_FACTS=0 stops planStrFacts proving anything at a loop head,
 * so each `s[i]` and each identity compare on a loop-invariant string local
 * guards its own operand again, every iteration. Default on. */
static bool jitStrFacts(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_STR_FACTS");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Measuring pass only: `slot` was read as one side of a string identity
 * compare here. Weighted as noteSlotIndexed is. */
void noteSlotStrEq(Emit *e, int slot) {
    if (!e->measuring || e->inlining || slot < 0 || slot > (int)JIT_MAX_SLOTS) {
        return;
    }
    unsigned w = 1u;
    if (e->loopDepth != NULL && e->curOffset < e->loopDepthCount) {
        unsigned d = e->loopDepth[e->curOffset];
        if (d > 6u) d = 6u;
        w = 1u << (2u * d);
    }
    e->slotEqUse[slot] += w;
    if (e->curOffset < e->slotEqLo[slot]) e->slotEqLo[slot] = e->curOffset;
    if (e->curOffset > e->slotEqHi[slot]) e->slotEqHi[slot] = e->curOffset;
}

static int strFactFor(const Emit *e, int slot) {
    if (slot < 0 || e->inlining || e->measuring) return -1;
    for (unsigned i = 0; i < e->strFactCount; i++) {
        if (e->strFact[i].slot != (uint8_t)slot) continue;
        if (e->curOffset < e->strFact[i].top) continue;
        if (e->curOffset >= e->strFact[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* Whether a loop head has already proved that local `slot` holds a string
 * whose every scalar is one byte -- the two guards `s[i]` opens with. */
bool strFactAscii(const Emit *e, int slot) {
    int f = strFactFor(e, slot);
    return f >= 0 && e->strFact[f].ascii;
}

/* Whether a loop head has already proved that local `slot` holds an interned
 * string -- the two guards each side of a string identity compare opens with. */
bool strFactInterned(const Emit *e, int slot) {
    int f = strFactFor(e, slot);
    return f >= 0 && e->strFact[f].interned;
}

static bool onlyBackEdgesEnter(const Chunk *c, uint32_t top, uint32_t end);

/* Register-free string facts, proved once at a loop head.
 *
 * A string local the loop never writes is the same string on every iteration,
 * and a string never changes: not its type, not its bytes, not its scalar
 * count once counted, and interning is never undone. So the guards `s[i]` and
 * `c == first` open with -- "is a string", "every scalar is one byte", "is
 * interned" -- are loop invariants, and need proving once per entry to the
 * loop rather than once per character.
 *
 * Unlike planHoists' headers these need no register, so a call inside the loop
 * does not disqualify them. What a call could do is write the local, and the
 * only way it can is through a by-reference capture; those slots are never
 * given a register (chunkByRefCaptures), which is why one is required here.
 *
 * Offered only where the sample already passes, so a loop whose string is not
 * ASCII, or whose comparand is not interned, keeps its per-site guards and
 * deopts exactly where it always did. A slot whose header was hoisted over the
 * same stretch needs no ascii fact; the header's own guards already are one. */
void planStrFacts(Emit *e, ObjFunction *fn) {
    if (e->measuring || !e->osr || !jitStrFacts()) return;
    const Chunk *c = &fn->chunk;
    for (unsigned s = 0; s < e->locals && s <= JIT_MAX_SLOTS; s++) {
        if (e->strFactCount >= JIT_MAX_STR_FACTS) break;
        if (e->localKind[s] != SLOT_OBJ) continue;
        if (e->slotXReg[s] == 0) continue;
        Value sv = seenLocal(e, s);
        if (!IS_STRING(sv)) continue;
        ObjString *ss = AS_STRING(sv);
        bool wantAscii = e->slotIndexUse[s] != 0 &&
                         ss->scalars == ss->length;
        bool wantInterned = e->slotEqUse[s] != 0 && JAI_STR_INTERNED(ss);
        if (!wantAscii && !wantInterned) continue;
        uint32_t lo = UINT32_MAX, hi = 0;
        if (wantAscii) {
            if (e->slotIndexLo[s] < lo) lo = e->slotIndexLo[s];
            if (e->slotIndexHi[s] > hi) hi = e->slotIndexHi[s];
        }
        if (wantInterned) {
            if (e->slotEqLo[s] < lo) lo = e->slotEqLo[s];
            if (e->slotEqHi[s] > hi) hi = e->slotEqHi[s];
        }
        uint32_t bestTop = 0, bestEnd = 0;
        for (int at = (int)e->osrTop; at < (int)e->osrEnd;) {
            int len = instructionLength(c, at);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)at;
            uint32_t le = loopBodyEnd(c, lt);
            at += len;
            if (le == 0 || le <= lt || le > e->osrEnd) continue;
            if (lo < lt || hi >= le) continue;
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        /* The header, where there is one over this stretch, already proves
         * both of the ascii fact's halves. */
        if (wantAscii) {
            for (unsigned h = 0; h < e->hoistCount; h++) {
                if (e->hoist[h].str && e->hoist[h].slot == (uint8_t)s &&
                    e->hoist[h].top <= bestTop && e->hoist[h].end >= bestEnd) {
                    wantAscii = false;
                }
            }
        }
        if (!wantAscii && !wantInterned) continue;
        unsigned k = e->strFactCount++;
        e->strFact[k].top      = bestTop;
        e->strFact[k].end      = bestEnd;
        e->strFact[k].slot     = (uint8_t)s;
        e->strFact[k].ascii    = wantAscii;
        e->strFact[k].interned = wantInterned;
    }
}

/* JAITHON_JIT_STR_HOIST=0 stops planHoists offering string locals, so every
 * `s[i]` in a loop goes back to proving, per character, that `s` is a string
 * and that it is all one-byte scalars, and to reloading `chars` and `length`.
 * Default on. */
static bool jitStrHoist(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_STR_HOIST");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
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

/* Whether the instruction at `at` runs on every pass of the loop [top, end)
 * that comes back to the head: nothing between the head and `at` branches
 * past it to somewhere still inside the loop, or back to the head (a
 * `continue`). A branch that leaves the loop is fine -- that pass is never
 * guarded again. A guard moved to the head from a site that passes this
 * misses on the first pass exactly when the site's own guard would have; one
 * moved from a site in a rarely taken arm would send every pass of the loop
 * to the interpreter for a miss the site might never have seen. */
static bool runsEveryPass(const Chunk *c, uint32_t top, uint32_t end,
                          uint32_t at) {
    if (at < top || at >= end) return false;
    for (int off = (int)top; off < (int)at;) {
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        int rel = jaiOpBranchOperandAt(c->code[off]);
        if (rel >= 0) {
            int32_t to = (int32_t)(off + len) +
                         jaiReadI16(c->code + off + 1 + rel);
            if (to > (int32_t)at && to < (int32_t)end) return false;
            if (to == (int32_t)top) return false;
        }
        off += len;
    }
    return true;
}

/* The appends in [lt, le) that a header of `slot` hoisted over that loop
 * would have to be proved distinct from, or false when no proof is possible:
 * an append to `slot` itself, to a list with no local behind it, or to a
 * local the loop reassigns (its value at the hoist is not the one appended
 * to). A comprehension's accumulator needs no proof -- no local can hold it
 * while its loop runs. Duplicate targets are listed once. */
static bool hoistAliasSet(const Emit *e, const SlotKind *kinds,
                          unsigned slot, uint32_t lt,
                          uint32_t le, uint8_t *out, uint8_t *nOut) {
    unsigned n = 0;
    if (e->pushSpill) return false;
    for (unsigned i = 0; i < e->pushCount; i++) {
        if (e->pushOff[i] < lt || e->pushOff[i] >= le) continue;
        int p = e->pushSlot[i];
        if (p == JIT_PUSH_FRESH) continue;
        if (p < 0 || p > (int)JIT_MAX_SLOTS) return false;
        if ((unsigned)p == slot) return false;
        if (e->slotWriteHi[p] >= lt && e->slotWriteLo[p] < le) return false;
        if (kinds[p] != SLOT_LIST) return false;
        bool seen = false;
        for (unsigned k = 0; k < n; k++) {
            if (out[k] == (uint8_t)p) { seen = true; break; }
        }
        if (seen) continue;
        if (n >= JIT_MAX_HOIST_ALIAS) return false;
        out[n++] = (uint8_t)p;
    }
    *nOut = (uint8_t)n;
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
 * is a call out, and so is every collection -- all but an inlined append whose
 * grow stub keeps the registers (jitGrowKeeps), which hoistAliasSet answers
 * for instead; an in-place `xs[i] = v` moves neither `items` nor `count`. So across a call-free stretch a header is
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
/* The register a hoisted list's local lives in, or 0. The loop tier's homes
 * are slotXReg; the function tier's are its fixed x19.. registers unless the
 * frame was planned, when they are slotXReg too. A dynamic slot has none. */
static unsigned hoistListReg(const Emit *e, unsigned slot) {
    if (slot > JIT_MAX_SLOTS || e->dynamicLocal[slot]) return 0;
    if (e->osr || e->spilled) return e->slotXReg[slot];
    return localHomeX(e, slot);
}

/* JAITHON_JIT_PUSH_REG: a loop's one push keeps its list's count and bumped
 * version in registers. See Emit::pushHoist. */
static bool jitPushReg(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_PUSH_REG");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Each push costs a load-add-store of `count` and another of `version`, and
 * the next push's load waits on this one's store: a dependency through memory
 * on every iteration of a loop that builds a list. Kept in registers instead,
 * over the outermost loop around the push that (a) calls nothing -- a keeping
 * grow stub (jitGrowKeeps) is not a call, and it puts the registers back --
 * (b) never rebinds the local, and (c) holds NO other append of any kind: two
 * pushes might name one list through two locals, and each would then count
 * from its own copy. A store into the list through another local changes
 * neither the count nor, as far as any iterator can tell, the version: it
 * moves it past every snapshot either way (see Emit::hoist's version note). */
static void planPushHoists(Emit *e, const Chunk *c, const SlotKind *kinds,
                           uint32_t regionLo, uint32_t regionHi) {
    e->pushHoistCount = 0;
    if (!jitPushReg() || !jitGrowKeeps() || e->pushSpill) return;
    for (unsigned p = 0; p < e->pushCount && e->pushHoistCount < 2u; p++) {
        int s = e->pushSlot[p];
        if (s < 0 || s > (int)JIT_MAX_SLOTS) continue;
        if (kinds[s] != SLOT_LIST || hoistListReg(e, (unsigned)s) == 0) continue;
        uint32_t at = e->pushOff[p];
        uint32_t bestTop = 0, bestEnd = 0;
        for (int off = (int)regionLo; off < (int)regionHi;) {
            int len = instructionLength(c, off);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)off;
            uint32_t le = loopBodyEnd(c, lt);
            off += len;
            if (le == 0 || le <= lt || le > regionHi) continue;
            if (at < lt || at >= le) continue;
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            if (regionCalls(e, lt, le)) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            unsigned inside = 0;
            for (unsigned q = 0; q < e->pushCount; q++) {
                if (e->pushOff[q] >= lt && e->pushOff[q] < le) inside++;
            }
            if (inside != 1) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        if (e->hoistPoolCount - e->hoistTaken < 2u) return;
        unsigned rc = e->hoistPool[e->hoistTaken++];
        unsigned rv = e->hoistPool[e->hoistTaken++];
        if (rc < e->scratchRoom) e->scratchRoom = rc;
        if (rv < e->scratchRoom) e->scratchRoom = rv;
        unsigned k = e->pushHoistCount++;
        e->pushHoist[k].top = bestTop;
        e->pushHoist[k].end = bestEnd;
        e->pushHoist[k].slot = (uint8_t)s;
        e->pushHoist[k].countReg = (uint8_t)rc;
        e->pushHoist[k].verReg = (uint8_t)rv;
    }
}

/* JAITHON_JIT_ITER_IDX_REG: see Emit::iterHoist. */
static bool jitIterIdxReg(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_ITER_IDX_REG");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* A for-in head whose loop calls nothing: the iterator's index can live in a
 * register for as long as the loop runs, because only this head's own step
 * writes it and a keeping grow stub puts the register back. Not the loop
 * tier's own head, which has registers of its own for that. */
static void planIterHoists(Emit *e, const Chunk *c, uint32_t regionLo,
                           uint32_t regionHi) {
    e->iterHoistCount = 0;
    if (!jitIterIdxReg() || !jitGrowKeeps()) return;
    for (int off = (int)regionLo; off < (int)regionHi;) {
        int len = instructionLength(c, off);
        if (len <= 0) return;
        uint32_t lt = (uint32_t)off;
        off += len;
        if (c->code[lt] != OP_FOR_ITER_BIND) continue;
        if (e->osr && lt == e->osrTop) continue;
        uint32_t le = loopBodyEnd(c, lt);
        if (le == 0 || le <= lt || le > regionHi) continue;
        if (regionCalls(e, lt, le)) continue;
        if (!onlyBackEdgesEnter(c, lt, le)) continue;
        if (e->iterHoistCount >= 2u) return;
        if (e->hoistPoolCount - e->hoistTaken < 1u) return;
        unsigned r = e->hoistPool[e->hoistTaken++];
        if (r < e->scratchRoom) e->scratchRoom = r;
        unsigned k = e->iterHoistCount++;
        e->iterHoist[k].top  = lt;
        e->iterHoist[k].end  = le;
        e->iterHoist[k].reg  = (uint8_t)r;
        e->iterHoist[k].live = false;
    }
}

/* The iterator hoist whose index register was loaded for the head at `top`,
 * or -1. */
int iterHoistAt(const Emit *e, uint32_t top) {
    if (e->measuring || e->inlining) return -1;
    for (unsigned i = 0; i < e->iterHoistCount; i++) {
        if (e->iterHoist[i].top == top && e->iterHoist[i].live) return (int)i;
    }
    return -1;
}

/* The push hoist covering a push of `slot`'s list at the current offset, or
 * -1. */
int pushHoistFor(const Emit *e, int slot) {
    if (slot < 0 || e->measuring) return -1;
    uint32_t at = e->inlining ? e->inlIp : e->curOffset;
    for (unsigned i = 0; i < e->pushHoistCount; i++) {
        if (e->pushHoist[i].slot != (uint8_t)slot) continue;
        if (at < e->pushHoist[i].top || at >= e->pushHoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* JAITHON_JIT_CLOSURE_HOIST: see Emit::closHoist. Default on. */
static bool jitClosureHoist(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_CLOSURE_HOIST");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The measuring pass inlined a closure read out of local `slot` at `off`. */
void noteClosureSite(Emit *e, int slot, uint32_t off, ObjClosure *sample) {
    if (!e->measuring || e->inlining || sample == NULL) return;
    if (slot < 0 || slot > (int)JIT_MAX_SLOTS) return;
    if (e->closSiteCount >= JIT_MAX_CLOS_SITES) return;
    e->closSite[e->closSiteCount].off = off;
    e->closSite[e->closSiteCount].slot = (uint8_t)slot;
    e->closSite[e->closSiteCount].sample = sample;
    e->closSiteCount++;
}

/* The closure hoist that proved local `slot` holds a closure over `fn` for a
 * loop containing `at`, or -1. */
int closHoistFor(const Emit *e, int slot, uint32_t at, const ObjFunction *fn) {
    if (e->measuring || slot < 0) return -1;
    for (unsigned i = 0; i < e->closHoistCount; i++) {
        if (e->closHoist[i].slot != (uint8_t)slot) continue;
        if (e->closHoist[i].fn != fn) continue;
        if (at < e->closHoist[i].top || at >= e->closHoist[i].end) continue;
        return (int)i;
    }
    return -1;
}

/* Inlining `f(x)` where `f` is a local closure leaves, on every iteration,
 * a guard that `f` is still over the function the body was inlined from
 * (four instructions), and for each upvalue the body reads, a four-load
 * chain to the cell and a tag check (seven more): eleven of closure_calls'
 * twenty-two per iteration, every one of them loop-invariant. Proved once at
 * the head instead, over the OUTERMOST loop that
 *   - writes no value to the local (so the closure is the one the head saw,
 *     and its `fn` is immutable),
 *   - calls nothing (regionCalls: no code can run that might store into the
 *     captured cell, and x13..x17 survive), and
 *   - is entered only through its head.
 * Compiled code has no OP_SET_UPVALUE arm and an inlined body admits none,
 * so with no call nothing can write the cell while the loop runs. A cell can
 * still be OPEN -- a live frame's stack slot -- and the one frame that could
 * write such a slot without a call is this one, through a local its chunk
 * captures by reference; a loop that writes one keeps its upvalue reads
 * per-site and hoists only the guard (loopWritesCapturedLocal). Upvalues are hoisted as scalars only, so a register
 * holds no reference the collector would have to see. */
/* Whether any local of this chunk that a closure captures BY REFERENCE is
 * written inside [lo, hi). Such a local is the one cell this frame can write
 * without a call -- the loop tier keeps it in its frame slot for exactly that
 * reason (chunkByRefCaptures in jit_osr.c) -- so a loop that writes one keeps
 * every upvalue read per-site. True when the chunk cannot be decoded. */
static bool loopWritesCapturedLocal(const Emit *e, const Chunk *c,
                                    uint32_t lo, uint32_t hi) {
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return true;
        if (c->code[off] == OP_CLOSURE) {
            for (int u = off + 4; u + 3 <= off + len; u += 3) {
                uint8_t how = c->code[u];
                if ((how & 1u) == 0 || (how & 2u) != 0) continue;
                unsigned slot = jaiReadU16(c->code + u + 1);
                if (slot > JIT_MAX_SLOTS) return true;
                if (e->slotWriteHi[slot] >= lo && e->slotWriteLo[slot] < hi) {
                    return true;
                }
            }
        }
        off += len;
    }
    return false;
}

static void planClosureHoists(Emit *e, ObjFunction *fn, const SlotKind *kinds,
                              uint32_t regionLo, uint32_t regionHi) {
    e->closHoistCount = 0;
    if (!jitClosureHoist() || e->noInline) return;
    const Chunk *c = &fn->chunk;
    for (unsigned i = 0; i < e->closSiteCount; i++) {
        if (e->closHoistCount >= JIT_MAX_CLOS_HOIST) return;
        unsigned s = e->closSite[i].slot;
        ObjClosure *sample = e->closSite[i].sample;
        ObjFunction *cfn = sample->fn;
        uint32_t at = e->closSite[i].off;
        if (s > JIT_MAX_SLOTS || kinds[s] != SLOT_OBJ) continue;
        if (hoistListReg(e, s) == 0) continue;
        if (closHoistFor(e, (int)s, at, cfn) >= 0) continue;   /* covered */
        uint32_t bestTop = 0, bestEnd = 0;
        for (int off = (int)regionLo; off < (int)regionHi;) {
            int len = instructionLength(c, off);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)off;
            uint32_t le = loopBodyEnd(c, lt);
            off += len;
            if (le == 0 || le <= lt || le > regionHi) continue;
            if (at < lt || at >= le) continue;
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            if (regionCalls(e, lt, le)) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            if (!runsEveryPass(c, lt, le, at)) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        /* A second hoist of the same local over an overlapping loop would
         * guard one function at the head and another inside it. */
        bool clash = false;
        for (unsigned h = 0; h < e->closHoistCount; h++) {
            if (e->closHoist[h].slot == (uint8_t)s &&
                e->closHoist[h].top < bestEnd && bestTop < e->closHoist[h].end) {
                clash = true;
            }
        }
        if (clash) continue;
        unsigned k = e->closHoistCount;
        e->closHoist[k].top = bestTop;
        e->closHoist[k].end = bestEnd;
        e->closHoist[k].slot = (uint8_t)s;
        e->closHoist[k].fn = cfn;
        e->closHoist[k].upCount = 0;
        bool upOk = !loopWritesCapturedLocal(e, c, bestTop, bestEnd);
        const Chunk *cc = &cfn->chunk;
        for (int o = 0; upOk && o < cc->count;) {
            int len = instructionLength(cc, o);
            if (len <= 0) break;
            if (cc->code[o] == OP_GET_UPVALUE) {
                unsigned idx = cc->code[o + 1];
                bool dup = false;
                for (unsigned u = 0; u < e->closHoist[k].upCount; u++) {
                    if (e->closHoist[k].upIdx[u] == idx) dup = true;
                }
                Value seen = NULL_VAL;
                if (idx < (unsigned)sample->upvalueCount &&
                    sample->upvalues[idx] != NULL) {
                    seen = *sample->upvalues[idx]->location;
                }
                SlotKind uk = IS_INT(seen) ? SLOT_INT
                            : IS_FLOAT(seen) ? SLOT_FLOAT
                            : IS_BOOL(seen) ? SLOT_BOOL : SLOT_OPAQUE;
                if (!dup && uk != SLOT_OPAQUE &&
                    e->closHoist[k].upCount < JIT_MAX_CLOS_UP &&
                    e->hoistPoolCount - e->hoistTaken >= 1u) {
                    unsigned r = e->hoistPool[e->hoistTaken++];
                    if (r < e->scratchRoom) e->scratchRoom = r;
                    unsigned u = e->closHoist[k].upCount++;
                    e->closHoist[k].upIdx[u] = (uint8_t)idx;
                    e->closHoist[k].upReg[u] = (uint8_t)r;
                    e->closHoist[k].upKind[u] = (uint8_t)uk;
                }
            }
            o += len;
        }
        e->closHoistCount++;
    }
}

/* JAITHON_JIT_GLOBAL_GUARD_HOIST: see planGuardHoists. Default on. */
static bool jitGlobalGuardHoist(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_GLOBAL_GUARD_HOIST");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* Can anything in [lo, hi) run Jaithon code -- and so define, delete or
 * rehash a module global? Every such site is a clobber, but not every
 * clobber is such a site: an instance allocation of a simple-init class
 * calls only the allocator and, at worst, the collector, and the collector
 * never touches a module's globals table (it prunes only the intern table). */
static bool regionRebinds(const Emit *e, uint32_t lo, uint32_t hi) {
    if (e->clobberSpill) return true;
    for (unsigned i = 0; i < e->clobberCount; i++) {
        if (e->clobberOff[i] < lo || e->clobberOff[i] >= hi) continue;
        if (!e->clobberAlloc[i]) return true;
    }
    return false;
}

/* emitGlobalsGuard runs before EVERY module-global access: three loads and a
 * compare against the table's keyVersion, which moves only when a key is
 * added, removed or the table rehashed. A loop at module scope pays it for
 * every read and write of its counters -- about a hundred of alloc_churn's
 * 291 instructions an iteration went on global access. Over a loop that runs
 * no Jaithon code (regionRebinds) and is entered only at its head, nothing
 * can move the keys between the head and any access in it, so the guard is
 * proved once at the head, the OUTERMOST such loop around each site. A miss
 * resumes the interpreter at the head. The value is still loaded and its tag
 * checked at every access; only the key check moves. */
static void planGuardHoists(Emit *e, ObjFunction *fn, uint32_t regionLo,
                            uint32_t regionHi) {
    e->guardHoistCount = 0;
    if (!jitGlobalGuardHoist() || e->globalsTable == NULL) return;
    if (e->globalSiteSpill) return;
    const Chunk *c = &fn->chunk;
    for (unsigned i = 0; i < e->globalSiteCount; i++) {
        uint32_t at = e->globalOff[i];
        bool covered = false;
        for (unsigned h = 0; h < e->guardHoistCount; h++) {
            if (at >= e->guardHoist[h].top && at < e->guardHoist[h].end) {
                covered = true;
            }
        }
        if (covered) continue;
        uint32_t bestTop = 0, bestEnd = 0;
        for (int off = (int)regionLo; off < (int)regionHi;) {
            int len = instructionLength(c, off);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)off;
            uint32_t le = loopBodyEnd(c, lt);
            off += len;
            if (le == 0 || le <= lt || le > regionHi) continue;
            if (at < lt || at >= le) continue;
            if (regionRebinds(e, lt, le)) continue;
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            if (!runsEveryPass(c, lt, le, at)) continue;
            if (bestEnd == 0 || le - lt > bestEnd - bestTop) {
                bestTop = lt; bestEnd = le;
            }
        }
        if (bestEnd == 0) continue;
        /* An inner loop chosen first for another site is swallowed by this
         * one: drop it, so no head is guarded twice. */
        unsigned k = 0;
        for (unsigned h = 0; h < e->guardHoistCount; h++) {
            if (e->guardHoist[h].top >= bestTop &&
                e->guardHoist[h].end <= bestEnd) {
                continue;
            }
            e->guardHoist[k++] = e->guardHoist[h];
        }
        e->guardHoistCount = k;
        if (e->guardHoistCount >= JIT_MAX_GUARD_HOIST) return;
        e->guardHoist[e->guardHoistCount].top = bestTop;
        e->guardHoist[e->guardHoistCount].end = bestEnd;
        e->guardHoistCount++;
    }
}

/* JAITHON_JIT_GLOBAL_PROMOTE: see Emit::tagProof's reg. Default on. */
static bool jitGlobalPromote(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_GLOBAL_PROMOTE");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The hoist register a proved global's value lives in at this access, or
 * -1. */
int globalPromotedReg(const Emit *e, JaiEntry *slot, SlotKind kind) {
    if (e->measuring) return -1;
    uint32_t at = e->inlining ? e->inlIp : e->curOffset;
    for (unsigned i = 0; i < e->tagProofCount; i++) {
        if (e->tagProof[i].slot != slot) continue;
        if (at < e->tagProof[i].top || at >= e->tagProof[i].end) continue;
        if ((SlotKind)e->tagProof[i].kind != kind) return -1;
        return e->tagProof[i].reg != 0 ? (int)e->tagProof[i].reg : -1;
    }
    return -1;
}

/* After an allocation's slow path has called out: every promoted global of
 * a loop around this site is reloaded from its entry, which writing through
 * keeps current. Nothing else is in x13..x17 in such a loop. JIT_SCRATCH_C
 * holds the new instance and is left alone. */
void emitPromotedReload(Emit *e) {
    if (e->measuring) return;
    uint32_t at = e->inlining ? e->inlIp : e->curOffset;
    for (unsigned i = 0; i < e->tagProofCount; i++) {
        if (e->tagProof[i].reg == 0) continue;
        if (at < e->tagProof[i].top || at >= e->tagProof[i].end) continue;
        emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)e->tagProof[i].slot);
        if ((SlotKind)e->tagProof[i].kind == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(e->tagProof[i].reg, JIT_SCRATCH_D,
                                  (unsigned)offsetof(JaiEntry, value) + 8u));
        } else {
            emit(e, jaiA64LdrX(e->tagProof[i].reg, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value) + 8u));
        }
    }
}

/* JAITHON_JIT_GLOBAL_TAG_PROOF: see Emit::tagProof. Default on. */
static bool jitGlobalTagProof(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_GLOBAL_TAG_PROOF");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

void noteGlobalAccess(Emit *e, JaiEntry *slot, SlotKind kind, bool write) {
    if (!e->measuring) return;
    if (e->globalAccCount >= JIT_MAX_GLOBAL_ACC) {
        e->globalAccSpill = true;
        return;
    }
    unsigned i = e->globalAccCount++;
    e->globalAcc[i].off = e->inlining ? e->inlIp : e->curOffset;
    e->globalAcc[i].slot = slot;
    e->globalAcc[i].kind = (uint8_t)kind;
    e->globalAcc[i].write = write;
}

/* Whether the access about to be emitted is to a global whose tag a loop
 * head around it has already proved to be `kind`. */
bool globalTagProven(const Emit *e, JaiEntry *slot, SlotKind kind) {
    if (e->measuring) return false;
    uint32_t at = e->inlining ? e->inlIp : e->curOffset;
    for (unsigned i = 0; i < e->tagProofCount; i++) {
        if (e->tagProof[i].slot != slot) continue;
        if (at < e->tagProof[i].top || at >= e->tagProof[i].end) continue;
        return (SlotKind)e->tagProof[i].kind == kind;
    }
    return false;
}

/* Over a loop whose globals guard is already proved at the head
 * (planGuardHoists), a module global's tag can change only through a store
 * the loop itself makes: nothing in it runs Jaithon code, and the only other
 * writers are Jaithon code. So when every access of a global inside the loop
 * -- every read and every compiled store -- is of one scalar kind, checking
 * the tag once at the head proves it for the whole loop: each read drops its
 * tag check, and each store its check that the old value was an object (it
 * cannot be) and its tag store (the tag is already the one it would write).
 * Every access of the loop must have been recorded; a spill proves nothing. */
static void planTagProofs(Emit *e, ObjFunction *fn) {
    const Chunk *c = &fn->chunk;
    e->tagProofCount = 0;
    if (!jitGlobalTagProof() || e->globalAccSpill) return;
    for (unsigned h = 0; h < e->guardHoistCount; h++) {
        uint32_t lt = e->guardHoist[h].top, le = e->guardHoist[h].end;
        for (unsigned i = 0; i < e->globalAccCount; i++) {
            if (e->globalAcc[i].off < lt || e->globalAcc[i].off >= le) continue;
            JaiEntry *slot = e->globalAcc[i].slot;
            SlotKind k = (SlotKind)e->globalAcc[i].kind;
            if (k != SLOT_INT && k != SLOT_FLOAT && k != SLOT_BOOL) continue;
            bool seen = false, ok = true;
            for (unsigned t = 0; t < e->tagProofCount; t++) {
                if (e->tagProof[t].slot == slot && e->tagProof[t].top == lt) {
                    seen = true;
                }
            }
            if (seen) continue;
            for (unsigned j = 0; j < e->globalAccCount; j++) {
                if (e->globalAcc[j].slot != slot) continue;
                if (e->globalAcc[j].off < lt || e->globalAcc[j].off >= le) continue;
                if ((SlotKind)e->globalAcc[j].kind != k) ok = false;
            }
            if (!ok) continue;
            /* The head's check stands in for a read's only when that read
             * would have made it on the first pass anyway: one that runs
             * every pass, with no store of the loop's ahead of it. */
            bool anchored = false;
            for (unsigned j = 0; j < e->globalAccCount && !anchored; j++) {
                uint32_t ro = e->globalAcc[j].off;
                if (e->globalAcc[j].slot != slot || e->globalAcc[j].write) continue;
                if (!runsEveryPass(c, lt, le, ro)) continue;
                bool stored = false;
                for (unsigned w = 0; w < e->globalAccCount; w++) {
                    if (e->globalAcc[w].slot == slot && e->globalAcc[w].write &&
                        e->globalAcc[w].off >= lt && e->globalAcc[w].off < ro) {
                        stored = true;
                    }
                }
                if (!stored) anchored = true;
            }
            if (!anchored) continue;
            if (e->tagProofCount >= JIT_MAX_TAG_PROOF) return;
            e->tagProof[e->tagProofCount].top = lt;
            e->tagProof[e->tagProofCount].end = le;
            e->tagProof[e->tagProofCount].slot = slot;
            e->tagProof[e->tagProofCount].kind = (uint8_t)k;
            e->tagProof[e->tagProofCount].reg = 0;
            e->tagProof[e->tagProofCount].allocs = false;
            /* A module-scope loop's counters and accumulators are carried
             * from one iteration to the next THROUGH their entries, so each
             * one is a store and a load on the loop's critical path. In a
             * loop with no call at all the value can sit in a hoist register
             * instead; the store stays, off the chain. A loop whose only
             * calls are allocations (regionRebinds already said nothing else
             * calls) takes x13..x17 alone and reloads after each slow path. */
            if (jitGlobalPromote()) {
                bool calls = regionCalls(e, lt, le);
                for (unsigned q = e->hoistTaken; q < e->hoistPoolCount; q++) {
                    unsigned r = e->hoistPool[q];
                    if (calls && (r < JIT_FREE_FIRST ||
                                  r >= JIT_FREE_FIRST + JIT_FREE_COUNT)) {
                        continue;
                    }
                    /* Taken out of the pool in place: swap to the front. */
                    e->hoistPool[q] = e->hoistPool[e->hoistTaken];
                    e->hoistPool[e->hoistTaken++] = (uint8_t)r;
                    if (r < e->scratchRoom) e->scratchRoom = r;
                    e->tagProof[e->tagProofCount].reg = (uint8_t)r;
                    e->tagProof[e->tagProofCount].allocs = calls;
                    break;
                }
            }
            e->tagProofCount++;
        }
    }
}

void planHoists(Emit *e, ObjFunction *fn, const SlotKind *kinds) {
    if (e->measuring) return;
    const Chunk *c = &fn->chunk;
    /* The function tier walks the whole chunk; the loop tier its own loop. */
    uint32_t regionLo = e->osr ? e->osrTop : 0u;
    uint32_t regionHi = e->osr ? e->osrEnd : (uint32_t)c->count;

    struct {
        uint32_t top, end, use; uint8_t slot; bool str;
        uint8_t aliasCount; uint8_t alias[JIT_MAX_HOIST_ALIAS];
        bool inside;
    } cand[JIT_MAX_SLOTS + 1];
    unsigned ncand = 0;

    for (unsigned s = 0; s < e->base + e->locals && s <= JIT_MAX_SLOTS; s++) {
        /* A string local hoists for the same reasons a list does, and for one
         * more: a string never changes at all, so the only thing that can
         * make its header stale is a write to the LOCAL -- the same range
         * test below. What the head proves for it is that the local holds a
         * string and that every scalar in it is one byte, which is what
         * each `s[i]` would otherwise prove per character.
         *
         * Offered only when the sample already passes both, so a loop over a
         * string that is not ASCII is left exactly as it was: its per-site
         * guards deopt where they always did, rather than the head deopting
         * every entry. `scalars` is UINT32_MAX until something asks, and an
         * unknown count is not a pass -- the per-site path is what fills it
         * in. The loop tier only: the string header is read off slotXReg. */
        bool str = false;
        if (kinds[s] == SLOT_OBJ) {
            if (!e->osr) continue;
            Value sv = seenLocal(e, s);
            if (!jitStrHoist() || !IS_STRING(sv)) continue;
            ObjString *ss = AS_STRING(sv);
            if (ss->scalars != ss->length) continue;
            str = true;
        } else if (kinds[s] != SLOT_LIST) {
            continue;
        }
        if (hoistListReg(e, s) == 0) continue;   /* no register to load from */
        if (e->slotIndexUse[s] == 0) continue;
        uint32_t bestTop = 0, bestEnd = 0;
        uint8_t bestAlias[JIT_MAX_HOIST_ALIAS];
        uint8_t bestAliasCount = 0;
        bool bestCovers = false;
        bool bestInside = false;
        for (int at = (int)regionLo; at < (int)regionHi;) {
            int len = instructionLength(c, at);
            if (len <= 0) break;
            uint32_t lt = (uint32_t)at;
            uint32_t le = loopBodyEnd(c, lt);
            at += len;
            if (le == 0 || le <= lt || le > regionHi) continue;
            /* Every subscript of this slot inside the loop -- or, with
             * jitHoistPartialOn, at least some: a site outside the hoist's
             * range never asks for it (hoistFor checks the range) and reloads
             * the header itself, so `let d = dist[node]` above an inner loop
             * that subscripts `dist` per element no longer costs the inner
             * loop its hoist. Only a hoist with every subscript inside proves
             * bounds at its head, because the span it would prove is the
             * slot's over ALL its subscripts. */
            bool inside = e->slotIndexLo[s] >= lt && e->slotIndexHi[s] < le;
            if (!inside) {
                if (!jitHoistPartialOn()) continue;
                if (e->slotIndexHi[s] < lt || e->slotIndexLo[s] >= le) continue;
            }
            /* ...and no write to it anywhere in the loop. */
            if (e->slotWriteHi[s] >= lt && e->slotWriteLo[s] < le) continue;
            /* ...and nothing in the loop that could resize the list or take
             * back the registers the header is being put in. */
            if (regionCalls(e, lt, le)) continue;
            /* ...and no append that could be to this same list. An append is
             * the one resize that is not a call (see jitGrowKeeps), so it is
             * asked about separately: never to this slot, and to any other
             * list only if the hoist can prove at run time it is another. */
            uint8_t alias[JIT_MAX_HOIST_ALIAS];
            uint8_t aliasCount = 0;
            if (!hoistAliasSet(e, kinds, s, lt, le, alias, &aliasCount)) {
                continue;
            }
            if (!onlyBackEdgesEnter(c, lt, le)) continue;
            /* Outermost is not the whole preference. emitHoistsAt proves a
             * slot's bounds at the head of the loop it hoists out of, and only
             * when that head's counter is the variable the subscripts are
             * shaped on -- so hoisting `b` out of matrix_mul's `j` loop
             * rather than its `k` loop saves one header load per `j` and
             * puts a compare and branch back on every `b[k]`. A loop whose
             * head can prove the bounds outranks any that cannot. */
            bool covers = jitGrowKeeps() && inside &&
                          lt + 9u <= (uint32_t)c->count &&
                          c->code[lt] == OP_FOR_RANGE_BIND &&
                          e->spanSeen[s] && e->spanOk[s] &&
                          e->spanLo[s] <= e->spanHi[s] &&
                          e->spanBase[s] == jaiReadU16(c->code + lt + 3);
            if (bestEnd == 0 || (covers && !bestCovers) ||
                (covers == bestCovers && le - lt > bestEnd - bestTop)) {
                bestTop = lt; bestEnd = le;
                bestCovers = covers;
                bestInside = inside;
                bestAliasCount = aliasCount;
                memcpy(bestAlias, alias, sizeof alias);
            }
        }
        if (bestEnd == 0) continue;
        cand[ncand].top  = bestTop;
        cand[ncand].end  = bestEnd;
        cand[ncand].use  = e->slotIndexUse[s];
        cand[ncand].slot = (uint8_t)s;
        cand[ncand].str  = str;
        cand[ncand].aliasCount = bestAliasCount;
        memcpy(cand[ncand].alias, bestAlias, sizeof bestAlias);
        cand[ncand].inside = bestInside;
        ncand++;
    }

    /* Lean: every candidate gets its items register first, then a list that
     * the loop stores into gets its version register, and only what is left
     * goes to counts -- busiest first each time. A stencil's four rows need
     * eight registers the old way and the pool has seven, so the row being
     * WRITTEN went without, which is the one whose per-element load-add-store
     * of `version` became the loop's critical path once everything else was
     * hoisted. */
    bool lean = jitHoistLean();
    unsigned need = lean ? 1u : 2u;
    while (e->hoistCount < JIT_MAX_HOIST &&
           e->hoistPoolCount - e->hoistTaken >= need) {
        unsigned pick = ncand, bestUse = 0;
        for (unsigned i = 0; i < ncand; i++) {
            if (cand[i].use > bestUse) { bestUse = cand[i].use; pick = i; }
        }
        if (pick == ncand) break;
        cand[pick].use = 0;                  /* taken */
        /* A string header is `chars` and `length` together, always: the
         * string arms read the length out of countReg (see hoistForStr). */
        bool wantCount = !lean || cand[pick].str;
        if (wantCount && e->hoistPoolCount - e->hoistTaken < 2u) continue;
        unsigned rI = e->hoistPool[e->hoistTaken++];
        unsigned rC = 0;
        if (wantCount) rC = e->hoistPool[e->hoistTaken++];
        if (rI < e->scratchRoom) e->scratchRoom = rI;
        if (wantCount && rC < e->scratchRoom) e->scratchRoom = rC;
        e->hoist[e->hoistCount].hasCount = wantCount;
        e->hoist[e->hoistCount].hasVer   = false;
        e->hoist[e->hoistCount].verReg   = 0;
        e->hoist[e->hoistCount].top      = cand[pick].top;
        e->hoist[e->hoistCount].end      = cand[pick].end;
        e->hoist[e->hoistCount].slot     = cand[pick].slot;
        e->hoist[e->hoistCount].itemsReg = (uint8_t)rI;
        e->hoist[e->hoistCount].countReg = (uint8_t)rC;
        e->hoist[e->hoistCount].rangeOk  = false;
        e->hoist[e->hoistCount].str      = cand[pick].str;
        e->hoist[e->hoistCount].inside   = cand[pick].inside;
        e->hoist[e->hoistCount].aliasCount = cand[pick].aliasCount;
        memcpy(e->hoist[e->hoistCount].aliasSlot, cand[pick].alias,
               sizeof cand[pick].alias);
        /* Only where there is a live list to read the storage off -- a slot
         * holding anything else predicts BOXED, which every entry would then
         * refute -- and only over a loop nothing in which can change a
         * storage: no call (planHoists required that already), and no
         * OP_ELEM_KIND stamping an empty list. */
        e->hoist[e->hoistCount].stgPin = false;
        e->hoist[e->hoistCount].stg = (uint8_t)LIST_STORE_BOXED;
        /* The loop tier only: the function tier's sample is the FIRST call's
         * arguments, and a later call with another storage would deoptimise
         * at this head every time. Its accesses keep their own dispatch. */
        if (e->osr && !cand[pick].str) {
            unsigned hs = cand[pick].slot;
            if (jitHoistPinOn() && !e->localStgPin[hs] &&
                e->observed != NULL && IS_LIST(e->observed[hs]) &&
                !regionStamps(e, cand[pick].top, cand[pick].end)) {
                e->hoist[e->hoistCount].stgPin = true;
                e->hoist[e->hoistCount].stg = AS_LIST(e->observed[hs])->stg;
            }
        }
        uint32_t ht = cand[pick].top;
        if (ht + 9u <= (uint32_t)c->count && c->code[ht] == OP_FOR_RANGE_BIND) {
            e->hoist[e->hoistCount].rangeOk = true;
            e->hoist[e->hoistCount].rVar = jaiReadU16(c->code + ht + 3);
            e->hoist[e->hoistCount].rCur = jaiReadU16(c->code + ht + 5);
            e->hoist[e->hoistCount].rEnd = jaiReadU16(c->code + ht + 7);
        }
        e->hoistCount++;
    }
    planPushHoists(e, c, kinds, regionLo, regionHi);
    planIterHoists(e, c, regionLo, regionHi);
    planClosureHoists(e, fn, kinds, regionLo, regionHi);
    planGuardHoists(e, fn, regionLo, regionHi);
    planTagProofs(e, fn);
    if (!lean) return;
    /* Versions, for lists stored into inside their own region. */
    for (unsigned h = 0; h < e->hoistCount; h++) {
        if (e->hoistPoolCount - e->hoistTaken < 1u) break;
        unsigned sl = e->hoist[h].slot;
        if (e->hoist[h].str) continue;
        if (e->slotStoreLo[sl] > e->slotStoreHi[sl]) continue;
        if (e->slotStoreLo[sl] < e->hoist[h].top) continue;
        if (e->slotStoreHi[sl] >= e->hoist[h].end) continue;
        unsigned rV = e->hoistPool[e->hoistTaken++];
        if (rV < e->scratchRoom) e->scratchRoom = rV;
        e->hoist[h].hasVer = true;
        e->hoist[h].verReg = (uint8_t)rV;
    }
    /* Counts, busiest first, with whatever is left. */
    for (;;) {
        if (e->hoistPoolCount - e->hoistTaken < 1u) break;
        int pick = -1;
        unsigned bestUse = 0;
        for (unsigned h = 0; h < e->hoistCount; h++) {
            if (e->hoist[h].hasCount) continue;
            unsigned use = e->slotIndexUse[e->hoist[h].slot];
            if (pick < 0 || use > bestUse) { pick = (int)h; bestUse = use; }
        }
        if (pick < 0) break;
        unsigned rC = e->hoistPool[e->hoistTaken++];
        if (rC < e->scratchRoom) e->scratchRoom = rC;
        e->hoist[pick].countReg = (uint8_t)rC;
        e->hoist[pick].hasCount = true;
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
    if (h < 0 || !e->hoist[h].rangeOk || !e->hoist[h].inside) return false;
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
    for (unsigned i = 0; i < e->strFactCount; i++) {
        if (e->strFact[i].top != off) continue;
        /* See planStrFacts. A miss resumes the interpreter at the head with
         * nothing run, which is where the per-site guard would have sent it
         * the first time through. */
        unsigned rs = e->slotXReg[e->strFact[i].slot];
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
        branchOnDeoptAt(e, JAI_A64_NE, off, false);
        if (e->strFact[i].ascii) {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                               (unsigned)offsetof(ObjString, length)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, rs,
                               (unsigned)offsetof(ObjString, scalars)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
        }
        if (e->strFact[i].interned) {
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, rs,
                                  (unsigned)offsetof(Obj, subFlag)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            branchOnDeoptAt(e, JAI_A64_EQ, off, false);
        }
    }
    /* The index of the list iterator on top of the stack -- only when it IS
     * one (OP_GET_ITER's list arm, shape 1): any other iterator kind keeps
     * its index in memory and this hoist simply stays unused. */
    for (unsigned i = 0; i < e->iterHoistCount; i++) {
        if (e->iterHoist[i].top != off) continue;
        if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER ||
            e->stackShape[e->depth - 1] != 1u) {
            continue;
        }
        unsigned rIt = pushReg(e) - 1;
        emit(e, jaiA64LdrX(e->iterHoist[i].reg, rIt,
                           (unsigned)offsetof(ObjIter, index)));
        e->iterHoist[i].live = true;
    }
    for (unsigned i = 0; i < e->guardHoistCount; i++) {
        if (e->guardHoist[i].top != off) continue;
        /* emitGlobalsGuard's own compare, resuming at the head. */
        uint32_t kv = e->globalsKeyVersion;
        emitConst64(e, JIT_SCRATCH_D,
                    (int64_t)(uintptr_t)&e->globalsTable->keyVersion);
        emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D, 0));
        if (kv <= 0xfffu) {
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, kv));
        } else {
            emitConst64(e, JIT_SCRATCH_B, (int64_t)kv);
            emit(e, jaiA64SubsX(31, JIT_SCRATCH_C, JIT_SCRATCH_B));
        }
        branchOnDeoptAt(e, JAI_A64_NE, off, false);
        /* After the key check, which is what keeps these addresses valid. */
        for (unsigned t = 0; t < e->tagProofCount; t++) {
            if (e->tagProof[t].top != off) continue;
            SlotKind tk = (SlotKind)e->tagProof[t].kind;
            unsigned tag = tk == SLOT_INT ? VAL_INT
                         : tk == SLOT_FLOAT ? VAL_FLOAT : VAL_BOOL;
            emitConst64(e, JIT_SCRATCH_D,
                        (int64_t)(uintptr_t)e->tagProof[t].slot);
            emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, tag));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
            unsigned pr = e->tagProof[t].reg;
            if (pr != 0) {
                if (tk == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(pr, JIT_SCRATCH_D,
                                          (unsigned)offsetof(JaiEntry, value) + 8u));
                } else {
                    emit(e, jaiA64LdrX(pr, JIT_SCRATCH_D,
                                       (unsigned)offsetof(JaiEntry, value) + 8u));
                }
            }
        }
    }
    for (unsigned i = 0; i < e->closHoistCount; i++) {
        if (e->closHoist[i].top != off) continue;
        /* See planClosureHoists. A miss resumes the interpreter at the head
         * with nothing run; the per-site guard would have sent it to the
         * call, which is no further than the head can reach without it. */
        unsigned rf = hoistListReg(e, e->closHoist[i].slot);
        emit(e, jaiA64SubsXImm(31, rf, 0));
        branchOnDeoptAt(e, JAI_A64_EQ, off, false);
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rf, (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_CLOSURE));
        branchOnDeoptAt(e, JAI_A64_NE, off, false);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, rf,
                           (unsigned)offsetof(ObjClosure, fn)));
        emitConstCmp(e, JIT_SCRATCH_B,
                     (int64_t)(uintptr_t)e->closHoist[i].fn);
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
        branchOnDeoptAt(e, JAI_A64_NE, off, false);
        for (unsigned u = 0; u < e->closHoist[i].upCount; u++) {
            SlotKind uk = (SlotKind)e->closHoist[i].upKind[u];
            unsigned tag = uk == SLOT_INT ? VAL_INT
                         : uk == SLOT_FLOAT ? VAL_FLOAT : VAL_BOOL;
            unsigned r = e->closHoist[i].upReg[u];
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, rf,
                               (unsigned)offsetof(ObjClosure, upvalues)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                               e->closHoist[i].upIdx[u] * 8u));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                               (unsigned)offsetof(ObjUpvalue, location)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, tag));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
            if (uk == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(r, JIT_SCRATCH_A, 8));
            } else {
                emit(e, jaiA64LdrX(r, JIT_SCRATCH_A, 8));
            }
        }
    }
    for (unsigned i = 0; i < e->pushHoistCount; i++) {
        if (e->pushHoist[i].top != off) continue;
        unsigned pl = hoistListReg(e, e->pushHoist[i].slot);
        emit(e, jaiA64LdrW(e->pushHoist[i].countReg, pl,
                           (unsigned)offsetof(ObjList, count)));
        emit(e, jaiA64LdrW(e->pushHoist[i].verReg, pl,
                           (unsigned)offsetof(ObjList, version)));
        emit(e, jaiA64AddXImm(e->pushHoist[i].verReg,
                              e->pushHoist[i].verReg, 1));
    }
    for (unsigned i = 0; i < e->hoistCount; i++) {
        if (e->hoist[i].top != off) continue;
        if (e->hoist[i].str) {
            /* The two facts every `s[i]` in the loop would otherwise prove
             * for itself, proved once: the local holds a string, and its
             * scalar count equals its byte count. Strings are immutable and
             * the collector does not move them, so neither fact -- nor
             * `chars` or `length` -- can change while the local is not
             * written, which planHoists has already established for this
             * range. A miss resumes the interpreter at the head with nothing
             * run, exactly where the per-site guard would have sent it on
             * the first character. */
            unsigned rs = e->slotXReg[e->hoist[i].slot];
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rs,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
            emit(e, jaiA64LdrW(e->hoist[i].countReg, rs,
                               (unsigned)offsetof(ObjString, length)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, rs,
                               (unsigned)offsetof(ObjString, scalars)));
            emit(e, jaiA64SubsXReg(31, e->hoist[i].countReg, JIT_SCRATCH_B));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
            emit(e, jaiA64LdrX(e->hoist[i].itemsReg, rs,
                               (unsigned)offsetof(ObjString, chars)));
            continue;
        }
        unsigned hList = hoistListReg(e, e->hoist[i].slot);
        /* The loop appends to these lists, and a header that is one of them
         * goes stale at the first append. planHoists ruled out the same
         * LOCAL; the same LIST under another name is settled here, once per
         * entry, by pointer. Equal deoptimises to the loop head, so the
         * interpreter runs a loop that aliases -- correct, and rare enough
         * that its price is no concern. */
        for (unsigned a = 0; a < e->hoist[i].aliasCount; a++) {
            unsigned other = localIn(e, e->hoist[i].aliasSlot[a],
                                     JIT_SCRATCH_B);
            emit(e, jaiA64SubsXReg(31, hList, other));
            branchOnDeoptAt(e, JAI_A64_EQ, off, false);
        }
        if (e->hoist[i].stgPin) {
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, hList,
                                  (unsigned)offsetof(ObjList, stg)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, e->hoist[i].stg));
            branchOnDeoptAt(e, JAI_A64_NE, off, false);
        }
        /* The count for the bounds guard below: its own register when it has
         * one, otherwise a scratch that is dead once the guard is past. */
        unsigned hCount = e->hoist[i].hasCount ? e->hoist[i].countReg
                                               : JIT_SCRATCH_D;
        emitListHeader(e, hList, e->hoist[i].itemsReg, hCount);
        if (e->hoist[i].hasVer) {
            unsigned rv = e->hoist[i].verReg;
            emit(e, jaiA64LdrW(rv, hList, (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64AddXImm(rv, rv, 1));
        }

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
        if (!e->hoist[i].rangeOk || !e->hoist[i].inside) continue;
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
        emit(e, jaiA64SubsXUxtw(31, JIT_SCRATCH_A, hCount));
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
    /* In range is the common case, and below it was a TAKEN branch -- the
     * `b.lo` over the negative-index arm -- on every list access the loop
     * head had not proved. Out of line instead (jitBoundsColdOn): in range
     * falls through, and only a negative or out-of-range index leaves for the
     * arm, which adds the count, checks again, and comes back or deoptimises
     * from the record taken here. */
    if (jitBoundsColdOn() && e->coldCount < JIT_MAX_COLD &&
        e->fixupCount < JIT_MAX_FIXUPS) {
        int k = deoptRecordNow(e);
        if (k >= 0) {
            unsigned ci = e->coldCount++;
            e->fixups[e->fixupCount].instIndex    = (int)e->count;
            e->fixups[e->fixupCount].targetOffset = FIXUP_COLD - ci;
            e->fixups[e->fixupCount].conditional  = true;
            e->fixups[e->fixupCount].depth        = -1;
            e->fixupCount++;
            emit(e, jaiA64BCond(JAI_A64_HS, 0));
            e->cold[ci].stub     = -1;
            e->cold[ci].returnTo = (int)e->count;
            e->cold[ci].insn     = 0;
            e->cold[ci].kind     = 1;
            e->cold[ci].rOut     = (uint8_t)rOut;
            e->cold[ci].rCount   = (uint8_t)rCount;
            e->cold[ci].countW   = countW;
            e->cold[ci].deoptK   = k;
            return;
        }
    }
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
