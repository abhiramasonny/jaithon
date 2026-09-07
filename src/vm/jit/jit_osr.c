/* jit_osr.c -- the on-stack-replacement tier: compiling a hot loop out of a frame
 * that is already running, and entering it. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* Returns the bytecode offset the interpreter should continue from. */
#ifdef JAI_ALLOC_CENSUS
#define JAI_DEOPT_HIT_MAX 512
uint64_t    jaiDeoptHits[JAI_DEOPT_HIT_MAX];
const char *jaiDeoptHitName[JAI_DEOPT_HIT_MAX];
uint32_t    jaiDeoptHitTop[JAI_DEOPT_HIT_MAX];
uint32_t    jaiDeoptHitIp[JAI_DEOPT_HIT_MAX];
unsigned    jaiDeoptHitOrd[JAI_DEOPT_HIT_MAX];
unsigned    jaiDeoptHitCount;
static void jaiDeoptHitReport(void) {
    for (unsigned i = 0; i < jaiDeoptHitCount; i++)
        fprintf(stderr, "[hit] %s@%u guard#%u ip=%u fired=%llu\n",
                jaiDeoptHitName[i], jaiDeoptHitTop[i], jaiDeoptHitOrd[i],
                jaiDeoptHitIp[i], (unsigned long long)jaiDeoptHits[i]);
}
void jaiDeoptHitEmit(Emit *e, const char *fn, uint32_t top, unsigned ord,
                     uint32_t ip) {
    static bool armed = false;
    if (!armed) { armed = true; atexit(jaiDeoptHitReport); }
    if (jaiDeoptHitCount >= JAI_DEOPT_HIT_MAX) return;
    unsigned slot = jaiDeoptHitCount++;
    jaiDeoptHitName[slot] = fn;
    jaiDeoptHitTop[slot]  = top;
    jaiDeoptHitIp[slot]   = ip;
    jaiDeoptHitOrd[slot]  = ord;
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jaiDeoptHits[slot]);
    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
    emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
    emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
}
#endif

typedef int64_t (*OsrFn)(Value *slots);
typedef int64_t (*OsrFnIter)(Value *slots, ObjIter *iter);

int instructionLength(const Chunk *c, int off) {
    uint8_t op = c->code[off];
    if (op != OP_CLOSURE) {
        int size = jaiOpOperandSize((OpCode)op);
        return size < 0 ? 0 : 1 + size;
    }
    if (off + 4 > c->count) return 0;
    uint32_t index = jaiReadU24(c->code + off + 1);
    if (index >= (uint32_t)c->constants.count) return 0;
    Value fnv = c->constants.data[index];
    if (!IS_FUNCTION(fnv)) return 0;
    return 4 + 3 * (int)AS_FUNCTION(fnv)->upvalueCount;
}

/* A list comprehension's accumulator, if this loop region has exactly one.
 *
 * `OP_BUILD_LIST` pushes the accumulator, then `OP_GET_ITER` the iterator, and
 * only then does the loop head follow -- so at an `OP_LIST_APPEND` inside the
 * body the container sits at an operand depth the OSR model never saw. The
 * model's depth at an offset is `chunkDepth[off] - chunkDepth[top]`, so the
 * container is out of its reach exactly when that difference is <= `back`, and
 * the append arm has to read the frame instead.
 *
 * The operand stack begins at slot `locals`, so the container's absolute frame
 * slot is `locals + chunkDepth[off] - 1 - back`. Confirmed against the
 * interpreter, which knows the answer outright: on `[c for c in src]` the
 * compile-time formula and `&PEEK(back) - frame->slots` both give 10.
 *
 * Refuses (returns false, so the append keeps refusing as before) when the
 * region holds appends to two DIFFERENT slots, because only one register is
 * reserved. Must be called BEFORE the measuring probe, which may shrink
 * `e.locals` to the slots actually used and would then give a different -- and
 * wrong -- absolute slot. */
static bool findComprehensionAcc(const ObjFunction *fn, const int *chunkDepth,
                                 uint32_t top, uint32_t end, unsigned locals,
                                 int *slotOut) {
    int found = -1;
    if (chunkDepth == NULL || top >= end) return false;
    int dTop = chunkDepth[top];
    if (dTop < 0) return false;
    for (uint32_t off = top; off < end; ) {
        int len = instructionLength(&fn->chunk, (int)off);
        if (len <= 0) return false;
        if (fn->chunk.code[off] == OP_LIST_APPEND) {
            unsigned back = jaiReadU16(fn->chunk.code + off + 1);
            int d = chunkDepth[off];
            if (d < 0) return false;
            if (d - dTop > (int)back) { off += (uint32_t)len; continue; }
            int pos = d - 1 - (int)back;
            if (pos < 0) return false;
            int abs = (int)locals + pos;
            if (abs < 0 || abs > JIT_MAX_SLOTS) return false;
            if (found >= 0 && found != abs) return false;
            found = abs;
        }
        off += (uint32_t)len;
    }
    if (found < 0) return false;
    *slotOut = found;
    return true;
}

static bool chunkByRefCaptures(const Chunk *c, bool *byRef, unsigned nslots) {
    for (unsigned i = 0; i < nslots; i++) byRef[i] = false;
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        if (c->code[off] == OP_CLOSURE) {
            for (int u = off + 4; u + 3 <= off + len; u += 3) {
                uint8_t how = c->code[u];
                if ((how & 1u) == 0 || (how & 2u) != 0) continue;
                unsigned slot = jaiReadU16(c->code + u + 1);
                if (slot < nslots) byRef[slot] = true;
            }
        }
        off += len;
    }
    return true;
}

static bool isInstructionStart(const Chunk *c, uint32_t top) {
    for (int off = 0; off < c->count;) {
        if ((uint32_t)off == top) return true;
        int len = instructionLength(c, off);
        if (len <= 0) return false;
        off += len;
    }
    return false;
}

static uint32_t findLoopEnd(const Chunk *c, uint32_t top, bool wholeBody) {
    uint32_t end = 0;
    for (int off = (int)top; off < c->count;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return end;
        if (op == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            if ((uint32_t)((int32_t)(off + 3) + jump) == top) {
                end = (uint32_t)(off + 3);
                if (!wholeBody) return end;
            }
        }
        off += len;
    }
    return end;
}

#define JIT_MAX_DEPTH_MAP 8192
static uint8_t gLoopDepth[JIT_MAX_DEPTH_MAP];

static unsigned loopDepthTable(const Chunk *c) {
    int n = c->count + 1 < JIT_MAX_DEPTH_MAP ? c->count + 1 : JIT_MAX_DEPTH_MAP;
    for (int i = 0; i < n; i++) gLoopDepth[i] = 0;
    for (int off = 0; off < c->count;) {
        int len = instructionLength(c, off);
        if (len <= 0) break;
        if (c->code[off] == OP_LOOP) {
            int16_t jump = jaiReadI16(c->code + off + 1);
            int target = off + 3 + jump;
            if (target >= 0 && target <= off && off < n) {
                for (int j = target; j <= off; j++) {
                    if (gLoopDepth[j] < 200) gLoopDepth[j]++;
                }
            }
        }
        off += len;
    }
    return (unsigned)n;
}

const uint8_t *loopDepthFor(const Chunk *c, unsigned *count) {
    *count = loopDepthTable(c);
    return gLoopDepth;
}

static int osrNo(ObjFunction *fn, uint32_t top, const char *why) {
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u stopped: %s\n",
                jitFnLabel(fn), top, why);
    }
    return 0;
}

static int osrNoSlot(ObjFunction *fn, uint32_t top, unsigned slot,
                     SlotKind want, Value held) {
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr,
                "[jit] osr %s at %u stopped: slot %u holds %s, but the form "
                "was compiled for %s\n",
                jitFnLabel(fn), top, slot,
                jaiTypeNameStatic(held), slotKindName(want));
    }
    return 0;
}

static unsigned osrFormCap(void) {
    static unsigned cap;
    if (cap == 0) {
        const char *e = getenv("JAITHON_JIT_OSR_FORMS");
        unsigned want = (e != NULL) ? (unsigned)atoi(e) : JAI_OSR_MAX;
        cap = (want >= 1 && want <= JAI_OSR_MAX) ? want : JAI_OSR_MAX;
    }
    return cap;
}

static unsigned osrSlotCap(void) {
    static unsigned cap;
    if (cap == 0) {
        const char *e = getenv("JAITHON_JIT_OSR_SLOTS");
        unsigned want = (e != NULL) ? (unsigned)atoi(e) : JAI_OSR_SLOTS;
        cap = (want >= 1 && want <= JAI_OSR_SLOTS) ? want : JAI_OSR_SLOTS;
    }
    return cap;
}

static bool osrNoB(ObjFunction *fn, uint32_t top, const char *why) {
    (void)osrNo(fn, top, why);
    return false;
}

static bool compileOsrOnce(ObjClosure *closure, uint32_t top, Value *slots,
                           uint8_t iterKind, Value elemSample, bool elemMixed,
                           uint8_t elemStg,
                           bool wholeBody, bool noInline,
                           const bool *nullable, bool *needNullable,
                           const bool *dynamic, bool *needDynamic);

static bool compileOsr(ObjClosure *closure, uint32_t top, Value *slots,
                       uint8_t iterKind, Value elemSample, bool elemMixed,
                       uint8_t elemStg,
                       bool wholeBody, bool noInline) {
    bool nullable[JIT_MAX_SLOTS + 1];
    bool needNullable[JIT_MAX_SLOTS + 1];

    bool dynamic[JIT_MAX_SLOTS + 1];
    bool needDynamic[JIT_MAX_SLOTS + 1];
    memset(nullable, 0, sizeof nullable);
    memset(dynamic, 0, sizeof dynamic);
    for (int attempt = 0; attempt < 3; attempt++) {
        memset(needNullable, 0, sizeof needNullable);
        memset(needDynamic, 0, sizeof needDynamic);
        if (compileOsrOnce(closure, top, slots, iterKind, elemSample, elemMixed,
                           elemStg, wholeBody, noInline, nullable,
                           needNullable, dynamic, needDynamic)) {
            return true;
        }
        bool grew = false;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (needNullable[i] && !nullable[i]) { nullable[i] = true; grew = true; }
            if (needDynamic[i] && !dynamic[i]) { dynamic[i] = true; grew = true; }
        }
        if (!grew) return false;
    }
    return false;
}

static bool compileOsrOnce(ObjClosure *closure, uint32_t top, Value *slots,
                           uint8_t iterKind, Value elemSample, bool elemMixed,
                           uint8_t elemStg,
                           bool wholeBody, bool noInline,
                           const bool *nullable, bool *needNullable,
                           const bool *dynamic, bool *needDynamic) {
    bool hasIter = iterKind != 0;
    ObjFunction *fn = closure->fn;
    if (!isInstructionStart(&fn->chunk, top))
        return osrNoB(fn, top, "the loop head is not an instruction boundary");
    uint32_t end = findLoopEnd(&fn->chunk, top, wholeBody);
    if (end == 0 || end <= top)
        return osrNoB(fn, top, "no loop body could be found from this head");
    if (fn->maxSlots < 1 || (unsigned)fn->maxSlots > osrSlotCap()) {
        char said[96];
        snprintf(said, sizeof said,
                 "the body declares %u slots and an entry record holds %u",
                 (unsigned)fn->maxSlots, osrSlotCap());
        return osrNoB(fn, top, said);
    }

    JaiCodeArena *arena = jaiJitArena();
    if (arena == NULL) return false;

    int *map = JAI_ALLOC(int, fn->chunk.count + 1);
    int64_t *depths = JAI_ALLOC(int64_t, fn->chunk.count + 1);
    int *chunkDepth = chunkDepthTable(fn);
    for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }

    static Emit e;
    memset(&e, 0, sizeof e);
    e.osr = true;
    e.loopDepth = gLoopDepth;
    e.loopDepthCount = loopDepthTable(&fn->chunk);
    e.hasIter = hasIter;
    e.iterKind = iterKind;
    e.elemSample = elemSample;
    e.elemMixed  = elemMixed;
    e.elemStg    = elemStg;
    if (top + 9u <= (uint32_t)fn->chunk.count &&
        fn->chunk.code[top] == OP_FOR_RANGE_BIND) {
        e.rangeHead = true;
        e.rangeVar  = jaiReadU16(fn->chunk.code + top + 3);
        e.rangeCur  = jaiReadU16(fn->chunk.code + top + 5);
        e.rangeEnd  = jaiReadU16(fn->chunk.code + top + 7);
    }
    e.osrTop = top;
    e.osrEnd = end;
    e.base = 0;
    e.locals = (unsigned)fn->maxSlots;
    e.limitLiteral = -1;
    e.bailBlock = -1;
    e.exceptionExit = -1;
    e.callsOut = true;
    e.scratchRoom = JIT_SCRATCH_BANK_COUNT;
    e.noInline = noInline;
    e.observed = slots;
    e.offsetToInst = map;
    e.offsetToDepth = depths;
    e.chunkDepth = chunkDepth;
    e.chunkDepthCount = fn->chunk.count + 1;
    {
        int accSlot = 0;
        if (jitCompAcc() &&
            findComprehensionAcc(fn, chunkDepth, top, end, e.locals, &accSlot)) {
            e.accSlot = accSlot;
            e.accWanted = true;
        }
    }
    e.savedCount = JIT_MAX_SAVED;
    memcpy(e.nullableLocal, nullable, sizeof e.nullableLocal);
    memcpy(e.dynamicLocal, dynamic, sizeof e.dynamicLocal);
    for (unsigned i = 0; i < e.locals; i++) {
        Value v = slots[i];
        e.localTyped[i] = true;
        if (IS_OBJ(v) && AS_OBJ(v) == NULL) {
            e.localKind[i] = SLOT_OPAQUE;
            e.localTyped[i] = false;
            continue;
        }
        if (IS_INT(v))        e.localKind[i] = SLOT_INT;
        else if (IS_FLOAT(v)) e.localKind[i] = SLOT_FLOAT;
        else if (IS_BOOL(v))  e.localKind[i] = SLOT_BOOL;
        else if (IS_LIST(v))  e.localKind[i] = SLOT_LIST;
        else if (IS_INSTANCE(v) && AS_INSTANCE(v)->klass != NULL) {
            e.localKind[i]  = nullable[i] ? SLOT_MAYBE_INST : SLOT_INST;
            e.localClass[i] = AS_INSTANCE(v)->klass;
            e.localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_OBJ(v)) {
            e.localKind[i] = SLOT_OBJ;
        } else {
            e.localKind[i] = SLOT_OPAQUE;
            e.localTyped[i] = false;
        }
    }

    unsigned probeMaxValue = 0;
    unsigned probeMaxValueAll = 0;
    unsigned probeStranded = 0;
    unsigned probeClobberDepth = 0;
    bool probeRan = false;
    {
        static Emit probe;
        memset(&probe, 0, sizeof probe);
        probe.osr = true; probe.measuring = true; probe.hasIter = hasIter;
        probe.iterKind = iterKind; probe.elemSample = elemSample;
        probe.elemMixed = elemMixed;
        probe.elemStg = elemStg;
        probe.rangeHead = e.rangeHead;
        probe.rangeVar  = e.rangeVar;
        probe.rangeCur  = e.rangeCur;
        probe.rangeEnd  = e.rangeEnd;
        probe.osrTop = top; probe.osrEnd = end; probe.base = 0;
        probe.noInline = noInline;
        probe.locals = e.locals; probe.callsOut = true; probe.observed = slots;
        probe.accSlot = e.accSlot; probe.accWanted = e.accWanted;
        probe.scratchRoom = JIT_SCRATCH_BANK_COUNT;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.chunkDepth = chunkDepth; probe.chunkDepthCount = fn->chunk.count + 1;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        probe.loopDepth = gLoopDepth;
        probe.loopDepthCount = e.loopDepthCount;
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            probe.slotWriteLo[i] = UINT32_MAX;
            probe.slotIndexLo[i] = UINT32_MAX;
            probe.spanLo[i]   = INT32_MAX;
            probe.spanHi[i]   = INT32_MIN;
            probe.spanOk[i]   = true;
            probe.spanSeen[i] = false;
        }
        memcpy(probe.nullableLocal, nullable, sizeof probe.nullableLocal);
        memcpy(probe.dynamicLocal, dynamic, sizeof probe.dynamicLocal);
        for (unsigned i = 0; i < e.locals; i++) {
            probe.localKind[i]  = e.localKind[i];
            probe.localShape[i] = e.localShape[i];
            probe.localClass[i] = e.localClass[i];
            probe.localTyped[i] = e.localTyped[i];
            probe.localSeen[i]  = e.localSeen[i];
        }
        if (compileBody(&probe, closure) && !probe.failed) {
            unsigned used = probe.maxSlotUsed + 1u;
            if (used < e.locals) e.locals = used;
            probeMaxValue = probe.maxValue;
            probeMaxValueAll = probe.maxValueAll;
            probeClobberDepth = probe.clobberDepth;
            probeRan = true;
            e.scratchValues = !probe.clobbersScratch &&
                              probe.maxValueAll <= JIT_INL_COUNT;
            if (jitSplitStress()) e.scratchValues = false;
            unsigned wantSplit = probe.clobberDepth;
            if (jitSplitStress() && wantSplit == 0) wantSplit = 1;
            if (!e.scratchValues && !probe.inlined &&
                (probe.clobbersScratch || jitSplitStress()) &&
                probe.maxValue > wantSplit &&
                probe.maxValue - wantSplit <= JIT_SCRATCH_BANK_COUNT) {
                e.splitAt = wantSplit;
            }
            unsigned overhead = osrReserved(&e) +
                                (e.scratchValues ? 0u
                                 : e.splitAt != 0 ? e.splitAt
                                                  : probe.maxValue);
            unsigned availX = overhead < JIT_MAX_SAVED
                                  ? JIT_MAX_SAVED - overhead : 0u;

            bool byRef[JIT_MAX_SLOTS + 1];
            bool decoded = chunkByRefCaptures(&fn->chunk, byRef, e.locals);
            bool skip[JIT_MAX_SLOTS + 1];
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                skip[i] = !decoded || i >= e.locals || byRef[i] ||
                          probe.dynamicLocal[i] || probe.slotUse[i] == 0;
            }
            planSlotRegisters(&e, &probe, availX, skip, &probeStranded);

            e.bodyCalls = probe.clobbersScratch;
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.slotWriteLo[i]  = probe.slotWriteLo[i];
                e.slotWriteHi[i]  = probe.slotWriteHi[i];
                e.slotIndexLo[i]  = probe.slotIndexLo[i];
                e.slotIndexHi[i]  = probe.slotIndexHi[i];
                e.slotIndexUse[i] = probe.slotIndexUse[i];
                e.spanLo[i]       = probe.spanLo[i];
                e.spanHi[i]       = probe.spanHi[i];
                e.spanOk[i]       = probe.spanOk[i];
                e.spanBase[i]     = probe.spanBase[i];
                e.spanSeen[i]     = probe.spanSeen[i];
            }
            e.clobberCount = probe.clobberCount;
            e.clobberSpill = probe.clobberSpill;
            for (unsigned i = 0; i < probe.clobberCount; i++) {
                e.clobberOff[i] = probe.clobberOff[i];
            }

            bool bodyCallsOut = regionCalls(&e, top, end);
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.localStgPin[i] =
                    !bodyCallsOut &&
                    !(e.slotWriteHi[i] >= top && e.slotWriteLo[i] < end);
            }
            e.elemStgPin = !bodyCallsOut;
            for (unsigned r = 0; r < JIT_FREE_COUNT; r++) {
                e.hoistPool[e.hoistPoolCount++] = (uint8_t)(JIT_FREE_FIRST + r);
            }
            if (e.scratchValues) {
                for (unsigned r = probe.maxValueAll;
                     r < JIT_SCRATCH_BANK_COUNT; r++) {
                    e.hoistPool[e.hoistPoolCount++] =
                        (uint8_t)(JIT_INL_BANK + r);
                }
            }
        } else {
            memcpy(needNullable, probe.needNullable, sizeof probe.needNullable);
            memcpy(needDynamic, probe.needDynamic, sizeof probe.needDynamic);
        }
        for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    }

    unsigned frame = 16u + 8u * JIT_MAX_SAVED + (unsigned)sizeof(JitCallDesc);
    e.descOffset = 16u + 8u * JIT_MAX_SAVED;
    e.iterFrameOffset = (frame + 7u) & ~7u;
    frame = e.iterFrameOffset + 16u;
    e.fpSaveOffset = frame;
    frame += 8u * JIT_FP_MAX_SAVED;
    e.frameBytes = (frame + 15u) & ~15u;

    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    emitFpSaveRestore(&e, true);
    emit(&e, jaiA64MovX(JIT_SLOTS_REG, 0));
    for (unsigned i = 0; i < e.locals; i++) {
        if (e.slotXReg[i] != 0 && e.localKind[i] == SLOT_BOOL) {
            emit(&e, jaiA64LdrByte(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotXReg[i] != 0) {
            emit(&e, jaiA64LdrX(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotFpReg[i] != 0) {
            emit(&e, jaiA64LdrD(e.slotFpReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        }
    }
    if (hasIter && (iterKind == 3 || iterKind == 4)) {
        emit(&e, jaiA64MovX(JIT_PAIR_ITER_REG, 1));
    } else if (hasIter) {
        unsigned rIter = iterKind == 1 ? 1u : JIT_ITER_REG;
        if (iterKind == 1) {
            emit(&e, jaiA64StrX(1, 31, e.iterFrameOffset));
        } else {
            emit(&e, jaiA64MovX(JIT_ITER_REG, 1));
        }
        emit(&e, jaiA64LdrX(JIT_IDX_REG, rIter,
                            (unsigned)offsetof(ObjIter, index)));
        emit(&e, jaiA64LdrX(JIT_LIM_REG, rIter,
                            (unsigned)offsetof(ObjIter, limit)));
        if (iterKind == 1) {
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                (unsigned)offsetof(ObjRange, start)));
            emit(&e, jaiA64AddX(JIT_IDX_REG, JIT_IDX_REG, JIT_SCRATCH_A));
            emit(&e, jaiA64AddX(JIT_LIM_REG, JIT_LIM_REG, JIT_SCRATCH_A));
        } else {
            emit(&e, jaiA64LdrX(JIT_START_REG, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
        }
    }

    planHoists(&e, fn);
    if (getenv("JAI_JIT_WHY")) {
        if (probeRan) fprintf(stderr,
                "[jit] osr %s at %u registers: %u reserved, %u stack "
                "(%u incl. inlined), %u deep at a call, %u x-locals, "
                "%u fp-locals, %u stranded, bank %s, of %u; %s\n",
                jitFnLabel(fn), top,
                osrReserved(&e), probeMaxValue, probeMaxValueAll,
                probeClobberDepth, e.xLocals, e.fpLocals, probeStranded,
                e.scratchValues ? "x0"
                                : e.splitAt != 0 ? "split" : "callee-saved",
                JIT_MAX_SAVED, e.bodyCalls ? "calls" : "call-free");
        for (unsigned i = 0; i < e.hoistCount; i++) {
            fprintf(stderr,
                    "[jit] osr at %u hoists slot %u's header out of %u..%u "
                    "into x%u/x%u\n",
                    top, e.hoist[i].slot, e.hoist[i].top, e.hoist[i].end,
                    e.hoist[i].itemsReg, e.hoist[i].countReg);
        }
    }

    if (!compileBody(&e, closure) || e.failed) {
        for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
            if (e.needNullable[i]) needNullable[i] = true;
            if (e.needDynamic[i])  needDynamic[i]  = true;
        }
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] osr %s at %u stopped: %s\n",
                    jitFnLabel(fn), top,
                    declineReason(&e));
        }
        jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
        return false;
    }

    /* Falling off the end of the compiled range is the loop exiting there. */
    if (e.exitCount >= JIT_MAX_EXIT) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }

/* Every way out writes back what the loop was holding: the iterator's index,
 * and the locals if they were living in registers. Miss one and the
 * interpreter carries on from stale values. */
#define OSR_SYNC_ITER()                                                        \
    do {                                                                       \
        /* A pair head (3 and 4) keeps nothing in a register but the pointer,  \
         * and the step has already stored the index it advanced -- so every   \
         * way out of one finds the ObjIter already current. The enumerate     \
         * head (5) is not one of those: its index rides in JIT_IDX_REG and    \
         * comes back here, as a list head's does. */                          \
        if (hasIter && iterKind != 3 && iterKind != 4) {                       \
            /* A range head left the iterator in the frame rather than in a    \
             * register, so it comes back here. Every stub this expands into   \
             * is a way out of the loop, so the load is off the hot path and   \
             * JIT_SCRATCH_A is dead at the top of all of them. */             \
            unsigned rIt = JIT_ITER_REG;                                       \
            unsigned rIx = JIT_IDX_REG;                                        \
            if (iterKind == 1) {                                               \
                rIt = JIT_SCRATCH_A;                                           \
                emit(&e, jaiA64LdrX(rIt, 31, e.iterFrameOffset));              \
                /* The index register carries the yielded value, not the       \
                 * index; ObjIter::index is zero-based, so the start comes      \
                 * back off. Two loads and a subtract, once per way out. */    \
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, rIt,                        \
                                    (unsigned)offsetof(ObjIter, source) + 8)); \
                emit(&e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,              \
                                    (unsigned)offsetof(ObjRange, start)));     \
                emit(&e, jaiA64SubsXReg(JIT_SCRATCH_B, JIT_IDX_REG,            \
                                        JIT_SCRATCH_B));                       \
                rIx = JIT_SCRATCH_B;                                           \
            }                                                                  \
            emit(&e, jaiA64StrX(rIx, rIt,                                      \
                                (unsigned)offsetof(ObjIter, index)));          \
        }                                                                      \
        for (unsigned li = 0; li < e.locals; li++) {                           \
            /* Slots still in the frame were written through as they went. */  \
            unsigned lx = e.slotXReg[li], lf = e.slotFpReg[li];                \
            if (lx == 0 && lf == 0) continue;                                  \
            SlotKind lk = e.localKind[li];                                     \
            unsigned lt = lk == SLOT_INT   ? VAL_INT                           \
                        : lk == SLOT_FLOAT ? VAL_FLOAT                         \
                        : lk == SLOT_BOOL  ? VAL_BOOL                          \
                        : lk == SLOT_OPAQUE ? VAL_NULL                         \
                                           : VAL_OBJ;                          \
            /* A d-resident slot the walk decided was not a float after all:   \
             * the home is bit-exact either way, so it only has to come back   \
             * through an X register to be tagged. */                          \
            unsigned rp = lx;                                                  \
            if (lf != 0 && (lk == SLOT_MAYBE_INST ||                           \
                            lk == SLOT_MAYBE_OBJ || lt != VAL_FLOAT)) {        \
                rp = JIT_SCRATCH_B;                                            \
                emit(&e, jaiA64FmovXD(rp, lf));                                \
                lf = 0;                                                        \
            }                                                                  \
            if (lk == SLOT_MAYBE_INST || lk == SLOT_MAYBE_OBJ) {               \
                emitTagFor(&e, lk, rp, JIT_SCRATCH_D, JIT_SCRATCH_C);          \
                emit(&e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, li * 16u));  \
                emit(&e, jaiA64StrX(rp, JIT_SLOTS_REG, li * 16u + 8u));        \
                continue;                                                      \
            }                                                                  \
            if (lt == VAL_NULL) continue;                                      \
            emit(&e, jaiA64MovzX(JIT_SCRATCH_D, lt, 0));                       \
            emit(&e, jaiA64StrW(JIT_SCRATCH_D, JIT_SLOTS_REG, li * 16u));      \
            if (lf != 0) {                                                     \
                emit(&e, jaiA64StrD(lf, JIT_SLOTS_REG, li * 16u + 8u));        \
            } else {                                                           \
                emit(&e, jaiA64StrX(rp, JIT_SLOTS_REG, li * 16u + 8u));        \
            }                                                                  \
        }                                                                      \
    } while (0)

    e.bailBlock = (int)e.count;
    OSR_SYNC_ITER();
    emitConst64(&e, 0, (int64_t)-1);          /* -1: could not continue */
    emitEpilogue(&e, 0);
    e.exceptionExit = (int)e.count;
    OSR_SYNC_ITER();
    emitConst64(&e, 0, (int64_t)-2);          /* -2: an exception is pending */
    emitEpilogue(&e, 0);

    for (unsigned i = 0; i < 3; i++) {
        if (!e.overflowUsed[i]) { e.overflowStub[i] = -1; continue; }
        e.overflowStub[i] = (int)e.count;
        OSR_SYNC_ITER();
        emit(&e, jaiA64MovzX(0, i, 0));
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jitThrowOverflow);
        emit(&e, jaiA64Blr(JIT_SCRATCH_A));
        emitConst64(&e, 0, (int64_t)-2);
        emitEpilogue(&e, 0);
    }

    for (unsigned i = 0; i < e.exitCount; i++) {
        e.exitStub[i] = (int)e.count;
        OSR_SYNC_ITER();
        emitConst64(&e, JIT_SCRATCH_A,
                    (int64_t)(uintptr_t)&gDeopt.nstack);
        emitConst64(&e, JIT_SCRATCH_B, 0);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        /* Leaving by the iterator's own exit means it is exhausted, and the
         * interpreter drops it there; any other way out leaves it in place. */
        emitConst64(&e, JIT_SCRATCH_A,
                    (int64_t)(uintptr_t)&gDeopt.base);
        emitConst64(&e, JIT_SCRATCH_B,
                    (e.hasIter && e.exitOffset[i] == e.iterExit) ? 1 : 0);
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, 0));
        emitConst64(&e, 0, (int64_t)e.exitOffset[i]);
        emitEpilogue(&e, 0);
    }

    emitSelfSlowStubs(&e, closure);

    emitGrowStubs(&e);

    for (unsigned k = 0; k < e.deoptCount; k++) {
        e.deopt[k].stub = (int)e.count;
        OSR_SYNC_ITER();
        emitConst64(&e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);
        unsigned valueSeen = 0;
        for (unsigned i = 0; i < e.deopt[k].depth; i++) {
            SlotKind kind = e.deopt[k].kinds[i];
            unsigned at = (unsigned)offsetof(JitDeoptRecord, stack) +
                          i * (unsigned)sizeof(Value);
            if (kind == SLOT_CLASS || kind == SLOT_SELF ||
                kind == SLOT_FUNC || kind == SLOT_NATIVE) {
                uintptr_t pv = kind != SLOT_SELF
                                   ? (uintptr_t)e.deopt[k].classes[i]
                                   : (uintptr_t)closure;
                emit(&e, jaiA64MovzX(JIT_SCRATCH_B, VAL_OBJ, 0));
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emitConst64(&e, JIT_SCRATCH_B, (int64_t)pv);
                emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8));
                continue;
            }
            if (e.deopt[k].lastFromDesc && i + 1 == e.deopt[k].depth) {
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
            if (e.deopt[k].fpLive & (1u << valueSeen)) {
                emit(&e, jaiA64FmovXD(reg0, fpRegAt(&e, valueSeen)));
            }
            /* Both nullable kinds. The function tier's twin of this ladder
             * (jit_compile.c) already reads both; this one named only the
             * instance, which is how a fix lands in one tier and not the
             * other -- and the tiers are complements, so a bug that survives
             * in one is a bug that ships. */
            if (kind == SLOT_MAYBE_INST || kind == SLOT_MAYBE_OBJ) {
                emitTagFor(&e, kind, reg0, JIT_SCRATCH_B, JIT_SCRATCH_C);
                emit(&e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
                emit(&e, jaiA64StrX(reg0, JIT_SCRATCH_A, at + 8));
                valueSeen++;
                continue;
            }
            /* Same SLOT_NULL hazard as the function tier's stub -- see there. */
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
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, e.deopt[k].depth, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, nstack)));
        emit(&e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
        emit(&e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                            (unsigned)offsetof(JitDeoptRecord, base)));
#ifdef JAI_ALLOC_CENSUS
        /* Which guard actually fires. A stub that resumes at the same ip as
         * three others says nothing on its own; this makes each one countable,
         * which is the difference between "the nested for-in arm deopts" and
         * knowing WHICH of its four guards is the one that never holds. */
        jaiDeoptHitEmit(&e, closure->fn->name ? closure->fn->name->chars : "?",
                        top, k, (uint32_t)e.deopt[k].ip);
#endif
        emitConst64(&e, 0, (int64_t)e.deopt[k].ip);
        emitEpilogue(&e, 0);
    }
#undef OSR_SYNC_ITER

    if (e.failed) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }

    for (unsigned i = 0; i < e.fixupCount; i++) {
        const Fixup *f = &e.fixups[i];
        int target;
        if (f->targetOffset == FIXUP_BAIL) target = e.bailBlock;
        else if (f->targetOffset == FIXUP_THREW) target = e.exceptionExit;
        else if (f->targetOffset <= FIXUP_EXIT &&
                 f->targetOffset > FIXUP_EXIT - JIT_MAX_EXIT)
            target = e.exitStub[FIXUP_EXIT - f->targetOffset];
        else if (f->targetOffset <= FIXUP_SELFSLOW &&
                 f->targetOffset > FIXUP_SELFSLOW - JIT_MAX_SELF_SLOW)
            target = e.selfSlow[FIXUP_SELFSLOW - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_DEOPT &&
                 f->targetOffset > FIXUP_DEOPT - JIT_MAX_DEOPT)
            target = e.deopt[FIXUP_DEOPT - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_GROW &&
                 f->targetOffset > FIXUP_GROW - JIT_MAX_GROW)
            target = e.grow[FIXUP_GROW - f->targetOffset].stub;
        else if (f->targetOffset <= FIXUP_OVF && f->targetOffset >= FIXUP_OVF - 2u)
            target = e.overflowStub[FIXUP_OVF - f->targetOffset];
        else {
            if (f->targetOffset > (uint32_t)fn->chunk.count) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
            target = map[f->targetOffset];
            if (target < 0 && getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] %s stopped: a branch to %u, which is "
                                "not an instruction this compiled\n",
                        jitFnLabel(fn), f->targetOffset);
            }
            if (target < 0 ||
                (f->depth >= 0 && depths[f->targetOffset] != f->depth)) {
                jitFree(map, depths, chunkDepth, fn->chunk.count + 1);
                return false;
            }
        }
        if (target < 0) { jitFree(map, depths, chunkDepth, fn->chunk.count + 1); return false; }
        int rel = target - f->instIndex;
        uint32_t word = e.code[f->instIndex];
        if ((word & 0xfc000000u) == 0x94000000u) e.code[f->instIndex] = jaiA64Bl(rel);
        else if (f->conditional) e.code[f->instIndex] = jaiA64BCond(word & 0xfu, rel);
        else e.code[f->instIndex] = jaiA64B(rel);
    }
    jitFree(map, depths, chunkDepth, fn->chunk.count + 1);

    if (!jaiCodeArenaUnseal(arena)) return false;
    /* Same 32-alignment as the function tier above, for the same two reasons. */
    uint8_t *entry = arenaEmit(arena, e.code, e.count);
    if (entry == NULL) return false;

    if (fn->osrCount >= osrFormCap()) return false;
    /* First form this function has ever recorded: fewer than 2% of functions
     * ever get one, so the table is allocated here rather than inline on
     * ObjFunction. Sized JAI_OSR_MAX regardless of osrFormCap()'s runtime cap
     * -- the cap only restricts how many of these entries get used, the same
     * as it did when the array was inline and fixed at this width. */
    if (fn->osrForms == NULL)
        fn->osrForms = JAI_ALLOC_ZEROED(JaiOsrForm, JAI_OSR_MAX);
    JaiOsrForm *form = &fn->osrForms[fn->osrCount];
    form->code  = entry;
    form->top   = top;
    form->slots = (uint8_t)e.locals;
    form->iterKind = iterKind;
    /* Kind in the low nibble, ListStore in the high one. See JaiOsrForm::kinds:
     * a list slot's storage was pinned when its element loads were emitted, so
     * the entry guard has to prove it the same way it proves a class. */
    for (unsigned i = 0; i < e.locals; i++) {
        /* Only a slot the emission actually believed needs proving. An
         * unpinned one was compiled at the boxed stride behind a guard of its
         * own, and demanding a storage of it here would refuse entry to a loop
         * the body would have run correctly. */
        uint8_t stg = e.localKind[i] == SLOT_LIST && e.localStgPin[i]
                          ? localStgOf(&e, i)
                          : (uint8_t)LIST_STG_ANY;
        form->kinds[i] = (uint8_t)((unsigned)e.localKind[i] | ((unsigned)stg << 4));
    }
    form->iterStg = e.elemStgPin ? e.elemStg : (uint8_t)LIST_STG_ANY;

    /* Pin the class of every instance slot the compile specialised on. Without
     * this the entry guard checks only IS_INSTANCE, and a loop compiled for one
     * class entered holding another reads a field at the wrong slot index --
     * see JaiOsrForm's note for the probe. Refuse rather than compile
     * unguarded if the pairs do not fit. */
    form->shapeCount = 0;
    for (unsigned i = 0; i < e.locals; i++) {
        SlotKind k = e.localKind[i];
        if (k != SLOT_INST && k != SLOT_MAYBE_INST) continue;
        if (e.localShape[i] == 0) continue;   /* no class pinned to this slot */
        if (form->shapeCount >= jitShapeLimit()) {
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] osr %s at %u stopped: more than %u "
                        "instance slots to pin\n",
                        jitFnLabel(fn), top,
                        jitShapeLimit());
            }
            return false;
        }
        form->shapeSlot[form->shapeCount] = (uint8_t)i;
        form->shapeId[form->shapeCount]   = e.localShape[i];
        form->shapeCount++;
    }

    fn->osrCount++;
    fn->osrHot = true;
    fn->jitOsrModuleVersion = fn->module != NULL ? fn->module->version : 0;
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u: %u instructions iter=%u\n",
                jitFnLabel(fn), top, e.count,
                (unsigned)iterKind);
        /* Same reason as the function tier's line: a loop that walked two
         * instructions and interpreted the other forty reported success. */
        if (e.unarmedOp != 0) {
            fprintf(stderr,
                    "[jit] osr %s at %u walked only to %s%s at %u -- the rest of "
                    "the loop is interpreted\n",
                    jitFnLabel(fn), top,
                    jaiOpName((OpCode)e.unarmedOp),
                    unarmedDetail(fn, e.unarmedOp, e.unarmedAt), e.unarmedAt);
        }
    }
    /* The same dump the whole-function tier has. A compiled loop is where most
     * of the time goes, so it is the code most worth reading, and until now it
     * was the only tier that could not be read at all. */
    {
        const char *dump = getenv("JAI_JIT_DUMP");
        if (dump != NULL && fn->name != NULL &&
            strcmp(dump, fn->name->chars) == 0) {
            char path[256];
            snprintf(path, sizeof path, "jit_osr_%s_%u.bin",
                     fn->name->chars, top);
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
    return true;
}

/* Whether this form's PINNED STORAGES describe the lists in hand.
 *
 * Storage takes part in choosing a form, not merely in admitting one. A form
 * pinned to F64 is not WRONG for a boxed list -- it simply does not apply --
 * and rejecting it after it has already been selected leaves the head with
 * nothing, for ever: the search below breaks on the first (top, iterKind)
 * match, so a second form for the other storage is never compiled. One
 * jaiListBox -- a sort, a slice assignment, an int put into a `list[float]` --
 * then cost that loop its compiled form for the rest of the program. Skipping
 * the form here instead lets the head compile another one. */
static bool osrFormStorageFits(const JaiOsrForm *form, const Value *slots,
                               uint8_t iterKind, uint8_t elemStg) {
    if ((iterKind == 2 || iterKind == 5) && form->iterStg != LIST_STG_ANY &&
        form->iterStg != elemStg) {
        return false;
    }
    for (unsigned i = 0; i < form->slots; i++) {
        uint8_t want = (uint8_t)(form->kinds[i] >> 4);
        if (want == LIST_STG_ANY) continue;
        if ((SlotKind)(form->kinds[i] & 0x0Fu) != SLOT_LIST) continue;
        if (!IS_LIST(slots[i])) return false;
        if (AS_LIST(slots[i])->stg != want) return false;
    }
    return true;
}

int jaiJitEnterOsr(ObjClosure *closure, uint32_t top, uint32_t *resumeAt) {
    ObjFunction *fn = closure->fn;
    CallFrame *frame = &vm.frames[vm.frameCount - 1];
    /* A for-loop head keeps its iterator on the stack. Only a range from zero
     * in unit steps is taken, because that is what makes the yielded value the
     * index and lets the body run against a plain counter. */
    bool pairTop = top < (uint32_t)fn->chunk.count &&
                   fn->chunk.code[top] == OP_FOR_ITER_PAIR;
    bool hasIter = pairTop ||
                   (top < (uint32_t)fn->chunk.count &&
                    fn->chunk.code[top] == OP_FOR_ITER_BIND);
    ObjIter *iter = NULL;
    uint8_t iterKind = 0;
    Value elemSample = NULL_VAL;
    bool  elemMixed = false;
    uint8_t elemStg = (uint8_t)LIST_STORE_BOXED;
    if (hasIter) {
        if (vm.stackTop <= frame->slots)
            return osrNo(fn, top, "a for-loop head with an empty stack");
        Value it = vm.stackTop[-1];
        if (!IS_ITER(it))
            return osrNo(fn, top, "a for-loop head whose stack top is not an iterator");
        iter = AS_ITER(it);
        if (pairTop) {
            if (iter->kind == ITER_LIST_ENUM && IS_LIST(iter->source) &&
                jaiLazyEnumerateOn()) {
                ObjList *esrc = AS_LIST(iter->source);
                int eat = (int)iter->index;
                if (eat < 0 || eat >= esrc->count) {
                    return osrNo(fn, top,
                                 "an enumerate loop already past its last "
                                 "element");
                }
                if (esrc->stg != LIST_STORE_BOXED) {
                    return osrNo(fn, top,
                                 "an enumerate snapshot that is not boxed");
                }
                if (!jitListHeadSample(esrc, eat, &elemSample, &elemMixed)) {
                    return osrNo(fn, top,
                                 "an enumerate loop whose elements are too "
                                 "often null");
                }
                elemStg = (uint8_t)LIST_STORE_BOXED;
                iterKind = 5;
            } else if (iter->kind == ITER_DICT_ITEMS && IS_DICT(iter->source)) {
                elemSample = iter->source;
                iterKind = 3;
            } else if (jitPairListOn() && iter->kind == ITER_LIST &&
                       IS_LIST(iter->source)) {
                ObjList *psrc = AS_LIST(iter->source);
                int pat = (int)iter->index;
                if (pat < 0 || pat >= psrc->count) {
                    return osrNo(fn, top,
                                 "a pair loop already past its last element");
                }
                Value pel = jaiListGet(psrc, pat);
                if (!IS_TUPLE(pel) || AS_TUPLE(pel)->count != 2) {
                    return osrNo(fn, top,
                                 "a pair loop over a list of something other "
                                 "than 2-tuples");
                }
                {
                    const ObjTuple *st = AS_TUPLE(pel);
                    if (IS_NULL(st->items[0]) || IS_NULL(st->items[1])) {
                        return osrNo(fn, top,
                                     "a pair loop whose sampled tuple holds "
                                     "a null component");
                    }
                    int scan = psrc->count < 1024 ? psrc->count : 1024;
                    int pnulls = 0, pother = 0;
                    for (int i = 0; i < scan; i++) {
                        Value v = jaiListGet(psrc, i);
                        if (!IS_TUPLE(v) || AS_TUPLE(v)->count != 2) {
                            pother++;
                            continue;
                        }
                        for (unsigned c = 0; c < 2; c++) {
                            Value w = AS_TUPLE(v)->items[c];
                            if (IS_NULL(w)) pnulls++;
                            else if (w.type != st->items[c].type) pother++;
                        }
                    }
                    if (pnulls * 64 > scan) {
                        return osrNo(fn, top,
                                     "a pair loop whose tuple components are "
                                     "too often null");
                    }
                    if (pother * 64 > scan) {
                        return osrNo(fn, top,
                                     "a pair loop whose tuple components too "
                                     "often change kind");
                    }
                }
                elemSample = pel;
                elemStg = psrc->stg;
                iterKind = 4;
            } else {
                return osrNo(fn, top,
                             iter->kind == ITER_LIST_ENUM
                                 ? "a pair loop over an enumerate snapshot, "
                                   "with the lazy enumerate switched off"
                             : iter->kind == ITER_LIST
                                 ? "a pair loop over a list, with the list "
                                   "pair head switched off"
                             : (iter->kind == ITER_USER ||
                                iter->kind == ITER_TRAIT)
                                 ? "a pair loop over a user iterator"
                                 : "a pair loop over something other than a "
                                   "live dict view or a list of 2-tuples");
            }
        } else if (iter->kind == ITER_RANGE && IS_RANGE(iter->source)) {
            /* Unit steps only -- that is what makes the yielded value start
             * plus the index. The start need not be 0; it is loaded at entry. */
            if (AS_RANGE(iter->source)->step != 1)
                return osrNo(fn, top, "a range with a step other than 1");
            /* Compiled head runs on `start + index` in one register, so the limit is biased by the start too (see
             * osrReserved). `limit` saturates at INT64_MAX for a range spanning the whole type, and a biased limit that wrapped would compare the wrong way round -- such a loop is left to the interpreter. */
            {
                int64_t biased;
                if (__builtin_add_overflow(AS_RANGE(iter->source)->start,
                                           iter->limit, &biased)) {
                    return osrNo(fn, top,
                                 "a range whose start-biased limit overflows");
                }
            }
            iterKind = 1;
        } else if (iter->kind == ITER_LIST && IS_LIST(iter->source)) {
            ObjList *src = AS_LIST(iter->source);
            /* Element the loop is about to bind, taken from the list itself: reading the loop variable's slot
             * instead gives whatever the previous iteration left there (nothing on the first entry), aiming the kind guard at the wrong type -- what crashed the first attempt at this. */
            int at = (int)iter->index;
            if (at < 0 || at >= src->count)
                return osrNo(fn, top, "a list loop already past its last element");
            elemStg = src->stg;
            iterKind = 2;
            /* Whether the list holds one class or several (jitListHeadSample,
             * which the enumerate head and the function tier's enumerate arm
             * share) -- and whether a `list[T?]` holds nulls beside its
             * instances. The element guard this form emits is a tag check
             * against VAL_OBJ, so a null bails out of the compiled loop -- and
             * the loop is re-entered on the next element, so the bail is paid
             * per null and not once.
             *
             * Refusing on PRESENCE is the obvious answer and it is the wrong
             * one; the question is DENSITY. Same probe throughout -- a 2M
             * list[Node?], twenty passes of
             * `for x in xs { if x is null { t += 1 } }`, whole process, best
             * of three alternating runs under scripts/gpu_lock.sh:
             *
             *                    no nulls   1 in 3    1 in 1000
             *   declined            635       661        661
             *   no refusal          223     11604        296
             *   refuse on presence  219       661        661
             *   refuse past 1/64    223       662        280
             *
             * One null in three costs 17.6x with no refusal at all, which is
             * why a refusal has to exist. One null in a thousand is 2.36x
             * FASTER pinned than interpreted, which is what refusing on
             * presence throws away. The threshold keeps both ends.
             *
             * The proper fix is to widen the element to SLOT_MAYBE_INST so
             * the guard accepts a null and no bail happens at all, which
             * would retire this whole test. Until then, this. */
            if (!jitListHeadSample(src, at, &elemSample, &elemMixed)) {
                return osrNo(fn, top, "a list loop whose elements are too often null");
            }
        } else if (iter->kind == ITER_STRING && IS_STRING(iter->source) &&
                   jitStringHead()) {
            /* The head steps ASCII inline and DEOPTS on a byte >= 0x80, so the
             * question a new head owes is density, not presence
             * (a-new-loop-head-needs-a-density-census): a bail leaves the
             * compiled loop and re-enters it, so a string that is mostly
             * multi-byte would pay that per character and come out slower than
             * refusing outright.
             *
             * Same 1-in-64 threshold the list head uses for nulls, and for the
             * same reason: source text, identifiers, JSON keys and log lines
             * are ASCII with the occasional accent, while text that is mostly
             * non-ASCII is better left to the interpreter's decoder. Sampled
             * from the CURRENT index forward, because that is the part the
             * compiled loop will actually walk. */
            ObjString *src = AS_STRING(iter->source);
            int64_t at = iter->index;
            if (at < 0 || at >= (int64_t)src->length)
                return osrNo(fn, top, "a string loop already past its last byte");
            int64_t scan = (int64_t)src->length - at;
            if (scan > 4096) scan = 4096;
            int64_t wide = 0;
            for (int64_t i = 0; i < scan; i++)
                if ((unsigned char)src->chars[at + i] >= 0x80u) wide++;
            if (wide * 64 > scan)
                return osrNo(fn, top, "a string loop that is mostly non-ASCII");
            {
                ObjString *empty = jaiStringIntern("", 0);
                if (empty == NULL)
                    return osrNo(fn, top, "no interned empty string to sample");
                elemSample = OBJ_VAL((Obj *)empty);
            }
            iterKind = 6;
        } else {
            return osrNo(fn, top, "an iterator kind with no loop-head arm");
        }
    }

    JaiOsrForm *form = NULL;
    for (unsigned i = 0; i < fn->osrCount; i++) {
        /* Kind as well as offset: a form compiled for a range head, entered with a list iterator, would read
          * ObjRange::start out of an ObjList -- `for x in cond ? xs : 0..n` is enough to arrange it. */
        if (fn->osrForms[i].top == top &&
            fn->osrForms[i].iterKind == iterKind &&
            osrFormStorageFits(&fn->osrForms[i], frame->slots, iterKind,
                               elemStg)) {
            form = &fn->osrForms[i];
            break;
        }
    }
    if (form == NULL) {
        if (fn->osrRefused)
            return osrNo(fn, top, "the tier has given up on this body's loops");
        if (fn->osrCount >= osrFormCap())
            return osrNo(fn, top, "no room left to record another loop form");
        /* This head's own share of the budget. See ObjFunction::osrMissTop for
         * why the count cannot be per function: the first loop to run spends
         * it, and every later loop in the same body is then refused a look. */
        unsigned miss = fn->osrMissCount;
        for (unsigned i = 0; i < fn->osrMissCount; i++) {
            if (fn->osrMissTop[i] == top) { miss = i; break; }
        }
        if (miss < osrFormCap() && fn->osrMissAttempts[miss] >= 5 * osrFormCap()) {
            /* This head is spent; other heads in the same body are not. */
            return osrNo(fn, top, "this loop head is out of compile attempts");
        }
        /* The whole body first, then just the part before the first
         * `continue`.
         *
         * A body that reaches past a `continue` is the one worth having --
         * without it the tail runs interpreted and a loop whose `continue`
         * rarely fires is twenty times slower than the same loop written with
         * an `if`. But the tail can also hold something the tier will not
         * compile, an uncompiled callee most often, and then the whole loop
         * declines and NOTHING is compiled. tests/bench's contour follower is
         * exactly that shape. So the prefix stays as the fallback: some of the
         * loop compiled beats none of it. */
        if (!compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, true, false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, true, true) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, false, false) &&
            !compileOsr(closure, top, frame->slots, iterKind, elemSample,
                        elemMixed, elemStg, false, true)) {
            /* Inlining widens live ranges; a loop that will not fit with it
             * may fit without, and a compiled call beats no compile at all. */
            if (miss == fn->osrMissCount && miss < osrFormCap()) {
                fn->osrMissTop[miss]      = top;
                fn->osrMissAttempts[miss] = 0;
                fn->osrMissCount++;
            }
            if (miss < osrFormCap()) fn->osrMissAttempts[miss]++;
            /* The whole-function backstop, for a body with more uncompilable
             * heads than the table holds: without it those heads share the
             * untracked path and would be retried for the life of the run. */
            if (++fn->osrAttempts >= 5 * osrFormCap() * osrFormCap()) {
                fn->osrRefused = true;
            } else if (fn->osrMissCount >= osrFormCap()) {
                bool allSpent = true;
                for (unsigned i = 0; i < osrFormCap(); i++) {
                    if (fn->osrMissAttempts[i] < 5 * osrFormCap()) {
                        allSpent = false;
                        break;
                    }
                }
                if (allSpent) fn->osrRefused = true;
            }
            /* The compile printed its own reason on the way out. What it could
             * not know is whether that was the attempt that retired the body,
             * which is the difference between a loop that will be tried again
             * and one nothing will look at for the rest of the run. */
            if (fn->osrRefused) {
                return osrNo(fn, top,
                             "that attempt retired this body's loops for good");
            }
            return 0;
        }
        form = &fn->osrForms[fn->osrCount - 1];
    }
    if (fn->module == NULL || fn->module->version != fn->jitOsrModuleVersion)
        return osrNo(fn, top, "the form was compiled against an older module");

    /* Every slot must still hold what it held when this was compiled. */
    for (unsigned i = 0; i < form->slots; i++) {
        Value v = frame->slots[i];
        SlotKind want = (SlotKind)(form->kinds[i] & 0x0Fu);
        if (want == SLOT_MAYBE_INST) {
            if (!IS_NULL(v) && !IS_INSTANCE(v))
                return osrNoSlot(fn, top, i, want, v);
            continue;
        }
        /* The same check without the class. `default` used to admit this kind
         * on the premise that an unnamed slot is never read, which is false
         * here: the prologue loads every slot with a register home eight bytes
         * wide. A slot compiled as object-or-null and entered holding an int
         * would put that int in the home, `== null` would compare it against
         * zero and call 7 non-null, and the first tag written off that payload
         * says VAL_OBJ for a pointer of 7. Exactly what the SLOT_OBJ arm below
         * records having been fixed once for the neighbouring kind. */
        if (want == SLOT_MAYBE_OBJ) {
            if (!IS_NULL(v) && !IS_OBJ(v))
                return osrNoSlot(fn, top, i, want, v);
            continue;
        }
        switch (want) {
        case SLOT_INT:   if (!IS_INT(v))   return osrNoSlot(fn, top, i, want, v); break;
        case SLOT_FLOAT: if (!IS_FLOAT(v)) return osrNoSlot(fn, top, i, want, v); break;
        case SLOT_BOOL:  if (!IS_BOOL(v))  return osrNoSlot(fn, top, i, want, v); break;
        /* Storage is settled by osrFormStorageFits, which ran as part of
         * choosing this form: a body compiled against a boxed list reads a
         * Value every sixteen bytes, and the same body entered holding a
         * `list[int]` would read two elements as one tagged pair. */
        case SLOT_LIST:  if (!IS_LIST(v))  return osrNoSlot(fn, top, i, want, v); break;
        case SLOT_INST:  if (!IS_INSTANCE(v)) return osrNoSlot(fn, top, i, want, v); break;
        /* jitArgIn (function tier) has always checked this; the loop tier let it through via `default`, so a
         * slot compiled as "some object" could be entered holding an int and have its payload read as a pointer. Unexercised until something emitted a load off such a slot (the invoke arm's receiver guard, below). */
        case SLOT_OBJ:   if (!IS_OBJ(v))   return osrNoSlot(fn, top, i, want, v); break;
        default: break;   /* opaque: never read */
        }
    }

    /* And it must be the same CLASS, not merely an instance. The kind check
     * above passes any instance; the compiled code reads fields at slot indices
     * baked in from one particular class. */
    for (unsigned i = 0; i < form->shapeCount; i++) {
        Value v = frame->slots[form->shapeSlot[i]];
        if (IS_NULL(v)) continue;    /* SLOT_MAYBE_INST, and it is the null */
        if (!IS_INSTANCE(v))
            return osrNo(fn, top, "a slot pinned to a class no longer holds an instance");
        ObjClass *klass = AS_INSTANCE(v)->klass;
        if (klass == NULL || klass->shapeId != form->shapeId[i])
            return osrNo(fn, top, "a slot pinned to a class now holds a different one");
    }

    gDeopt.nstack = 0;
    gDeopt.base = 0;
    /* An OSR form's locals ARE the frame slots and its stub writes none, so
     * nothing here fills the mask -- clear it rather than leave the last
     * function-tier stub's behind for whoever reads the record next. */
    gDeopt.skipLocals = 0;
    int64_t at = ((OsrFnIter)(uintptr_t)form->code)(frame->slots, iter);
    if (at == -1)
        return osrNo(fn, top, "the compiled loop bailed out at entry");
    if (at == -2) return 2;              /* an exception is pending */
    if (gDeopt.base != 0) vm.stackTop--;   /* the exhausted iterator */
    for (int64_t i = 0; i < gDeopt.nstack; i++) *vm.stackTop++ = gDeopt.stack[i];
    *resumeAt = (uint32_t)at;
    return 1;
}

#else

/* 0 is "declined", so the interpreter runs the loop itself. */
int jaiJitEnterOsr(ObjClosure *closure, uint32_t top, uint32_t *resumeAt) {
    (void)closure; (void)top; (void)resumeAt; return 0;
}

#endif
