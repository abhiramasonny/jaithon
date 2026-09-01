/* jit_osr.c -- the on-stack-replacement tier: compiling a hot loop out of a frame
 * that is already running, and entering it. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
/* For jaiJitFieldReadFor: which builtins are one load from their receiver. */
#include "vm/jit/jit_field_read.h"
/* For jaiBuiltinMethod: resolving `xs.len()` to a native needs the runtime's name table. */
/* For jaiOpBranchOperandAt: says which opcodes carry a branch target. */
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* ------------------------------------------------------------------ */
/* On-stack replacement                                                 */
/* ------------------------------------------------------------------ */

/* Returns the bytecode offset the interpreter should continue from. */
typedef int64_t (*OsrFn)(Value *slots);
typedef int64_t (*OsrFnIter)(Value *slots, ObjIter *iter);

/* The OP_LOOP that jumps back to `top`, and so the end of the loop. Gives up
 * on OP_CLOSURE, whose length depends on its operands. */
/* findLoopEnd walks from `top` itself, so if that offset is mid-instruction the walk decodes operand
 * bytes as opcodes and can find a plausible-but-wrong OP_LOOP -- nbody's advance was once compiled from offset 128, inside a GET_LOCAL2's operands. Walking from the start (isInstructionStart, below) costs one scan per compile and removes the question. */
/* OP_CLOSURE is the one variable-length instruction (u24 constant index + 3 bytes/upvalue, count
 * from the function the index names). Treating it as undecodable once refused OSR for the WHOLE CHUNK (both walks below scan from offset 0), so a module declaring a class or function before its hot loop -- i.e. every one of them -- could never enter a compiled loop at module scope, silently: compileOsr was never even reached, so nothing was there to report a decline. */
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

/* A slot captured BY REFERENCE by a closure is aliased by an ObjUpvalue pointing straight into the
 * VM's slot array; caching it in a register during OSR makes them different storage for the loop's duration -- reads through the closure see a frozen value, writes through it are lost. (`var base=3; let f=|x| x+base; while .. { acc=f(acc); base+=1 }` once returned a different, always-short sum every run, since OSR triggers on a timer tick.) Scans the WHOLE enclosing function, not just the loop region, since the capture can be anywhere. `how` bit 1 = by-value capture (safe, copies into a closed cell); bit 0 alone = by-reference. Marks individual slots, not chunk-wide, since the register plan is per-slot. Returns false (keep everything in memory) if the chunk doesn't decode. */
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

/* Where the loop whose head is `top` ends: after its LAST back edge, not its
 * first.
 *
 * A loop has one back edge for the ordinary path off the bottom and one more
 * for every `continue`, and a `continue` comes FIRST -- it is written near the
 * top of the body, which is the point of it. Stopping at the first meant the
 * compiled region was the prefix before the `continue` and everything after it
 * ran interpreted, so the more often the `continue` was SKIPPED the slower the
 * loop got: measured over three hundred thousand elements, 0.25ms when every
 * iteration took the `continue` against 7.6ms when none did, for the same loop
 * written with an `if` in 0.29ms.
 *
 * Only this loop's own back edges name `top`: an inner loop's name the inner
 * head, and a later loop's name its own. So the last one is this loop's bottom
 * however many `continue`s are between. */
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

/* How many loops enclose each byte, so slots can be ranked by heat: every OP_LOOP is a back edge, the
 * range it jumps over is its body, and one pass counting those ranges is enough to put the innermost loop's variables ahead of the setup around them (doesn't need to be exact). File static since both Emit structures read the same table and compilation isn't reentrant; a chunk longer than JIT_MAX_DEPTH_MAP just goes unweighted. */
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

/* The table, filled and handed back, so the function tier can ask for it
 * without gLoopDepth being visible where it is declared. */
const uint8_t *loopDepthFor(const Chunk *c, unsigned *count) {
    *count = loopDepthTable(c);
    return gLoopDepth;
}

/* An OSR entry that refuses without a word makes the whole body invisible.
 * It never reaches the tier's log, so a ranking can only say "never
 * considered" and the reason is gone -- there is nowhere to look it up.
 * `good_features_to_track` sat at 16.6% of the jaicv benchmark in exactly
 * that state: hot, sampled, refused, and unnamed.
 *
 * Same wording and format as the body walker's refusals, so the report joins
 * these to the interpreted-work attribution without knowing they came from a
 * different path. */
static int osrNo(ObjFunction *fn, uint32_t top, const char *why) {
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr %s at %u stopped: %s\n",
                fn->name ? fn->name->chars : "<anon>", top, why);
    }
    return 0;
}

/* The entry guards, which say what the form wants and what the frame holds.
 * A form that exists and is never entered looks identical to one that was
 * never compiled unless the mismatch is named. */
static int osrNoSlot(ObjFunction *fn, uint32_t top, unsigned slot,
                     SlotKind want, Value held) {
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr,
                "[jit] osr %s at %u stopped: slot %u holds %s, but the form "
                "was compiled for %s\n",
                fn->name ? fn->name->chars : "<anon>", top, slot,
                jaiTypeNameStatic(held), slotKindName(want));
    }
    return 0;
}


/* How many compiled loops one body may keep. Only lets the old, smaller table
 * be put back in the same binary, so the change can be measured against
 * itself; the table itself is sized JAI_OSR_MAX. */
static unsigned osrFormCap(void) {
    static unsigned cap;
    if (cap == 0) {
        const char *e = getenv("JAITHON_JIT_OSR_FORMS");
        unsigned want = (e != NULL) ? (unsigned)atoi(e) : JAI_OSR_MAX;
        cap = (want >= 1 && want <= JAI_OSR_MAX) ? want : JAI_OSR_MAX;
    }
    return cap;
}

/* The record is sized JAI_OSR_SLOTS; this only lets the old, smaller cap be
 * put back in the same binary, so the change can be measured against itself. */
static unsigned osrSlotCap(void) {
    static unsigned cap;
    if (cap == 0) {
        const char *e = getenv("JAITHON_JIT_OSR_SLOTS");
        unsigned want = (e != NULL) ? (unsigned)atoi(e) : JAI_OSR_SLOTS;
        cap = (want >= 1 && want <= JAI_OSR_SLOTS) ? want : JAI_OSR_SLOTS;
    }
    return cap;
}

/* Same, for the compile path, which reports failure as a bool. Everything
 * before the emitter refused without a word, so a body that never got past
 * these looked to the log exactly like one that was never sampled. */
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

/* The loop tier had no retry at all, where the function tier has had one since
 * `nullableLocal` existed: a slot the walk cannot settle on one kind for simply
 * declined the whole loop. That is `var at = head` followed by `at = at.next`
 * -- an instance seeded from the slot, then a maybe-instance assigned into it
 * -- which is every list and tree walk there is, and the shape a parser's inner
 * loops are made of. Two attempts: the second seeds the slots the first asked
 * for as nullable.
 *
 * MEASURED, and only as a chain. On its own this changes nothing: widening the
 * local moves the refusal from the assignment to the `at.follow()` call, and
 * accepting a nullable RETURN kind (see emitMaybeInstResult) moves it back to
 * the assignment. All three together plus the maybe-instance receiver at
 * OP_INVOKE take a 200-node list walked 20000 times from 137 ms to 38 ms,
 * best of seven alternating runs under scripts/gpu_lock.sh with no overlap
 * between the two sets -- 3.6x, and it is a decline-to-compile transition,
 * not a micro-optimisation: `walk` goes from no compiled form at either loop
 * head to both.
 *
 * RULED OUT: the self-hosted compiler does not move. `check --no-cache` over
 * four compiler files is 2143 ms against 2146 ms median of nine, inside the
 * spread, even though the return-kind gate alone stops 64 sites in one file.
 * Those bodies clear it and stop at the next wall (OP_INVOKE on a receiver
 * whose class the model cannot pin, and OP_GET_FIELD_LOCAL). Kept for the
 * loops it does unblock, on the same reasoning the enum fold was. */
static bool compileOsr(ObjClosure *closure, uint32_t top, Value *slots,
                       uint8_t iterKind, Value elemSample, bool elemMixed,
                       uint8_t elemStg,
                       bool wholeBody, bool noInline) {
    bool nullable[JIT_MAX_SLOTS + 1];
    bool needNullable[JIT_MAX_SLOTS + 1];
    /* Same ledger the function tier has kept since it grew one: a slot given
     * two kinds does not give the loop up, it asks to carry its tag and the
     * walk runs again with that decided from the start. OSR had the nullable
     * half of this and not the dynamic half, and a comprehension is exactly
     * what needed it -- the front end reuses one slot for the loop variable and
     * for the result, so `[x * 2 for x in xs]` seeds the slot as a list from
     * the previous iteration and then binds an int into it. Three attempts,
     * because a body can want both widenings and neither implies the other. */
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
    /* The entry re-checks every slot, so this is the size of that record --
     * nbody's advance declares nineteen. */
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
    /* See Emit::rangeHead. */
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
    e.savedCount = JIT_MAX_SAVED;
    memcpy(e.nullableLocal, nullable, sizeof e.nullableLocal);
    memcpy(e.dynamicLocal, dynamic, sizeof e.dynamicLocal);
    /* Each slot takes the kind it holds right now. The entry re-checks them on
     * every later entry, so this is a specialisation, not an assumption. */
    for (unsigned i = 0; i < e.locals; i++) {
        Value v = slots[i];
        e.localTyped[i] = true;
        /* A frame's slots run to fn->maxSlots, and the ones the program has
         * not reached yet hold whatever the last frame at that depth left --
         * including a VAL_OBJ tag over a NULL pointer. IS_LIST and IS_INSTANCE
         * both dereference before they test, so without this the COMPILER
         * reads Obj::type off address zero: EXC_BAD_ACCESS inside
         * compileOsrOnce, once in eight runs of tests/bench/jaiframe/frameops
         * and never in the test suite, because it needs a slot that is both
         * stale and untouched at the moment a loop goes hot. */
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
            /* A slot an earlier attempt found sometimes-null takes the wider
             * kind from the start; the class still comes from what is in the
             * slot NOW, which a maybe-instance is a correct supertype of. */
            e.localKind[i]  = nullable[i] ? SLOT_MAYBE_INST : SLOT_INST;
            e.localClass[i] = AS_INSTANCE(v)->klass;
            e.localShape[i] = AS_INSTANCE(v)->klass->shapeId;
        } else if (IS_OBJ(v)) {
            e.localKind[i] = SLOT_OBJ;
        } else {
            /* Nothing recognisable in it yet -- a slot the loop assigns before
             * it reads. It takes its kind from the first thing bound to it. */
            e.localKind[i] = SLOT_OPAQUE;
            e.localTyped[i] = false;
        }
    }


    /* Runs AFTER the seeding above, not before: it used to measure a body where every slot was still an
     * untyped int (zeroed), a different program from the one being compiled, corrupting the maxValue the register decision rests on. Also narrows `locals` to what the loop actually names -- OSR started from the whole enclosing function's `maxSlots` (18-20 for real bodies), so `reserved + locals + maxValue <= 10` could never hold and register-resident locals were unreachable for any real function. */
    /* Registers for the slots that earn them, memory for the rest. The
     * reserved four (or one) plus the X locals plus the deepest expression
     * must all sit inside the ten callee-saved registers; a first pass
     * measures the last of those, and which slots are named at all, and how
     * often each of them is named inside the innermost loop. */
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
        probe.scratchRoom = JIT_SCRATCH_BANK_COUNT;
        probe.offsetToInst = map; probe.offsetToDepth = depths;
        probe.chunkDepth = chunkDepth; probe.chunkDepthCount = fn->chunk.count + 1;
        probe.limitLiteral = -1; probe.bailBlock = -1; probe.exceptionExit = -1;
        probe.loopDepth = gLoopDepth;
        probe.loopDepthCount = e.loopDepthCount;
        /* "Never seen" is an empty range, not offset zero. */
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

            /* A body that never calls out and never inlines can put its
             * operand stack in x0..x8 instead of above the locals, and then
             * the whole callee-saved bank is the locals'. This is the
             * difference between a five-deep expression leaving room for two
             * locals and a seven-deep one leaving room for none -- and the
             * loops that go seven deep are exactly the ones with several rows
             * or accumulators to keep. */
            e.scratchValues = !probe.clobbersScratch &&
                              probe.maxValueAll <= JIT_INL_COUNT;
            /* Under split stress the split is preferred to the whole-scratch
             * bank, because it is the two-run layout that has the boundary in
             * it and the whole-scratch one does not. */
            if (jitSplitStress()) e.scratchValues = false;
            /* A body that DOES call still need not keep its WHOLE stack in the
             * callee-saved bank -- only the part of it that can be live while
             * a helper runs, which is everything below the deepest the stack
             * ever is at a call. life's `step` goes five deep summing nine
             * neighbours and two deep at its `row.push`, so three of its five
             * operand registers were being held against a call that can never
             * see them, while five of its locals sat in memory.
             *
             * Not offered to a body that inlines: an inlined body takes x0..x8
             * for its own entries (inlineOwnBank), and both cannot have them.
             * Not offered to the function tier either, which does not run this
             * code -- x0..x3 are its arguments and the roadmap prices the same
             * change there at +-1%. */
            unsigned wantSplit = probe.clobberDepth;
            /* Stress: a body with no calls at all has clobberDepth 0, and a
             * split at 0 is no split. Moving the boundary up to 1 is still
             * sound there -- nothing can clobber x0..x8 in a body that never
             * calls -- and it is what puts the two-run layout under every
             * existing gate rather than under the few bodies that want it. */
            if (jitSplitStress() && wantSplit == 0) wantSplit = 1;
            if (!e.scratchValues && !probe.inlined &&
                (probe.clobbersScratch || jitSplitStress()) &&
                probe.maxValue > wantSplit &&
                probe.maxValue - wantSplit <= JIT_SCRATCH_BANK_COUNT) {
                e.splitAt = wantSplit;
            }
            /* What is left over after the loop's own reserved registers and
             * the deepest expression the body builds. maxValue is model state,
             * not a register number, so measuring it in memory mode and
             * spending it here is sound. Under a split only the half below
             * `splitAt` is charged to the callee-saved bank. */
            unsigned overhead = osrReserved(&e) +
                                (e.scratchValues ? 0u
                                 : e.splitAt != 0 ? e.splitAt
                                                  : probe.maxValue);
            unsigned availX = overhead < JIT_MAX_SAVED
                                  ? JIT_MAX_SAVED - overhead : 0u;
            /* Ranked by what a register SAVES, on the same ledgers and through
             * the same planner as the function tier -- not by how often the
             * body names the slot, which is what this used to do. Both counts
             * are weighted by loop nesting the same way, so the difference is
             * not "flat versus weighted": it is that one pooled counter had to
             * answer two questions it cannot tell apart. It could not say which
             * BANK a slot wants (a float read costs an `ldr d` from memory and
             * an `fmov` from an X home, so the two banks are not worth the same
             * to it), and it could not say that a WRITE is worth twice a read
             * here, because an OSR slot's memory home is the interpreter's own
             * Value and takes the tag as well as the payload. A loop variable
             * assigned every iteration is exactly the case both mistakes fell
             * on, which is the case OSR exists for.
             *
             * The slots the ledgers cannot rule out on their own are handed
             * over as `skip`: one captured by reference by a closure, which is
             * aliased by an ObjUpvalue pointing into the VM's slot array and so
             * must stay in memory whatever it saves (see chunkByRefCaptures);
             * one the probe found dynamic, which keeps its run-time tag and has
             * nowhere but the frame to put it; and, when the chunk would not
             * decode at all, every slot -- the by-reference scan is what makes
             * a register safe here, so failing it means registers for nobody.
             *
             * `slotUse` stays in that mask rather than being retired for the
             * ledgers, so the set of slots this can place is the old one
             * narrowed, never widened. The two disagree in one direction only:
             * slotUse goes uncounted past JIT_MAX_DEPTH_MAP (a chunk over 8KB),
             * where the ledgers still count at weight 1, so retiring it would
             * hand registers to slots in a region no gate has ever planned for.
             * Whatever the ledgers rank last is dropped by the zero-gain
             * exclusion anyway, which is the accurate half of the old test:
             * slotUse counts localInRange, a bounds CHECK made at more sites
             * than actually read or write the slot. */
            bool byRef[JIT_MAX_SLOTS + 1];
            bool decoded = chunkByRefCaptures(&fn->chunk, byRef, e.locals);
            bool skip[JIT_MAX_SLOTS + 1];
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                skip[i] = !decoded || i >= e.locals || byRef[i] ||
                          probe.dynamicLocal[i] || probe.slotUse[i] == 0;
            }
            /* `probeStranded` is the count of slots that earned a register and
             * found none left -- what a wider bank would buy, and 0 far more
             * often than the decline census suggests. */
            planSlotRegisters(&e, &probe, availX, skip, &probeStranded);

            /* What planHoists needs: where each slot was written, where it was
             * subscripted, where the body can destroy a caller-saved register,
             * and the registers nothing else claims. */
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

            /* Which sampled storages the real pass may emit against.
             *
             * A slot ASSIGNED inside the region cannot be pinned: the tier's
             * OP_ELEM_KIND arm writes elemKind without specialising, so a
             * `var f: list[bool] = []` at the top of a loop body makes a BOXED
             * list every round while the pin, taken from the frame before
             * entry, still says U8. A sieve written that way read every
             * element one byte wide out of a sixteen-byte array and printed a
             * different prime count on each run.
             *
             * A region that CALLS OUT cannot pin either: jaiListBox is
             * reachable from ordinary builtins -- a sort, a concat of two
             * storages, a put of a kind the store cannot hold -- and it
             * de-specialises the list with nothing the compiled body could
             * watch. The same test planHoists makes of a hoisted header, for
             * the same reason: the entry guard proves a fact, and this is
             * whether the body can invalidate it. */
            bool bodyCallsOut = regionCalls(&e, top, end);
            for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
                e.localStgPin[i] =
                    !bodyCallsOut &&
                    !(e.slotWriteHi[i] >= top && e.slotWriteLo[i] < end);
            }
            e.elemStgPin = !bodyCallsOut;
            /* x13..x17 are nobody's in any body -- a call destroys them, which
             * is why only a call-free REGION may hold anything there, and
             * planHoists is what tests that. Offering them unconditionally is
             * the whole per-region change: they used to be withheld from every
             * body containing a call, including the ones whose inner loop is
             * the entire benchmark. */
            for (unsigned r = 0; r < JIT_FREE_COUNT; r++) {
                e.hoistPool[e.hoistPoolCount++] = (uint8_t)(JIT_FREE_FIRST + r);
            }
            /* Whatever the operand stack left at the top of its own bank.
             * Only when it IS that bank -- otherwise those registers are
             * carrying operands, or an inlined body has x0..x8 to itself and
             * none of it is spare. `scratchValues` is a whole-body claim and
             * has to stay one: unlike x13..x17, these registers have another
             * owner outside the region. */
            if (e.scratchValues) {
                for (unsigned r = probe.maxValueAll;
                     r < JIT_SCRATCH_BANK_COUNT; r++) {
                    e.hoistPool[e.hoistPoolCount++] =
                        (uint8_t)(JIT_INL_BANK + r);
                }
            }
        } else {
            /* The probe is where a kind clash is found -- the real walk below
             * runs on the same seed and would only find it again -- so its
             * request for a wider one has to travel back to the retry loop
             * from here. */
            memcpy(needNullable, probe.needNullable, sizeof probe.needNullable);
            memcpy(needDynamic, probe.needDynamic, sizeof probe.needDynamic);
        }
        for (int i = 0; i <= fn->chunk.count; i++) { map[i] = -1; depths[i] = -1; }
    }

    unsigned frame = 16u + 8u * JIT_MAX_SAVED + (unsigned)sizeof(JitCallDesc);
    e.descOffset = 16u + 8u * JIT_MAX_SAVED;
    /* The ObjIter, for the heads that do not keep it in a register. Sixteen
     * bytes so the frame stays 16-aligned without a second rounding. */
    e.iterFrameOffset = (frame + 7u) & ~7u;   /* str/ldr scale the offset by 8 */
    frame = e.iterFrameOffset + 16u;
    e.fpSaveOffset = frame;
    frame += 8u * JIT_FP_MAX_SAVED;
    e.frameBytes = (frame + 15u) & ~15u;

    emitFrameEnter(&e);
    emitSaveRestore(&e, true);
    emitFpSaveRestore(&e, true);
    emit(&e, jaiA64MovX(JIT_SLOTS_REG, 0));
    /* Payloads only: the entry has already checked every slot's kind, so there
     * is nothing left to guard here.
     *
     * Except the width. Unlike the function tier, which marshals every argument
     * through jitArgIn and hands a bool over as a clean 0 or 1, these slots
     * are the interpreter's own and BOOL_VAL is a one-byte `strb` -- the seven
     * bytes above a bool are whatever that slot last held. An eight-byte load
     * puts that garbage in a SLOT_BOOL register, and every consumer of one
     * tests the whole word, so `if flag {` took the wrong arm whenever the
     * slot had previously held anything with a high byte set. Silent: the
     * program ran, and computed the other branch. */
    for (unsigned i = 0; i < e.locals; i++) {
        if (e.slotXReg[i] != 0 && e.localKind[i] == SLOT_BOOL) {
            emit(&e, jaiA64LdrByte(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotXReg[i] != 0) {
            emit(&e, jaiA64LdrX(e.slotXReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        } else if (e.slotFpReg[i] != 0) {
            emit(&e, jaiA64LdrD(e.slotFpReg[i], JIT_SLOTS_REG, i * 16u + 8u));
        }
    }
    if (hasIter && iterKind == 3) {
        /* A dict-items head keeps only the pointer: its index, limit and
         * version all live in the ObjIter and the step reads them there, so
         * there is nothing to hoist here and nothing to write back at an exit. */
        emit(&e, jaiA64MovX(JIT_PAIR_ITER_REG, 1));
    } else if (hasIter) {
        /* x1 is the iterator on entry. A range head reads everything it wants
         * out of it here and parks the pointer in the frame; a list head keeps
         * it, because its version guard reads the iterator every iteration. */
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
            /* A range yields start + index, so both bounds are shifted by the
             * start once, here, and the loop runs on the yielded value
             * directly. ObjIter.source is a Value, so the object pointer sits
             * eight bytes into it. jaiJitEnterOsr has already established that
             * start + limit does not overflow. */
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
            emit(&e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                (unsigned)offsetof(ObjRange, start)));
            emit(&e, jaiA64AddX(JIT_IDX_REG, JIT_IDX_REG, JIT_SCRATCH_A));
            emit(&e, jaiA64AddX(JIT_LIM_REG, JIT_LIM_REG, JIT_SCRATCH_A));
        } else {
            /* A list head keeps the ObjList itself in JIT_START_REG. */
            emit(&e, jaiA64LdrX(JIT_START_REG, rIter,
                                (unsigned)offsetof(ObjIter, source) + 8));
        }
    }

    planHoists(&e, fn);
    if (getenv("JAI_JIT_WHY")) {
        /* The register arithmetic, not just the verdict, and for a body that
         * COMPILES as much as one that does not: "declined for want of a
         * register" is only half the census -- the other half is a body that
         * fitted with nothing left over, which is what a bank decision is
         * chosen against. Printed here rather than in the failure arm because
         * this is the point where every number in it is final.
         *
         * Only when the measuring pass got through: a loop that declined
         * before the register plan has zeroed numbers, and printing them reads
         * as "nought entries deep", which is a different and false claim.
         *
         * Its own line, and without the words the decline census greps for, so
         * that adding it cannot invent census entries. */
        if (probeRan) fprintf(stderr,
                "[jit] osr %s at %u registers: %u reserved, %u stack "
                "(%u incl. inlined), %u deep at a call, %u x-locals, "
                "%u fp-locals, %u stranded, bank %s, of %u; %s\n",
                fn->name ? fn->name->chars : "<anon>", top,
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
                    fn->name ? fn->name->chars : "<anon>", top,
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
        /* iterKind 3 keeps nothing in a register but the pointer, and the step \
         * has already stored the index it advanced -- so every way out of a    \
         * dict-items loop finds the ObjIter already current. */                \
        if (hasIter && iterKind != 3) {                                        \
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
        /* An ordinary way out carries no operand stack of its own, and has to
         * say so.
         *
         * `gDeopt` is one global record, on the reasoning that only one body
         * can be deoptimising at a time. That holds for a body and its own
         * stubs; it does not hold across a call. A compiled callee that
         * deopts part way through this region writes the record, is put back
         * on its feet by its own return path, and leaves `nstack` behind --
         * and the C side reads `nstack` after every way out of the region,
         * not only after a deopt. So a region that had called such a callee
         * and then finished its loop normally pushed the callee's leftover
         * operand stack onto this frame's.
         *
         * What that cost: `for n in xs` around an inner loop calling a
         * compiled method left two extra values above the enclosing loop's
         * iterator, so the next `OP_FOR_ITER_BIND` peeked one of them and
         * the loop died with "expected an iterator, not 'int'". Only under
         * `--gc-stress`, which is what made the callee deopt reliably.
         *
         * Clearing it here is the narrow fix: every way out of a region now
         * states its own record rather than inheriting one. */
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
                /* Result of an already-happened call lives in the descriptor with whatever tag the callee actually
                 * returned. The function tier's stub knew this; this OSR one didn't, and alloc_churn came back 982406343 instead of 550770565 under deopt stress -- the entire reason this lastFromDesc branch exists. */
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
                        fn->name ? fn->name->chars : "<anon>", f->targetOffset);
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
                        fn->name ? fn->name->chars : "<anon>", top,
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
                fn->name ? fn->name->chars : "<anon>", top, e.count,
                (unsigned)iterKind);
        /* Same reason as the function tier's line: a loop that walked two
         * instructions and interpreted the other forty reported success. */
        if (e.unarmedOp != 0) {
            fprintf(stderr,
                    "[jit] osr %s at %u walked only to %s%s at %u -- the rest of "
                    "the loop is interpreted\n",
                    fn->name ? fn->name->chars : "<anon>", top,
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
    if (iterKind == 2 && form->iterStg != LIST_STG_ANY &&
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
            /* `for (k, v) in d.items()` at the top of the loop being entered.
             * Only the lazy dict view: a pair loop over a LIST of tuples has no
             * head arm, and letting it through here would enter a form compiled
             * for a dict with an ObjList in the register. The sample is the
             * dict itself -- the head arm reads the first live entry out of it
             * for the component kinds, as the list head reads items[index]. */
            if (iter->kind != ITER_DICT_ITEMS || !IS_DICT(iter->source)) {
                return osrNo(fn, top,
                             "a pair loop over something other than a live dict view");
            }
            elemSample = iter->source;
            iterKind = 3;
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
            elemSample = jaiListGet(src, at);
            elemStg = src->stg;
            iterKind = 2;
            /* Whether the list holds one class or several. One sample cannot
             * say, and getting it wrong is not merely a slower loop: a form
             * compiled pinned to the sampled class fails its own entry guard
             * on every later class, so the loop runs interpreted for the rest
             * of the program with no second chance to notice. Capped because
             * this runs per compile attempt and a list can be enormous; past
             * the cap the pinned form is compiled as before and deoptimises if
             * it was wrong, which is where this started. */
            if (IS_INSTANCE(elemSample)) {
                const ObjClass *first = AS_INSTANCE(elemSample)->klass;
                int scan = src->count < 1024 ? src->count : 1024;
                int nulls = 0;
                for (int i = 0; i < scan; i++) {
                    Value v = jaiListGet(src, i);
                    if (IS_NULL(v)) { nulls++; continue; }
                    if (!IS_INSTANCE(v)) continue;
                    if (AS_INSTANCE(v)->klass != first) { elemMixed = true; }
                }
                /* A `list[T?]` holding nulls beside instances. The element
                 * guard this form emits is a tag check against VAL_OBJ, so a
                 * null bails out of the compiled loop -- and the loop is
                 * re-entered on the next element, so the bail is paid per null
                 * and not once.
                 *
                 * Refusing on PRESENCE is the obvious answer and it is the
                 * wrong one; the question is DENSITY. Same probe throughout --
                 * a 2M list[Node?], twenty passes of
                 * `for x in xs { if x is null { t += 1 } }`, whole process,
                 * best of three alternating runs under scripts/gpu_lock.sh:
                 *
                 *                    no nulls   1 in 3    1 in 1000
                 *   declined            635       661        661
                 *   no refusal          223     11604        296
                 *   refuse on presence  219       661        661
                 *   refuse past 1/64    223       662        280
                 *
                 * One null in three costs 17.6x with no refusal at all, which
                 * is why a refusal has to exist. One null in a thousand is
                 * 2.36x FASTER pinned than interpreted, which is what refusing
                 * on presence throws away. The threshold keeps both ends.
                 *
                 * The proper fix is to widen the element to SLOT_MAYBE_INST so
                 * the guard accepts a null and no bail happens at all, which
                 * would retire this whole test. Until then, this.
                 *
                 * The class scan no longer stops at the first mismatch: the
                 * null count needs the whole prefix, and the cap is what
                 * bounds the cost. */
                if (nulls * 64 > scan)
                    return osrNo(fn, top, "a list loop whose elements are too often null");
            }
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
