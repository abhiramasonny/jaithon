/* jit_frame.c -- constants, prologue and epilogue, branches and their fixups, the
 * deopt sites, and the one-byte string ordering arm. */

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

/* JAITHON_JIT_MATCH=0 puts the four enum-`match` opcodes back on the unarmed
 * path, so what arming them is worth can be read off ONE binary. It has to be
 * one binary: alternating two invalidates __jaicache__ and every sample then
 * pays a stdlib recompile larger than the effect.
 *
 * Off is the exact pre-arm behaviour and not an approximation of it -- each arm
 * declines the way it declines any shape it cannot speak, into emitUnarmedDeopt
 * at its own offset.
 *
 * Default ON. Read once: a body compiled with the arms and one compiled without
 * must not coexist in a run. */
bool jitMatchArm(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_MATCH");
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

#else

#endif
