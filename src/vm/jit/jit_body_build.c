/* jit_body_build.c -- the arms that build an object: the list, dict, set and
 * tuple literals, the f-string, and the element stamp that follows a literal. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitBuildList(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        unsigned n = jaiReadU16(code + off + 1);
        if (n > JIT_MAX_ARGS_OUT) return false;
        if (!e->callsOut) return false;
        if (e->depth < n) return false;
        Value elemSeen = NULL_VAL;
        if (!buildListExemplar(e, e->depth - n, n, &elemSeen)) {
            elemSeen = NULL_VAL;
        }
        if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                            (void *)&jitBuildList)) {
            return false;
        }
        for (unsigned i = 0; i < n; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (!pushValue(e, SLOT_LIST, 0, NULL)) return false;
        /* Computed BEFORE the pops above, since it reads the entries they
         * remove. See buildListExemplar for why the list itself cannot
         * answer this. */
        e->stackElem[e->depth - 1] = elemSeen;
        /* The same exemplar as a KIND, which survives a bind into a local
         * where the Value does not -- there is nowhere to root a Value per
         * local, and a byte needs no rooting.
         *
         * A prediction, not a fact, unlike the OP_ELEM_KIND route: an
         * undeclared literal gets boxed storage, so a later append may put
         * anything in it. That is safe on the same terms as stackElem
         * itself -- a boxed element is tag-checked at every read, so a
         * changed kind deoptimises. `min_area_rect` builds
         * `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` with no declared type and then
         * subscripts it, which is the case that needed this. */
        e->stackElemDecl[e->depth - 1] =
            IS_INT(elemSeen)   ? (uint8_t)(FIELD_KIND_INT   + 1)
          : IS_FLOAT(elemSeen) ? (uint8_t)(FIELD_KIND_FLOAT + 1)
          : IS_BOOL(elemSeen)  ? (uint8_t)(FIELD_KIND_BOOL  + 1)
                               : (uint8_t)0;
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        e->wroteHeap = true;
        off += 3;
        break;
    } while (0);
    *offp = off;
    return true;
}

JitArmResult emitBuildDictSet(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* The remaining container literals, on the OP_BUILD_LIST template.
         *
         * Both were top of the partial-walk census over the self-hosted
         * parser: `_node(kind, span, fields: dict = {})` builds a dict for
         * every AST node, and the walk stopped there sixteen times in one
         * file. */
        bool isDict = code[off] == OP_BUILD_DICT;
        unsigned n = jaiReadU16(code + off + 1);
        unsigned operands = isDict ? n * 2u : n;
        if (!jitTuple() || operands > JIT_MAX_ARGS_OUT || !e->callsOut ||
            e->depth < operands) {
            goto unarmedOpcode;
        }
        if (!emitDescriptor(e, NULL_VAL, e->depth - operands, operands,
                            isDict ? (void *)&jitBuildDict
                                   : (void *)&jitBuildSet)) {
            return false;
        }
        for (unsigned i = 0; i < operands; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
        /* SLOT_OBJ does not say which container this is, and OP_ELEM_KIND
         * comes straight after a literal and has to know. */
        e->stackObjType[e->depth - 1] =
            (uint8_t)((isDict ? OBJ_DICT : OBJ_SET) + 1);
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        e->wroteHeap = true;
        off += 3;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitBuildTuple(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* Same shape as OP_BUILD_LIST above, and simpler: jaiTupleNew
         * copies the operands itself and cannot throw. No exemplar is
         * kept -- a tuple has no element arm to feed, so the entry is a
         * plain SLOT_OBJ.
         *
         * Worth an arm only because there was none: a tuple build ENDED
         * THE WALK, and `let p = (x, y)` in a loop body is common enough
         * that the whole body after it ran interpreted. */
        unsigned n = jaiReadU16(code + off + 1);
        if (!jitTuple() || n > JIT_MAX_ARGS_OUT || !e->callsOut ||
            e->depth < n) {
            goto unarmedOpcode;
        }
        if (!emitDescriptor(e, NULL_VAL, e->depth - n, n,
                            (void *)&jitBuildTuple)) {
            return false;
        }
        for (unsigned i = 0; i < n; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
        e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_TUPLE + 1);
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        e->wroteHeap = true;
        off += 3;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

JitArmResult emitElemKind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        uint8_t packed = code[off + 1];
        if (e->depth == 0) return false;
        /* The STATIC kind, not the sampled value: the measuring pass runs
         * with no sample, so keying on stackSeen declined every time and
         * cost the whole function. OP_BUILD_LIST pushes SLOT_LIST, which is
         * exactly what the emitter puts this opcode after. */
        if (e->stack[e->depth - 1] != SLOT_LIST) {
            /* The interpreter stamps a dict's two nibbles as well, and
             * does NOTHING for any other container -- "an unstamped
             * container is simply unguarded". Both of those are arms.
             *
             * They became reachable the day the dict and set literals got
             * arms of their own: before that the walk stopped AT the
             * literal, so this opcode was never reached with a non-list on
             * top. `var d: dict[str, int] = {}` in a hot body then
             * declined the WHOLE function, which is strictly worse than
             * the partial walk it replaced. */
            uint8_t built = e->stackObjType[e->depth - 1];
            if (built == (uint8_t)(OBJ_SET + 1) ||
                built == (uint8_t)(OBJ_TUPLE + 1)) {
                off += 2;
                break;
            }
            if (built != (uint8_t)(OBJ_DICT + 1)) {
                goto unarmedOpcode;
            }
            unsigned dr = valueXReg(e, e->valueDepth - 1);
            /* The prediction came from the build instruction just below,
             * but a guard costs two instructions and does not depend on
             * the emitter keeping them adjacent. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, dr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeoptInstStart(e, JAI_A64_NE);
            emitConst64(e, JIT_SCRATCH_A, (int64_t)((packed >> 4) & 0xFu));
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                  (unsigned)offsetof(ObjDict, keyKind)));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, dr,
                                  (unsigned)offsetof(ObjDict, valKind)));
            e->wroteHeap = true;
            off += 2;
            break;
        }
        unsigned r = valueXReg(e, e->valueDepth - 1);
        /* The arm was already computing this byte and throwing it away.
         * Keeping it is what lets a subscript of this list choose a load
         * when no sample of it can exist. */
        e->stackElemDecl[e->depth - 1] = (uint8_t)((packed & 0xFu) + 1u);
        emitConst64(e, JIT_SCRATCH_A, (int64_t)(packed & 0xFu));
        emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                              (unsigned)offsetof(ObjList, elemKind)));
        /* And the storage, on the same terms jaiListSpecialise takes: an
         * empty list with nothing reserved, which is what a `[]` literal
         * is. Six instructions rather than a call, and no allocation --
         * that is the whole reason the interpreter's half refuses a
         * non-empty list too. Without this the two tiers build the same
         * literal at different widths and a pinned loop form is denied
         * entry for half the lists it meets; see jaiListSpecialise. */
        uint8_t kStg = listAltFor(
            (packed & 0xFu) == FIELD_KIND_INT   ? SLOT_INT
          : (packed & 0xFu) == FIELD_KIND_FLOAT ? SLOT_FLOAT
          : (packed & 0xFu) == FIELD_KIND_BOOL  ? SLOT_BOOL
                                                : SLOT_OPAQUE);
        if (kStg != LIST_STORE_BOXED && jaiListUnboxOn()) {
            noteStorageStamp(e);
            /* JIT_SCRATCH_A only: this arm has always used one scratch,
             * and e->scratchRoom is what says how many the body actually
             * reserved -- reaching for a second clobbered a live value
             * register and miscompiled the self-hosted emitter. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, r,
                               (unsigned)offsetof(ObjList, count)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            int kA = (int)e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, r,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 0));
            int kB = (int)e->count;
            emit(e, jaiA64BCond(JAI_A64_NE, 0));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)kStg);
            emit(e, jaiA64StrByte(JIT_SCRATCH_A, r,
                                  (unsigned)offsetof(ObjList, stg)));
            e->code[kA] = jaiA64BCond(JAI_A64_NE,
                                      (int32_t)((int)e->count - kA));
            e->code[kB] = jaiA64BCond(JAI_A64_NE,
                                      (int32_t)((int)e->count - kB));
        }
        e->wroteHeap = true;
        off += 2;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

/* JAITHON_JIT_FMT_LEAF=0 sends every f-string back through its descriptor
 * call, for a one-binary A/B of emitFormatLeaf. */
static bool jitFmtLeafOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FMT_LEAF");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_FMT_INT_LEAF=0 sends the one-int-hole shape to the general
 * f-string leaf too, for a one-binary A/B of jaiValueFormatIntLeaf. */
static bool jitFormatIntLeaf(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FMT_INT_LEAF");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* JAITHON_JIT_FMT_MEMO=0 emits no probe of jaiFmtMemo in front of the
 * one-int-hole leaf, and calls the leaf that fills nothing, for a one-binary
 * A/B of the memo. Nothing else fills it. */
static bool jitFmtMemoOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_FMT_MEMO");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

/* The memo probe (see jaiFmtMemo in value.c) for an int in `rN` between
 * runs in `rPre` and `rPost` (31, the zero register, for an absent one): the
 * entry for this (pre, n, post), and when all three match, its string is the
 * answer with no call at all --
 *
 *     x10, x17 <- entries, mask
 *     x9  <- (n ^ pre >> 4 ^ post >> 5) & mask    jaiFmtMemoIndex
 *     x10, x11, x12, x0 <- the entry's pre, post, n, s
 *     cmp pre; ccmp post; ccmp n;  b.eq <answered>
 *
 * The index of that b.eq, for the caller to aim at where the leaf's answer
 * is taken from x0, or -1 with nothing emitted. Every register it writes is
 * one the leaf call behind it clobbers anyway, and the operands must be in
 * callee-saved registers (leafRegOk), so the miss path finds them as they
 * were. An empty entry's pre is JAI_FMT_MEMO_EMPTY, which no operand equals.
 * The caller calls jaiValueFormatIntLeafMemo on a miss, which is what fills
 * the table. */
int emitFmtMemoProbe(Emit *e, unsigned rPre, unsigned rN, unsigned rPost) {
    if (!jitFmtMemoOn()) return -1;
    noteScratchClobber(e);
    emitConst64(e, 16, (int64_t)(uintptr_t)&jaiFmtMemo);
    emit(e, jaiA64LdpOff(10, 17, 16, 0));
    /* An absent run contributes nothing to the index. */
    unsigned rIdx = rN;
    if (rPre != 31u) {
        emit(e, jaiA64EorXLsr(9, rIdx, rPre, 4));
        rIdx = 9;
    }
    if (rPost != 31u) {
        emit(e, jaiA64EorXLsr(9, rIdx, rPost, 5));
        rIdx = 9;
    }
    emit(e, jaiA64AndX(9, rIdx, 17));
    emit(e, jaiA64AddXLsl(16, 10, 9, 5));
    emit(e, jaiA64LdpOff(10, 11, 16, 0));
    emit(e, jaiA64LdpOff(12, 0, 16, 16));
    emit(e, jaiA64SubsXReg(31, 10, rPre));
    emit(e, jaiA64CcmpX(11, rPost, 0, JAI_A64_EQ));
    emit(e, jaiA64CcmpX(12, rN, 0, JAI_A64_EQ));
    int at = (int)e->count;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    return at;
}

/* After a call to jaiValueFormatIntLeafMemo has returned a non-NULL x0: an
 * answer marked as an intern hit (bit 0) goes out of line to be filed --
 *
 *         tbnz x0, #0, fill
 *   ...   <the caller's answered path, then its branch over the slow path>
 *   fill: sub x0, x0, #1;  jaiFmtMemoFill(pre, n, post, x0);  b answered
 *
 * emitFmtMemoFillTest places the test and returns its index (-1 for none);
 * emitFmtMemoFillStub, called once the caller has emitted its branch over
 * the slow path, places the stub and aims the test at it, rejoining at
 * `answered`. pre, n and post are still in their callee-saved registers. */
int emitFmtMemoFillTest(Emit *e, int memoHit) {
    if (memoHit < 0) return -1;
    int at = (int)e->count;
    emit(e, jaiA64Tbnz(0, 0, 0));
    return at;
}

void emitFmtMemoFillStub(Emit *e, int test, int answered, unsigned rPre,
                         unsigned rN, unsigned rPost) {
    if (test < 0 || e->count > JIT_MAX_INSTS) return;
    e->code[test] = jaiA64Tbnz(0, 0, (int32_t)((int)e->count - test));
    emit(e, jaiA64SubXImm(3, 0, 1));
    emit(e, jaiA64MovX(0, rPre));
    emit(e, jaiA64MovX(1, rN));
    emit(e, jaiA64MovX(2, rPost));
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&jaiFmtMemoFill);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64B((int32_t)(answered - (int)e->count)));
}

/* Aims a probe's hit branch at the current instruction. */
void fmtMemoHitHere(Emit *e, int at) {
    if (at < 0 || at >= (int)e->count || e->count > JIT_MAX_INSTS) return;
    e->code[at] = jaiA64BCond(JAI_A64_EQ, (int32_t)((int)e->count - at));
}

/* The f-string through jaiValueFormatLeaf, in front of the descriptor call to
 * jitFormat, which stays behind it as the slow path -- the same layout as the
 * dict leaves (see emitDictLeafGet):
 *
 *        args <- the parts;  blr jaiValueFormatLeaf
 *        NULL ---------------------------------------\
 *        result <- x0;  b done                         |
 *   slow: <the descriptor call, unchanged>  <---------/
 *   done: <load the result out of the descriptor>
 *
 * The leaf never collects, so the parts are written as plain arguments and
 * nothing is rooted: no root fill, no root-range push and pop, no wrapper.
 * Emitted only when every part is something formatShortInto renders or may
 * be (an int, a bool, an object that may be a string), since a part it cannot
 * render would take the slow path every time and pay for both. */
static void emitFormatLeaf(Emit *e, unsigned parts, LeafFix *fx) {
    fx->on = false;
    fx->slow[0] = fx->slow[1] = fx->done = -1;
    if (!jitFmtLeafOn() || !jaiValueFormatShortOn() || e->inlining) return;
    if (parts > JIT_MAX_ARGS_OUT || e->valueDepth < parts) return;
    unsigned first = e->depth - parts;
    size_t seen = 0;
    for (unsigned i = 0; i < parts; i++) {
        SlotKind k = e->stack[first + i];
        if (k != SLOT_INT && k != SLOT_BOOL && k != SLOT_OBJ &&
            k != SLOT_MAYBE_OBJ) {
            return;
        }
        /* A literal part's sample is the pool's own string (OP_CONST pushes
         * it as one), so its length is a fact; a local's is the value it held
         * when the body was compiled, a prediction. An int is at least one
         * digit. */
        Value v = e->stackSeen[first + i];
        if (IS_STRING(v)) {
            seen += AS_STRING(v)->length;
        } else if (k == SLOT_INT) {
            seen += 1;
        }
    }
    /* A result already past the short limit on what the compiler can see:
     * the leaf would copy up to the limit, give up and leave the work to the
     * descriptor path, every time -- the log line, the message, the path
     * built with an f-string. A site whose samples were long and whose later
     * values are short loses only the leaf, never an answer. */
    if (seen > JAI_INTERN_MAX) return;
    unsigned vfirst = e->valueDepth - parts;
    for (unsigned i = 0; i < parts; i++) {
        if (!leafRegOk(valueXReg(e, vfirst + i))) return;
    }
    unsigned argsAt = e->descOffset + (unsigned)offsetof(JitCallDesc, args);
    unsigned rat = e->descOffset + (unsigned)offsetof(JitCallDesc, result);
    if (argsAt > 4095u) return;
    fpSyncAll(e);
    settleAll(e);

    /* One int hole between at most one object on either side -- `f"k{i}"`
     * and its kin -- goes to jaiValueFormatIntLeaf in three registers; the
     * leaf checks that the objects are strings. Everything else writes its
     * parts out for the general leaf. */
    int hole = -1;
    int memoHit = -1;
    unsigned fmtPre = 31u, fmtN = 31u, fmtPost = 31u;
    bool oneIntHole = parts <= 3 && jitFormatIntLeaf();
    for (unsigned i = 0; i < parts && oneIntHole; i++) {
        SlotKind k = e->stack[first + i];
        if (k == SLOT_INT) {
            if (hole >= 0) oneIntHole = false;
            hole = (int)i;
        } else if (k != SLOT_OBJ) {
            oneIntHole = false;
        }
    }
    if (oneIntHole && hole >= 0 && (parts - (unsigned)hole) <= 2u &&
        hole <= 1) {
        unsigned rPre = hole == 1 ? valueXReg(e, vfirst) : 31u;
        unsigned rN = valueXReg(e, vfirst + (unsigned)hole);
        unsigned rPost = (unsigned)hole + 1 < parts
                             ? valueXReg(e, vfirst + (unsigned)hole + 1)
                             : 31u;
        memoHit = emitFmtMemoProbe(e, rPre, rN, rPost);
        fmtPre = rPre;
        fmtN = rN;
        fmtPost = rPost;
        /* `mov x, xzr` is the NULL for an absent run; register 31 here is
         * the zero register, which jaiA64MovX encodes as `orr x, xzr, xzr`. */
        emit(e, jaiA64MovX(0, rPre));
        emit(e, jaiA64MovX(1, rN));
        emit(e, jaiA64MovX(2, rPost));
        emitConst64(e, JIT_SCRATCH_A,
                    memoHit >= 0
                        ? (int64_t)(uintptr_t)&jaiValueFormatIntLeafMemo
                        : (int64_t)(uintptr_t)&jaiValueFormatIntLeaf);
    } else {
        for (unsigned i = 0; i < parts; i++) {
            unsigned reg = valueXReg(e, vfirst + i);
            unsigned at = argsAt + i * (unsigned)sizeof(Value);
            emitTagFor(e, e->stack[first + i], reg, JIT_SCRATCH_B,
                       JIT_SCRATCH_A);
            emit(e, jaiA64StrW(JIT_SCRATCH_B, 31, at));
            emit(e, jaiA64StrX(reg, 31, at + 8));
        }
        emit(e, jaiA64AddXImm(0, 31, argsAt));
        emit(e, jaiA64MovzX(1, parts, 0));
        emitConst64(e, JIT_SCRATCH_A,
                    (int64_t)(uintptr_t)&jaiValueFormatLeaf);
    }
    noteScratchClobber(e);
    emit(e, jaiA64Blr(JIT_SCRATCH_A));
    emit(e, jaiA64SubsXImm(31, 0, 0));
    fx->slow[0] = (int)e->count;
    fx->cond[0] = JAI_A64_EQ;
    emit(e, jaiA64BCond(JAI_A64_EQ, 0));
    /* An intern hit to file goes out of line; a memo hit joins after the
     * test, with its string already in x0. */
    int fillTest = emitFmtMemoFillTest(e, memoHit);
    int answered = (int)e->count;
    fmtMemoHitHere(e, memoHit);
    /* Straight into the register the result will occupy -- the first part's,
     * once the parts are popped and the string pushed -- and past the load
     * the descriptor path ends with (emitFormat places the join after it).
     * Through the descriptor it was two stores and a load on the way from
     * the leaf to whatever consumes the string, a store-forwarding round
     * trip on the path that runs on into the dict probe. */
    if (jitLeafInReg()) {
        emit(e, jaiA64MovX(valueXReg(e, vfirst), 0));
    } else {
        emit(e, jaiA64MovzX(JIT_SCRATCH_A, VAL_OBJ, 0));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, 31, rat));
        emit(e, jaiA64StrX(0, 31, rat + 8));
    }
    fx->done = (int)e->count;
    emit(e, jaiA64B(0));
    emitFmtMemoFillStub(e, fillTest, answered, fmtPre, fmtN, fmtPost);
    fx->on = true;
}

bool emitFormat(Emit *e, ObjClosure *closure, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* Largest single refusal reason across the benchmark census (ninety) -- every f-string is one, and
         * dict_ops, word_freq and string_build all build their keys with one. */
        unsigned parts = code[off + 1];
        if (parts == 0 || parts > JIT_MAX_ARGS_OUT) {
            e->whyNot = "an f-string with more parts than the descriptor holds";
            return false;
        }
        if (e->depth < parts) return false;
        /* `str` bound in the module means every part goes through it
         * instead, which is a call this does not make. */
        {
            ObjModule *fmod = closure->fn->module;
            Value bound;
            ObjString *sname = jaiStringIntern("str", 3);
            if (fmod == NULL || sname == NULL ||
                jaiTableGetInterned(&fmod->globals, sname, &bound)) {
                e->whyNot = "the module binds its own str";
                return false;
            }
        }
        LeafFix ffx;
        emitFormatLeaf(e, parts, &ffx);
        leafSlowHere(e, &ffx);
        if (!emitDescriptor(e, NULL_VAL, e->depth - parts, parts,
                            (void *)&jitFormat)) {
            return false;
        }
        if (!jitLeafInReg()) leafDoneHere(e, &ffx);
        for (unsigned i = 0; i < parts; i++) {
            unsigned drop;
            if (!popValue(e, &drop, NULL)) return false;
        }
        if (!pushValue(e, SLOT_OBJ, 0, NULL)) return false;
        /* jitFormat always builds a string, and there is no Value to carry
         * as a sample, so the expectation is recorded instead: without it
         * `f"{a}-{b}".len()` declined the loop around it at the very next
         * instruction ("an invoke on an object with nothing to look at"). */
        e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_STRING + 1);
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* After the load: the leaf put its answer in this register itself. */
        if (jitLeafInReg()) leafDoneHere(e, &ffx);
        e->wroteHeap = true;
        /* count u8, litmask u24, name u24, cache u16 -- nine after the
         * opcode. */
        off += 10;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
