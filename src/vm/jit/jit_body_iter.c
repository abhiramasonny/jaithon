/* jit_body_iter.c -- the iterator arms of the opcode walk. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitGetIter(Emit *e, int *offp) {
    int off = *offp;
    do {
        if (!e->pendingRange) {
            if (e->depth == 0) return false;
            Value itSeen = e->stackSeen[e->depth - 1];
            /* `for c in text`. The iterator itself is the same call the
             * list path makes -- jaiGetIter takes anything iterable -- so
             * the only new thing a string needs is an element exemplar and
             * a proof that it IS a string.
             *
             * Worth an arm because it is a ONE-LINK chain on a hot body:
             * `JAI_JIT_CHAIN=1` says `_scan_identifier` compiles the moment
             * this is cleared, and character loops are what a lexer is
             * made of. */
            const bool strIter = jitStrIter() &&
                                 e->stack[e->depth - 1] == SLOT_OBJ &&
                                 IS_STRING(itSeen);
            if (e->stack[e->depth - 1] != SLOT_LIST && !strIter) {
                /* Named: 4.3% of parser.jai's interpreted work sat behind
                 * this and it said nothing about WHAT was being iterated,
                 * which is the entire question. */
                return subWhy(e, "iterating a %s, not a list or a range",
                              IS_OBJ(itSeen)
                                  ? jaiTypeNameStatic(itSeen)
                                  : slotKindName(e->stack[e->depth - 1]));
            }
            if (!e->callsOut) return false;
            /* Carry one element forward: the loop variable's kind comes
             * from it, and the iterator itself says nothing about what it
             * will yield. */
            Value srcv = itSeen;
            Value sample = NULL_VAL;
            if (IS_LIST(srcv) && AS_LIST(srcv)->count > 0) {
                sample = jaiListGet(AS_LIST(srcv), 0);
            }
            if (strIter) {
                /* Prove OBJ_STRING before the exemplar claims the elements
                 * are strings. VAL_OBJ is every heap object, and an
                 * exemplar the value does not match is how the next arm
                 * comes to read the wrong header.
                 *
                 * The exemplar is an interned EMPTY string, not the first
                 * character: a string yields one-CHARACTER elements, and a
                 * character is not always one byte. Claiming one byte would
                 * send multi-byte text down the one-byte arm, which guards
                 * the length and would therefore deopt once per iteration.
                 * Empty says "a string" and claims nothing else. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                                   valueXReg(e, e->valueDepth - 1),
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
                branchOnDeopt(e, JAI_A64_NE);
                ObjString *empty = jaiStringIntern("", 0);
                if (empty == NULL) return false;
                sample = OBJ_VAL((Obj *)empty);
            }
            /* A list this body built has no live sample to read an element
             * off -- it does not exist yet -- but OP_BUILD_LIST recorded
             * what went into it. `[expr for x in [a, b, c]]` is the shape
             * that wanted this: the source list is a literal built one
             * instruction earlier, and without it every comprehension over
             * one ran interpreted. */
            if (IS_NULL(sample)) sample = e->stackElem[e->depth - 1];
            if (IS_NULL(sample)) {
                e->whyNot = "iterating a list with nothing to look at";
                return false;
            }
            if (!emitDescriptor(e, NULL_VAL, e->depth - 1, 1,
                                (void *)&jitMakeIter)) {
                return false;
            }
            unsigned rdrop;
            if (!popValue(e, &rdrop, NULL)) return false;
            /* Shape 1 marks an iterator the runtime has to step; a range is 0 and gets the inline path. Rides on
             * the stack entry, not the Emit, since a function can build both -- nbody's `advance` runs two range loops then a list loop, and one whole-compile flag made the path taken depend on what came before it. */
            /* Shape 5 is a STRING, which shape 1 used to cover. It needs one of
             * its own because the sample cannot tell the two apart -- a list OF
             * strings carries a string sample too -- and the step arm reads
             * ObjString's header either way. Under shape 1 that arm guarded
             * `kind == ITER_LIST`, so every `for c in text` bailed at the loop
             * head on every call and ran interpreted. */
            if (!pushValue3(e, SLOT_ITER, strIter ? 5u : 1u, NULL, sample, -1))
                return false;
            emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                               e->descOffset +
                                   (unsigned)offsetof(JitCallDesc, result) + 8));
            e->wroteHeap = true;
            off += 1;
            break;
        }
        if (!e->callsOut) return false;
        if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
        emitConst64(e, pushReg(e) - 1, e->rangeInclusive ? 1 : 0);
        if (!emitDescriptor(e, NULL_VAL, e->depth - 3, 3,
                            (void *)&jitMakeRangeIter)) {
            return false;
        }
        for (unsigned i = 0; i < 3; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        /* Shape 2 says this body built the range itself, so its step is 1
         * by construction; shape 3 adds a start the emitter knows, carried
         * as the entry's sample. Shape 0 stays the general form, for an
         * ObjIter that arrived from anywhere else. */
        if (!pushValue3(e, SLOT_ITER, e->rangeStartKnown ? 3u : 2u, NULL,
                        e->rangeStartKnown ? INT_VAL(e->rangeStartVal)
                                           : NULL_VAL,
                        -1)) {
            return false;
        }
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        e->wroteHeap = true;
        e->pendingRange = false;
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitIterRange(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `for x in a..b` opened, with neither object built. The two ints
         * this writes are the whole loop, so there is no descriptor, no
         * root fill and no call out -- and, being ordinary int locals, they
         * compete for registers on the same terms as everything else
         * instead of reserving four the way an ObjIter head does.
         *
         * `end` is one past the last value, WRAPPING, which is what makes
         * `a..=INT64_MAX` terminate: the counter meets INT64_MIN there.
         * Same arithmetic as the interpreter's, deliberately -- see
         * OP_ITER_RANGE in chunk.h. */
        bool     inclusive = code[off + 1] != 0;
        unsigned curSlot   = jaiReadU16(code + off + 2);
        unsigned endSlot   = jaiReadU16(code + off + 4);
        if (e->depth < 2) return false;
        if (e->stack[e->depth - 1] != SLOT_INT) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;
        if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
            return false;
        }
        if (!adoptLocalKind(e, curSlot, SLOT_INT, 0, NULL)) return false;
        if (!adoptLocalKind(e, endSlot, SLOT_INT, 0, NULL)) return false;
        if (curSlot == 0 || endSlot == 0) e->usesSlot0 = true;

        unsigned rHi, rLo;
        if (!popValue(e, &rHi, NULL)) return false;
        if (!popValue(e, &rLo, NULL)) return false;
        /* Both ends stay read-only: a popped register may be a local's own,
         * borrowed, and writing it would rewrite the local. */
        if (inclusive) {
            emit(e, jaiA64AddXImm(JIT_SCRATCH_A, rHi, 1));
        } else {
            emit(e, jaiA64MovX(JIT_SCRATCH_A, rHi));
        }
        emit(e, jaiA64SubsXReg(31, rHi, rLo));
        /* An empty range ends where it begins, so the first test already
         * fails and the body never runs. */
        emit(e, jaiA64CselX(JIT_SCRATCH_B, rLo, JIT_SCRATCH_A, JAI_A64_LT));
        localOut(e, endSlot, JIT_SCRATCH_B);
        localOut(e, curSlot, rLo);
        off += 6;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitForRangeBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* One step of that loop: a compare, a branch, a bind and an add,
         * with nothing to load from the heap and nothing to guard. The two
         * slots are written by OP_ITER_RANGE and by this instruction and by
         * nothing else -- the emitter hands out fresh temporaries for them
         * -- so their kind is a fact of the shape rather than a sample, and
         * this arm has no deopt of its own. A bail lands on this
         * instruction with the counter unadvanced. */
        int16_t  jump    = jaiReadI16(code + off + 1);
        unsigned slot    = jaiReadU16(code + off + 3);
        unsigned curSlot = jaiReadU16(code + off + 5);
        unsigned endSlot = jaiReadU16(code + off + 7);
        if (!localInRange(e, slot)) return false;
        if (!localInRange(e, curSlot) || !localInRange(e, endSlot)) {
            return false;
        }
        if (e->localKind[curSlot] != SLOT_INT ||
            e->localKind[endSlot] != SLOT_INT) {
            return false;
        }
        if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) {
            return subWhy(e, "loop variable in local %u has kind %s, "
                             "not int", slot,
                          slotKindName(e->localKind[slot]));
        }
        if (slot == 0) e->usesSlot0 = true;

        unsigned rCur = localIn(e, curSlot, JIT_SCRATCH_A);
        unsigned rEnd = localIn(e, endSlot, JIT_SCRATCH_B);
        emit(e, jaiA64SubsXReg(31, rCur, rEnd));
        /* Nothing of this loop's is on the operand stack, so the exit is
         * reached at exactly the depth this branch leaves from. */
        branchTo(e, (uint32_t)((int32_t)(off + 9) + jump), true,
                 JAI_A64_EQ);
        /* Bind before stepping: in register mode the counter's home IS
         * rCur, so the add would destroy the value about to be bound. */
        localOut(e, slot, rCur);
        {
            unsigned dst = localDest(e, curSlot);
            emit(e, jaiA64AddXImm(dst, rCur, 1));
            localOut(e, curSlot, dst);
        }
        off += 9;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitGetIterItems(Emit *e, const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        /* `for (a, b) in X.items()`. The emitter cannot know X's type, so
         * it plants this ahead of an ordinary `INVOKE items; GET_ITER` and
         * lets the opcode jump over the pair when X turns out to be a dict,
         * building a lazy ITER_DICT_ITEMS instead. Unarmed, it declined the
         * whole enclosing function -- which is why dict_iter's two loops
         * ran interpreted end to end.
         *
         * The dict case is specialised and the branch is resolved HERE, by
         * walking on at the target rather than by emitting a jump: the
         * skipped `INVOKE items` never executes in this form, so its inline
         * cache is empty and compiling it as dead code would decline. That
         * is only sound because the region really is the emitter's own, so
         * nothing branches into it -- checked below, and backstopped by the
         * fixup resolver, which declines a branch to an offset the walk
         * never reached rather than mis-resolving it.
         *
         * Specialising rather than falling through to the eager `items()`
         * is required, not merely faster: the lazy view raises when the
         * dict changes under the loop and the materialised list does not,
         * so a compiled body that took the other path would answer
         * differently from the interpreter. */
        if (e->depth == 0) return false;
        unsigned sidx = e->depth - 1;
        if (e->stack[sidx] != SLOT_OBJ || !IS_DICT(e->stackSeen[sidx])) {
            e->whyNot = "items() on something that is not a dict";
            return false;
        }
        if (!e->callsOut) return false;
        /* Read before the entry is popped below, not through the model
         * afterwards: the push that replaces it overwrites this cell. */
        Value    itemsDict = e->stackSeen[sidx];
        int16_t  ijump = jaiReadI16(code + off + 1);
        int32_t  after = (int32_t)(off + 3) + ijump;
        /* The emitter's shape exactly: OP_INVOKE (7 bytes) then
         * OP_GET_ITER (1), and the head that follows must be the pair form,
         * since that is the only one shape 4 has an arm for. */
        if (after != off + 11 || after >= stop ||
            code[off + 3] != OP_INVOKE || code[off + 10] != OP_GET_ITER ||
            code[after] != OP_FOR_ITER_PAIR) {
            e->whyNot = "an items() head this tier does not recognise";
            return false;
        }

        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - 1,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
        branchOnDeopt(e, JAI_A64_NE);

        if (!emitDescriptor(e, NULL_VAL, sidx, 1,
                            (void *)&jitMakeItemsIter)) {
            return false;
        }
        unsigned rdrop;
        if (!popValue(e, &rdrop, NULL)) return false;
        /* Shape 4 is an ITER_DICT_ITEMS, and it carries the DICT as its
         * sample rather than an element: the pair head reads the first live
         * entry off it for the component kinds, exactly as the list form
         * reads items[0]. */
        if (!pushValue3(e, SLOT_ITER, 4, NULL, itemsDict, -1)) {
            return false;
        }
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        e->wroteHeap = true;
        off = after;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitForIterBind(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* A loop this body built the iterator for: the index lives in the
         * iterator, so every iteration loads and stores it, and a deopt
         * needs nothing -- what is on the stack is already current. */
        /* Only the loop at the OSR entry point owns the reserved iterator registers -- a nested FOR_ITER_BIND
         * built its own iterator and is an ordinary one. Refusing it stopped the outer loops of spectral, mandelbrot, matrix_mul and life from compiling at all, while their inner loops compiled fine. */
        if (!e->osr || !e->hasIter || (uint32_t)off != e->osrTop) {
            if (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER) {
                return false;
            }
            int16_t  fjump = jaiReadI16(code + off + 1);
            unsigned fslot = jaiReadU16(code + off + 3);
            if (!localInRange(e, fslot)) return false;
            unsigned rIt = pushReg(e) - 1;

            /* 0 is a range, 1 an iterator the runtime has to step, 2 and 3
             * ranges this body built (see OP_GET_ITER). Anything else is
             * not something the inline range form may assume, so it keeps
             * the stepped path this test has always sent it down. */
            uint32_t iterShape = e->stackShape[e->depth - 1];
            /* Shape 4 is a dict-items view, and the arm below would read
             * ObjList's offsets out of an ObjDict were the kind guard not
             * there to stop it. OP_GET_ITER_ITEMS only makes one when a
             * pair head follows, so this is unreachable -- kept because the
             * fact lives in another arm and a decline is the cheap side. */
            if (iterShape == 4) {
                e->whyNot = "a non-destructuring loop over dict items";
                return false;
            }
            /* `for c in <string>`, stepped inline: the interpreter's
             * ITER_STRING fast path (object_iter.c), instruction for
             * instruction. `index` is a BYTE offset and `limit` the byte
             * length, so the ASCII step is one ldrb and an add -- and it
             * allocates nothing, because the character it yields is the shared
             * one-byte string jaiVMInit filled every slot of.
             *
             * A byte >= 0x80 deoptimises rather than decoding UTF-8 inline: the
             * interpreter's arm allocates a fresh ObjString for a multi-byte
             * scalar, which is not something this can emit, and the resume
             * point is this instruction with nothing yet advanced. Source text
             * is overwhelmingly ASCII, so the bail is rare -- and it is paid
             * per non-ASCII character, not per loop, the same way the list
             * head's null bail is.
             *
             * Worth an arm because BOTH tiers used to give up here: the OSR
             * head refuses ITER_STRING outright ("an iterator kind with no
             * loop-head arm") and the function tier compiled the body but
             * deoptimised at the loop head on every call. A per-character loop
             * measured 290ms against 30ms for the same work written as an
             * indexed `while`. */
            if (iterShape == 5) {
                Value sample = e->stackSeen[e->depth - 1];
                if (!IS_STRING(sample)) {
                    e->whyNot = "a string loop with no string to look at";
                    return false;
                }
                if (!adoptLocalKindSeen(e, fslot, SLOT_OBJ, 0, NULL, sample)) {
                    return subWhy(e, "loop variable in local %u has kind "
                                     "%s, not a string", fslot,
                                  slotKindName(e->localKind[fslot]));
                }

                /* Only OP_GET_ITER's string arm makes a shape-5 SLOT_ITER, and
                 * SLOT_ITER is never adopted into a local. Checked anyway, one
                 * load: reading ObjString's header off an ObjList is the one
                 * failure mode this tier is not allowed. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, kind)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_STRING));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                   (unsigned)offsetof(ObjIter, source) + 8));

                /* `limit` is the byte length sampled when the iterator was
                 * built. A string is immutable, so unlike the list arm there is
                 * no version to check. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, limit)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves from. */
                branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                              JAI_A64_GE,
                              (int)stackSignatureAt(e, e->depth - 1));

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjString, chars)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A, 0));
                emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
                /* 128 is an imm12, so the compare needs no register. */
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 128));
                branchOnDeopt(e, JAI_A64_HS);

                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_A, 1));
                emit(e, jaiA64StrX(JIT_SCRATCH_C, rIt,
                                   (unsigned)offsetof(ObjIter, index)));

                /* All 128 slots are filled from the end of jaiVMInit, so this
                 * is a load and not a load plus a null test. */
                emitConst64(e, JIT_SCRATCH_C,
                            (int64_t)(uintptr_t)jaiAsciiCharTable());
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_B, 3));
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                localOut(e, fslot, JIT_SCRATCH_A);
                /* The index store is a heap write, as the call-out this
                 * replaced was. */
                e->wroteHeap = true;
                off += 5;
                break;
            }

            if (iterShape != 0 && iterShape != 2 && iterShape != 3) {
                /* A list iterator, stepped inline (jaiIterNext's ITER_LIST
                 * case, instruction for instruction) rather than through
                 * jitIterStep. The kind still comes from the element the
                 * list is holding now, but the tag of what the step
                 * actually produces is GUARDED, and every guard runs
                 * BEFORE the index advances.
                 *
                 * Calling out cannot be made sound: jitIterStep advances
                 * the iterator before it returns, so a guard on its result
                 * has nowhere to resume -- this instruction would re-run
                 * and SKIP an element, and under
                 * JAITHON_JIT_DEOPT_STRESS (where branchOnDeopt is
                 * unconditional) it would skip one on every iteration of
                 * every list loop. Reading the payload out of the
                 * descriptor with no tag check at all was worse: a list
                 * sampled as int and later pushed a str bound the string's
                 * POINTER as an integer (probe: 41080394656 where the
                 * interpreter raises TypeError), and a float's IEEE bits
                 * likewise. Inline, nothing has happened when a guard
                 * fires, so the resume point is this instruction and the
                 * interpreter does the raise. */
                Value sample = e->stackSeen[e->depth - 1];
                SlotKind ek; unsigned etag; uint32_t esh = 0;
                ObjClass *ecl = NULL;
                if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
                else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
                else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
                else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
                else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
                else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass) {
                    ek = SLOT_INST; etag = VAL_OBJ;
                    ecl = AS_INSTANCE(sample)->klass;
                    esh = ecl->shapeId;
                } else { e->whyNot = "element kind unknown"; return false; }

                if (!adoptLocalKindSeen(e, fslot, ek, esh, ecl, sample)) {
                    return subWhy(e, "loop variable in local %u has kind "
                                     "%s, not %s", fslot,
                                  slotKindName(e->localKind[fslot]),
                                  slotKindName(ek));
                }

                /* Only OP_GET_ITER's list arm makes a shape-nonzero
                 * SLOT_ITER, and SLOT_ITER is never adopted into a local,
                 * so this is an ITER_LIST over an ObjList. Checked anyway,
                 * one load: the alternative is reading an ObjString or an
                 * ObjDict through ObjList's offsets if another iterator
                 * shape is ever added above, and a wrong answer is the one
                 * failure mode this tier is not allowed. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, kind)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
                branchOnDeopt(e, JAI_A64_NE);

                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                   (unsigned)offsetof(ObjIter, source) + 8));
                /* Nothing names this list -- it is whatever the iterator was
                 * built over -- so the storage is dispatched rather than
                 * pinned. This was `emitListBoxedGuard` alone, and a
                 * push-built `list[int]` is LIST_STORE_I64, so that guard
                 * failed on EVERY entry of every nested loop over one: the
                 * form left compiled code and the inner loop ran interpreted.
                 * Same program, same output, 55ms with a boxed literal and
                 * 200ms with the identical list built by `push`. It is the
                 * mistake listAccessFor's own comment records having made
                 * once already ("ruinous rather than merely slower"). */
                ListAccess nAcc;
                if (jitIterStorage()) {
                    nAcc = listAccessFor(e, JIT_SCRATCH_C, -1, ek,
                                         JIT_SCRATCH_A);
                } else {
                    nAcc.stg = (uint8_t)LIST_STORE_BOXED;
                    nAcc.alt = (uint8_t)LIST_STORE_BOXED;
                    nAcc.dynamic = false;
                    emitListBoxedGuard(e, JIT_SCRATCH_C, JIT_SCRATCH_A);
                }

                /* Mutation first, as jaiIterNext tests it: a list that grew
                 * or shrank under the loop must raise, and the version is
                 * the only thing that says so. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjList, version)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, version)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);

                /* `limit` is the snapshot count, not the live one -- the
                 * version guard above owns any disagreement between them. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, limit)));
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                /* The exhausted arm drops the iterator, so the target is
                 * reached one entry shallower than this branch leaves
                 * from. */
                branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                              JAI_A64_GE,
                              (int)stackSignatureAt(e, e->depth - 1));

                /* Reload items rather than hoisting: a reallocation bumps
                 * the version, which the guard above covers. */
                /* JIT_SCRATCH_D is the dispatch scratch: JIT_SCRATCH_A still
                 * holds the index and must reach the advance below. */
                int nSkip = listDispatchBegin(e, &nAcc, JIT_SCRATCH_C,
                                              JIT_SCRATCH_D);
                unsigned nBase = JIT_SCRATCH_C;
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, nBase,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_SCRATCH_A, listStgShift(nAcc.stg)));

                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, etag));
                branchOnDeopt(e, JAI_A64_NE);

                if (ek == SLOT_INST) {
                    /* VAL_OBJ is every heap object, so the object type is
                     * checked before `klass` is read -- otherwise a list
                     * that gained a string reads `klass` one word past an
                     * ObjString's header. JIT_SCRATCH_A still holds the
                     * index and must survive to the store below. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_INSTANCE));
                    branchOnDeopt(e, JAI_A64_NE);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_D, (int64_t)esh);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_B, JIT_SCRATCH_D));
                    branchOnDeopt(e, JAI_A64_NE);
                } else if (ek == SLOT_LIST) {
                    /* Same contract as OP_GET_INDEX's own SLOT_LIST arm:
                     * VAL_OBJ is every heap object, not specifically a
                     * list, so the object type is confirmed here, once,
                     * before a SLOT_LIST consumer trusts it with no check
                     * of its own. JIT_SCRATCH_A still holds the index and
                     * must survive to the store below. */
                    emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C, 8));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                                       (unsigned)offsetof(Obj, type)));
                    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, OBJ_LIST));
                    branchOnDeopt(e, JAI_A64_NE);
                }

                /* Both arms leave JIT_SCRATCH_C ON the payload, as the OSR
                 * head's own list arm does, so one load below serves either.
                 * The unboxed arm reloads the list out of the iterator
                 * because the boxed arm above overwrote the pointer with
                 * `items`; JIT_START_REG holds it for the head's arm, and
                 * there is no such register here. Nothing to check on an
                 * unboxed element: no tag, and no object behind it. */
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
                if (nSkip >= 0) {
                    int nJoin = listDispatchElse(e, nSkip);
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                       (unsigned)offsetof(ObjIter, source) + 8));
                    emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjList, items)));
                    emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                          JIT_SCRATCH_A,
                                          listStgShift(nAcc.alt)));
                    listDispatchEnd(e, nJoin);
                }

                /* Past the last guard: advance, then bind. The advance goes
                 * first because localOut may use JIT_SCRATCH_C/D for the
                 * tag and the index has to be stored out of a register the
                 * write cannot touch. One byte for a bool: see the note in
                 * OP_GET_INDEX -- `strb` is what BOOL_VAL compiles to, so
                 * the rest of the payload word is stale. */
                emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_A, 1));
                emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                                   (unsigned)offsetof(ObjIter, index)));
                if (ek == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
                }
                localOut(e, fslot, JIT_SCRATCH_A);
                /* The index store is a heap write, as the call-out this
                 * replaced was: a bail after it would re-run the loop from
                 * the top with the iterator already advanced. */
                e->wroteHeap = true;
                off += 5;
                break;
            }

            emit(e, jaiA64LdrX(JIT_SCRATCH_A, rIt,
                               (unsigned)offsetof(ObjIter, index)));
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIt,
                               (unsigned)offsetof(ObjIter, limit)));
            if (!adoptLocalKind(e, fslot, SLOT_INT, 0, NULL)) return false;
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            /* The exhausted arm drops the iterator, so the target is
             * reached one entry shallower than this branch leaves from. */
            branchToDepth(e, (uint32_t)((int32_t)(off + 5) + fjump),
                          JAI_A64_GE,
                          (int)stackSignatureAt(e, e->depth - 1));
            /* A range yields `start + index * step`, not the index itself (see jaiIterNext's ITER_RANGE case) --
             * the index is always zero-based, so using it directly is only right for `0..n` in unit steps. `for j in i + 1..n` once counted from zero instead of i+1, a plausible wrong answer (nested loops summed the wrong pairs), not a crash. The dead-after-compare limit register carries the index across to the increment. */
            emit(e, jaiA64MovX(JIT_SCRATCH_B, JIT_SCRATCH_A));
            /* Both halves of that map are loop-invariant, and for a range
             * this body built (shape 2) the step is 1 by construction --
             * jitMakeRangeIter has no step argument. With the start a
             * literal too (shape 3) nothing about the ObjRange has to be
             * read at all, which is five loads and a multiply off the back
             * of every nested `for k in 0..n`: matrix_mul spends fourteen
             * of its innermost forty-nine instructions on this counter. */
            /* Shape 3's constant travels in the entry's sample, and an
             * entry can reach here having been through a local, where the
             * sample need not have come with it. No sample, no shortcut:
             * the general form below is right for any range. */
            if (iterShape == 3 && !IS_INT(e->stackSeen[e->depth - 1])) {
                iterShape = 2;
            }
            if (iterShape == 3) {
                int64_t k = AS_INT(e->stackSeen[e->depth - 1]);
                if (k > 0 && k <= 4095) {
                    emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                          (unsigned)k));
                } else if (k < 0 && k >= -4095) {
                    emit(e, jaiA64SubXImm(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                          (unsigned)(-k)));
                } else if (k != 0) {
                    emitConst64(e, JIT_SCRATCH_D, k);
                    emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                       JIT_SCRATCH_A));
                }
            } else {
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIt,
                                   (unsigned)offsetof(ObjIter, source) + 8));
                if (iterShape != 2) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                       (unsigned)offsetof(ObjRange, step)));
                    emit(e, jaiA64MulX(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                       JIT_SCRATCH_D));
                }
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjRange, start)));
                emit(e, jaiA64AddX(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   JIT_SCRATCH_A));
            }
            localOut(e, fslot, JIT_SCRATCH_A);
            emit(e, jaiA64AddXImm(JIT_SCRATCH_B, JIT_SCRATCH_B, 1));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, rIt,
                               (unsigned)offsetof(ObjIter, index)));
            off += 5;
            break;
        }
        /* Only as the head of the loop being compiled, and only for a range from zero in unit steps -- that's
         * what makes the yielded value the index itself. Everything else about the iterator is a runtime fact, checked at entry. */
        if (!e->osr || !e->hasIter) return false;
        if ((uint32_t)off != e->osrTop) return false;
        int16_t  jump = jaiReadI16(code + off + 1);
        unsigned slot = jaiReadU16(code + off + 3);
        if (!localInRange(e, slot)) return false;

        if (e->iterKind == 6) {
            /* A STRING at the loop head. The reserved registers mean what they
             * do for a list, with one substitution: JIT_IDX_REG and JIT_LIM_REG
             * are BYTE offsets, not element counts, because that is what
             * ObjIter carries for an ITER_STRING (object_iter.c). JIT_START_REG
             * holds the ObjString.
             *
             * No version guard, unlike the list head: a string is immutable, so
             * there is nothing that can change under the loop.
             *
             * A byte >= 0x80 deoptimises rather than decoding UTF-8 inline, for
             * the reason emitForIterBind's shape-5 arm gives: the interpreter
             * allocates a fresh ObjString for a multi-byte scalar, and nothing
             * has been advanced when the guard fires, so the resume point is
             * this instruction. The DENSITY question a new head owes
             * (jitStringHeadSample, called from compileOsr) is answered before
             * the form is built rather than here.
             *
             * Without this the whole shape was refused -- "an iterator kind
             * with no loop-head arm" -- so a per-character loop long enough to
             * reach the OSR tier ran entirely interpreted: 290ms against 30ms
             * for the same work written as an indexed `while`. */
            Value sample = e->elemSample;
            if (!IS_STRING(sample)) {
                return subWhy(e, "a string head with no string to look at");
            }
            if (!adoptLocalKindSeen(e, slot, SLOT_OBJ, 0, NULL, sample)) {
                return subWhy(e, "loop variable in local %u has kind %s, "
                                 "not a string", slot,
                              slotKindName(e->localKind[slot]));
            }
            e->iterSlot = slot;
            e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

            emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
            branchTo(e, e->iterExit, true, JAI_A64_GE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                               (unsigned)offsetof(ObjString, chars)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_IDX_REG, 0));
            emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C, 0));
            /* 128 is an imm12, so the compare needs no register. */
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 128));
            branchOnDeopt(e, JAI_A64_HS);

            /* All 128 slots are filled from the end of jaiVMInit, so this is a
             * load and not a load plus a null test. What the table holds is
             * interned by construction. */
            emitConst64(e, JIT_SCRATCH_C,
                        (int64_t)(uintptr_t)jaiAsciiCharTable());
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_SCRATCH_B, 3));
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            localOut(e, slot, JIT_SCRATCH_A);
            emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
            off += 5;
            break;
        }

        if (e->iterKind == 2) {
            /* A list at the loop head: reserved registers mean what they do for a range, except JIT_START_REG
             * holds the ObjList instead of a first value. Without this, a top-level `for x in xs` was never even attempted -- the gate refused anything not a range before compileOsr ran, so not even a decline was recorded. */
            Value sample = e->elemSample;
            SlotKind ek;
            unsigned etag;
            uint32_t esh = 0;
            ObjClass *ecl = NULL;
            if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
            else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
            else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
            else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
            else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
            else if (IS_INSTANCE(sample) &&
                     AS_INSTANCE(sample)->klass != NULL) {
                /* The object type is checked before the class is read,
                 * because VAL_OBJ is every heap object and a list holding
                 * a string beside the sampled instance would otherwise
                 * read `klass` one word past an ObjString's header. */
                ek = SLOT_INST; etag = VAL_OBJ;
                ecl = AS_INSTANCE(sample)->klass;
                esh = ecl->shapeId;
            } else {
                e->whyNot = "list element kind unknown";
                return false;
            }
            /* The list holds more than one class, so the loop variable is
             * an instance of no particular one. Two classes are not two
             * KINDS -- the representation is the same untagged pointer --
             * so the slot widens rather than the compile failing, and the
             * call sites inside dispatch by name. Only sound at THIS
             * instruction: an OSR body is walked from its loop head, so
             * nothing has been emitted against the class being dropped. */
            if (ek == SLOT_INST && e->elemMixed) {
                esh = 0;
                ecl = NULL;
                e->localTyped[slot] = false;
            }
            if (!adoptLocalKindSeen(e, slot, ek, esh, ecl, sample)) {
                return subWhy(e, "loop variable in local %u has kind %s, "
                                 "not %s", slot,
                              slotKindName(e->localKind[slot]),
                              slotKindName(ek));
            }
            e->iterSlot = slot;
            e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

            /* Mutation first: a list that grew or shrank under the loop
             * must raise, and the version is the only thing that says so.
             * Nothing has happened yet, so this resumes at this very
             * instruction and the interpreter raises it properly. */
            /* Storage is pinned per form, not checked here: jaiJitEnterOsr
             * matches JaiOsrForm::iterStg against the list this head is
             * about to walk, so by the time the body runs the stride below
             * is already the right one. */
            /* Unpinned is unproved, and the form records LIST_STG_ANY
             * for the head -- but a deopt guard here is not the answer.
             * `e->elemStgPin` is false for any body that calls out, and a
             * `push` is a call, so `for x in xs { out.push(f(x)) }` over a
             * `list[int]` would fail that guard on its FIRST element and
             * on every entry after: 11x slower than boxed, and the head's
             * give-up counter never fires because a bail is not a decline.
             * So the head dispatches like every other site. */
            ListAccess iAcc;
            iAcc.stg = e->elemStgPin ? e->elemStg
                                     : (uint8_t)LIST_STORE_BOXED;
            iAcc.alt = e->elemStgPin ? iAcc.stg : listAltFor(ek);
            iAcc.dynamic = !e->elemStgPin && iAcc.alt != LIST_STORE_BOXED;
            uint8_t iStg = iAcc.stg;
            if (iStg != LIST_STORE_BOXED && ek != listStgKind(iStg)) {
                return subWhy(e, "element kind %d is not storage %u's",
                              (int)ek, iStg);
            }

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_START_REG,
                               (unsigned)offsetof(ObjList, version)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_ITER_REG,
                               (unsigned)offsetof(ObjIter, version)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
            branchTo(e, e->iterExit, true, JAI_A64_GE);

            /* Reload items each time rather than hoisting: a reallocation
             * bumps the version so the guard above covers it, and one ldr
             * removes the question entirely. */
            int iSkip = listDispatchBegin(e, &iAcc, JIT_START_REG,
                                          JIT_SCRATCH_A);
            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                               (unsigned)offsetof(ObjList, items)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_IDX_REG, listStgShift(iStg)));

            /* Nothing to check on an unboxed element: no tag, and no
             * object behind it whose type could surprise the arms below. */
            if (iStg == LIST_STORE_BOXED) {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, etag));
            branchOnDeopt(e, JAI_A64_NE);

            if (ek == SLOT_INST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                /* Same hazard as above: VAL_OBJ covers every heap object, so the type is checked before `klass` is read. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeopt(e, JAI_A64_NE);
                /* The class is checked only when one was pinned. A widened
                 * slot has no class to check against, and that it is an
                 * instance at all -- which the guard above settles -- is
                 * everything a by-name call needs of it. */
                if (esh != 0) {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                       (unsigned)offsetof(ObjInstance, klass)));
                    emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                       (unsigned)offsetof(ObjClass, shapeId)));
                    emitConst64(e, JIT_SCRATCH_A, (int64_t)esh);
                    emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
                    branchOnDeopt(e, JAI_A64_NE);
                }
            } else if (ek == SLOT_LIST) {
                /* Same contract as OP_GET_INDEX's own SLOT_LIST arm: VAL_OBJ
                 * is every heap object, not specifically a list, so the
                 * object type is confirmed here, once, before a SLOT_LIST
                 * consumer trusts it with no check of its own. */
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
            }

            /* Both arms leave JIT_SCRATCH_C on the payload, as
             * OP_GET_INDEX's do, so the load below serves either. */
            if (iStg == LIST_STORE_BOXED) {
                emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
            }
            if (iSkip >= 0) {
                int iJoin = listDispatchElse(e, iSkip);
                emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                                   (unsigned)offsetof(ObjList, items)));
                emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                      JIT_IDX_REG,
                                      listStgShift(iAcc.alt)));
                listDispatchEnd(e, iJoin);
            }

            /* One byte for a bool: see the note in OP_GET_INDEX. `strb` is
             * what BOOL_VAL compiles to, so the rest of the payload word is
             * stale, and a SLOT_BOOL register must hold 0 or 1. */
            unsigned eAt = 0;
            if (ek == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
            } else {
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, eAt));
            }
            localOut(e, slot, JIT_SCRATCH_A);
            emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
            off += 5;
            break;
        }

        if (!adoptLocalKind(e, slot, SLOT_INT, 0, NULL)) return false;
        e->iterSlot = slot;
        e->iterExit = (uint32_t)((int32_t)(off + 5) + jump);

        emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
        branchTo(e, e->iterExit, true, JAI_A64_GE);
        /* Value is start + index, not the index (`for j in i + 1..n` is nbody advance's inner loop): both
         * registers were biased by the start in the prologue, so the value IS the index register and the add that used to be here is gone. */
        localOut(e, slot, JIT_IDX_REG);
        emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
        off += 5;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool jitListHeadSample(const ObjList *src, int at, Value *sample, bool *mixed) {
    *sample = jaiListGet(src, at);
    *mixed = false;
    if (!IS_INSTANCE(*sample)) {
        /* A non-instance sample still needs the NULL census, and skipping it
         * was a real regression rather than a tidiness point: a head compiled
         * for `int` off a `list[int?]` fails its tag compare on every null and
         * bails, and on a 2M list with one null in three that is 3,333,335
         * bails and 1.15x SLOWER than refusing -- worse than never compiling.
         * The class half below cannot apply here, so only the count runs. */
        int scan = src->count < 1024 ? src->count : 1024;
        int nulls = 0;
        for (int i = 0; i < scan; i++) {
            if (IS_NULL(jaiListGet(src, i))) nulls++;
        }
        return nulls * 64 <= scan;
    }
    /* Whether the list holds one class or several. One sample cannot say,
     * and getting it wrong is not merely a slower loop: a form compiled
     * pinned to the sampled class fails its own entry guard on every later
     * class, so the loop runs interpreted for the rest of the program with
     * no second chance to notice. Capped because this runs per compile
     * attempt and a list can be enormous; past the cap the pinned form is
     * compiled as before and deoptimises if it was wrong, which is where
     * this started. The scan does not stop at the first mismatch: the null
     * count needs the whole prefix, and the cap is what bounds the cost. */
    const ObjClass *first = AS_INSTANCE(*sample)->klass;
    int scan = src->count < 1024 ? src->count : 1024;
    int nulls = 0;
    for (int i = 0; i < scan; i++) {
        Value v = jaiListGet(src, i);
        if (IS_NULL(v)) { nulls++; continue; }
        if (!IS_INSTANCE(v)) continue;
        if (AS_INSTANCE(v)->klass != first) *mixed = true;
    }
    return nulls * 64 <= scan;
}

/* `for (i, x) in xs.enumerate()` over the ITER_LIST_ENUM snapshot OP_INVOKE
 * built: the index is the first component, read off the iterator, and the
 * element the second, guarded exactly as OP_FOR_ITER_BIND's list arms guard
 * theirs -- tag, then object type, then class where one is pinned -- with
 * every guard BEFORE anything is written, so a deopt resumes at this very
 * instruction. As the head of the OSR loop (iterKind 5) the list, index and
 * limit ride in the reserved registers a list head uses; nested in a compiled
 * function the index lives in the ObjIter, as it does for every other
 * iterator the body built itself. No version guard either way: the snapshot
 * is the iterator's own and nothing else can reach it. */
static bool emitForIterPairEnum(Emit *e, unsigned pslotA, unsigned pslotB,
                                int16_t pjump, int off, bool head) {
    Value sample = head ? e->elemSample : e->stackSeen[e->depth - 1];
    SlotKind ek;
    unsigned etag;
    uint32_t esh = 0;
    ObjClass *ecl = NULL;
    if (IS_INT(sample))        { ek = SLOT_INT;   etag = VAL_INT; }
    else if (IS_FLOAT(sample)) { ek = SLOT_FLOAT; etag = VAL_FLOAT; }
    else if (IS_BOOL(sample))  { ek = SLOT_BOOL;  etag = VAL_BOOL; }
    else if (IS_LIST(sample))  { ek = SLOT_LIST;  etag = VAL_OBJ; }
    else if (rawObjValue(sample)) { ek = SLOT_OBJ; etag = VAL_OBJ; }
    else if (IS_INSTANCE(sample) && AS_INSTANCE(sample)->klass != NULL) {
        ek = SLOT_INST; etag = VAL_OBJ;
        ecl = AS_INSTANCE(sample)->klass;
        esh = ecl->shapeId;
    } else {
        e->whyNot = "enumerating an element of a kind with no slot";
        return false;
    }
    /* Several classes in the list: the element is an instance of no class
     * in particular. Only the head may widen -- see the list head arm. */
    if (ek == SLOT_INST && head && e->elemMixed) {
        esh = 0;
        ecl = NULL;
        e->localTyped[pslotB] = false;
    }
    if (!adoptLocalKindSeen(e, pslotA, SLOT_INT, 0, NULL, INT_VAL(0)) ||
        !adoptLocalKindSeen(e, pslotB, ek, esh, ecl, sample)) {
        return subWhy(e, "an enumerate loop's variables (locals %u and %u) "
                         "have kinds %s and %s", pslotA, pslotB,
                      slotKindName(e->localKind[pslotA]),
                      slotKindName(e->localKind[pslotB]));
    }
    uint32_t exit = (uint32_t)((int32_t)(off + 7) + pjump);

    if (head) {
        e->iterSlot = pslotB;
        e->iterExit = exit;
        emit(e, jaiA64SubsXReg(31, JIT_IDX_REG, JIT_LIM_REG));
        branchTo(e, exit, true, JAI_A64_GE);
        /* Boxed by construction (jaiIterNewListEnum), and jaiJitEnterOsr
         * refused anything else at entry, so the stride is 16. */
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_START_REG,
                           (unsigned)offsetof(ObjList, items)));
        emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_IDX_REG, 4));
    } else {
        unsigned rIter = pushReg(e) - 1;
        int exitDepth = (int)stackSignatureAt(e, e->depth - 1);
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                           (unsigned)offsetof(ObjIter, kind)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST_ENUM));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                           (unsigned)offsetof(ObjIter, source) + 8));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                           (unsigned)offsetof(ObjIter, index)));
        emit(e, jaiA64LdrX(JIT_SCRATCH_D, rIter,
                           (unsigned)offsetof(ObjIter, limit)));
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
        /* The exhausted arm drops the iterator, so the target is reached
         * one entry shallower than this branch leaves from. */
        branchToDepth(e, exit, JAI_A64_GE, exitDepth);
        emitListBoxedGuard(e, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                           (unsigned)offsetof(ObjList, items)));
        emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_B, JIT_SCRATCH_C, 4));
        /* From here both forms hold the element's address in JIT_SCRATCH_C;
         * the nested one keeps the index in the iterator and re-reads it
         * once the guards are past. */
    }

    emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
    emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, etag));
    branchOnDeopt(e, JAI_A64_NE);
    if (ek == SLOT_INST) {
        /* VAL_OBJ is every heap object, so the type is checked before
         * `klass` is read; the class only when one was pinned. */
        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
        branchOnDeopt(e, JAI_A64_NE);
        if (esh != 0) {
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)esh);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_NE);
        }
    } else if (ek == SLOT_LIST) {
        emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 8));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeopt(e, JAI_A64_NE);
    }

    /* Past the last guard: from here nothing may fail. localOut spends
     * JIT_SCRATCH_C and JIT_SCRATCH_D on a frame-resident slot's tag, so the
     * element is loaded and bound first, off the address in C, and the index
     * comes from a register localOut cannot touch: the reserved one at the
     * head, and a fresh read of the iterator otherwise -- the advance below
     * is the only heap write, and it goes before either bind. A bool is one
     * byte (see OP_GET_INDEX); the rest of its payload word is stale. */
    if (!head) {
        unsigned rIter = pushReg(e) - 1;
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                           (unsigned)offsetof(ObjIter, index)));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_D, JIT_SCRATCH_B, 1));
        emit(e, jaiA64StrX(JIT_SCRATCH_D, rIter,
                           (unsigned)offsetof(ObjIter, index)));
    }
    if (ek == SLOT_BOOL) {
        emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
    } else {
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_C, 8));
    }
    localOut(e, pslotB, JIT_SCRATCH_A);
    if (head) {
        localOut(e, pslotA, JIT_IDX_REG);
        emit(e, jaiA64AddXImm(JIT_IDX_REG, JIT_IDX_REG, 1));
    } else {
        localOut(e, pslotA, JIT_SCRATCH_B);
        e->wroteHeap = true;
    }
    return true;
}

bool emitForIterPair(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `for (a, b) in xs` over a list of 2-tuples, stepped inline.
         *
         * Deliberately not a call-out to a helper that steps the iterator.
         * OP_FOR_ITER_BIND's list arm used to be one, and it is inline for
         * the same reason this is: such a helper advances the iterator
         * before returning, so a component-kind guard after it would deopt
         * to an instruction that has already happened -- and under
         * JAITHON_JIT_DEOPT_STRESS every guard is turned into an
         * unconditional branch, which would then skip an element on every
         * pass. Every guard here is placed BEFORE anything is written, so
         * resuming at this very instruction is exact whether the guard
         * failed for a real reason or because the stress flag forced it.
         *
         * The reachable iterator is always a list one -- shape != 0 comes
         * only from OP_GET_ITER's jitMakeIter, over a SLOT_LIST -- but the
         * kind is guarded anyway rather than assumed, because that fact
         * lives two arms away. */
        /* As the head of an OSR loop the iterator is not on the modelled
         * operand stack at all -- it arrives in a reserved register and
         * stays on the interpreter's stack, which is what lets an exit
         * leave without unwinding anything. Only a PAIR head gets here --
         * iterKind 3 (a dict-items view), 4 (a list of 2-tuples) or 5 (an
         * enumerate snapshot); a range or a plain list head is an
         * OP_FOR_ITER_BIND. */
        bool pairHead = e->osr && e->hasIter &&
                        (e->iterKind == 3 || e->iterKind == 4) &&
                        (uint32_t)off == e->osrTop;
        /* iterKind 5 is the enumerate head, and it is kept apart from 3 and 4
         * rather than added to them: those two read the index out of the
         * ObjIter every iteration, while this one is a list head whose list,
         * index and limit ride in the reserved registers the prologue filled.
         * Everything below the arm belongs to the other two. */
        bool enumHead = e->osr && e->hasIter && e->iterKind == 5 &&
                        (uint32_t)off == e->osrTop;
        if (!pairHead && !enumHead &&
            (e->depth == 0 || e->stack[e->depth - 1] != SLOT_ITER)) {
            return false;
        }
        if (!pairHead && !enumHead && e->stackShape[e->depth - 1] == 0) {
            /* A range head yields ints, which never destructure. */
            e->whyNot = "destructuring what a range yields";
            return false;
        }
        int16_t  pjump = jaiReadI16(code + off + 1);
        unsigned pslotA = jaiReadU16(code + off + 3);
        unsigned pslotB = jaiReadU16(code + off + 5);
        if (!localInRange(e, pslotA) || !localInRange(e, pslotB)) {
            return false;
        }
        if (pslotA == pslotB) {
            /* `for (x, x) in …`: legal, and the second write wins. Not
             * worth a special case; the interpreter keeps it. */
            e->whyNot = "a pair loop binding one slot twice";
            return false;
        }
        /* Away from a head, SLOT_ITER shape 5 is the enumerate snapshot
         * OP_INVOKE's arm pushed. That numbering is the function tier's own
         * and has nothing to do with iterKind: shape 4 there is the dict view,
         * whose head is iterKind 3. */
        if (enumHead || (!pairHead && e->stackShape[e->depth - 1] == 5)) {
            if (!emitForIterPairEnum(e, pslotA, pslotB, pjump, off,
                                     enumHead)) {
                return false;
            }
            off += 7;
            break;
        }

        /* Component kinds come from the pair the source was holding when
         * the iterator was built (OP_GET_ITER / OP_GET_ITER_ITEMS carries
         * it forward), and the guards below are what make that a
         * specialisation rather than an assumption. */
        /* At a head the container is settled by the iterKind the driver chose
         * and there is no operand-stack entry to ask; away from a head it is
         * settled by the shape OP_GET_ITER left. Kept as two questions rather
         * than one OR, because a head's iterKind 4 is a LIST and would
         * otherwise be read as the shape-4 dict it happens to share a digit
         * with. */
        bool pairIsDict = pairHead ? e->iterKind == 3
                                   : e->stackShape[e->depth - 1] == 4;
        Value psample = pairHead ? e->elemSample : e->stackSeen[e->depth - 1];
        SlotKind pk[2];
        unsigned ptag[2];
        Value pseen[2];
        if (pairIsDict) {
            /* Shape 4 carries the dict itself, so the sample is its first
             * live entry -- the one the loop is about to yield. */
            if (!IS_DICT(psample) ||
                !firstLiveEntry(&AS_DICT(psample)->table,
                                &pseen[0], &pseen[1])) {
                e->whyNot = "iterating a dict with nothing to look at";
                return false;
            }
        } else {
            if (!IS_TUPLE(psample) || AS_TUPLE(psample)->count != 2) {
                e->whyNot = "pair element is not a 2-tuple";
                return false;
            }
            pseen[0] = AS_TUPLE(psample)->items[0];
            pseen[1] = AS_TUPLE(psample)->items[1];
        }
        for (unsigned i = 0; i < 2; i++) {
            Value v = pseen[i];
            if (IS_INT(v))        { pk[i] = SLOT_INT;   ptag[i] = VAL_INT; }
            else if (IS_FLOAT(v)) { pk[i] = SLOT_FLOAT; ptag[i] = VAL_FLOAT; }
            else if (IS_BOOL(v))  { pk[i] = SLOT_BOOL;  ptag[i] = VAL_BOOL; }
            else if (IS_OBJ(v) && AS_OBJ(v) != NULL) {
                /* Held raw, like any other SLOT_OBJ: the tag guard is the
                 * whole of what this promises, and an arm that wants to
                 * know WHICH object type checks that itself. */
                pk[i] = SLOT_OBJ; ptag[i] = VAL_OBJ;
            } else {
                e->whyNot = "pair component kind unknown";
                return false;
            }
        }
        if (!adoptLocalKindSeen(e, pslotA, pk[0], 0, NULL, pseen[0]) ||
            !adoptLocalKindSeen(e, pslotB, pk[1], 0, NULL, pseen[1])) {
            return subWhy(e, "a pair's loop variables (locals %u and %u) "
                             "have kinds %s and %s", pslotA, pslotB,
                          slotKindName(e->localKind[pslotA]),
                          slotKindName(e->localKind[pslotB]));
        }

        unsigned rIter = pairHead ? JIT_PAIR_ITER_REG : pushReg(e) - 1;
        uint32_t pairExit = (uint32_t)((int32_t)(off + 7) + pjump);
        /* A head's exit leaves the model at the depth it is already at --
         * the iterator it drops was never in the model. Registering it as
         * iterExit is what makes the exit stub tell the interpreter to pop
         * the exhausted iterator off its own stack. */
        int pairExitDepth = pairHead
                                ? (int)stackSignature(e)
                                : (int)stackSignatureAt(e, e->depth - 1);
        if (pairHead) e->iterExit = pairExit;

        if (pairIsDict) {
            /* iterStepPairFast's ITER_DICT_ITEMS case plus the jaiTableNext
             * it calls, inline. Same discipline as the list arm below:
             * every guard, and the whole scan, runs before the index is
             * written back, so a deopt -- forced or real -- resumes at this
             * instruction with the iterator exactly as the interpreter left
             * it and re-does the scan. */
            _Static_assert(sizeof(JaiEntry) == 48,
                           "the dict-items step scales the order index by "
                           "hand: slot * 16 * 3");
            const unsigned tOff = (unsigned)offsetof(ObjDict, table);

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                               (unsigned)offsetof(ObjIter, kind)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_DICT_ITEMS));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                               (unsigned)offsetof(ObjIter, source) + 8));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);

            /* A dict that changed under the loop must raise, and only the
             * version says so. jaiIterNext owns that message, so the guard
             * hands the whole instruction back unadvanced. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               tOff + (unsigned)offsetof(JaiTable, version)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, version)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_B,
                               tOff +
                                   (unsigned)offsetof(JaiTable, orderCount)));

            /* The scan. `order` holds an entry index per insertion
             * position, negative where a delete left a hole, so a dict with
             * deletions in it costs one extra pass per hole and nothing
             * else. orderCount is hoisted because only a mutation can move
             * it and the version guard above has already excluded one.
             *
             * branchToDepth inside a loop is sound only because it settles
             * nothing here: the branchOnDeopt three lines up fails the
             * compile outright if a deferred value is live, so the settle
             * it performs is a no-op and cannot be re-executed. */
            unsigned scanTop = e->count;
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
            /* The exhausted arm drops the iterator, so the target is
             * reached one entry shallower than this branch leaves from. */
            branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               tOff + (unsigned)offsetof(JaiTable, order)));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                  JIT_SCRATCH_C, 2));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A, 0));
            emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
            /* A hole: the slot is int32 and negative, which after the
             * zero-extending load is bit 31 set. Measured against e->count
             * so an instruction added above cannot rot the distance. */
            emit(e, jaiA64Tbnz(JIT_SCRATCH_A, 31,
                               (int32_t)scanTop - (int32_t)e->count));

            /* entries + slot * sizeof(JaiEntry): slot << 4, then + itself
             * twice over, which is the 48 the assert above pins. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_B,
                               tOff + (unsigned)offsetof(JaiTable, entries)));
            emit(e, jaiA64LslX(JIT_SCRATCH_A, JIT_SCRATCH_A, 4));
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                  JIT_SCRATCH_A, 1));
            emit(e, jaiA64AddX(JIT_SCRATCH_B, JIT_SCRATCH_D, JIT_SCRATCH_A));

            /* Key and value carry the kinds sampled off the first live
             * entry; a dict that later holds another kind fails here with
             * nothing written. */
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                     : (unsigned)offsetof(JaiEntry, value);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
                branchOnDeopt(e, JAI_A64_NE);
            }

            /* Past the last guard. The index goes first because localOut
             * spends JIT_SCRATCH_C and JIT_SCRATCH_D on a frame-resident
             * slot's tag; only JIT_SCRATCH_B survives it. */
            emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                               (unsigned)offsetof(ObjIter, index)));
            for (unsigned i = 0; i < 2; i++) {
                unsigned at = (i == 0 ? (unsigned)offsetof(JaiEntry, key)
                                      : (unsigned)offsetof(JaiEntry, value))
                              + 8u;
                /* A bool is one byte (see OP_GET_INDEX) -- the rest of its
                 * payload word is stale, and a SLOT_BOOL register must hold
                 * 0 or 1. */
                if (pk[i] == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                } else {
                    emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
                }
                localOut(e, i == 0 ? pslotA : pslotB, JIT_SCRATCH_A);
            }
            e->wroteHeap = true;
            off += 7;
            break;
        }

        /* Really a list iterator, and its source really a list. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rIter,
                           (unsigned)offsetof(ObjIter, kind)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ITER_LIST));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, rIter,
                           (unsigned)offsetof(ObjIter, source) + 8));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
        branchOnDeopt(e, JAI_A64_NE);

        /* A list that grew or shrank under the loop must raise, and the
         * version is the only thing that says so. Nothing has happened
         * yet, so the interpreter raises it from this instruction. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(ObjList, version)));
        emit(e, jaiA64LdrW(JIT_SCRATCH_C, rIter,
                           (unsigned)offsetof(ObjIter, version)));
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
        branchOnDeopt(e, JAI_A64_NE);

        emit(e, jaiA64LdrX(JIT_SCRATCH_C, rIter,
                           (unsigned)offsetof(ObjIter, index)));
        emit(e, jaiA64LdrX(JIT_SCRATCH_D, rIter,
                           (unsigned)offsetof(ObjIter, limit)));
        emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_C, JIT_SCRATCH_D));
        /* The exhausted arm drops the iterator, so the target is reached
         * one entry shallower than this branch leaves from. */
        branchToDepth(e, pairExit, JAI_A64_GE, pairExitDepth);

        /* items is reloaded rather than hoisted: a reallocation bumps the
         * version, which the guard above covers, and one ldr removes the
         * question. */
        emitListBoxedGuard(e, JIT_SCRATCH_B, JIT_SCRATCH_A);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                           (unsigned)offsetof(ObjList, items)));
        emit(e, jaiA64AddXLsl(JIT_SCRATCH_B, JIT_SCRATCH_B,
                              JIT_SCRATCH_C, 4));

        /* The element is a 2-tuple whose components have the sampled
         * kinds. Object type is checked before `count` is read: VAL_OBJ
         * covers every heap object, so a string beside the sampled tuple
         * would otherwise have `count` read out of an ObjString. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, 0));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_B, 8));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_TUPLE));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(ObjTuple, count)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 2));
        branchOnDeopt(e, JAI_A64_NE);
        for (unsigned i = 0; i < 2; i++) {
            unsigned at = (unsigned)offsetof(ObjTuple, items) +
                          i * (unsigned)sizeof(Value);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ptag[i]));
            branchOnDeopt(e, JAI_A64_NE);
        }

        /* Past the last guard: from here nothing may fail, and every write
         * below is what the interpreter would have left behind. The index
         * goes first because localOut spends JIT_SCRATCH_C and
         * JIT_SCRATCH_D on the tag of a slot that lives in the frame; only
         * the element pointer in JIT_SCRATCH_B survives it. */
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 1));
        emit(e, jaiA64StrX(JIT_SCRATCH_C, rIter,
                           (unsigned)offsetof(ObjIter, index)));
        /* A bool is one byte (see OP_GET_INDEX) -- the rest of its payload
         * word is stale, and a SLOT_BOOL register must hold 0 or 1. */
        for (unsigned i = 0; i < 2; i++) {
            unsigned at = (unsigned)offsetof(ObjTuple, items) +
                          i * (unsigned)sizeof(Value) + 8u;
            if (pk[i] == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
            } else {
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, at));
            }
            localOut(e, i == 0 ? pslotA : pslotB, JIT_SCRATCH_A);
        }
        e->wroteHeap = true;
        off += 7;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitBuildRange(Emit *e, const uint8_t *code, int *offp, int count) {
    int off = *offp;
    do {
        /* Deferred: the range is only worth building alongside its
         * iterator, which the next instruction asks for. */
        if (e->depth < 2) return false;
        if (e->stack[e->depth - 1] != SLOT_INT) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;
        /* OP_BUILD_RANGE carries one operand byte, so the next opcode is
         * two along. */
        if (off + 2 >= count || code[off + 2] != OP_GET_ITER) {
            e->whyNot = "a range that is not immediately iterated";
            return false;
        }
        e->rangeInclusive = code[off + 1] != 0;
        e->pendingRange = true;
        e->rangeBuildIp = (uint32_t)off;
        /* Both ends hold registers, so the low end is the entry one below
         * the top in the value bank as well as on the stack. */
        {
            unsigned lo = e->valueDepth - 2;
            e->rangeStartKnown = (e->kKnown & (1u << lo)) != 0;
            e->rangeStartVal   = e->rangeStartKnown ? e->kKnownVal[lo] : 0;
        }
        off += 2;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
