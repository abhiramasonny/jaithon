/* jit_body_index.c -- the subscript and slice arms of the opcode walk. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/vm.h"

#include <stddef.h>
#include <stdint.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitGetIndex(Emit *e, const uint8_t *code, int *offp, int stop) {
    int off = *offp;
    do {
        /* `s[i]` on a string: every guard is a load+compare, and the result is a table lookup, not an
         * allocation (the 128 one-byte strings are made once and shared). Without this the whole loop around a character scan declines -- why `str_search` ran interpreted end to end, and every lexer scans one byte at a time. */
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
            e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_STRING(e->stackSeen[e->depth - 2])) {
            unsigned rIdx = pushReg(e) - 1;
            unsigned rStr = valueXReg(e, e->valueDepth - 2);

            /* Really a string, and not something else this object slot
             * happened to hold when the loop was compiled. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_STRING));
            branchOnDeopt(e, JAI_A64_NE);

            /* ASCII only: one scalar is one byte, so indexing is indexing.
             * `scalars` is UINT32_MAX until something asks, so the first
             * time through deopts and the interpreter fills it in. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rStr,
                               (unsigned)offsetof(ObjString, length)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_B, rStr,
                               (unsigned)offsetof(ObjString, scalars)));
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);

            /* jaiNormalizeIndex, then one unsigned compare for both ends.
             * `length` came from an `ldr w`, so it is already the whole
             * register. */
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                false);

            emit(e, jaiA64LdrX(JIT_SCRATCH_C, rStr,
                               (unsigned)offsetof(ObjString, chars)));
            emit(e, jaiA64AddX(JIT_SCRATCH_C, JIT_SCRATCH_C, JIT_SCRATCH_B));
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
            /* 128 is an imm12, so the compare needs no register: a
             * materialised constant on a body this hot is not free the way
             * a register copy is. */
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, 128));
            branchOnDeopt(e, JAI_A64_HS);

            /* The shared one-byte string. jaiVMInit fills all 128 slots, so
             * this is a load and not a load plus a null test -- see
             * jaiAsciiCharsFill. The scaled add folds the shift in. */
            emitConst64(e, JIT_SCRATCH_C,
                        (int64_t)(uintptr_t)jaiAsciiCharTable());
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, JIT_SCRATCH_C,
                                  JIT_SCRATCH_A, 3));
            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_C, 0));

            /* Carry a sample so later instructions know this is a
             * string: the receiver serves, since only its type is read.
             * Without one the interned-equality path below cannot tell
             * what it is holding and declines. */
            Value strSample = e->stackSeen[e->depth - 2];
            unsigned d1, d2;
            if (!popValue(e, &d1, NULL)) return false;
            if (!popValue(e, &d2, NULL)) return false;
            if (!pushValue3(e, SLOT_OBJ, 0, NULL, strSample, -1)) {
                return false;
            }
            /* What the table holds is interned by construction -- see
             * jaiStringChar -- so a consumer that would guard this for
             * being a string, and for being interned, need not. */
            e->stackAscii[e->depth - 1] = true;
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_C));
            off += 1;
            break;
        }
        /* `buf[i]` on a `bytes`: a length-guarded byte load, and the
         * result is a plain int, so nothing is allocated. Every binary
         * format in the language is read one byte at a time through this
         * -- the JPEG bit reader is a `bytes` index and nothing else --
         * and without it the whole function around one declined. */
        if (e->depth >= 2 && e->stack[e->depth - 1] == SLOT_INT &&
            e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_BYTES(e->stackSeen[e->depth - 2])) {
            unsigned rIdx = pushReg(e) - 1;
            unsigned rBuf = valueXReg(e, e->valueDepth - 2);

            /* Really a bytes, and not something else this object slot
             * happened to hold when the body was compiled. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_BYTES));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rBuf,
                               (unsigned)offsetof(ObjBytes, length)));
            emitBoundsNormalise(e, rIdx, JIT_SCRATCH_A, JIT_SCRATCH_B,
                                false);

            /* The payload is inline after the header, so the base needs no
             * load of its own -- unlike a string, which holds a pointer. */
            emit(e, jaiA64AddX(JIT_SCRATCH_C, rBuf, JIT_SCRATCH_B));
            emit(e, jaiA64LdrByte(JIT_SCRATCH_A, JIT_SCRATCH_C,
                                  (unsigned)offsetof(ObjBytes, data)));

            unsigned dByte1, dByte2;
            if (!popValue(e, &dByte1, NULL)) return false;
            if (!popValue(e, &dByte2, NULL)) return false;
            if (!pushValue(e, SLOT_INT, 0, NULL)) return false;
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_A));
            off += 1;
            break;
        }
        /* Index normalised as jaiNormalizeIndex does it, one unsigned compare covering both ends. Out of
         * range, or an element not the kind seen at compile time, goes back to the interpreter -- reading an element has no effect, so resuming at this instruction is always sound. */
        if (e->depth < 2) return subWhy(e, "the model is only %u deep", e->depth);
        if (e->stack[e->depth - 2] == SLOT_OBJ &&
            IS_DICT(e->stackSeen[e->depth - 2])) {
            /* `d[k]`, the read half of the OP_SET_INDEX dict arm below.
             * Without it a loop that reads a dict ran interpreted end to
             * end: `t += d["a"]` two million times was 16,280,472
             * interpreted instructions and 1,684 once this landed.
             *
             * Predicted off a live sample and guarded, as the list arm is,
             * except that the sample must be UNIFORM across the dict --
             * see dictUniformValue for why a dict is not a list here. */
            unsigned dsidx = e->depth - 2;
            Value dsample;
            if (!dictUniformValue(AS_DICT(e->stackSeen[dsidx]), &dsample)) {
                return subWhy(e, "the live dict is empty or holds more than "
                                 "one kind of value");
            }
            SlotKind dkind;
            unsigned dtag;
            ObjClass *dcls;
            uint32_t dshape;
            if (!exemplarKind(dsample, &dkind, &dtag, &dcls, &dshape)) {
                return subWhy(e, "a dict value of a kind the tier cannot hold");
            }
            /* SLOT_OBJ pins nothing, so the container is proved to be a
             * dict before anything is consumed: a miss resumes with the
             * dict and the key both still on the interpreter's stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A,
                               valueXReg(e, e->valueDepth - 2),
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);

            if (!emitDescriptor(e, NULL_VAL, dsidx, 2,
                                (void *)&jitGetIndexDict)) {
                return false;
            }
            for (unsigned i = 0; i < 2; i++) {
                unsigned drop;
                if (!popValue(e, &drop, NULL)) return false;
            }
            /* The sample travels with the entry, as the list arm's does:
             * without it `names["first"].len()` is an invoke on an object
             * the model cannot name, and the body declines one instruction
             * after the read it just learned to make. */
            if (!pushValue3(e, dkind, dshape, dcls, dsample, -1)) {
                return false;
            }

            unsigned drat = e->descOffset +
                            (unsigned)offsetof(JitCallDesc, result);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, 31, drat));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
            /* Resumes AFTER the read. The lookup itself is pure, but it may
             * have raised and been caught, and re-running it would be a
             * second probe of a table the handler could have changed. */
            branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
            unsigned drd = pushReg(e) - 1;
            if (dkind == SLOT_BOOL) {
                emit(e, jaiA64LdrByte(drd, 31, drat + 8));
            } else {
                emit(e, jaiA64LdrX(drd, 31, drat + 8));
            }
            if (dkind == SLOT_INST) {
                /* Two shapes in one dict cannot be told apart by the tag,
                 * and the walk above only sampled a prefix.
                 *
                 * The object type comes first, for the reason the shared
                 * return path gives: VAL_OBJ covers every heap object, and
                 * reading `klass` off a string lands in its length/hash and
                 * dereferences it. A dict holding a Box under one key and a
                 * str under another SEGFAULTED the VM from ordinary code --
                 * `d[k]` in any body hot enough to compile.
                 *
                 * The tag test above cannot stand in for this: it is the
                 * same test the sampled prefix already passed. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, drd,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), true);
                emit(e, jaiA64LdrX(JIT_SCRATCH_A, drd,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_A,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dshape));
                branchOnDeoptAt(e, JAI_A64_NE, (uint32_t)(off + 1), false);
            }
            e->wroteHeap = true;
            off += 1;
            break;
        }
        if (e->stack[e->depth - 2] != SLOT_LIST) {
            return subWhy(e, "the container has kind %s, not list",
                          slotKindName(e->stack[e->depth - 2]));
        }
        if (e->stack[e->depth - 1] != SLOT_INT) {
            return subWhy(e, "the subscript is kind %d, not an int",
                          (int)e->stack[e->depth - 1]);
        }
        unsigned rIdx = pushReg(e) - 1;
        unsigned rList = valueXReg(e, e->valueDepth - 2);
        bool gHoisted = false;

        Value seenList = e->stackSeen[e->depth - 2];
        SlotKind kind = SLOT_OPAQUE;
        unsigned tag = VAL_OBJ;
        ObjClass *elemClass = NULL;
        uint32_t  elemShape = 0;
        /* NULL_VAL on the declared route: there is no exemplar to carry,
         * which is the whole reason that route exists. */
        Value elem = NULL_VAL;
        /* No sample, but the list was DECLARED. See Emit::stackElemDecl:
         * for a body-local list filled through an alias there is nothing
         * to sample and never will be, so the declaration is the only
         * fact available -- and it is a fact, not a guess, because the
         * same byte pins ObjList::stg while the list is still empty.
         *
         * Safe even if it were wrong: listAccessFor rejects a kind the
         * pinned storage contradicts at compile time, and a dispatched
         * access still tag-checks the boxed arm at run time, so a bad
         * declaration deoptimises rather than misreading memory. */
        if (!IS_LIST(seenList) && elemDeclOn()) {
            SlotKind dk = SLOT_OPAQUE;
            switch ((unsigned)e->stackElemDecl[e->depth - 2]) {
            case FIELD_KIND_INT   + 1u: dk = SLOT_INT;   tag = VAL_INT;   break;
            case FIELD_KIND_FLOAT + 1u: dk = SLOT_FLOAT; tag = VAL_FLOAT; break;
            case FIELD_KIND_BOOL  + 1u: dk = SLOT_BOOL;  tag = VAL_BOOL;  break;
            default: break;
            }
            if (dk != SLOT_OPAQUE) {
                kind = dk;
                goto haveElemKind;
            }
        }
        if (!IS_LIST(seenList)) {
            return subWhy(e, "no live list to read an element kind off");
        }
        {
        ObjList *sl = AS_LIST(seenList);
        if (sl->count <= 0) {
            return subWhy(e, "the live list is empty, so there is no exemplar");
        }
        elem = jaiListGet(sl, 0);
        if (IS_INT(elem))        { kind = SLOT_INT;   tag = VAL_INT; }
        else if (IS_FLOAT(elem)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
        else if (IS_BOOL(elem))  { kind = SLOT_BOOL;  tag = VAL_BOOL; }
        else if (IS_LIST(elem)) {
            /* A list of lists. `matrix_mul` is `b[k][j]` in its innermost
             * loop and could not compile the outer half of it. */
            kind = SLOT_LIST;
            tag = VAL_OBJ;
        }
        else if (rawObjValue(elem)) {
            /* A list of strings, dicts, sets, or tuples, held raw: the same contract as a SLOT_OBJ
             * global or field (sample specialises, the tag guard below confirms, every consumer
             * re-checks Obj.type for itself). `str_search` builds text out of `chunks[seed %% 8]` and
             * declined that whole loop forty times over before the string case alone was admitted;
             * widened from IS_STRING to rawObjValue so every other raw-holdable element kind gets the
             * same treatment rather than only strings. */
            kind = SLOT_OBJ;
            tag = VAL_OBJ;
        }
        else if (IS_INSTANCE(elem) && AS_INSTANCE(elem)->klass != NULL) {
            /* A list of instances, all of one shape -- which the per-read
             * tag check cannot confirm on its own, so the class is checked
             * too. A list holding two shapes deoptimises on the second. */
            kind = SLOT_INST;
            tag = VAL_OBJ;
            elemClass = AS_INSTANCE(elem)->klass;
            elemShape = elemClass->shapeId;
        } else return false;
        }
    haveElemKind:

        /* One `ldp` for both header fields: `items` at +16, `count`/`capacity` the adjacent int32s at +24, so
         * the pair's second half is `count | capacity << 32` and the bounds test reads it with uxtw -- one instruction per element read (life does nine per cell). */
        noteSlotIndexed(e, e->stackLocal[e->depth - 2]);
        {
            int32_t gOff = 0;
            uint8_t gBase = 0;
            bool gShaped = boundsCoveredAtHead(e, e->stackLocal[e->depth - 2],
                                               e->valueDepth - 1, &gOff,
                                               &gBase);
            noteIndexSpan(e, e->stackLocal[e->depth - 2], gShaped, gOff,
                          gBase);
            gHoisted = gShaped;
        }
        ListAccess gAcc = listAccessFor(e, rList,
                                        e->stackLocal[e->depth - 2],
                                        kind, JIT_SCRATCH_D);
        /* The sampled element and a PINNED storage cannot disagree -- an
         * I64 store holds ints and nothing else -- but the kind is what
         * the loads below are emitted for, so it is checked rather than
         * assumed. A dispatched access picks its second arm from the kind,
         * so it cannot disagree by construction. */
        if (!gAcc.dynamic && gAcc.stg != LIST_STORE_BOXED &&
            kind != listStgKind(gAcc.stg)) {
            return subWhy(e, "element kind %d is not storage %u's",
                          (int)kind, gAcc.stg);
        }
        unsigned gItems = JIT_SCRATCH_C, gCount = JIT_SCRATCH_A;
        int gh = hoistFor(e, e->stackLocal[e->depth - 2]);
        if (gh >= 0) {
            gItems = e->hoist[gh].itemsReg;
            gCount = e->hoist[gh].countReg;
        } else {
            emitListHeader(e, rList, gItems, gCount);
        }
        if (gHoisted) {
            /* The head proved it. Only the normalisation copy is left, and
             * a shaped index is non-negative by that same proof, so even
             * that is just a move. */
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
        } else {
            emitBoundsNormalise(e, rIdx, gCount, JIT_SCRATCH_B, true);
        }

        /* Both arms below leave JIT_SCRATCH_C on the PAYLOAD rather than
         * on the element, which is what lets one load serve them: a boxed
         * element's payload is eight bytes into it, an unboxed element IS
         * its payload. */
        int gSkip = listDispatchBegin(e, &gAcc, rList, JIT_SCRATCH_D);

        emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems,
                              JIT_SCRATCH_B, listStgShift(gAcc.stg)));
        /* An unboxed element has no tag to check, and no object behind it
         * to confirm the type of: the storage already said what it is. */
        if (gAcc.stg == LIST_STORE_BOXED) {
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_C, 0));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
        branchOnDeopt(e, JAI_A64_NE);
        emit(e, jaiA64AddXImm(JIT_SCRATCH_C, JIT_SCRATCH_C, 8));
        if (kind == SLOT_INST) {
            /* The tag says "an object", not "an object of this class" --
             * and not even "an instance" yet: VAL_OBJ is every heap
             * object, so a list holding an instance beside a string must
             * have its object type confirmed before `klass` is read,
             * exactly as OP_FOR_ITER_BIND's SLOT_INST arms already do.
             * Without this, a list whose sampled element is an instance
             * but a later element is (say) a string reads that string's
             * header bytes as an ObjInstance's `klass` pointer and
             * segfaults dereferencing it -- not merely a wrong answer. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjInstance, klass)));
            emit(e, jaiA64LdrW(JIT_SCRATCH_D, JIT_SCRATCH_D,
                               (unsigned)offsetof(ObjClass, shapeId)));
            emitConst64(e, JIT_SCRATCH_A, (int64_t)elemShape);
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_D, JIT_SCRATCH_A));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (kind == SLOT_LIST) {
            /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
             * its own, so the object type is confirmed here, once, before the kind is handed out --
             * same contract, same check, as OP_GET_FIELD_LOCAL's SLOT_LIST arm. Without this a
             * heterogeneous list (`[[1, 2], "not a list"]`) passes the generic VAL_OBJ tag check on
             * either element and reads the second one's bytes through ObjList's field offsets. */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, JIT_SCRATCH_C, 0));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
        }
        }
        if (gSkip >= 0) {
            int gJoin = listDispatchElse(e, gSkip);
            emit(e, jaiA64AddXLsl(JIT_SCRATCH_C, gItems, JIT_SCRATCH_B,
                                  listStgShift(gAcc.alt)));
            listDispatchEnd(e, gJoin);
        }

        unsigned d1, d2;
        if (!popValue(e, &d1, NULL)) return false;
        if (!popValue(e, &d2, NULL)) return false;
        if (!pushValue3(e, kind, elemShape, elemClass, elem, -1)) return false;
        /* Bool payload is one byte (`BOOL_VAL` compiles to `strb`), so the other seven bytes are stale --
         * an 8-byte load would hand a SLOT_BOOL register (required to hold exactly 0 or 1, since every consumer does `cbnz` on the whole word) garbage. */
        if (kind == SLOT_BOOL) {
            emit(e, jaiA64LdrByte(pushReg(e) - 1, JIT_SCRATCH_C, 0));
        } else if (kind == SLOT_FLOAT &&
                   fpWorthLoading(e, code, off + 1, stop)) {
            /* Straight into the FP bank, for the same reason a float local
             * goes there: `ldr x` followed by `fmov d, x` puts a
             * cross-register-file move between the load and the multiply
             * that wants it, and `ai[k] * b[k][j]` had two of them. */
            unsigned idx = e->valueDepth - 1;
            emit(e, jaiA64LdrD(fpRegAt(e, idx), JIT_SCRATCH_C, 0));
            fpClaim(e, idx);
        } else {
            emit(e, jaiA64LdrX(pushReg(e) - 1, JIT_SCRATCH_C, 0));
        }
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSetIndex(Emit *e, int *offp) {
    int off = *offp;
    do {
        /* Write half of OP_GET_INDEX, normalised the same way; every guard runs before the store, so a deopt
         * here still resumes at an instruction that hasn't happened yet. Sixteen refusals across the benchmarks came from its absence -- `queens` couldn't compile the function that does the work. */
        if (e->depth < 3) return false;
        if (e->stack[e->depth - 3] == SLOT_OBJ) {
            /* `d[k] = v`: a dict is as ordinary a container here as a list -- without this, dict_ops' loop just
             * moved its decline from `get` to this store (a loop that declines anywhere runs interpreted end to end). Object type guarded before anything is consumed, so a miss resumes with container/key/value all still on the interpreter's stack. */
            unsigned sidx = e->depth - 3;
            if (!IS_DICT(e->stackSeen[sidx])) {
                e->whyNot = "an index store into an object that is not a dict";
                return false;
            }
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, valueXReg(e, e->valueDepth - 3),
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_DICT));
            branchOnDeopt(e, JAI_A64_NE);
            if (!emitDescriptor(e, NULL_VAL, sidx, 3,
                                (void *)&jitSetIndexDict)) {
                return false;
            }
            for (unsigned i = 0; i < 3; i++) {
                unsigned r;
                if (!popValue(e, &r, NULL)) return false;
            }
            e->wroteHeap = true;
            off += 1;
            break;
        }
        if (e->stack[e->depth - 3] != SLOT_LIST) return false;
        if (e->stack[e->depth - 2] != SLOT_INT) return false;
        SlotKind vk = e->stack[e->depth - 1];
        unsigned vtag = vk == SLOT_INT   ? VAL_INT
                      : vk == SLOT_FLOAT ? VAL_FLOAT
                      : vk == SLOT_BOOL  ? VAL_BOOL
                      : (vk == SLOT_INST || vk == SLOT_LIST ||
                         vk == SLOT_OBJ)  ? VAL_OBJ
                                          : 0xffffffffu;
        if (vtag == 0xffffffffu) return false;
        unsigned rVal = pushReg(e) - 1;
        unsigned rIdx = valueXReg(e, e->valueDepth - 2);
        unsigned rList = valueXReg(e, e->valueDepth - 3);

        noteSlotIndexed(e, e->stackLocal[e->depth - 3]);
        bool sHoisted;
        {
            int32_t sOff = 0;
            uint8_t sBase = 0;
            sHoisted = boundsCoveredAtHead(e, e->stackLocal[e->depth - 3],
                                           e->valueDepth - 2, &sOff,
                                           &sBase);
            noteIndexSpan(e, e->stackLocal[e->depth - 3], sHoisted, sOff,
                          sBase);
        }
        ListAccess sAcc = listAccessFor(e, rList, e->stackLocal[e->depth - 3],
                                        vk, JIT_SCRATCH_D);
        /* Exactly what jaiListStoreAccepts allows, and for its reason: an
         * int written into a `list[float]` de-specialises the list in the
         * interpreter, which is not something this can do inline. The
         * dispatched form cannot hit it -- its second arm is the storage
         * that holds a `vk` and no other. */
        if (!sAcc.dynamic && sAcc.stg != LIST_STORE_BOXED &&
            vk != listStgKind(sAcc.stg)) {
            return subWhy(e, "storing kind %d into storage %u",
                          (int)vk, sAcc.stg);
        }
        unsigned sItems = JIT_SCRATCH_C, sCount = JIT_SCRATCH_A;
        int sh = hoistFor(e, e->stackLocal[e->depth - 3]);
        if (sh >= 0) {
            sItems = e->hoist[sh].itemsReg;
            sCount = e->hoist[sh].countReg;
        } else {
            emitListHeader(e, rList, sItems, sCount);
        }
        if (sHoisted) {
            emit(e, jaiA64MovX(JIT_SCRATCH_B, rIdx));
        } else {
            emitBoundsNormalise(e, rIdx, sCount, JIT_SCRATCH_B, true);
        }

        int sSkip = listDispatchBegin(e, &sAcc, rList, JIT_SCRATCH_A);
        emitElemStoreAt(e, sAcc.stg, sItems, JIT_SCRATCH_B, vtag, rVal);
        if (sSkip >= 0) {
            int sJoin = listDispatchElse(e, sSkip);
            emitElemStoreAt(e, sAcc.alt, sItems, JIT_SCRATCH_B, vtag, rVal);
            listDispatchEnd(e, sJoin);
        }
        /* jaiListTouch: the count has not changed, so only the version
         * tells an iterator that the list moved under it. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, rList,
                           (unsigned)offsetof(ObjList, version)));
        emit(e, jaiA64AddXImm(JIT_SCRATCH_A, JIT_SCRATCH_A, 1));
        emit(e, jaiA64StrW(JIT_SCRATCH_A, rList,
                           (unsigned)offsetof(ObjList, version)));
        e->wroteHeap = true;

        unsigned d1, d2, d3;
        if (!popValue(e, &d1, NULL)) return false;
        if (!popValue(e, &d2, NULL)) return false;
        if (!popValue(e, &d3, NULL)) return false;
        off += 1;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitGetSlice(Emit *e, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        /* `xs[a:b]` out to the runtime: the clamp arithmetic has three
         * throwing exits and lives in one place, and the work itself is an
         * O(n) copy against which the descriptor's stores are noise. */
        unsigned flags = code[off + 1];
        unsigned nops  = ((flags & 1u) != 0) + ((flags & 2u) != 0) +
                         ((flags & 4u) != 0);
        unsigned nargs = 1u + nops;
        if (e->depth < nargs) return false;
        unsigned cidx = e->depth - nargs;
        Value cseen = e->stackSeen[cidx];
        /* A string slices as readily as a list -- same runtime call, same
         * "the guard pins the type so the result kind follows" argument --
         * and `s[a:b]` is what every hand-written scanner cuts tokens with.
         * Held as SLOT_OBJ, since that is what a string is here. */
        unsigned cType;
        SlotKind sliceKind;
        if (e->stack[cidx] == SLOT_LIST) {
            cType = OBJ_LIST; sliceKind = SLOT_LIST;
        } else if (e->stack[cidx] == SLOT_OBJ && IS_STRING(cseen)) {
            cType = OBJ_STRING; sliceKind = SLOT_OBJ;
        } else if (e->stack[cidx] == SLOT_OBJ && IS_TUPLE(cseen)) {
            /* `jitGetSlice` is a thin wrapper over `jaiSliceGet`, which
             * already handles a tuple container exactly like a list or a
             * string -- only this arm's own type guard was narrower than
             * what the call it makes actually supports. */
            cType = OBJ_TUPLE; sliceKind = SLOT_OBJ;
        } else {
            e->whyNot = "slicing a container this tier does not model";
            return false;
        }

        /* Guard the container, not the result: with its object type pinned
         * the arm jaiSliceGet takes is settled, so the result's kind
         * follows. Before the descriptor and before any pop, so a miss
         * resumes here with everything still on the interpreter's stack. */
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, pushReg(e) - nargs,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, cType));
        branchOnDeopt(e, JAI_A64_NE);

        emit(e, jaiA64MovzX(JIT_SCRATCH_A, flags, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, aux)));
        if (!emitDescriptorStatus(e, NULL_VAL, cidx, nargs,
                                  (void *)&jitGetSlice, false, -1)) {
            return false;
        }
        for (unsigned i = 0; i < nargs; i++) {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
        }
        /* The container's own sample types the slice: a slice of a list of
         * ints is a list of ints, a slice of a string is a string, and
         * every element read re-checks its own tag, so this is a hint and
         * not an assumption. */
        if (!pushValue3(e, sliceKind, 0, NULL, cseen, -1)) return false;
        emit(e, jaiA64LdrX(pushReg(e) - 1, 31,
                           e->descOffset +
                               (unsigned)offsetof(JitCallDesc, result) + 8));
        /* Deliberately not e->wroteHeap: the only effect is a fresh object
         * and an interpreted re-run would make another. Setting it would
         * decline the next self-call, which is the shape `sort` has. */
        off += 2;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
