/* jit_body_build.c -- the arms that build an object: the list, dict, set and
 * tuple literals, the f-string, and the element stamp that follows a literal. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "runtime/runtime.h"
#include "vm/vm.h"

#include <stddef.h>
#include <stdint.h>
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
        if (!emitDescriptor(e, NULL_VAL, e->depth - parts, parts,
                            (void *)&jitFormat)) {
            return false;
        }
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
