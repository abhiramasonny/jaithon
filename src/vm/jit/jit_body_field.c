/* jit_body_field.c -- the field read/write and type-guard arms of the opcode walk. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

bool emitTypeGuard(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp)
{
    int off = *offp;
    do {
        /* A declared boundary (parameter or return type). The kind is already known here, so the guard is
         * either nothing at all or the int-to-float widening the interpreter does at the same place (spec 2.2). Anything the kinds can't settle is declined, not guessed: `evalA` in spectral was refused outright for want of this. */
        uint32_t idx = jaiReadU24(code + off + 1);
        if (idx >= (uint32_t)fn->chunk.constants.count) return false;
        Value t = fn->chunk.constants.data[idx];
        if (!IS_STRING(t)) return false;
        if (e->depth == 0) return false;
        const char *tn = AS_STRING(t)->chars;
        SlotKind k = e->stack[e->depth - 1];
        if (strcmp(tn, "float") == 0) {
            if (k == SLOT_INT) {
                unsigned r = pushReg(e) - 1;
                emit(e, jaiA64ScvtfDX(JIT_FSCRATCH_A, r));
                emit(e, jaiA64FmovXD(r, JIT_FSCRATCH_A));
                e->stack[e->depth - 1]      = SLOT_FLOAT;
                e->stackShape[e->depth - 1] = 0;
                e->stackClass[e->depth - 1] = NULL;
                e->stackSeen[e->depth - 1]  = NULL_VAL;
                e->stackLocal[e->depth - 1] = -1;
                e->stackAscii[e->depth - 1] = false;
                e->stackNullLit[e->depth - 1] = false;
                e->stackUnit[e->depth - 1]  = false;
                e->stackObjType[e->depth - 1] = 0;
                e->stackElem[e->depth - 1] = NULL_VAL;
            } else if (k != SLOT_FLOAT) {
                e->whyNot = "a type guard the kinds cannot settle";
                return false;
            }
        } else if ((strcmp(tn, "int") == 0 && k == SLOT_INT) ||
                   (strcmp(tn, "bool") == 0 && k == SLOT_BOOL)) {
        } else if (jitAnyGuard() && strcmp(tn, "list") == 0 &&
                   k == SLOT_OBJ) {
            /* A declared `list` boundary on a value the model has only as
             * "some heap object". Unlike the cases below this one is not a
             * no-op: it NARROWS. Proving OBJ_LIST here turns the entry into
             * a SLOT_LIST, and every list arm downstream -- indexing,
             * iteration, `len` -- asks for exactly that.
             *
             * The guard is what makes the narrowing honest. VAL_OBJ covers
             * every heap object, so without the Obj.type check the next arm
             * would read ObjList's header off whatever this is, which is
             * the hole that segfaulted the VM through the dict-index arm.
             *
             * A deopt here resumes AT this instruction with the operand
             * still on the interpreter's stack, so a miss costs nothing
             * beyond the exit.
             *
             * WORTH ZERO ON ITS OWN, and recorded as such: it takes the
             * "a `list` guard on a object" refusal in the jaicv benchmark
             * from four to none, and `min_enclosing_circle`'s attributed
             * interpreted work does not move -- the body goes from
             * declining HERE to declining one instruction later, at
             * `OP_GET_INDEX: the container has kind int, not list`, after
             * which JAI_JIT_CHAIN says it compiles. Kept because it is six
             * guarded lines that narrow rather than guess, and because the
             * link it exposes is the one being worked next. */
            unsigned gr = valueXReg(e, e->valueDepth - 1);
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, gr,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
            branchOnDeopt(e, JAI_A64_NE);
            e->stack[e->depth - 1] = SLOT_LIST;
        } else if (jitAnyGuard() && strcmp(tn, "list") == 0 &&
                   k == SLOT_LIST) {
            /* SLOT_LIST is a guarded fact, not a hope: every arm that puts
             * one on the stack has already proved OBJ_LIST, because VAL_OBJ
             * covers every heap object and the shared return path says so
             * in as many words. `jaiTypeNameStatic` calls an ObjList "list"
             * (value.c), which is the name this guard compares against, so
             * there is nothing left to check.
             *
             * Same exposure as the `float`/`int`/`bool` cases above and no
             * more: all four reason from the type NAME, and a module global
             * shadowing one of those names with a class would change what
             * the interpreter does. That is the arm's existing contract. */
        } else if (jitAnyGuard() && k == SLOT_INST &&
                   e->stackClass[e->depth - 1] != NULL &&
                   e->stackClass[e->depth - 1]->name != NULL &&
                   strcmp(e->stackClass[e->depth - 1]->name->chars, tn) == 0) {
            /* The pinned class IS the one the boundary names. Like
             * SLOT_LIST this is a guarded fact -- every arm that puts a
             * SLOT_INST on the stack with a class pinned has emitted the
             * shapeId check that proves it -- so the guard has nothing left
             * to do.
             *
             * An exact name match is sufficient, not necessary:
             * jaiValueMatchesType also accepts a subclass and a trait
             * implementer, so anything this does not recognise still
             * declines rather than being waved through. The names that
             * turn up are `Tensor`, `Mat` and `NDArray` -- the ML packages
             * annotate their boundaries, so one of these sat in the middle
             * of a hot body and declined all of it. */
        } else if (jitAnyGuard() && strcmp(tn, "any") == 0) {
            /* `any` is satisfied by every value, so this guard is a no-op
             * for every kind -- not a narrowing the tier is guessing at.
             * jaiValueMatchesType returns true for the name "any" before
             * looking at the subject at all (vm.c), which is what makes
             * emitting nothing here the same thing the interpreter does.
             *
             * It is common enough to matter because `dict[str, any]` is how
             * this compiler carries AST records: a parameter or return
             * declared `any` put one of these in the middle of a body and
             * declined all of it. */
        } else {
            return subWhy(e, "a `%s` guard on a %s", tn, slotKindName(k));
        }
        off += 4;
        break;
    } while (0);
    *offp = off;
    return true;
}

JitArmResult emitGetFieldLocal(Emit *e, ObjFunction *fn, const uint8_t *code,
                               int *offp, int stop) {
    int off = *offp;
    do {
        unsigned slot    = jaiReadU16(code + off + 1);
        uint32_t nameIdx = jaiReadU24(code + off + 3);

        /* A maybe-instance reads like an instance once known not-null. The program has usually just tested
         * it (`if node == null { return 0 }`) but the tier doesn't track that, so the guard stands and costs one compare against zero; a null arriving for real just deopts. */
        if (e->localKind[slot] != SLOT_INST &&
            e->localKind[slot] != SLOT_MAYBE_INST) {
            return subWhy(e, "receiver local %u has kind %s, not instance",
                          slot, slotKindName(e->localKind[slot]));
        }
        if (e->localClass[slot] == NULL) {
            return subWhy(e, "receiver local %u has no pinned class", slot);
        }
        /* The name is resolved BEFORE the receiver guard is emitted, so
         * that a name this arm cannot read reaches `unarmedOpcode` with
         * nothing half-emitted -- the unarmed path takes over at the start
         * of an instruction. */
        if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
            return subWhy(e, "the field name is not in the pool");
        }
        Value nameVal = fn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nameVal)) return subWhy(e, "the field name is not a string");

        const FieldInfo *info =
            jaiClassFieldInfo(e->localClass[slot], AS_STRING(nameVal));
        /* The commonest of the four by a wide margin, and the one that
         * reads as a puzzle without being named: `self.method` is a METHOD,
         * and jaiClassFieldInfo only knows fields, so every
         * `return self.m(x)` written as a value lands here.
         *
         * It is 88 of resolve.jai's 104 OP_GET_FIELD_LOCAL declines, and
         * every one is the same shape: a keyword call to a private method,
         * `self._error(code, start, end, msg, help: "...")`. A keyword
         * argument rules out OP_INVOKE, so codegen has to materialise the
         * callee, and reading a method name as a value lands here.
         *
         * Those calls are on ERROR paths -- the lexer reaching a malformed
         * token -- which is what makes the unarmed path the right answer
         * rather than a decline, and the same reason OP_GET_GLOBAL takes it
         * for `throw ValueError(...)`. Softening a HOT refusal is a loss:
         * the same change on OP_GET_INDEX's `d[k]` measured 18% slower,
         * because a body that deopts every iteration costs more than one
         * that is simply interpreted. Cold is the whole condition.
         *
         * `!e->osr` is the same rule again, and measured too: inside a loop
         * nothing is cold. Dropping that guard compiles six MORE bodies in
         * resolve.jai (277 -> 283) and cancels the win -- 0.272s becomes
         * 0.274-0.294s against 0.274-0.284s for the hard refusal, where the
         * non-OSR-only form was a clean 3%. A loop that deopts inside
         * itself pays the entry and the exit every iteration. */
        if (info == NULL) {
            if (!e->osr && jitSoftField()) goto unarmedOpcode;
            return subWhy(e, "`%s` is not a field of %s",
                          AS_STRING(nameVal)->chars,
                          e->localClass[slot]->name
                              ? e->localClass[slot]->name->chars : "?");
        }
        if (e->localKind[slot] == SLOT_MAYBE_INST) {
            unsigned rcv = localIn(e, slot, JIT_SCRATCH_A);
            emit(e, jaiA64SubsXImm(31, rcv, 0));
            branchOnDeopt(e, JAI_A64_EQ);
        }
        if (slot == 0) e->usesSlot0 = true;
        if (info->isStatic) {
            return subWhy(e, "`%s` is a static", AS_STRING(nameVal)->chars);
        }

        /* Field type is read off the LIVE receiver, so the tier specialises to what the program actually
         * stores rather than a declaration -- only possible for a parameter (hence the arity cap above): a local assigned further in has no value yet to look at. */
        Value seen = seenLocal(e, slot);
        if (!IS_INSTANCE(seen)) {
            /* No sample -- but the DECLARATION may still say enough, and
             * the receiver is a proven instance either way: localKind is
             * SLOT_INST or SLOT_MAYBE_INST with a pinned class, and the
             * null check above has already run. Only the field's own kind
             * was missing.
             *
             * The sibling arm at OP_GET_FIELD has had this since the
             * accessor-then-field-read chain was fixed; this one did not,
             * and the asymmetry was worth **5.2% of parser.jai's
             * interpreted work** by exact attribution -- the largest single
             * reason left once the declared-ANY fields were given a code.
             *
             * Everything below is the same shape as there: the tag is
             * guarded, a list or str is proved by Obj.type as well because
             * VAL_OBJ is every heap object, and a str carries an interned
             * empty string as a sample so the arms downstream that ask
             * `stringOperand` can fire. */
            SlotKind lkind;
            unsigned ltag;
            if (!jitDeclaredFieldKindEnabled() ||
                !declaredScalarFieldKind(info->typeId, &lkind, &ltag)) {
                return subWhy(e,
                    "no live receiver to read local %u's field off, and its "
                    "declared kind (%u) is not one this predicts",
                    slot, (unsigned)info->typeId);
            }
            unsigned lbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned lrr = localIn(e, slot, JIT_SCRATCH_C);
            /* Guarded before anything is pushed, so a deopt resumes at this
             * instruction with the interpreter's stack untouched. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, lrr, lbase));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, ltag));
            branchOnDeopt(e, JAI_A64_NE);
            Value lprobe = NULL_VAL;
            if (lkind == SLOT_LIST || info->typeId == FIELD_KIND_STR) {
                unsigned want = lkind == SLOT_LIST ? (unsigned)OBJ_LIST
                                                   : (unsigned)OBJ_STRING;
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, lrr, lbase + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, want));
                branchOnDeopt(e, JAI_A64_NE);
                if (want == (unsigned)OBJ_STRING) {
                    ObjString *empty = jaiStringIntern("", 0);
                    if (empty == NULL) return false;
                    lprobe = OBJ_VAL((Obj *)empty);
                }
            }
            if (!pushValue3(e, lkind, 0, NULL, lprobe, -1)) return false;
            if (!IS_NULL(lprobe)) {
                e->stackObjType[e->depth - 1] = (uint8_t)(OBJ_STRING + 1);
            }
            if (lkind == SLOT_BOOL) {
                /* One byte: BOOL_VAL writes only the union's bool member. */
                emit(e, jaiA64LdrByte(pushReg(e) - 1, lrr, lbase + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, lrr, lbase + 8));
            }
            off += 8;
            break;
        }
        ObjInstance *inst = AS_INSTANCE(seen);
        if (info->slot >= inst->fieldCount) {
            return subWhy(e, "field slot %u is past the live receiver's %d",
                          (unsigned)info->slot, inst->fieldCount);
        }
        Value fieldVal = inst->fields[info->slot];

        SlotKind kind;
        unsigned tag;
        ObjClass *fcls = NULL;
        if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
        else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
        else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
            /* A tree's `left` is a Node on one call and null on the next; a leaf has no class to read, so the
             * receiver's own class is the guess (a nullable instance field is usually a linked structure) -- safe because the class guard below just deopts if wrong, never miscompiles. */
            kind = SLOT_MAYBE_INST;
            tag  = VAL_OBJ;
            fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                         : e->localClass[slot];
            if (fcls == NULL) return false;
        }
        /* A list field earns the stronger kind: SLOT_OBJ can be passed and stored but not iterated,
         * indexed or pushed to, and `for x in self.items` / `self.items.push(v)` is the commonest thing
         * a class holding a list does. Paid for with the OBJ_LIST check below, since VAL_OBJ alone
         * would let a str reach a header read. */
        else if (IS_LIST(fieldVal)) { kind = SLOT_LIST; tag = VAL_OBJ; }
        /* A str/dict/set field, held raw -- the same contract as a SLOT_OBJ global or list element:
         * the tag guard below says "an object", the sample says which type it was, and every consumer
         * (index, invoke, compare) re-checks Obj.type for itself before it does anything type-specific.
         * Refusing this declined the whole enclosing FUNCTION, which is most object-oriented code. */
        /* A bool field. Most classes carry one, and without this arm the
         * whole enclosing function declined -- `_check_live` in jaicv's
         * cascade is a method whose only field read is a bool, and it cost
         * 107 ms against 11.8 ms for the byte-identical method over an int
         * field. The load below is a `ldrb`, because BOOL_VAL writes only
         * the union's one-byte member. */
        else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
        else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ; tag = VAL_OBJ; }
        else return false;

        unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                        (unsigned)info->slot * (unsigned)sizeof(Value);
        /* The tag is checked every time, unless this body wrote the field
         * itself. A field is not typed by the runtime, so a later int
         * where a float was seen must bail rather than be read as one. */
        unsigned recv = localIn(e, slot, JIT_SCRATCH_C);
        SlotKind already = knownFieldKind(e, (int)slot, info->slot);
        if (kind == SLOT_MAYBE_INST) {
            /* Three guards, branch-free via a trick: when the field is null, the loads below are redirected at
             * the receiver instead (already a live instance, per the entry guard/null check above) -- a real object of the right type, so nothing ever dereferences zero. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));       /* tag */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));   /* value */
            emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
            emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                JIT_SCRATCH_B, JAI_A64_EQ));
            emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                JIT_SCRATCH_A, JAI_A64_EQ));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
            emit(e, jaiA64CselX(JIT_SCRATCH_B, recv, JIT_SCRATCH_D,
                                JAI_A64_EQ));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjInstance, klass)));
            emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
            if (fcls != e->localClass[slot]) {
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
            }
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (already != SLOT_SELF) {
            kind = already;
        } else {
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, recv, base));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnDeopt(e, JAI_A64_NE);
            /* "an object" is not "a list": every SLOT_LIST consumer reads the header with no check of
             * its own, so the object type is confirmed here, once, before the kind is handed out.
             * `already` is never SLOT_LIST (see recordFieldStore's caller), so this arm sees them all. */
            if (kind == SLOT_LIST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, recv, base + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
        }

        if (kind == SLOT_MAYBE_INST || kind == SLOT_LIST) {
            if (!pushValue3(e, kind,
                            kind == SLOT_MAYBE_INST ? fcls->shapeId : 0,
                            kind == SLOT_MAYBE_INST ? fcls : NULL,
                            kind == SLOT_LIST ? fieldVal
                            : IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                            -1)) {
                return false;
            }
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
        } else {
            /* SLOT_OBJ needs the sample carried: it is the only record of which object type this was,
             * and a consumer with nothing to look at declines. NULL_VAL when `already` overrode the
             * observation, since then the stored kind and the sampled value need not agree. */
            if (!pushValue3(e, kind, 0, NULL,
                            kind == SLOT_OBJ && rawObjValue(fieldVal)
                                ? fieldVal : NULL_VAL,
                            -1)) {
                return false;
            }
            /* A float field goes straight to the FP bank when something
             * downstream will read it there: `ldr d` instead of `ldr x`
             * followed by the `fmov` its consumer would emit. */
            if (kind == SLOT_FLOAT &&
                fpWorthLoading(e, code, off + 8, stop)) {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), recv, base + 8));
                fpClaim(e, idx);
            } else if (kind == SLOT_BOOL) {
                /* One byte: BOOL_VAL writes only the union's `boolean`
                 * member, so the seven above it are whatever the field
                 * held before, and every SLOT_BOOL consumer tests the
                 * whole register. */
                emit(e, jaiA64LdrByte(pushReg(e) - 1, recv, base + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, recv, base + 8));
            }
        }
        off += 8;
        break;
    } while (0);
    *offp = off;
    return JIT_ARM_OK;
unarmedOpcode:
    *offp = off;
    return JIT_ARM_UNARMED;
}

bool emitGetField(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp,
                  int stop) {
    int off = *offp;
    do {
        /* Same as OP_GET_FIELD_LOCAL, but the receiver arrives on the
         * operand stack -- which is what a compound assignment emits,
         * since it pushes the receiver twice. */
        uint32_t nameIdx = jaiReadU24(code + off + 1);
        if (e->depth == 0) return false;
        ObjClass *klass = e->stackClass[e->depth - 1];
        Value seen = e->stackSeen[e->depth - 1];
        int fromLocal = e->stackLocal[e->depth - 1];

        /* `Enum.Variant` for a payload-less variant, folded to the one
         * value that variant will ever have (EnumVariant::unit, made on
         * first mention and shared from then on so that `is` keeps
         * meaning identity). Without it the receiver is a global enum,
         * which globalKind classifies SLOT_OBJ, and the SLOT_INST
         * requirement below declined the whole enclosing function.
         *
         * The guard is the enum's shapeId, not its address: shapeId comes
         * from a monotonic counter precisely so that a freed enum whose
         * address is reused cannot be mistaken for the original, which is
         * what makes baking `unit` as a constant safe. `rcv` holds this
         * iteration's load of the global, so reading through it touches a
         * live object.
         *
         * The methods table is consulted first because enumMember does:
         * a method may shadow a variant name, and folding past that would
         * change what the program means.
         *
         * MEASURED. A loop dispatching on `k == Kind.X` over a token list
         * went 654 ms to 62 ms (10.5x, best of five alternating), and a
         * bare `Color.Red`/`Color.Green` compare loop 208 ms to 31 ms:
         * before this the whole loop was interpreted, and this was the
         * tier's single largest refusal on the self-hosted front end --
         * 189 distinct declining sites against 90 for the next reason,
         * the declined list being the whole of the lexer and the parser.
         *
         * RULED OUT: the front end itself does not get faster. It drops
         * 594 distinct declines to 540 and 469M interpreted instructions
         * to 459M (-2.0%), and 56 parser functions newly compile, but
         * `check` over ten large sources measured 3253 ms against 3253 ms
         * and `fmt --check lib` 3514 against 3471 -- both inside the
         * run-to-run spread. The functions that hold the time
         * (_parse_expression, _scan_punct, _punct_kind, keyword_lookup)
         * clear this refusal only to stop at the next one. Kept for the
         * loops it does unblock and as the prerequisite for that work,
         * not as a win on the compiler.
         *
         * REACH. Only an operand-stack enum receiver, which in practice
         * means a global: `Kind.X` written against a module-level enum.
         * The same mention through a LOCAL holding the enum arrives at
         * OP_GET_FIELD_LOCAL, which has no fold and still declines. That
         * is the next site, and the arm here ports to it unchanged. */
        if (e->stack[e->depth - 1] == SLOT_OBJ &&
            IS_OBJ_TYPE(seen, OBJ_ENUM) &&
            nameIdx < (uint32_t)fn->chunk.constants.count &&
            IS_STRING(fn->chunk.constants.data[nameIdx])) {
            ObjEnum *en = (ObjEnum *)AS_OBJ(seen);
            ObjString *vname = AS_STRING(fn->chunk.constants.data[nameIdx]);
            Value shadow;
            int tag = -1;
            if (!jaiTableGetInterned(&en->methods, vname, &shadow)) {
                for (uint16_t vi = 0; vi < en->variantCount; vi++) {
                    if (en->variants[vi].name == vname ||
                        jaiStringEquals(en->variants[vi].name, vname)) {
                        tag = (int)vi;
                        break;
                    }
                }
            }
            if (tag >= 0 && en->variants[tag].arity == 0 &&
                en->variants[tag].unit != NULL) {
                ObjEnumVal *unit = en->variants[tag].unit;
                /* This path guards, so nothing may still be deferred. */
                settleAll(e);
                unsigned rcv = valueXReg(e, e->valueDepth - 1);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_ENUM));
                branchOnDeopt(e, JAI_A64_NE);
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, rcv,
                                   (unsigned)offsetof(ObjEnum, shapeId)));
                emitConst64(e, JIT_SCRATCH_B, (int64_t)en->shapeId);
                emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
                branchOnDeopt(e, JAI_A64_NE);
                unsigned droppedEnum;
                if (!popValue(e, &droppedEnum, NULL)) return false;
                if (!pushValue3(e, SLOT_OBJ, 0, NULL, OBJ_VAL(unit), -1)) {
                    return false;
                }
                e->stackUnit[e->depth - 1] = true;
                emitConst64(e, pushReg(e) - 1, (int64_t)(uintptr_t)unit);
                off += 6;
                break;
            }
        }

        /* `Klass.STATIC_NAME`: a static field read off a class object
         * known at compile time. The receiver came from OP_GET_GLOBAL's
         * `globalClass` arm (the only producer of SLOT_CLASS -- see its
         * comment: "resolved now and pinned by the module version check
         * at entry" and "occupies no register, baked into the call
         * sequence"), so `klass` here is not a guess.
         *
         * What is NOT known at compile time is the static's VALUE: it
         * lives in `klass->statics`, a plain JaiTable exactly like a
         * module's globals table, reassignable at any point after class
         * definition (see staticFieldSlot's comment -- `isLet` is not
         * enforced by the interpreter or the checker for a static, only
         * for an INSTANCE field written through `self`). So this reads it
         * the same way OP_GET_GLOBAL reads a plain module global BY
         * ADDRESS: the JaiEntry* is baked, and two guards (the table
         * hasn't rehashed, the value still has the kind observed at
         * compile time) stand between the load and trusting it. A wrong
         * guess, or a rebind between compile and this call, deopts; nothing
         * here can return a stale or mistyped answer.
         *
         * REACH. `TokenFlags.AFTER_NEWLINE` in Lexer._push
         * (lib/jaithon/compile/lexer.jai) -- 10/10 attempts hit this
         * exact gap with no secondary reason (docs/agents/fix-1.md). */
        if (jitStaticFieldEnabled() && e->stack[e->depth - 1] == SLOT_CLASS) {
            if (klass == NULL) {
                return subWhy(e, "a static receiver with no class pinned");
            }
            if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
                return subWhy(e, "the field name is not in the pool");
            }
            Value sNameVal = fn->chunk.constants.data[nameIdx];
            if (!IS_STRING(sNameVal)) {
                return subWhy(e, "the field name is not a string");
            }
            ObjString *sname = AS_STRING(sNameVal);

            const FieldInfo *sinfo = jaiClassFieldInfo(klass, sname);
            if (sinfo == NULL) {
                return subWhy(e, "`%s` is not a field of %s", sname->chars,
                              klass->name ? klass->name->chars : "?");
            }
            if (!sinfo->isStatic) {
                return subWhy(e, "`%s` is an instance field, not a static",
                              sname->chars);
            }

            JaiEntry *sslot = staticFieldSlot(e, klass, sname);
            if (sslot == NULL) {
                return subWhy(e, "`%s` has no static storage, or is a "
                              "second class's statics in one body",
                              sname->chars);
            }
            Value sseen = sslot->value;

            SlotKind sk = SLOT_OPAQUE;
            uint32_t sshape = 0;
            ObjClass *sfcls = NULL;
            if (!globalKind(sseen, &sk, &sshape, &sfcls)) {
                return subWhy(e, "static `%s` is a kind this tier cannot "
                              "hold", sname->chars);
            }

            /* Named ahead of the guards, same reason as OP_GET_GLOBAL's
             * BY-ADDRESS arm: they run against the model as it is now. */
            unsigned dst = valueXReg(e, e->valueDepth);
            unsigned stag = sk == SLOT_INT   ? VAL_INT
                          : sk == SLOT_FLOAT ? VAL_FLOAT
                          : sk == SLOT_BOOL  ? VAL_BOOL
                                             : VAL_OBJ;

            emitStaticsGuard(e);
            emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)sslot);
            emit(e, jaiA64LdrW(JIT_SCRATCH_C, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_C, stag));
            branchOnDeopt(e, JAI_A64_NE);
            emit(e, jaiA64LdrX(JIT_SCRATCH_C, JIT_SCRATCH_D,
                               (unsigned)offsetof(JaiEntry, value) + 8u));
            if (sk == SLOT_INST || sk == SLOT_LIST) {
                emit(e, jaiA64LdrByte(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                      (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B,
                                       sk == SLOT_INST ? OBJ_INSTANCE
                                                        : OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
            if (sk == SLOT_INST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SCRATCH_C,
                                   (unsigned)offsetof(ObjInstance, klass)));
                emit(e, jaiA64LdrW(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                   (unsigned)offsetof(ObjClass, shapeId)));
                emitConst64(e, JIT_SCRATCH_A, (int64_t)sshape);
                emit(e, jaiA64SubsX(31, JIT_SCRATCH_B, JIT_SCRATCH_A));
                branchOnDeopt(e, JAI_A64_NE);
            }

            /* The receiver held no register (SLOT_CLASS -- see its
             * comment on the enum): dropping it is the whole of popping
             * it. popValue would refuse it via holdsRegister. */
            e->depth--;
            if (!pushValue3(e, sk, sshape, sfcls, sseen, -1)) return false;
            emit(e, jaiA64MovX(dst, JIT_SCRATCH_C));
            off += 6;
            break;
        }

        /* And, as at the local-receiver arm, a maybe-instance reads like an
         * instance once known not-null -- which is what `a.b.c` needs, since
         * `.b` came back SLOT_MAYBE_INST and `.c` wants a receiver.
         *
         * This arm and the SLOT_MAYBE_INST field arm below were the two
         * halves of the same hole: the whole chain `a.b.c.d.v` declined its
         * enclosing loop, and it now compiles. Measured on a loop reading
         * one four-deep chain per iteration, alternating old/new binaries
         * and warmed by the clock, best of five each: 0.4356 s -> 0.1107 s,
         * 3.9x. What did NOT move is a self-hosted compile (`check` over
         * parser.jai + emit.jai): 1.4253 s -> 1.4353 s, inside the 2% a
         * base-against-base control shows -- only eleven of the compiler's
         * bodies decline for these two reasons. The enum receiver that stops
         * 110 more of them is the arm directly above, and it did not move
         * the compile either. */
        /* `module.MEMBER`. The receiver is a module the walk has a live
         * sample of, which is every imported module: OP_GET_GLOBAL just
         * read it out of this body's own globals. Reading a constant off
         * one used to decline the whole function. */
        if (moduleFieldOn() && IS_MODULE(seen) &&
            nameIdx < (uint32_t)fn->chunk.constants.count &&
            IS_STRING(fn->chunk.constants.data[nameIdx])) {
            ObjString *mname = AS_STRING(fn->chunk.constants.data[nameIdx]);
            JaiEntry *mslot = moduleMemberSlot(e, AS_MODULE(seen), mname);
            Value held = (mslot != NULL) ? mslot->value : NULL_VAL;
            SlotKind mk = SLOT_OPAQUE;
            unsigned mtag = VAL_OBJ;
            if (mslot != NULL) {
                if (IS_INT(held))        { mk = SLOT_INT;   mtag = VAL_INT; }
                else if (IS_FLOAT(held)) { mk = SLOT_FLOAT; mtag = VAL_FLOAT; }
                else if (IS_BOOL(held))  { mk = SLOT_BOOL;  mtag = VAL_BOOL; }
                /* Objects are deliberately left out. A module member that
                 * is a closure, class or native is a CALLEE, and the arms
                 * that bake one by value rely on ObjModule::version, which
                 * only retires a form at its next entry -- handing one out
                 * through this path would be a wrong answer rather than a
                 * decline. Scalars are inert and have no such problem. */
            }
            if (mk != SLOT_OPAQUE) {
                emitModuleGuard(e);
                unsigned mdrop;
                if (!popValue(e, &mdrop, NULL)) return false;
                if (!pushValue3(e, mk, 0, NULL, held, -1)) return false;
                unsigned mrd = pushReg(e) - 1;
                emitConst64(e, JIT_SCRATCH_D, (int64_t)(uintptr_t)mslot);
                /* The tag is re-checked every read: a module global is
                 * assignable, so `math.PI = 3` must deoptimise rather than
                 * hand back the old kind's payload. */
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(JaiEntry, value)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, mtag));
                branchOnDeopt(e, JAI_A64_NE);
                if (mk == SLOT_BOOL) {
                    emit(e, jaiA64LdrByte(mrd, JIT_SCRATCH_D,
                                          (unsigned)offsetof(JaiEntry, value) + 8u));
                } else {
                    emit(e, jaiA64LdrX(mrd, JIT_SCRATCH_D,
                                       (unsigned)offsetof(JaiEntry, value) + 8u));
                }
                off += 6;
                break;
            }
        }
        if (e->stack[e->depth - 1] != SLOT_INST &&
            e->stack[e->depth - 1] != SLOT_MAYBE_INST) {
            return subWhy(e, "a receiver of kind %s, not an instance",
                          slotKindName(e->stack[e->depth - 1]));
        }
        if (klass == NULL) {
            return subWhy(e, "an instance receiver with no class pinned");
        }
        if (e->stack[e->depth - 1] == SLOT_MAYBE_INST) {
            emit(e, jaiA64SubsXImm(31, valueBankReg(e, e->valueDepth - 1), 0));
            branchOnDeopt(e, JAI_A64_EQ);
        }
        if (nameIdx >= (uint32_t)fn->chunk.constants.count) {
            return subWhy(e, "the field name is not in the pool");
        }
        Value nameVal = fn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nameVal)) {
            return subWhy(e, "the field name is not a string");
        }

        const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
        if (info == NULL) {
            return subWhy(e, "`%s` is not a field of %s",
                          AS_STRING(nameVal)->chars,
                          klass->name ? klass->name->chars : "that class");
        }
        if (info->isStatic) {
            return subWhy(e, "`%s` is a static, not an instance field",
                          AS_STRING(nameVal)->chars);
        }
        if (!IS_INSTANCE(seen)) {
            /* No sample to classify the field from. The commonest cause
             * is a receiver OP_INVOKE itself predicted: `klass` came from
             * observedReturnKind's shape (or a compiled callee's
             * jitReturnShape) by way of jaiClassForShape, which pins the
             * class exactly, but pushValue leaves `seen` at NULL_VAL
             * because there is no actual instance behind a prediction,
             * only a shape id -- `self._peek().kind`, `self._chunk().depth`
             * and every other zero-arg-accessor-then-field-read chain in
             * the self-hosted front end is this shape, and it declined
             * outright before this arm existed. See declaredScalarFieldKind
             * for why only a scalar can be predicted from here. */
            SlotKind dkind;
            unsigned dtag;
            if (!jitDeclaredFieldKindEnabled() ||
                !declaredScalarFieldKind(info->typeId, &dkind, &dtag)) {
                /* The declared kind is NAMED, because which one it is
                 * decides the work: a scalar wants a row in
                 * declaredScalarFieldKind, an ANY wants nothing at all,
                 * and the census could not tell them apart. */
                return subWhy(e,
                    "no live receiver to read `%s` off, and its declared "
                    "kind (%u) is not one this predicts",
                    AS_STRING(nameVal)->chars, (unsigned)info->typeId);
            }
            unsigned dbase = (unsigned)offsetof(ObjInstance, fields) +
                             (unsigned)info->slot * (unsigned)sizeof(Value);
            unsigned drr = valueBankReg(e, e->valueDepth - 1);
            /* Guard BEFORE the receiver comes off the model, as the
             * sampled path below does: a deopt here resumes at this
             * instruction, and the interpreter's stack still has the
             * receiver on it. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, drr, dbase));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, dtag));
            branchOnDeopt(e, JAI_A64_NE);
            Value dprobe = NULL_VAL;
            if (dkind == SLOT_LIST || info->typeId == FIELD_KIND_STR) {
                /* VAL_OBJ said "a heap object" and no more. Prove the
                 * actual type before the entry claims to be one, or the
                 * next arm reads ObjList's count out of a string's header
                 * -- the hole that segfaulted the VM through the
                 * dict-index arm. */
                unsigned want = dkind == SLOT_LIST ? (unsigned)OBJ_LIST
                                                   : (unsigned)OBJ_STRING;
                emit(e, jaiA64LdrX(JIT_SCRATCH_B, drr, dbase + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, want));
                branchOnDeopt(e, JAI_A64_NE);
                if (want == (unsigned)OBJ_STRING) {
                    /* A probe, not an observation: the arms below ask
                     * `stringOperand`, which reads the SAMPLE rather than
                     * the kind, so a string entry without one is a string
                     * nothing can do anything with. The guard just emitted
                     * is what makes the probe honest -- it chooses an arm
                     * and never removes a check. Same device the invoke
                     * arm uses for a receiver it knows only the type of. */
                    ObjString *empty = jaiStringIntern("", 0);
                    if (empty == NULL) return false;
                    dprobe = OBJ_VAL((Obj *)empty);
                }
            }
            unsigned dpopped;
            SlotKind dkr;
            if (!popValue(e, &dpopped, &dkr)) return false;
            if (!pushValue3(e, dkind, 0, NULL, dprobe, -1)) return false;
            if (!IS_NULL(dprobe)) {
                e->stackObjType[e->depth - 1] =
                    (uint8_t)(OBJ_STRING + 1);
            }
            if (dkind == SLOT_FLOAT &&
                fpWorthLoading(e, code, off + 6, stop)) {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), drr, dbase + 8));
                fpClaim(e, idx);
            } else if (dkind == SLOT_BOOL) {
                /* One byte, for the reason given at the local-receiver
                 * arm: BOOL_VAL writes only the union's bool member. */
                emit(e, jaiA64LdrByte(pushReg(e) - 1, drr, dbase + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, drr, dbase + 8));
            }
            off += 6;
            break;
        }
        ObjInstance *inst = AS_INSTANCE(seen);
        if (info->slot >= inst->fieldCount) {
            return subWhy(e, "the sampled instance has fewer fields than "
                             "its class declares");
        }
        Value fieldVal = inst->fields[info->slot];

        SlotKind kind;
        unsigned tag;
        ObjClass *fcls = NULL;
        if (IS_INT(fieldVal))        { kind = SLOT_INT;   tag = VAL_INT; }
        else if (IS_FLOAT(fieldVal)) { kind = SLOT_FLOAT; tag = VAL_FLOAT; }
        /* A nullable instance field, same guess and same guards as the
         * local-receiver arm: a leaf's null has no class to read, so the
         * receiver's own class stands in and the class guard below deopts
         * if that was wrong. Without this arm `rawObjValue` (which excludes
         * OBJ_INSTANCE) let an instance-valued field fall off the end of
         * the chain and decline the whole function. */
        else if (IS_INSTANCE(fieldVal) || IS_NULL(fieldVal)) {
            kind = SLOT_MAYBE_INST;
            tag  = VAL_OBJ;
            fcls = IS_INSTANCE(fieldVal) ? AS_INSTANCE(fieldVal)->klass
                                         : klass;
            if (fcls == NULL) {
                return subWhy(e, "a nullable field with no class to guard");
            }
        }
        /* As in OP_GET_FIELD_LOCAL: an object-typed field is held raw
         * rather than declining the enclosing function, and a list earns
         * the stronger kind at the price of an OBJ_LIST check. */
        else if (IS_LIST(fieldVal))     { kind = SLOT_LIST; tag = VAL_OBJ; }
        /* A bool field. Most classes carry one, and without this arm the
         * whole enclosing function declined -- `_check_live` in jaicv's
         * cascade is a method whose only field read is a bool, and it cost
         * 107 ms against 11.8 ms for the byte-identical method over an int
         * field. The load below is a `ldrb`, because BOOL_VAL writes only
         * the union's one-byte member. */
        else if (IS_BOOL(fieldVal)) { kind = SLOT_BOOL; tag = VAL_BOOL; }
        else if (rawObjValue(fieldVal)) { kind = SLOT_OBJ;  tag = VAL_OBJ; }
        else {
            return subWhy(e, "a field holding %s, which has no arm",
                          jaiTypeNameStatic(fieldVal));
        }

        unsigned fbase = (unsigned)offsetof(ObjInstance, fields) +
                         (unsigned)info->slot * (unsigned)sizeof(Value);
        unsigned rr = valueBankReg(e, e->valueDepth - 1);
        SlotKind already = knownFieldKind(e, fromLocal, info->slot);
        if (kind == SLOT_MAYBE_INST) {
            /* The same three guards the local arm emits, branch-free by the
             * same trick: a null payload redirects the loads at the receiver,
             * which the entry check above has already proved is a live
             * instance, so nothing ever dereferences zero. All of it stands
             * before the receiver comes off the model, since every deopt
             * here resumes at this instruction with it still on the stack. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));       /* tag */
            emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));   /* value */
            emit(e, jaiA64MovzX(JIT_SCRATCH_B, 0, 0));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
            emit(e, jaiA64CselX(JIT_SCRATCH_D, JIT_SCRATCH_D,
                                JIT_SCRATCH_B, JAI_A64_EQ));
            emit(e, jaiA64CselX(JIT_SCRATCH_B, JIT_SCRATCH_B,
                                JIT_SCRATCH_A, JAI_A64_EQ));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_B, 0));
            branchOnDeopt(e, JAI_A64_NE);      /* neither object nor null */

            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
            emit(e, jaiA64CselX(JIT_SCRATCH_B, rr, JIT_SCRATCH_D,
                                JAI_A64_EQ));
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(Obj, type)));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
            branchOnDeopt(e, JAI_A64_NE);

            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                               (unsigned)offsetof(ObjInstance, klass)));
            emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)fcls);
            if (fcls != klass) {
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_D, 0));
                emit(e, jaiA64CselX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                                    JIT_SCRATCH_A, JAI_A64_EQ));
            }
            emit(e, jaiA64SubsXReg(31, JIT_SCRATCH_A, JIT_SCRATCH_B));
            branchOnDeopt(e, JAI_A64_NE);
        } else if (already != SLOT_SELF) {
            kind = already;
        } else {
            /* Guard BEFORE the receiver comes off the model: a deopt here
             * resumes at this instruction, and the interpreter's stack
             * still has the receiver on it. */
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, rr, fbase));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchOnDeopt(e, JAI_A64_NE);
            /* And that it is a list, not merely an object -- same reason as
             * in OP_GET_FIELD_LOCAL, and likewise while the receiver is
             * still on the model, since this guard resumes here too. */
            if (kind == SLOT_LIST) {
                emit(e, jaiA64LdrX(JIT_SCRATCH_D, rr, fbase + 8));
                emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_D,
                                   (unsigned)offsetof(Obj, type)));
                emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_LIST));
                branchOnDeopt(e, JAI_A64_NE);
            }
        }
        unsigned popped;
        SlotKind kr;
        if (!popValue(e, &popped, &kr)) return false;
        if (kind == SLOT_MAYBE_INST) {
            if (!pushValue3(e, kind, fcls->shapeId, fcls,
                            IS_INSTANCE(fieldVal) ? fieldVal : NULL_VAL,
                            -1)) {
                return false;
            }
            /* Already in hand: the guards above loaded the payload (or a
             * zero) into D, so there is nothing left to read. */
            emit(e, jaiA64MovX(pushReg(e) - 1, JIT_SCRATCH_D));
        } else {
            if (!pushValue3(e, kind, 0, NULL,
                            (kind == SLOT_OBJ && rawObjValue(fieldVal)) ||
                                    (kind == SLOT_LIST && IS_LIST(fieldVal))
                                ? fieldVal : NULL_VAL,
                            -1)) {
                return false;
            }
            if (kind == SLOT_FLOAT && fpWorthLoading(e, code, off + 6, stop)) {
                unsigned idx = e->valueDepth - 1;
                emit(e, jaiA64LdrD(fpRegAt(e, idx), rr, fbase + 8));
                fpClaim(e, idx);
            } else if (kind == SLOT_BOOL) {
                /* One byte, for the reason given at the local-receiver arm. */
                emit(e, jaiA64LdrByte(pushReg(e) - 1, rr, fbase + 8));
            } else {
                emit(e, jaiA64LdrX(pushReg(e) - 1, rr, fbase + 8));
            }
        }
        off += 6;
        break;
    } while (0);
    *offp = off;
    return true;
}

bool emitSetField(Emit *e, ObjFunction *fn, const uint8_t *code, int *offp) {
    int off = *offp;
    do {
        uint32_t nameIdx = jaiReadU24(code + off + 1);
        /* Receiver then value, both dropped. The receiver's class comes from its STACK entry, not guessed
         * from the locals: two instance locals of different classes would make any guess a silently wrong field offset. */
        if (e->depth < 2) return false;
        ObjClass *klass = e->stackClass[e->depth - 2];
        int recvLocal = e->stackLocal[e->depth - 2];

        unsigned rv, rr;
        SlotKind kv, kr;
        /* A float already in the FP bank is stored from there (`str d`, not popValue's `fmov x,d` + `str x`)
         * -- captured before the pop, since popping is what clears the bit and renames the index. */
        bool vIsFp = e->depth >= 1 && e->valueDepth >= 1 &&
                     e->stack[e->depth - 1] == SLOT_FLOAT &&
                     (e->fpLive & (1u << (e->valueDepth - 1))) != 0;
        unsigned dv = vIsFp ? fpHeldIn(e, e->valueDepth - 1) : 0u;
        if (vIsFp) {
            if (!popValueRaw(e, &rv, &kv)) return false;
        } else if (!popValue(e, &rv, &kv)) {
            return false;
        }
        if (!popValue(e, &rr, &kr)) return false;
        if (kr != SLOT_INST) return false;
        /* An object goes in as readily as a number. The collector is a plain mark-sweep with no write
         * barrier and nothing moves, so the only question is reachability: the receiver is rooted (it
         * is a live SLOT_INST here), and after the store the value hangs off it, while before the store
         * it was rooted in its own right by emitRootFill. What is refused is a kind with no payload
         * register to store (class/function/native/self) and SLOT_ITER, whose index lives in memory. */
        if (kv != SLOT_INT && kv != SLOT_FLOAT && kv != SLOT_BOOL &&
            kv != SLOT_OBJ && kv != SLOT_LIST && kv != SLOT_INST &&
            kv != SLOT_MAYBE_INST) {
            e->whyNot = "storing a field kind this tier cannot write";
            return false;
        }
        if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
        Value nameVal = fn->chunk.constants.data[nameIdx];
        if (!IS_STRING(nameVal)) return false;

        if (klass == NULL) return false;
        const FieldInfo *info = jaiClassFieldInfo(klass, AS_STRING(nameVal));
        if (info == NULL || info->isStatic) return false;

        unsigned base = (unsigned)offsetof(ObjInstance, fields) +
                        (unsigned)info->slot * (unsigned)sizeof(Value);
        /* Not a constant tag any more: a maybe-instance's is null-or-object,
         * read off the payload, which is exactly what emitTagFor does. */
        emitTagFor(e, kv, rv, JIT_SCRATCH_A, JIT_SCRATCH_B);
        emit(e, jaiA64StrW(JIT_SCRATCH_A, rr, base));
        if (vIsFp) emit(e, jaiA64StrD(dv, rr, base + 8));
        else       emit(e, jaiA64StrX(rv, rr, base + 8));
        /* Only a kind a later read can replay EXACTLY is remembered; the rest merely retire what was
         * known of this field (local -1), because the read side would otherwise take SLOT_INST with no
         * class behind it -- a shape every field offset it then resolved would be resolved against. */
        bool replayable = kv == SLOT_INT || kv == SLOT_FLOAT ||
                          kv == SLOT_BOOL || kv == SLOT_OBJ;
        recordFieldStore(e, replayable ? recvLocal : -1, info->slot, kv);
        e->wroteHeap = true;
        off += 6;
        break;
    } while (0);
    *offp = off;
    return true;
}

#endif /* __aarch64__ || __arm64__ */
