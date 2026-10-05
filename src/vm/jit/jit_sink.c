/* jit_sink.c -- allocation sinking in the OSR tier (JAITHON_JIT_SINK).
 *
 * A loop that builds an instance of a simple-init class into a local and then
 * only reads its fields -- directly, or through methods the tier inlines --
 * never needs the object at all. Such a local is SUNK: the construction stores
 * its arguments into frame slots (the fields' homes), a field read is one load
 * from there, and nothing is allocated.
 *
 * The object only has to exist when the interpreter takes over, and every way
 * out of an OSR form -- a loop exit, a deopt, a bail, a raised overflow -- runs
 * the same OSR_SYNC_ITER. That writes each sink's fields into gDeopt, and
 * jaiJitEnterOsr allocates the object and puts it in the local (and in every
 * operand-stack entry that named it) once the frame and stack are back in the
 * interpreter's hands and rooted. A field is an int or a float, so the record
 * itself holds nothing the collector has to see.
 *
 * What makes it sound is planSinks, before anything is emitted:
 *
 *  * every instruction in the loop is one whose local operands it can read,
 *    and the local is named only by OP_BIND (straight after an OP_CALL), by
 *    OP_GET_FIELD_LOCAL, and by local reads that feed an OP_INVOKE's receiver
 *    or arguments with nothing but pushes in between;
 *  * the first such instruction is the bind, and no branch skips it or lands
 *    behind it, so every read in the loop sees an object this iteration built
 *    (the local is dead at the head -- nothing is ever loaded from the frame);
 *  * the class's init only stores its arguments into all of its fields, and
 *    the sample object in the frame says each field is an int or a float.
 *
 * Everything the plan cannot prove from the bytecode is checked at the site --
 * the callee is that class, the arguments are those kinds, the call that
 * consumes a reference is inlined -- and a site that fails declines the
 * compile, which compileOsrAny then retries with sinking off. So a wrong guess
 * costs a compile, never an answer. */

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

static bool storeConstruction(Emit *e, unsigned j, unsigned argc);
static bool pushRef(Emit *e, unsigned j);

bool jitSinkOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_SINK");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* JAITHON_JIT_SINK_FP_HOMES=0 keeps every sunk float field in its frame
 * slot. On, a float field lives in a callee-saved d register: a loop-carried
 * object's chain (`acc = acc.add(step)`) is then an fadd and a move rather
 * than a store and a reload per field per iteration. */
static bool jitSinkFpHomesOn(void) {
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("JAITHON_JIT_SINK_FP_HOMES");
        on = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return on != 0;
}

/* Once the float locals have their registers: the rest of v8..v15 to sunk
 * float fields, in plan order. Counted into fpLocals, so the prologue saves
 * and every epilogue restores them like any float local's home. */
void sinkPlanFpHomes(Emit *e) {
    if (!jitSinkFpHomesOn()) return;
    for (unsigned j = 0; j < e->sinkCount; j++) {
        for (unsigned f = 0; f < e->sink[j].nfields; f++) {
            e->sink[j].fpHome[f] = 0;
            if (e->sink[j].kind[f] != SLOT_FLOAT) continue;
            if (e->fpLocals >= JIT_FP_MAX_SAVED) continue;
            e->sink[j].fpHome[f] =
                (uint8_t)(JIT_FP_FIRST_SAVED + e->fpLocals++);
        }
    }
}

/* A sunk construction reads its arguments out of the FP bank itself
 * (storeConstruction), so the walker need not sync the bank to X before it:
 * on a loop-carried object that sync was a round trip through X per field
 * per iteration, on the chain. */
bool sinkFpFast(const Emit *e, const uint8_t *code, int off, uint8_t op) {
    if (e->sinkCount == 0) return false;
    if (op == OP_TAIL_CALL) return e->inlining && e->inlSinkBind != 0;
    /* Only a construction sinkConstructs takes: any other call reads its
     * arguments out of X, so the bank must be synced for it as usual. */
    if (op == OP_CALL && !e->inlining && code[off + 2] == OP_BIND) {
        unsigned slot = jaiReadU16(code + off + 3);
        unsigned argc = code[off + 1];
        if (slot > JIT_MAX_SLOTS || e->sinkOf[slot] == 0) return false;
        if (e->depth < argc + 1u) return false;
        unsigned cidx = e->depth - argc - 1u;
        return e->stack[cidx] == SLOT_CLASS &&
               e->stackClass[cidx] == e->sink[e->sinkOf[slot] - 1u].cls;
    }
    return false;
}

enum { USE_NONE = 0, USE_BIND = 1, USE_FIELD = 2, USE_REF = 4, USE_OTHER = 8 };

/* Where a jump at `off` lands, or -1 if `op` is not a jump. */
static int64_t jumpTarget(const uint8_t *code, int off, uint8_t op, int len) {
    switch (op) {
    case OP_JUMP: case OP_JUMP_IF_FALSE: case OP_JUMP_IF_TRUE:
    case OP_JUMP_IF_FALSE_KEEP: case OP_JUMP_IF_TRUE_KEEP:
    case OP_LOOP: case OP_FOR_RANGE_BIND: case OP_FOR_ITER_BIND:
        return (int64_t)off + len + jaiReadI16(code + off + 1);
    case OP_JUMP_IF_CMP_FALSE:
        return (int64_t)off + len + jaiReadI16(code + off + 2);
    case OP_JUMP_IF_CMP_LOCAL_K:
        return (int64_t)off + len + jaiReadI16(code + off + 7);
    default:
        return -1;
    }
}

/* Whether the reference a local read pushes at `off` (with `after` of that
 * instruction's pushes above it) is read by the OP_GET_FIELD that follows, or
 * reaches an OP_INVOKE's receiver or arguments with only pushes of plain
 * values in between. */
static bool refFeedsInvoke(const Chunk *c, int off, unsigned after,
                           uint32_t end) {
    int at = off + instructionLength(c, off);
    if (after == 0 && (uint32_t)at < end && c->code[at] == OP_GET_FIELD) {
        return true;
    }
    while ((uint32_t)at < end) {
        uint8_t op = c->code[at];
        int len = instructionLength(c, at);
        if (len <= 0) return false;
        switch (op) {
        case OP_GET_LOCAL: case OP_CONST: case OP_INT:
            after += 1;
            break;
        case OP_GET_LOCAL2:
            after += 2;
            break;
        case OP_INVOKE:
            return after <= c->code[at + 4];
        default:
            return false;
        }
        at += len;
    }
    return false;
}

static void skipWhy(uint32_t top, unsigned slot, const char *why) {
    if (getenv("JAI_JIT_WHY")) {
        fprintf(stderr, "[jit] osr at %u does not sink local %u: %s\n", top,
                slot, why);
    }
}

void planSinks(Emit *e, const ObjFunction *fn, uint32_t top, uint32_t end,
               const Value *slots, const bool *byRef) {
    e->sinkCount = 0;
    memset(e->sinkOf, 0, sizeof e->sinkOf);
    if (!jitSinkOn() || e->locals == 0) return;
    const Chunk *c = &fn->chunk;
    uint8_t use[JIT_MAX_SLOTS + 1];
    uint32_t firstAt[JIT_MAX_SLOTS + 1];
    uint8_t firstUse[JIT_MAX_SLOTS + 1];
    int bindArgc[JIT_MAX_SLOTS + 1];
    memset(use, 0, sizeof use);
    memset(firstUse, 0, sizeof firstUse);
    for (unsigned i = 0; i <= JIT_MAX_SLOTS; i++) {
        firstAt[i] = UINT32_MAX;
        bindArgc[i] = -1;
    }

#define NOTE(slot_, how_)                                                    \
    do {                                                                     \
        unsigned s_ = (slot_);                                               \
        if (s_ > JIT_MAX_SLOTS) return;                                      \
        use[s_] |= (uint8_t)(how_);                                          \
        if (firstAt[s_] == UINT32_MAX) {                                     \
            firstAt[s_] = (uint32_t)off;                                     \
            firstUse[s_] = (uint8_t)(how_);                                  \
        }                                                                    \
    } while (0)

    int prevOff = -1;
    for (int off = (int)top; (uint32_t)off < end;) {
        uint8_t op = c->code[off];
        int len = instructionLength(c, off);
        if (len <= 0) return;
        switch (op) {
        case OP_GET_GLOBAL: case OP_SET_GLOBAL: case OP_CONST: case OP_INT:
        case OP_NULL: case OP_TRUE: case OP_FALSE: case OP_POP:
        case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD:
        case OP_NEG: case OP_NOT: case OP_MOD_INT_CONST:
        case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT: case OP_GE:
        case OP_CALL: case OP_INVOKE: case OP_GET_FIELD:
        case OP_JUMP: case OP_JUMP_IF_FALSE: case OP_JUMP_IF_TRUE:
        case OP_JUMP_IF_FALSE_KEEP: case OP_JUMP_IF_TRUE_KEEP:
        case OP_JUMP_IF_CMP_FALSE: case OP_LOOP:
        case OP_TYPE_GUARD: case OP_FLOORDIV: case OP_POW: case OP_POS:
        case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL: case OP_SHR:
        case OP_BNOT: case OP_GET_INDEX:
            break;
        case OP_GET_LOCAL: {
            unsigned s = jaiReadU16(c->code + off + 1);
            NOTE(s, refFeedsInvoke(c, off, 0, end) ? USE_REF : USE_OTHER);
            break;
        }
        case OP_GET_LOCAL2: {
            unsigned a = jaiReadU16(c->code + off + 1);
            unsigned b = jaiReadU16(c->code + off + 3);
            NOTE(a, refFeedsInvoke(c, off, 1, end) ? USE_REF : USE_OTHER);
            NOTE(b, refFeedsInvoke(c, off, 0, end) ? USE_REF : USE_OTHER);
            break;
        }
        case OP_GET_FIELD_LOCAL:
            NOTE(jaiReadU16(c->code + off + 1), USE_FIELD);
            break;
        case OP_BIND: {
            unsigned s = jaiReadU16(c->code + off + 1);
            bool ctor = prevOff >= 0 && prevOff + 2 == off &&
                        c->code[prevOff] == OP_CALL;
            /* Bound to a method's result: only an inlined body that ends by
             * constructing the class can hand it one (sinkConstructInline);
             * anything else refuses at the bind. */
            bool viaInvoke = prevOff >= 0 && prevOff + 7 == off &&
                             c->code[prevOff] == OP_INVOKE;
            NOTE(s, ctor || viaInvoke ? USE_BIND : USE_OTHER);
            if (ctor && s <= JIT_MAX_SLOTS) {
                int argc = c->code[prevOff + 1];
                if (bindArgc[s] >= 0 && bindArgc[s] != argc) {
                    use[s] |= USE_OTHER;
                }
                bindArgc[s] = argc;
            }
            break;
        }
        case OP_SET_LOCAL: case OP_ADD_INT_CONST: case OP_SUB_INT_CONST:
        case OP_MUL_INT_CONST: case OP_INC_LOCAL: case OP_CMP_LOCAL_CONST_LT:
        case OP_ADD_BIND: case OP_SUB_BIND: case OP_MUL_BIND:
            NOTE(jaiReadU16(c->code + off + 1), USE_OTHER);
            break;
        case OP_ADD_LOCALS:
            NOTE(jaiReadU16(c->code + off + 1), USE_OTHER);
            NOTE(jaiReadU16(c->code + off + 3), USE_OTHER);
            break;
        case OP_JUMP_IF_CMP_LOCAL_K:
            NOTE(jaiReadU16(c->code + off + 2), USE_OTHER);
            break;
        case OP_FOR_ITER_BIND:
            NOTE(jaiReadU16(c->code + off + 3), USE_OTHER);
            break;
        case OP_FOR_RANGE_BIND:
            NOTE(jaiReadU16(c->code + off + 3), USE_OTHER);
            NOTE(jaiReadU16(c->code + off + 5), USE_OTHER);
            NOTE(jaiReadU16(c->code + off + 7), USE_OTHER);
            break;
        default:
            /* An instruction whose local operands this does not read. */
            if (getenv("JAI_JIT_WHY")) {
                fprintf(stderr, "[jit] osr at %u plans no sink: %s at %d\n",
                        top, jaiOpName((OpCode)op), off);
            }
            return;
        }
        prevOff = off;
        off += len;
    }
#undef NOTE

    if (byRef == NULL) return;   /* the captures could not be decoded */

    /* Why a local bound from a construction is not sunk, for JAI_JIT_WHY.
     * A bare block, not do/while(0): `continue` must reach the for loop. */
#define SKIP(why_)                                                           \
    {                                                                        \
        skipWhy(top, s, (why_));                                             \
        continue;                                                            \
    }

    for (unsigned s = 0; s < e->locals && e->sinkCount < JIT_MAX_SINK; s++) {
        if ((use[s] & USE_BIND) == 0) continue;
        if ((use[s] & USE_OTHER) != 0) SKIP("a use other than a field read");
        if (byRef[s]) SKIP("captured by a closure");
        /* Read before it is bound: the object it holds at entry is unpacked
         * there, so no dominance is needed -- the homes always hold the
         * local's current fields. */
        bool entryLive = firstUse[s] != USE_BIND;
        if (entryLive) goto planned_dominance;
        /* Dominance: nothing before the first bind may leave for anywhere but
         * out of the loop, and nothing may land between the head and it. */
        bool dominated = true;
        for (int off = (int)top; (uint32_t)off < end && dominated;) {
            uint8_t op = c->code[off];
            int len = instructionLength(c, off);
            int64_t t = jumpTarget(c->code, off, op, len);
            if (t >= 0) {
                bool inLoop = t >= (int64_t)top && t < (int64_t)end;
                if ((uint32_t)off < firstAt[s] && inLoop) dominated = false;
                if (t > (int64_t)top && t <= (int64_t)firstAt[s]) {
                    dominated = false;
                }
            }
            off += len;
        }
        if (!dominated) SKIP("a branch skips its bind");
    planned_dominance:;

        Value v = slots[s];
        if (!IS_INSTANCE(v)) SKIP("not holding an instance at entry");
        ObjInstance *inst = AS_INSTANCE(v);
        ObjClass *cls = inst->klass;
        int argc = bindArgc[s];
        if (argc < 0 && cls != NULL) argc = (int)cls->fieldCount;
        if (cls == NULL || argc <= 0 || (unsigned)argc > JIT_SINK_FIELDS ||
            cls->fieldCount != (unsigned)argc ||
            inst->fieldCount != (unsigned)argc) {
            SKIP("a field count the construction does not fill");
        }
        uint16_t map[JIT_MAX_ARGS_OUT];
        if (!jitSimpleInitSlots(cls, (unsigned)argc, map)) {
            SKIP("an init that does more than store its arguments");
        }
        bool ok = true;
        uint32_t seenSlots = 0;
        unsigned j = e->sinkCount;
        for (int i = 0; i < argc && ok; i++) {
            if (map[i] >= (unsigned)argc || (seenSlots & (1u << map[i]))) {
                ok = false;
                break;
            }
            seenSlots |= 1u << map[i];
            Value fv = inst->fields[map[i]];
            if (IS_INT(fv))        e->sink[j].kind[map[i]] = SLOT_INT;
            else if (IS_FLOAT(fv)) e->sink[j].kind[map[i]] = SLOT_FLOAT;
            else ok = false;
            e->sink[j].argSlot[i] = map[i];
        }
        if (!ok) SKIP("a field that is not an int or a float");
        e->sink[j].cls = cls;
        e->sink[j].local = s;
        e->sink[j].nfields = (unsigned)argc;
        e->sink[j].homeOff = 16u;   /* the real pass's frame says where */
        e->sink[j].entryLive = entryLive;
        e->sinkOf[s] = (uint8_t)(j + 1u);
        e->sinkCount++;
        if (getenv("JAI_JIT_WHY")) {
            fprintf(stderr, "[jit] osr at %u sinks local %u (%s)\n", top, s,
                    cls->name ? cls->name->chars : "?");
        }
    }
#undef SKIP
}

/* A read of sunk local `slot`: an entry that names the sink, in no register. */
bool sinkPushRef(Emit *e, unsigned slot) {
    unsigned j = e->sinkOf[slot] - 1u;
    if (!e->osr || e->inlining) {
        e->whyNot = "a sunk local read outside its loop";
        return false;
    }
    if (e->depth >= JIT_MAX_STACK) {
        e->whyNot = "the operand stack is deeper than the model allows";
        return false;
    }
    unsigned d = e->depth;
    e->stackShape[d]   = e->sink[j].cls->shapeId;
    e->stackClass[d]   = e->sink[j].cls;
    e->stackSeen[d]    = NULL_VAL;
    e->stackLocal[d]   = -1;
    e->stackAscii[d]   = false;
    e->stackNullLit[d] = false;
    e->stackUnit[d]    = false;
    e->stackPinned[d]  = false;
    e->stackObjType[d] = 0;
    e->stackElemDecl[d] = 0;
    e->stackSunk[d]    = (uint8_t)(j + 1u);
    e->stack[e->depth++] = SLOT_VREF;
    return true;
}

/* Field `nameIdx` of sink `sink`: one load from its home. */
bool sinkFieldRead(Emit *e, unsigned sink, const ObjFunction *fn,
                   uint32_t nameIdx, const uint8_t *code, int next, int stop) {
    if (sink >= e->sinkCount) return false;
    if (nameIdx >= (uint32_t)fn->chunk.constants.count) return false;
    Value nv = fn->chunk.constants.data[nameIdx];
    if (!IS_STRING(nv)) return false;
    const FieldInfo *fi = jaiClassFieldInfo(e->sink[sink].cls, AS_STRING(nv));
    if (fi == NULL || fi->isStatic || fi->slot >= e->sink[sink].nfields) {
        e->whyNot = "a sunk instance read for something not one of its fields";
        return false;
    }
    SlotKind kind = e->sink[sink].kind[fi->slot];
    unsigned at = e->sink[sink].homeOff + 8u * fi->slot;
    if (!pushValue(e, kind, 0, NULL)) return false;
    /* A copy, never a borrow: the next construction writes the home, and a
     * borrow still live then (`Vec2(p.y, p.x)`) would read the new value. */
    unsigned fh = e->sink[sink].fpHome[fi->slot];
    if (fh != 0) {
        if (fpWorthLoading(e, code, next, stop)) {
            unsigned idx = e->valueDepth - 1u;
            emit(e, jaiA64FmovDD(fpRegAt(e, idx), fh));
            fpClaim(e, idx);
        } else {
            emit(e, jaiA64FmovXD(pushReg(e) - 1u, fh));
        }
        return true;
    }
    /* Straight to the FP bank when the consumer reads it there, as a heap
     * field read does. */
    if (kind == SLOT_FLOAT && fpWorthLoading(e, code, next, stop)) {
        unsigned idx = e->valueDepth - 1u;
        emit(e, jaiA64LdrD(fpRegAt(e, idx), 31, at));
        fpClaim(e, idx);
        return true;
    }
    emit(e, jaiA64LdrX(pushReg(e) - 1u, 31, at));
    return true;
}

/* OP_CALL at `off` building the object a sunk local is bound to: its
 * arguments go to the fields' homes and the OP_BIND after it is consumed.
 * 1 when handled (the caller skips both instructions), 0 when this call is
 * not one, -1 when it should have been and is not. */
int sinkConstructs(Emit *e, const uint8_t *code, int off) {
    if (e->sinkCount == 0 || e->inlining) return 0;
    if (code[off + 2] != OP_BIND) return 0;
    unsigned slot = jaiReadU16(code + off + 3);
    if (slot > JIT_MAX_SLOTS || e->sinkOf[slot] == 0) return 0;
    unsigned j = e->sinkOf[slot] - 1u;
    /* A function, not a class: its body may end by constructing one
     * (sinkConstructInline, once inlined), and if it does not the bind
     * refuses (sinkBindResult). */
    unsigned argc = code[off + 1];
    if (e->depth < argc + 1u || e->stack[e->depth - argc - 1u] != SLOT_CLASS) {
        return 0;
    }
    if (!storeConstruction(e, j, argc)) {
        e->whyNot = "a sunk local bound to something its plan did not expect";
        return -1;
    }
    return 1;
}

/* OP_INVOKE with a sunk reference among its receiver and arguments: it must
 * be inlined, since the object does not exist to be passed. UNARMED when no
 * reference is involved, so the ordinary arm runs. */
JitArmResult sinkInvoke(Emit *e, ObjFunction *fn, const uint8_t *code,
                        int *offp) {
    int off = *offp;
    if (e->sinkCount == 0) return JIT_ARM_UNARMED;
    unsigned argc = code[off + 4];
    if (e->depth < argc + 1u) return JIT_ARM_UNARMED;
    unsigned ridx = e->depth - argc - 1u;
    bool any = false;
    for (unsigned i = ridx; i < e->depth; i++) {
        if (e->stack[i] == SLOT_VREF) any = true;
    }
    if (!any) return JIT_ARM_UNARMED;
    uint32_t nameIdx = jaiReadU24(code + off + 1);
    ObjClass *rcls = e->stackClass[ridx];
    Value method;
    if ((e->stack[ridx] != SLOT_VREF && e->stack[ridx] != SLOT_INST) ||
        rcls == NULL || nameIdx >= (uint32_t)fn->chunk.constants.count ||
        !IS_STRING(fn->chunk.constants.data[nameIdx]) ||
        !jaiClassFindMethod(rcls, AS_STRING(fn->chunk.constants.data[nameIdx]),
                            &method) ||
        !IS_CLOSURE(method) ||
        !inlineMethodCall(e, fn, AS_CLOSURE(method), argc, (uint32_t)off)) {
        if (e->whyNot == NULL) {
            e->whyNot = "a sunk instance reaching a call that is not inlined";
        }
        e->failed = true;
        return JIT_ARM_REFUSED;
    }
    *offp = off + 7;
    return JIT_ARM_OK;
}

/* Zero each sink's "bound yet" word: a way out before the first bind leaves
 * the local as the interpreter had it. Then unpack each entry-live sink's
 * object: an instance of exactly the planned class whose fields have the
 * planned kinds, or the form bails before it has run anything. Emitted once
 * the frame, the locals and the iterator are in place, so the bail's way out
 * finds them as it expects. */
void sinkEmitEntry(Emit *e) {
    for (unsigned j = 0; j < e->sinkCount; j++) {
        emit(e, jaiA64StrX(31, 31,
                           e->sink[j].homeOff + 8u * e->sink[j].nfields));
    }
    for (unsigned j = 0; j < e->sinkCount; j++) {
        if (!e->sink[j].entryLive) continue;
        unsigned at = e->sink[j].local * 16u;
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SLOTS_REG, at));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, VAL_OBJ));
        branchTo(e, FIXUP_BAIL, true, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, JIT_SLOTS_REG, at + 8u));
        emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(Obj, type)));
        emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, OBJ_INSTANCE));
        branchTo(e, FIXUP_BAIL, true, JAI_A64_NE);
        emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B,
                           (unsigned)offsetof(ObjInstance, klass)));
        emitConst64(e, JIT_SCRATCH_C, (int64_t)(uintptr_t)e->sink[j].cls);
        emit(e, jaiA64SubsX(31, JIT_SCRATCH_A, JIT_SCRATCH_C));
        branchTo(e, FIXUP_BAIL, true, JAI_A64_NE);
        for (unsigned f = 0; f < e->sink[j].nfields; f++) {
            unsigned fo = (unsigned)offsetof(ObjInstance, fields) +
                          f * (unsigned)sizeof(Value);
            unsigned tag = e->sink[j].kind[f] == SLOT_FLOAT ? VAL_FLOAT
                                                            : VAL_INT;
            emit(e, jaiA64LdrW(JIT_SCRATCH_A, JIT_SCRATCH_B, fo));
            emit(e, jaiA64SubsXImm(31, JIT_SCRATCH_A, tag));
            branchTo(e, FIXUP_BAIL, true, JAI_A64_NE);
            if (e->sink[j].fpHome[f] != 0) {
                emit(e, jaiA64LdrD(e->sink[j].fpHome[f], JIT_SCRATCH_B,
                                   fo + 8u));
                continue;
            }
            emit(e, jaiA64LdrX(JIT_SCRATCH_A, JIT_SCRATCH_B, fo + 8u));
            emit(e, jaiA64StrX(JIT_SCRATCH_A, 31,
                               e->sink[j].homeOff + 8u * f));
        }
    }
}

bool sinkHasEntryLive(const Emit *e) {
    for (unsigned j = 0; j < e->sinkCount; j++) {
        if (e->sink[j].entryLive) return true;
    }
    return false;
}

/* The sunk local the instruction after the call at `callOff` binds its
 * result to, + 1, or 0: what lets an inlined body's closing construction be
 * sunk (sinkConstructInline). */
unsigned sinkBindAfter(const Emit *e, const ObjFunction *caller,
                       uint32_t callOff) {
    if (e->sinkCount == 0 || e->inlining) return 0;
    const Chunk *c = &caller->chunk;
    if (callOff >= (uint32_t)c->count) return 0;
    int len = instructionLength(c, (int)callOff);
    if (len <= 0 || callOff + (uint32_t)len + 3u > (uint32_t)c->count) return 0;
    const uint8_t *at = c->code + callOff + len;
    if (at[0] != OP_BIND) return 0;
    unsigned slot = jaiReadU16(at + 1);
    if (slot > JIT_MAX_SLOTS || e->sinkOf[slot] == 0) return 0;
    return slot + 1u;
}

/* The pushed reference to sink `j`, without sinkPushRef's checks: an inlined
 * body's result, which only the caller's bind consumes. */
static bool pushRef(Emit *e, unsigned j) {
    if (e->depth >= JIT_MAX_STACK) return false;
    unsigned d = e->depth;
    e->stackShape[d]   = e->sink[j].cls->shapeId;
    e->stackClass[d]   = e->sink[j].cls;
    e->stackSeen[d]    = NULL_VAL;
    e->stackLocal[d]   = -1;
    e->stackAscii[d]   = false;
    e->stackNullLit[d] = false;
    e->stackUnit[d]    = false;
    e->stackPinned[d]  = false;
    e->stackObjType[d] = 0;
    e->stackElemDecl[d] = 0;
    e->stackSunk[d]    = (uint8_t)(j + 1u);
    e->stack[e->depth++] = SLOT_VREF;
    return true;
}

/* Stores the top `argc` entries into sink j's homes and drops them and the
 * class beneath; sets the bound word. */
static bool storeConstruction(Emit *e, unsigned j, unsigned argc) {
    unsigned home = e->sink[j].homeOff;
    if (argc != e->sink[j].nfields || e->depth < argc + 1u) return false;
    unsigned cidx = e->depth - argc - 1u;
    if (e->stack[cidx] != SLOT_CLASS || e->stackClass[cidx] != e->sink[j].cls) {
        return false;
    }
    for (unsigned i = 0; i < argc; i++) {
        if (e->stack[cidx + 1u + i] !=
            e->sink[j].kind[e->sink[j].argSlot[i]]) {
            return false;
        }
    }
    /* Every argument is in a register of its own before any home is
     * written, since an argument may be a copy of a home (`Vec2(p.y, p.x)`):
     * the copies are made at the reads, so the order of the writes below
     * cannot matter. */
    for (unsigned n = argc; n-- > 0;) {
        unsigned at = home + 8u * e->sink[j].argSlot[n];
        unsigned fh = e->sink[j].fpHome[e->sink[j].argSlot[n]];
        unsigned idx = e->valueDepth - 1u;
        if (e->stack[e->depth - 1u] == SLOT_FLOAT && !e->fpOff &&
            (e->fpLive & (1u << idx))) {
            unsigned held = fpHeldIn(e, idx);
            unsigned r2;
            SlotKind k2;
            if (!popValueRaw(e, &r2, &k2)) return false;
            emit(e, fh != 0 ? jaiA64FmovDD(fh, held)
                            : jaiA64StrD(held, 31, at));
        } else {
            unsigned r;
            if (!popValue(e, &r, NULL)) return false;
            emit(e, fh != 0 ? jaiA64FmovDX(fh, r) : jaiA64StrX(r, 31, at));
        }
    }
    e->depth--;                                    /* the class */
    emit(e, jaiA64MovzX(JIT_SCRATCH_A, 1, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_A, 31, home + 8u * e->sink[j].nfields));
    return true;
}

/* OP_TAIL_CALL closing an inlined body whose result the caller binds to a
 * sunk local: the construction goes to the homes and the body returns a
 * reference. 1 when done, -1 when it should have been and is not. */
int sinkConstructInline(Emit *e, unsigned argc) {
    if (!e->inlining || e->inlSinkBind == 0) return 0;
    unsigned slot = e->inlSinkBind - 1u;
    unsigned j = e->sinkOf[slot] - 1u;
    if (!storeConstruction(e, j, argc) || !pushRef(e, j)) {
        e->whyNot = "an inlined construction a sunk local cannot take";
        return -1;
    }
    return 1;
}

/* OP_BIND of sunk local `slot` after an inlined call: its result must be the
 * reference sinkConstructInline left, which is dropped -- the homes already
 * hold the fields. */
bool sinkBindResult(Emit *e, unsigned slot) {
    unsigned j = e->sinkOf[slot] - 1u;
    if (e->depth == 0 || e->stack[e->depth - 1u] != SLOT_VREF ||
        e->stackSunk[e->depth - 1u] != j + 1u) {
        e->whyNot = "a sunk local bound to something its plan did not expect";
        return false;
    }
    e->depth--;
    return true;
}

/* Part of OSR_SYNC_ITER, so on every way out: the sinks' fields into gDeopt
 * for jaiJitEnterOsr to build the objects from. JIT_SCRATCH_A and _B only. */
void sinkEmitSync(Emit *e) {
    if (e->sinkCount == 0) return;
    emitConst64(e, JIT_SCRATCH_A, (int64_t)(uintptr_t)&gDeopt);
    emit(e, jaiA64MovzX(JIT_SCRATCH_B, e->sinkCount, 0));
    emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                       (unsigned)offsetof(JitDeoptRecord, sinkCount)));
    for (unsigned j = 0; j < e->sinkCount; j++) {
        unsigned base = (unsigned)offsetof(JitDeoptRecord, sinks) +
                        j * (unsigned)sizeof gDeopt.sinks[0];
        unsigned home = e->sink[j].homeOff;
        unsigned n = e->sink[j].nfields;
        emit(e, jaiA64MovzX(JIT_SCRATCH_B, e->sink[j].local, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                           base + (unsigned)offsetof(__typeof__(gDeopt.sinks[0]), local)));
        emitConst64(e, JIT_SCRATCH_B, (int64_t)(uintptr_t)e->sink[j].cls);
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                           base + (unsigned)offsetof(__typeof__(gDeopt.sinks[0]), cls)));
        emit(e, jaiA64MovzX(JIT_SCRATCH_B, n, 0));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                           base + (unsigned)offsetof(__typeof__(gDeopt.sinks[0]), nfields)));
        emit(e, jaiA64LdrX(JIT_SCRATCH_B, 31, home + 8u * n));
        emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A,
                           base + (unsigned)offsetof(__typeof__(gDeopt.sinks[0]), live)));
        for (unsigned f = 0; f < n; f++) {
            unsigned at = base +
                          (unsigned)offsetof(__typeof__(gDeopt.sinks[0]), fields) +
                          f * (unsigned)sizeof(Value);
            /* Pinned `refuses` in tests/vm/jit_kind_tag.manifest: planSinks
             * admits only int and float fields, storeConstruction only an
             * argument of exactly that kind, and the entry check only an
             * object whose tags say so, so no other kind reaches this. */
            unsigned tag = e->sink[j].kind[f] == SLOT_FLOAT ? VAL_FLOAT : VAL_INT;
            emit(e, jaiA64MovzX(JIT_SCRATCH_B, tag, 0));
            emit(e, jaiA64StrW(JIT_SCRATCH_B, JIT_SCRATCH_A, at));
            if (e->sink[j].fpHome[f] != 0) {
                emit(e, jaiA64StrD(e->sink[j].fpHome[f], JIT_SCRATCH_A,
                                   at + 8u));
                continue;
            }
            emit(e, jaiA64LdrX(JIT_SCRATCH_B, 31, home + 8u * f));
            emit(e, jaiA64StrX(JIT_SCRATCH_B, JIT_SCRATCH_A, at + 8u));
        }
    }
}

#endif
