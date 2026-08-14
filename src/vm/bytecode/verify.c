/* verify.c -- the bytecode verifier: operands in range, jumps landing on
 * instruction boundaries, one stack depth per instruction whichever path
 * reaches it. The four helpers below must agree with chunk.h. */

#include "vm/bytecode/verify.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "vm/bytecode/chunk.h"
#include "vm/object/object.h"
#include "vm/vm.h"

/* Keep in sync with chunk.h. */
int jaiOpBranchOperandAt(uint8_t op) {
    switch (op) {
    case OP_JUMP:
    case OP_JUMP_IF_FALSE:
    case OP_JUMP_IF_TRUE:
    case OP_JUMP_IF_FALSE_KEEP:
    case OP_JUMP_IF_TRUE_KEEP:
    case OP_JUMP_IF_NULL:
    case OP_LOOP:
    case OP_FOR_ITER:
    case OP_FOR_ITER_BIND:
    case OP_FOR_ITER_PAIR:
    /* Branches past the ordinary `INVOKE items; GET_ITER` when its receiver
     * turns out to be a dict. Missing from this list when the opcode landed,
     * which left the verifier unable to see the edge at all. */
    case OP_GET_ITER_ITEMS:
    case OP_FOR_RANGE_BIND:
    case OP_PUSH_FINALLY:
    case OP_PUSH_HANDLER:
        return 0;
    case OP_JUMP_IF_CMP_FALSE:
        return 1;
    case OP_JUMP_IF_CMP_LOCAL_K:
        return 6;   /* after the u8 comparison, the u16 slot and the u24 K */
    case OP_MATCH_SEQ:
        return 2;
    case OP_MATCH_CONST:
    case OP_MATCH_TYPE:
    case OP_MATCH_FIELDS:
        return 3;
    case OP_MATCH_RANGE:
        return 7;
    default:
        return -1;
    }
}

/* True when `target` is a control-flow edge taken by the instruction itself.
 * PUSH_HANDLER/PUSH_FINALLY only *register* an address; no stack state flows
 * along that edge. */
static bool opBranchIsEdge(uint8_t op) {
    return jaiOpBranchOperandAt(op) >= 0 && op != OP_PUSH_HANDLER &&
           op != OP_PUSH_FINALLY;
}


bool jaiOpFallsThrough(uint8_t op) {
    switch (op) {
    case OP_JUMP:
    case OP_LOOP:
    case OP_RETURN:
    case OP_RETURN_NULL:
    case OP_POP_RETURN_NULL:
    case OP_THROW:
    case OP_RERAISE:
    case OP_HALT:
        return false;
    default:
        /* OP_TAIL_CALL normally ends the frame, but the emitter is free to
         * follow it with a RETURN, so keep the edge conservatively. */
        return true;
    }
}

/* Byte offsets of the u16 local-slot operands of `op`, if any. */
static int slotOperands(uint8_t op, int *out) {
    switch (op) {
    case OP_GET_LOCAL:
    case OP_SET_LOCAL:
    case OP_BIND:
    case OP_CLOSE_UPVALUE:
    case OP_ADD_INT_CONST:
    case OP_SUB_INT_CONST:
    case OP_INC_LOCAL:
    case OP_CMP_LOCAL_CONST_LT:
    case OP_GET_FIELD_LOCAL:
        out[0] = 0;
        return 1;
    case OP_FOR_ITER_BIND:
        out[0] = 2;   /* the slot follows the i16 jump */
        return 1;
    case OP_FOR_ITER_PAIR:
        out[0] = 2;   /* both slots follow the i16 jump */
        out[1] = 4;
        return 2;
    case OP_ITER_RANGE:
        out[0] = 1;   /* both slots follow the u8 inclusive flag */
        out[1] = 3;
        return 2;
    case OP_FOR_RANGE_BIND:
        out[0] = 2;   /* loop variable, counter and end follow the i16 jump */
        out[1] = 4;
        out[2] = 6;
        return 3;
    case OP_JUMP_IF_CMP_LOCAL_K:
        out[0] = 1;   /* the slot follows the u8 comparison */
        return 1;
    case OP_GET_LOCAL2:
    case OP_ADD_LOCALS:
        out[0] = 0;
        out[1] = 2;
        return 2;
    default:
        return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Verification                                                         */
/* ------------------------------------------------------------------ */

static bool verifyFail(char *buf, size_t size, const char *fmt, ...) JAI_PRINTF(3, 4);

static bool verifyFail(char *buf, size_t size, const char *fmt, ...) {
    if (buf != NULL && size > 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, size, fmt, args);
        va_end(args);
    }
    return false;
}

/* Constant-pool operands of `op`, as byte offsets into its operand run. */
static int constOperands(uint8_t op, int *out) {
    switch (op) {
    case OP_CONST:
    case OP_DEF_GLOBAL:
    case OP_GET_GLOBAL:
    case OP_SET_GLOBAL:
    case OP_CLASS:
    case OP_IMPL_TRAIT:
    case OP_METHOD:
    case OP_FIELD_DEF:
    case OP_GET_FIELD:
    case OP_SET_FIELD:
    case OP_GET_SUPER:
    case OP_NEW:
    case OP_ENUM_NEW:
    case OP_IS_INSTANCE:
    case OP_PUSH_DEFER:
    case OP_MATCH_EXC:
    case OP_MATCH_CONST:
    case OP_MATCH_TYPE:
    case OP_MATCH_FIELDS:
    case OP_IMPORT:
    case OP_IMPORT_FROM:
    case OP_EXPORT:
    case OP_ASSERT_FAIL:
    case OP_TYPE_GUARD:
    case OP_INVOKE:
    case OP_SUPER_INVOKE:
    case OP_CLOSURE:
        out[0] = 0;
        return 1;
    case OP_GET_MODULE:
    case OP_MATCH_RANGE:
        out[0] = 0;
        out[1] = 3;
        return 2;
    case OP_CALL_KW:
        out[0] = 1;
        return 1;
    case OP_JUMP_IF_CMP_LOCAL_K:   /* K follows the u8 comparison and u16 slot */
        out[0] = 3;
        return 1;
    case OP_PUSH_HANDLER:
    case OP_GET_FIELD_LOCAL:   /* the name follows the u16 slot */
        out[0] = 2;
        return 1;
    case OP_FORMAT:            /* the name follows the count and the mask */
        out[0] = 4;
        return 1;
    default:
        return 0;
    }
}

#define cacheOperandAt(op) jaiOpCacheOperand((OpCode)(op))

/* Number of entries in the tuple constant `k`, or -1 when it is not a tuple. */
static int tupleArity(const Chunk *chunk, uint32_t k) {
    if (k >= (uint32_t)chunk->constants.count) return -1;
    Value v = chunk->constants.data[k];
    if (!IS_TUPLE(v)) return -1;
    return (int)AS_TUPLE(v)->count;
}

static int popcount3(unsigned flags) {
    int n = 0;
    for (unsigned bit = 1; bit <= 4; bit <<= 1) {
        if (flags & bit) n++;
    }
    return n;
}

typedef struct {
    int  pops, pushes;
    bool known;
} Effect;

/* Pops and pushes of the instruction at `offset` on its fall-through path.
 * Where only the net effect is tabulated in chunk.c, pops is a lower bound,
 * to keep the depth check free of false positives. */
static Effect fallEffect(const Chunk *chunk, int offset, uint8_t op,
                         const uint8_t *a) {
    Effect e = {0, 0, true};
    (void)offset;

    switch (op) {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_FLOORDIV:
    case OP_MOD: case OP_POW: case OP_ADD_WRAP: case OP_SUB_WRAP:
    case OP_MUL_WRAP: case OP_BAND: case OP_BOR: case OP_BXOR: case OP_SHL:
    case OP_SHR: case OP_EQ: case OP_NE: case OP_LT: case OP_LE: case OP_GT:
    case OP_GE: case OP_IS: case OP_IS_NOT: case OP_IN: case OP_NOT_IN:
    case OP_CONCAT: case OP_GET_INDEX:
        e.pops = 2;
        e.pushes = 1;
        return e;
    /* OP_ENUM_TAG is not here: like OP_ENUM_FIELD it peeks and pushes, so the
     * enum stays underneath and the table's +1 is the truth. */
    case OP_NEG: case OP_POS: case OP_BNOT: case OP_NOT:
    case OP_GET_ITER: case OP_TYPE_GUARD: case OP_IS_INSTANCE:
    case OP_TO_FLOAT:
        e.pops = 1;
        e.pushes = 1;
        return e;
    case OP_SET_INDEX:
        e.pops = 3;
        return e;
    case OP_POPN:
        e.pops = a[0];
        return e;
    case OP_CALL:
    case OP_CALL_SPREAD:
    case OP_TAIL_CALL:
        e.pops = (int)a[0] + 1;
        e.pushes = 1;
        return e;
    case OP_CALL_KW: {
        int kw = tupleArity(chunk, jaiReadU24(a + 1));
        if (kw < 0) { e.known = false; return e; }
        e.pops = (int)a[0] + kw + 1;
        e.pushes = 1;
        return e;
    }
    case OP_INVOKE:
        e.pops = (int)a[3] + 1;   /* receiver plus arguments */
        e.pushes = 1;
        return e;
    case OP_SUPER_INVOKE:
        e.pops = (int)a[3] + 1;
        e.pushes = 1;
        return e;
    case OP_NEW:
        e.pops = a[3];
        e.pushes = 1;
        return e;
    case OP_ENUM_NEW:
        e.pops = a[4];
        e.pushes = 1;
        return e;
    case OP_BUILD_LIST:
    case OP_BUILD_SET:
    case OP_BUILD_TUPLE:
        e.pops = jaiReadU16(a);
        e.pushes = 1;
        return e;
    case OP_BUILD_DICT:
        e.pops = 2 * (int)jaiReadU16(a);
        e.pushes = 1;
        return e;
    case OP_GET_SLICE:
        e.pops = 1 + popcount3(a[0]);
        e.pushes = 1;
        return e;
    case OP_SET_SLICE:
        e.pops = 2 + popcount3(a[0]);
        return e;
    case OP_UNPACK:
        e.pops = 1;
        e.pushes = a[0];
        return e;
    case OP_FOR_ITER:
        e.pushes = 1;   /* the iterator stays under the produced value */
        return e;
    case OP_FORMAT:
        e.pops = a[0];
        e.pushes = 1;
        return e;
    case OP_MATCH_FIELDS: {
        int fields = tupleArity(chunk, jaiReadU24(a));
        if (fields < 0) { e.known = false; return e; }
        e.pushes = fields;
        return e;
    }
    default:
        break;
    }

    int net = jaiOpStackEffect((OpCode)op);
    if (net == INT32_MIN) {
        e.known = false;
        return e;
    }
    e.pops = net < 0 ? -net : 0;
    e.pushes = e.pops + net;
    return e;
}

/* `depthOut`, when given, receives pass 4's table on success -- see
 * jaiChunkStackDepths. Nothing is written unless the whole chunk verified. */
static bool verifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize,
                        int *depthOut) {
    if (errBuf != NULL && errBufSize > 0) errBuf[0] = '\0';
    if (fn == NULL) return verifyFail(errBuf, errBufSize, "function is null");

    const Chunk *chunk = &fn->chunk;
    if (chunk->code == NULL || chunk->count <= 0) {
        return verifyFail(errBuf, errBufSize, "chunk contains no code");
    }

    int n = chunk->count;
    bool ok = true;
    bool *boundary = JAI_ALLOC_ZEROED(bool, n + 1);
    int *depth = JAI_ALLOC(int, n + 1);
    int *work = JAI_ALLOC(int, n + 1);
    /* An offset whose depth came from an imprecise seed, or from one by
     * following edges. Pass 4 pins an exception handler and a default thunk at
     * 0 because nothing tells it what the unwinder will have left there, and
     * that answer is good enough to verify against but is NOT the truth. Only
     * built for a caller that asked for the table, which is the only one that
     * has to tell the two apart. */
    bool *approx = depthOut != NULL ? JAI_ALLOC_ZEROED(bool, n + 1) : NULL;
    int workCount = 0;
    for (int i = 0; i <= n; i++) depth[i] = -1;

#define VFAIL(...)                                                             \
    do {                                                                       \
        ok = verifyFail(errBuf, errBufSize, __VA_ARGS__);                      \
        goto done;                                                             \
    } while (0)

    /* Pass 1: decode linearly, checking operand widths and operand ranges. */
    for (int offset = 0; offset < n;) {
        boundary[offset] = true;
        uint8_t op = chunk->code[offset];
        if (op >= OP_COUNT) {
            VFAIL("offset %d: opcode 0x%02x is not a valid opcode (OP_COUNT = %d)",
                  offset, (unsigned)op, (int)OP_COUNT);
        }

        int operands = jaiOpOperandSize((OpCode)op);
        if (op == OP_CLOSURE) {
            if (offset + 4 > n) {
                VFAIL("offset %d: OP_CLOSURE constant index runs past the end "
                      "of the code", offset);
            }
            uint32_t k = jaiReadU24(chunk->code + offset + 1);
            if (k >= (uint32_t)chunk->constants.count) {
                VFAIL("offset %d: OP_CLOSURE constant index %u is out of range "
                      "(%d constants)", offset, (unsigned)k,
                      chunk->constants.count);
            }
            if (!IS_FUNCTION(chunk->constants.data[k])) {
                VFAIL("offset %d: OP_CLOSURE constant %u is not a function",
                      offset, (unsigned)k);
            }
            operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
        }
        if (operands < 0) {
            VFAIL("offset %d: %s has no known operand width", offset,
                  jaiOpName((OpCode)op));
        }
        if (offset + 1 + operands > n) {
            VFAIL("offset %d: %s reads %d operand byte%s past the end of the "
                  "code (%d bytes)", offset, jaiOpName((OpCode)op),
                  offset + 1 + operands - n,
                  offset + 1 + operands - n == 1 ? "" : "s", n);
        }

        const uint8_t *a = chunk->code + offset + 1;

        int constAt[2];
        int constCount = constOperands(op, constAt);
        for (int k = 0; k < constCount; k++) {
            uint32_t index = jaiReadU24(a + constAt[k]);
            if (index >= (uint32_t)chunk->constants.count) {
                VFAIL("offset %d: %s constant index %u is out of range "
                      "(%d constants)", offset, jaiOpName((OpCode)op),
                      (unsigned)index, chunk->constants.count);
            }
        }

        int cacheAt = cacheOperandAt(op);
        if (cacheAt >= 0) {
            uint16_t cache = jaiReadU16(a + cacheAt);
            if (cache >= (uint16_t)chunk->cacheCount) {
                VFAIL("offset %d: %s inline-cache index %u is out of range "
                      "(%d caches)", offset, jaiOpName((OpCode)op),
                      (unsigned)cache, chunk->cacheCount);
            }
        }

        int slotAt[3];
        int slotCount = slotOperands(op, slotAt);
        for (int k = 0; k < slotCount; k++) {
            uint16_t slot = jaiReadU16(a + slotAt[k]);
            if (slot >= fn->maxSlots) {
                VFAIL("offset %d: %s uses local slot %u but the frame has %u",
                      offset, jaiOpName((OpCode)op), (unsigned)slot,
                      (unsigned)fn->maxSlots);
            }
        }

        if (op == OP_GET_UPVALUE || op == OP_SET_UPVALUE) {
            if (a[0] >= fn->upvalueCount) {
                VFAIL("offset %d: %s uses upvalue %u but the closure has %u",
                      offset, jaiOpName((OpCode)op), (unsigned)a[0],
                      (unsigned)fn->upvalueCount);
            }
        }

        if (op == OP_CLOSURE) {
            int upvalues = (operands - 3) / 3;
            for (int u = 0; u < upvalues; u++) {
                const uint8_t *desc = a + 3 + u * 3;
                uint16_t index = jaiReadU16(desc + 1);
                if (desc[0] != 0 && index >= fn->maxSlots) {
                    VFAIL("offset %d: OP_CLOSURE captures local slot %u but the "
                          "frame has %u", offset, (unsigned)index,
                          (unsigned)fn->maxSlots);
                }
                if (desc[0] == 0 && index >= fn->upvalueCount) {
                    VFAIL("offset %d: OP_CLOSURE captures upvalue %u but the "
                          "closure has %u", offset, (unsigned)index,
                          (unsigned)fn->upvalueCount);
                }
            }
        }
        offset += 1 + operands;
    }
    boundary[n] = true;

    /* Pass 2: every code address lands on an instruction boundary. */
    for (int offset = 0; offset < n;) {
        uint8_t op = chunk->code[offset];
        int operands = jaiOpOperandSize((OpCode)op);
        if (op == OP_CLOSURE) {
            uint32_t k = jaiReadU24(chunk->code + offset + 1);
            operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
        }
        int at = jaiOpBranchOperandAt(op);
        if (at >= 0) {
            int16_t rel = jaiReadI16(chunk->code + offset + 1 + at);
            long target = (long)offset + 1 + operands + rel;
            if (target < 0 || target >= n) {
                VFAIL("offset %d: %s targets %ld, outside the code (%d bytes)",
                      offset, jaiOpName((OpCode)op), target, n);
            }
            if (!boundary[target]) {
                VFAIL("offset %d: %s targets %ld, which is not an instruction "
                      "boundary", offset, jaiOpName((OpCode)op), target);
            }
        }
        offset += 1 + operands;
    }

    /* Pass 3: exception table and default-thunk entry points. */
    for (int e = 0; e < (int)fn->exceptionCount; e++) {
        const ExceptionEntry *entry = fn->exceptions;
        if (entry == NULL) {
            VFAIL("exception table has %u entries but no storage",
                  (unsigned)fn->exceptionCount);
        }
        entry += e;
        if (entry->start > (uint32_t)n || !boundary[entry->start]) {
            VFAIL("exception entry %d: start %u is not an instruction boundary",
                  e, (unsigned)entry->start);
        }
        if (entry->end > (uint32_t)n || !boundary[entry->end]) {
            VFAIL("exception entry %d: end %u is not an instruction boundary",
                  e, (unsigned)entry->end);
        }
        if (entry->start >= entry->end) {
            VFAIL("exception entry %d: empty protected region [%u, %u)", e,
                  (unsigned)entry->start, (unsigned)entry->end);
        }
        if (entry->handler >= (uint32_t)n || !boundary[entry->handler]) {
            VFAIL("exception entry %d: handler %u is not an instruction "
                  "boundary", e, (unsigned)entry->handler);
        }
        if (entry->typeConst != UINT32_MAX &&
            entry->typeConst >= (uint32_t)chunk->constants.count) {
            VFAIL("exception entry %d: type constant %u is out of range "
                  "(%d constants)", e, (unsigned)entry->typeConst,
                  chunk->constants.count);
        }
    }
    for (int d = 0; d < (int)fn->defaultCount && fn->defaultOffsets != NULL; d++) {
        uint32_t at = fn->defaultOffsets[d];
        if (at >= (uint32_t)n || !boundary[at]) {
            VFAIL("default thunk %d: offset %u is not an instruction boundary",
                  d, (unsigned)at);
        }
    }

    /* Pass 4: stack depth. Every path into an offset must agree on the depth
     * there, and no path may pop more than it has. */
    depth[0] = 0;
    work[workCount++] = 0;

    /* Repeats until a round adds nothing, since a seeded block can contain
     * further entry points. Precise seeds (dynamic handlers) go first, so an
     * exception-table handler/default thunk (both imprecise, depth 0) only
     * seed once nothing precise is left. */
    for (;;) {
        while (workCount > 0) {
            int offset = work[--workCount];
            int here = depth[offset];
            uint8_t op = chunk->code[offset];
            const uint8_t *a = chunk->code + offset + 1;

            int operands = jaiOpOperandSize((OpCode)op);
            if (op == OP_CLOSURE) {
                uint32_t k = jaiReadU24(a);
                operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
            }
            int next = offset + 1 + operands;

            Effect e = fallEffect(chunk, offset, op, a);
            if (!e.known) continue;   /* unmodelled: stop, never guess */
            if (here < e.pops) {
                VFAIL("offset %d: %s pops %d value%s from a stack of depth %d",
                      offset, jaiOpName((OpCode)op), e.pops,
                      e.pops == 1 ? "" : "s", here);
            }
            int after = here - e.pops + e.pushes;

            int fallTo = -1, fallDepth = after;
            int jumpTo = -1, jumpDepth = after;

            if (jaiOpFallsThrough(op)) {
                /* OP_TAIL_CALL is modelled as falling through so that a
                 * following RETURN is verified, but it is also a legitimate
                 * last instruction. */
                if (next < n) {
                    fallTo = next;
                } else if (op != OP_TAIL_CALL) {
                    VFAIL("offset %d: %s is the last instruction and execution "
                          "falls off the end of the code", offset,
                          jaiOpName((OpCode)op));
                }
            }

            int at = jaiOpBranchOperandAt(op);
            if (at >= 0 && opBranchIsEdge(op)) {
                jumpTo = (int)((long)offset + 1 + operands +
                               jaiReadI16(chunk->code + offset + 1 + at));
                switch (op) {
                case OP_JUMP:
                case OP_LOOP:
                    jumpDepth = here;
                    break;
                case OP_JUMP_IF_FALSE:
                case OP_JUMP_IF_TRUE:
                    jumpDepth = here - 1;
                    break;
                case OP_JUMP_IF_CMP_FALSE:
                    /* Both operands go, on either path. */
                    jumpDepth = here - 2;
                    break;
                case OP_FOR_ITER:
                case OP_FOR_ITER_BIND:
                case OP_FOR_ITER_PAIR:
                    /* Exhaustion pops the iterator and leaves the loop. */
                    jumpDepth = here - 1;
                    break;
                case OP_FOR_RANGE_BIND:
                    /* A range loop keeps nothing on the operand stack, so
                     * unlike its three neighbours above there is nothing for
                     * exhaustion to pop. */
                    jumpDepth = here;
                    break;
                case OP_MATCH_FIELDS:
                    jumpDepth = here;   /* nothing destructured on no-match */
                    break;
                default:
                    jumpDepth = here;   /* the *_KEEP forms and MATCH_* peek */
                    break;
                }
                if (jumpDepth < 0) {
                    VFAIL("offset %d: %s leaves a negative stack depth on its "
                          "taken path", offset, jaiOpName((OpCode)op));
                }
            }

            for (int edge = 0; edge < 2; edge++) {
                int to = edge == 0 ? fallTo : jumpTo;
                int d = edge == 0 ? fallDepth : jumpDepth;
                if (to < 0) continue;
                if (d < 0) {
                    VFAIL("offset %d: stack depth %d is negative after %s",
                          offset, d, jaiOpName((OpCode)op));
                }
                if (depth[to] < 0) {
                    depth[to] = d;
                    if (approx != NULL) approx[to] = approx[offset];
                    work[workCount++] = to;
                } else if (depth[to] != d) {
                    VFAIL("offset %d: stack depth %d here disagrees with depth "
                          "%d reached by another path (from %d, %s)", to, d,
                          depth[to], offset, jaiOpName((OpCode)op));
                }
            }
        }

        /* A dynamic handler's target runs at the depth of its own PUSH (the
         * unwinder restores that), not at 0 like an exception-table handler. */
        int added = 0;
        for (int offset = 0; offset < n;) {
            uint8_t op = chunk->code[offset];
            int operands = jaiOpOperandSize((OpCode)op);
            if (op == OP_CLOSURE) {
                uint32_t k = jaiReadU24(chunk->code + offset + 1);
                operands = 3 + 3 *
                    (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
            }
            if ((op == OP_PUSH_FINALLY || op == OP_PUSH_HANDLER) &&
                depth[offset] >= 0) {
                int at = jaiOpBranchOperandAt(op);
                int to = (int)((long)offset + 1 + operands +
                               jaiReadI16(chunk->code + offset + 1 + at));
                if (depth[to] < 0) {
                    depth[to] = depth[offset];
                    if (approx != NULL) approx[to] = approx[offset];
                    work[workCount++] = to;
                    added++;
                }
            }
            offset += 1 + operands;
        }
        if (added > 0) continue;

        /* Exactly one imprecise seed per round: seeding them all at once can
         * pin a nested region's handler at depth 0 before the precise pass
         * reaches it -- e.g. a `finally` inside a loop, whose handler is
         * really at depth 1 (under the iterator). */
        for (int e = 0; e < (int)fn->exceptionCount && added == 0; e++) {
            uint32_t handler = fn->exceptions[e].handler;
            if (depth[handler] < 0) {
                depth[handler] = 0;
                if (approx != NULL) approx[handler] = true;
                work[workCount++] = (int)handler;
                added++;
            }
        }
        for (int d = 0; d < (int)fn->defaultCount && added == 0 &&
                        fn->defaultOffsets != NULL; d++) {
            uint32_t at = fn->defaultOffsets[d];
            if (depth[at] < 0) {
                depth[at] = 0;
                if (approx != NULL) approx[at] = true;
                work[workCount++] = (int)at;
                added++;
            }
        }
        if (added == 0) break;
    }
#undef VFAIL

done:
    if (ok && depthOut != NULL) {
        for (int i = 0; i <= n; i++) {
            depthOut[i] = approx[i] ? -1 : depth[i];
        }
    }
    JAI_FREE_ARRAY(bool, boundary, n + 1);
    JAI_FREE_ARRAY(int, depth, n + 1);
    JAI_FREE_ARRAY(int, work, n + 1);
    if (approx != NULL) JAI_FREE_ARRAY(bool, approx, n + 1);
    return ok;
}

bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize) {
    return verifyChunk(fn, errBuf, errBufSize, NULL);
}

bool jaiChunkStackDepths(const ObjFunction *fn, int *out) {
    char err[8];
    return verifyChunk(fn, err, sizeof err, out);
}
