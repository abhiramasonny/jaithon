/* verify_chunk.c — proves jaiVerifyChunk rejects malformed bytecode.
 *
 * Every case compiles a real source snippet, checks the chunk verifies, then
 * corrupts exactly one thing and checks the verifier names it. Hand-built
 * chunks would prove less: the point is that a chunk the emitter could
 * plausibly produce after a bad rewrite is caught, not that a synthetic one is.
 *
 * Built by `make verify-test`; not part of the shipped binary.
 */
#include "common/common.h"
#include "common/diag.h"
#include "runtime/runtime.h"
#include "vm/bytecode/chunk.h"
#include "vm/object/object.h"
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gFailures = 0;
static int gChecks = 0;

/* Compile `source` as a module body. The module and the body stay rooted for
 * the rest of the process; a test binary that never collects needs no more. */
static ObjFunction *compile(const char *source) {
    static int counter = 0;
    char name[64];
    snprintf(name, sizeof name, "vt%d", counter++);

    ObjModule *module = jaiModuleNew(jaiStringInternC(name),
                                     jaiStringInternC(name));
    jaiPushRoot(OBJ_VAL(module));

    CodegenOptions opts = jaiCodegenDefaults();
    ObjFunction *fn = jaiCompileSource(source, strlen(source), name, module,
                                       &opts);
    if (fn == NULL) {
        fprintf(stderr, "FATAL: could not compile:\n%s\n", source);
        exit(2);
    }
    /* Warnings are the front end's business, not this test's output. */
    jaiDiagReset(&gDiags);
    jaiPushRoot(OBJ_VAL(fn));
    return fn;
}

/* A function in `body`'s constant pool, for tests that need a chunk with
 * locals or an exception table of its own. `name` of NULL takes the first one,
 * which is how an anonymous `fn` literal has to be reached. */
static ObjFunction *nestedFunction(ObjFunction *body, const char *name) {
    for (int i = 0; i < body->chunk.constants.count; i++) {
        Value v = body->chunk.constants.data[i];
        if (!IS_FUNCTION(v)) continue;
        ObjFunction *fn = AS_FUNCTION(v);
        if (name == NULL) return fn;
        if (fn->name != NULL && strcmp(fn->name->chars, name) == 0) return fn;
    }
    fprintf(stderr, "FATAL: no nested function named %s\n",
            name != NULL ? name : "(any)");
    exit(2);
}

static int instrWidth(const Chunk *chunk, int offset) {
    uint8_t op = chunk->code[offset];
    int operands = jaiOpOperandSize((OpCode)op);
    if (op == OP_CLOSURE) {
        uint32_t k = jaiReadU24(chunk->code + offset + 1);
        operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
    }
    if (operands < 0) {
        fprintf(stderr, "FATAL: undecodable opcode %s\n", jaiOpName((OpCode)op));
        exit(2);
    }
    return 1 + operands;
}

static int findOp(const Chunk *chunk, uint8_t op, int from) {
    for (int offset = from; offset < chunk->count; offset += instrWidth(chunk, offset)) {
        if (chunk->code[offset] == op) return offset;
    }
    return -1;
}

/* The conditional branch an `if` compiles to, wherever the optimiser left it.
 * `-O2` fuses the comparison into the jump, so the shape these tests corrupt is
 * OP_JUMP_IF_CMP_FALSE and its i16 lives one byte further in; looking only for
 * OP_JUMP_IF_FALSE made three cases silently skip themselves. *`operandAt` is
 * set to the byte index of the displacement inside the operand run. */
static int findConditionalBranch(const Chunk *chunk, int from, int *operandAt) {
    for (int offset = from; offset < chunk->count; offset += instrWidth(chunk, offset)) {
        switch (chunk->code[offset]) {
        case OP_JUMP_IF_FALSE:     *operandAt = 0; return offset;
        case OP_JUMP_IF_CMP_FALSE: *operandAt = 1; return offset;
        default:                   break;
        }
    }
    return -1;
}

/* The first instruction at or after `from` that is more than one byte wide, so
 * that offset + 1 is guaranteed *not* to be an instruction boundary. */
static int findWideOp(const Chunk *chunk, int from) {
    for (int offset = from; offset < chunk->count; offset += instrWidth(chunk, offset)) {
        if (instrWidth(chunk, offset) > 1) return offset;
    }
    return -1;
}

static void expectValid(ObjFunction *fn, const char *what) {
    char err[256];
    gChecks++;
    if (jaiVerifyChunk(fn, err, sizeof err)) return;
    gFailures++;
    printf("  FAIL %s: expected a valid chunk, got \"%s\"\n", what, err);
}

/* The verifier must reject, and its message must mention `needle` so that a
 * different check firing by accident does not count as a pass. */
static void expectRejected(ObjFunction *fn, const char *what, const char *needle) {
    char err[256];
    gChecks++;
    if (jaiVerifyChunk(fn, err, sizeof err)) {
        gFailures++;
        printf("  FAIL %s: chunk accepted, expected rejection\n", what);
        return;
    }
    if (strstr(err, needle) == NULL) {
        gFailures++;
        printf("  FAIL %s: rejected for the wrong reason: \"%s\"\n", what, err);
        return;
    }
    printf("  ok   %s -> %s\n", what, err);
}

/* ---------------------------------------------------------------- cases */

/* A jump whose target is one byte past a boundary. */
static void caseJumpMidInstruction(void) {
    ObjFunction *fn = compile("var a = 1\nif a > 0 { a = 2 } else { a = 3 }\n");
    expectValid(fn, "if/else baseline");

    int operandAt;
    int at = findConditionalBranch(&fn->chunk, 0, &operandAt);
    if (at < 0) { printf("  SKIP no conditional branch emitted\n"); return; }
    int16_t rel = jaiReadI16(fn->chunk.code + at + 1 + operandAt);
    jaiChunkPatchI16(&fn->chunk, at + 1 + operandAt, (int16_t)(rel + 1));
    expectRejected(fn, "jump into the middle of an instruction",
                   "not an instruction boundary");
}

/* A jump past the end of the code is a distinct check from the one above. */
static void caseJumpOutOfRange(void) {
    ObjFunction *fn = compile("var a = 1\nif a > 0 { a = 2 } else { a = 3 }\n");
    int operandAt;
    int at = findConditionalBranch(&fn->chunk, 0, &operandAt);
    if (at < 0) { printf("  SKIP no conditional branch emitted\n"); return; }
    jaiChunkPatchI16(&fn->chunk, at + 1 + operandAt, (int16_t)fn->chunk.count);
    expectRejected(fn, "jump past the end of the code", "outside the code");
}

static void caseConstantOutOfRange(void) {
    ObjFunction *fn = compile("let s = \"hello\"\nprint(s)\n");
    expectValid(fn, "constant baseline");

    int at = findOp(&fn->chunk, OP_CONST, 0);
    if (at < 0) { printf("  SKIP no OP_CONST emitted\n"); return; }
    uint8_t *k = fn->chunk.code + at + 1;
    uint32_t bad = (uint32_t)fn->chunk.constants.count;
    k[0] = (uint8_t)(bad & 0xff);
    k[1] = (uint8_t)((bad >> 8) & 0xff);
    k[2] = (uint8_t)((bad >> 16) & 0xff);
    expectRejected(fn, "constant index one past the pool",
                   "constant index");
}

static void caseSlotOutOfRange(void) {
    ObjFunction *body =
        compile("fn f(n: int) -> int {\n"
                "    var a = n + 1\n"
                "    var b = a * 2\n"
                "    return a + b\n"
                "}\n"
                "print(f(1))\n");
    ObjFunction *fn = nestedFunction(body, "f");
    expectValid(fn, "local-slot baseline");

    int at = findOp(&fn->chunk, OP_GET_LOCAL, 0);
    if (at < 0) at = findOp(&fn->chunk, OP_SET_LOCAL, 0);
    if (at < 0) { printf("  SKIP no local access emitted\n"); return; }
    jaiChunkPatchU16(&fn->chunk, at + 1, fn->maxSlots);
    expectRejected(fn, "local slot equal to maxSlots", "local slot");
}

/* Two paths into one offset that disagree about how deep the stack is: the POP
 * that balances the then-arm becomes a NOP, which is the same one byte wide, so
 * that arm arrives at the merge one value deeper than the else-arm. */
static void caseUnbalancedJoin(void) {
    ObjFunction *fn =
        compile("var a = 1\n"
                "if a > 0 { print(1) } else { print(2) }\n");
    expectValid(fn, "join baseline");

    int operandAt;
    int branch = findConditionalBranch(&fn->chunk, 0, &operandAt);
    int join = findOp(&fn->chunk, OP_JUMP, branch < 0 ? 0 : branch);
    if (branch < 0 || join < 0) { printf("  SKIP no if/else shape emitted\n"); return; }

    int pop = findOp(&fn->chunk, OP_POP, branch);
    if (pop < 0 || pop > join) { printf("  SKIP no POP inside the then-arm\n"); return; }
    fn->chunk.code[pop] = OP_NOP;
    expectRejected(fn, "stack depth disagreement at a join", "disagrees with depth");
}

/* A pop from a stack that does not hold enough: the POP after a call becomes an
 * ADD, which wants two operands where only the call's result is live. */
static void caseStackUnderflow(void) {
    /* A second statement, so the first call's OP_POP is not immediately
     * followed by the module's own implicit epilogue -- the two would
     * otherwise fuse into a single OP_POP_RETURN_NULL, leaving no standalone
     * OP_POP for this case to find. */
    ObjFunction *fn = compile("print(1)\nprint(2)\n");
    expectValid(fn, "underflow baseline");

    int at = findOp(&fn->chunk, OP_POP, 0);
    if (at < 0) { printf("  SKIP no OP_POP emitted\n"); return; }
    fn->chunk.code[at] = OP_ADD;
    expectRejected(fn, "binary op with one operand on the stack",
                   "from a stack of depth 1");
}

/* An exception handler that does not land on an instruction boundary. Pointing
 * it one byte into a multi-byte instruction is what a table that was not fixed
 * up after a re-encode looks like. */
static void caseBadExceptionEntry(void) {
    ObjFunction *body =
        compile("fn g() -> int {\n"
                "    try {\n"
                "        throw ValueError(\"x\")\n"
                "    } catch _e: ValueError {\n"
                "        return 1\n"
                "    }\n"
                "}\n"
                "print(g())\n");
    ObjFunction *fn = nestedFunction(body, "g");
    expectValid(fn, "exception-table baseline");
    if (fn->exceptionCount == 0) { printf("  SKIP no exception entries\n"); return; }

    int wide = findWideOp(&fn->chunk, 0);
    if (wide < 0) { printf("  SKIP every instruction is one byte wide\n"); return; }
    fn->exceptions[0].handler = (uint32_t)wide + 1;
    expectRejected(fn, "handler off an instruction boundary",
                   "handler");
}

/* An upvalue index the closure does not have. */
static void caseBadUpvalue(void) {
    ObjFunction *body =
        compile("fn outer() -> int {\n"
                "    var x = 1\n"
                "    let inner = fn() -> int { return x }\n"
                "    return inner()\n"
                "}\n"
                "print(outer())\n");
    ObjFunction *outer = nestedFunction(body, "outer");
    ObjFunction *fn = nestedFunction(outer, NULL);
    expectValid(fn, "upvalue baseline");
    int at = findOp(&fn->chunk, OP_GET_UPVALUE, 0);
    if (at < 0) { printf("  SKIP no OP_GET_UPVALUE emitted\n"); return; }
    fn->chunk.code[at + 1] = (uint8_t)fn->upvalueCount;
    expectRejected(fn, "upvalue index past upvalueCount", "upvalue");
}

/* ------------------------------------------------- emitted shape */

/* The cases above corrupt a chunk and ask the verifier to notice. This last one
 * asks a different question of the same machinery: what did the emitter put
 * there in the first place? Nothing else in the suite can see an instruction
 * that should not have been emitted — a spurious close is invisible from the
 * outside, since closing a scope nothing captured is a no-op at runtime.
 */

static int countOp(const Chunk *chunk, uint8_t op) {
    int n = 0;
    for (int offset = 0; offset < chunk->count; offset += instrWidth(chunk, offset)) {
        if (chunk->code[offset] == op) n++;
    }
    return n;
}

static void expectOpCount(ObjFunction *fn, uint8_t op, int want, const char *what) {
    gChecks++;
    int got = countOp(&fn->chunk, op);
    if (got == want) {
        printf("  ok   %s -> %d %s\n", what, got, jaiOpName((OpCode)op));
        return;
    }
    gFailures++;
    printf("  FAIL %s: %d %s, expected %d\n", what, got, jaiOpName((OpCode)op), want);
}

/* Spec §5.2's per-scope freshness must be free unless it is used. */
static void caseBlockCloseShape(void) {
    ObjFunction *body =
        compile("fn plain(n: int) -> int {\n"
                "    var total = 0\n"
                "    if n > 0 {\n"
                "        let doubled = n * 2\n"
                "        total = total + doubled\n"
                "    }\n"
                "    {\n"
                "        let extra = 3\n"
                "        total = total + extra\n"
                "    }\n"
                "    return total\n"
                "}\n"
                "print(plain(4))\n");
    ObjFunction *fn = nestedFunction(body, "plain");
    expectValid(fn, "plain-block baseline");
    expectOpCount(fn, OP_CLOSE_UPVALUE, 0, "a block that captures nothing emits no close");

    /* A `let` is captured by value (spec §6): a snapshot needs no close. */
    body = compile("fn byvalue(c: bool) -> fn() -> int {\n"
                   "    var g = fn() -> int { return 0 }\n"
                   "    if c {\n"
                   "        let a = 1\n"
                   "        g = || a\n"
                   "    }\n"
                   "    return g\n"
                   "}\n"
                   "print(byvalue(true)())\n");
    fn = nestedFunction(body, "byvalue");
    expectValid(fn, "by-value-capture baseline");
    expectOpCount(fn, OP_CLOSE_UPVALUE, 0, "a by-value capture emits no close");

    /* A `var` captured by reference: one close, where the block ends. */
    body = compile("fn byref(c: bool) -> fn() -> int {\n"
                   "    var g = fn() -> int { return 0 }\n"
                   "    if c {\n"
                   "        var a = 1\n"
                   "        g = || a\n"
                   "    }\n"
                   "    var b = 99\n"
                   "    print(b)\n"
                   "    return g\n"
                   "}\n"
                   "print(byref(true)())\n");
    fn = nestedFunction(body, "byref");
    expectValid(fn, "by-reference-capture baseline");
    expectOpCount(fn, OP_CLOSE_UPVALUE, 1, "a block with a by-reference capture closes once");

    /* The loop's iteration close already covers its body block's base; both
     * would land at the same offset, one instruction of pure waste on the
     * hottest edge there is. */
    body = compile("fn bodies() -> list[fn() -> int] {\n"
                   "    var out: list[fn() -> int] = []\n"
                   "    for i in 0..3 {\n"
                   "        var v = i * 10\n"
                   "        out.push(|| v)\n"
                   "    }\n"
                   "    return out\n"
                   "}\n"
                   "print(bodies().len())\n");
    fn = nestedFunction(body, "bodies");
    expectValid(fn, "loop-body baseline");
    expectOpCount(fn, OP_CLOSE_UPVALUE, 1, "a loop body is not closed twice");
}

/* OP_MATCH_CONST_POP has a different net stack effect on each of its two
 * edges -- -1 falling through (the arm matched, the subject is popped), 0 on
 * the taken jump (no match, the subject stays for the next arm's own test).
 * A real match statement over literal patterns is the baseline; it must
 * verify exactly like any other chunk the front end produces. */
static void caseMatchConstPopBaseline(void) {
    ObjFunction *fn =
        compile("let n = 2\n"
                "print(match n { 0 => \"zero\", 1 => \"one\", _ => \"many\" })\n");
    expectValid(fn, "match over literal patterns");
    expectOpCount(fn, OP_MATCH_CONST_POP, 2, "one MATCH_CONST_POP per literal arm");
}

/* Redirect a MATCH_CONST_POP's no-match jump to land exactly where its own
 * fall-through does. The fall-through edge arrives one value shallower (the
 * subject just got popped); the redirected jump edge still carries the
 * unpopped depth the taken path is supposed to have. Two edges into one
 * offset that disagree by exactly the asymmetry this opcode exists for --
 * if `jaiOpBranchOperandAt` or the depth switch ever mis-modelled either
 * edge, this is what would stop catching it. */
static void caseMatchConstPopEdgesDisagree(void) {
    ObjFunction *fn =
        compile("let n = 2\n"
                "print(match n { 0 => \"zero\", 1 => \"one\", _ => \"many\" })\n");

    int at = findOp(&fn->chunk, OP_MATCH_CONST_POP, 0);
    if (at < 0) { printf("  SKIP no OP_MATCH_CONST_POP emitted\n"); return; }
    /* The jump displacement is relative to the byte after the whole
     * instruction (see chunk.c's own note on this), so 0 retargets the
     * no-match edge onto the very instruction fall-through already reaches. */
    jaiChunkPatchI16(&fn->chunk, at + 1 + 3, 0);
    expectRejected(fn, "MATCH_CONST_POP edges redirected to the same offset",
                   "disagrees with depth");
}

/* MATCH_TYPE_POP, MATCH_RANGE_POP and MATCH_SEQ_POP share MATCH_CONST_POP's
 * exact asymmetric shape and the same jaiOpBranchOperandAt/depth-switch
 * machinery -- one baseline each proves every site actually emits its fused
 * form, and one negative case (mirroring caseMatchConstPopEdgesDisagree)
 * proves the shared mechanism catches a real edge disagreement for a second
 * opcode, not just the one it was written against. */
static void caseMatchTypePopBaseline(void) {
    ObjFunction *fn =
        compile("enum Shape { Circle(r: int), Point }\n"
                "let s = Shape.Point\n"
                "print(match s { Shape.Point => \"pt\", _ => \"other\" })\n");
    expectValid(fn, "match over an enum's own type check");
    expectOpCount(fn, OP_MATCH_TYPE_POP, 1, "one MATCH_TYPE_POP for the tag<0 variant check");
}

static void caseMatchTypePopEdgesDisagree(void) {
    ObjFunction *fn =
        compile("enum Shape { Circle(r: int), Point }\n"
                "let s = Shape.Point\n"
                "print(match s { Shape.Point => \"pt\", _ => \"other\" })\n");

    int at = findOp(&fn->chunk, OP_MATCH_TYPE_POP, 0);
    if (at < 0) { printf("  SKIP no OP_MATCH_TYPE_POP emitted\n"); return; }
    jaiChunkPatchI16(&fn->chunk, at + 1 + 3, 0);
    expectRejected(fn, "MATCH_TYPE_POP edges redirected to the same offset",
                   "disagrees with depth");
}

static void caseMatchRangePopBaseline(void) {
    ObjFunction *fn =
        compile("let n = 5\n"
                "print(match n { 0..=9 => \"digit\", _ => \"other\" })\n");
    expectValid(fn, "match over a range pattern");
    expectOpCount(fn, OP_MATCH_RANGE_POP, 1, "one MATCH_RANGE_POP for the range arm");
}

static void caseMatchSeqPopBaseline(void) {
    ObjFunction *fn =
        compile("let v: any = [1, 2]\n"
                "print(match v { [a, b] => a + b, _ => 0 })\n");
    expectValid(fn, "match over a fixed-length list pattern");
    expectOpCount(fn, OP_MATCH_SEQ_POP, 1, "one MATCH_SEQ_POP for the list-shape arm");
}

/* `slotOperands` (verify.c) once had no case for OP_ADD_BIND/OP_SUB_BIND/
 * OP_MUL_BIND -- their own u16 slot operand went unchecked, so a malformed
 * `.jaic` naming a slot past `maxSlots` passed verification and the VM would
 * later write out of the frame (`slots[slot] = ...`, vm.c). SUB_BIND stands
 * in for its two siblings: they share this one case in the switch, so one
 * out-of-range instance covers the whole family. */
static void caseSubBindSlotOutOfRange(void) {
    ObjFunction *body =
        compile("fn f(n: int) -> int {\n"
                "    var a = n + 1\n"
                "    var b = a - 1\n"
                "    var c = 0\n"
                "    c = a - b\n"
                "    return c\n"
                "}\n"
                "print(f(3))\n");
    ObjFunction *fn = nestedFunction(body, "f");
    expectValid(fn, "sub-bind baseline");

    int at = findOp(&fn->chunk, OP_SUB_BIND, 0);
    if (at < 0) { printf("  SKIP no OP_SUB_BIND emitted\n"); return; }
    jaiChunkPatchU16(&fn->chunk, at + 1, fn->maxSlots);
    expectRejected(fn, "sub-bind slot equal to maxSlots", "local slot");
}

int main(void) {
    jaiDiagInit(&gDiags);
    gDiags.colorOutput = false;
    vm.optLevel = 2;
    jaiVMInit();
    jaiModulePathInit(".");
    if (!jaiLoadPrelude()) {
        fprintf(stderr, "FATAL: prelude failed to load\n");
        return 2;
    }

    printf("jaiVerifyChunk\n");
    caseJumpMidInstruction();
    caseJumpOutOfRange();
    caseConstantOutOfRange();
    caseSlotOutOfRange();
    caseUnbalancedJoin();
    caseStackUnderflow();
    caseBadExceptionEntry();
    caseBadUpvalue();
    caseBlockCloseShape();
    caseMatchConstPopBaseline();
    caseMatchConstPopEdgesDisagree();
    caseMatchTypePopBaseline();
    caseMatchTypePopEdgesDisagree();
    caseMatchRangePopBaseline();
    caseMatchSeqPopBaseline();
    caseSubBindSlotOutOfRange();

    printf("%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
