/* invoke_result_kind.c — what an OP_INVOKE site records about its own result.
 *
 * `InlineCache::resultKind` is a per-site, per-way prediction the compiled tier
 * turns into a slot kind, and it is the only thing standing between a
 * polymorphic call and a wrong guess. roadmap §"a megamorphic call-site arm
 * would be 5.7x SLOWER" prices a wrong prediction at 5.7x the decline it
 * replaces, so a record that quietly said "int" for a site returning two kinds
 * would be worse than no record at all.
 *
 * Nothing else can see this. It is not in a program's output, not in a
 * disassembly, and a JIT-level test would only observe it through whatever the
 * tier happened to do with it -- which is exactly the coupling that let the
 * builtin record sit at ONE observation for its whole life (§6). So this test
 * runs real source through the interpreter and then reads the bytes.
 *
 * Every probe function holds exactly one OP_INVOKE, which is asserted rather
 * than assumed: a probe whose site moved would otherwise pass while checking
 * nothing, and roadmap §9 records a fuzz shape that did exactly that.
 *
 * Built by `make invoke-result-test`; part of `make test`.
 */
#include "common/common.h"
#include "common/diag.h"
#include "runtime/runtime.h"
#include "vm/bytecode/chunk.h"
#include "vm/object/object.h"
#include "vm/vm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gFailures = 0;
static int gChecks = 0;

static void fail(const char *where, const char *what) {
    fprintf(stderr, "FAIL: %s: %s\n", where, what);
    gFailures++;
}

/* One OP_INVOKE per probe, so `n` is deliberately far above the observation
 * budget: a record that stopped early, or never started, shows up as a wrong
 * kind rather than as a missing one.
 *
 * `-> any` on every apply() is load-bearing. A declared `-> int` would let the
 * front end reject the str-returning bodies, and the whole question here is
 * what happens when one site sees two kinds. */
static const char kSource[] =
    "class IntOp {\n"
    "    pub fn apply(self, a: int) -> any { return a + 1 }\n"
    "}\n"
    "class StrOp {\n"
    "    pub fn apply(self, a: int) -> any { return \"s\" }\n"
    "}\n"
    "class FlipOp {\n"
    "    pub fn apply(self, a: int) -> any {\n"
    "        if a % 2 == 0 { return 1 }\n"
    "        return \"s\"\n"
    "    }\n"
    "}\n"
    "class NativeInitOp {\n"
    "    pub fn apply(self, a: int) -> any { return [a] }\n"
    "}\n"
    "\n"
    "fn site_mono_int(o: any, n: int) -> int {\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        o.apply(i)\n"
    "        i = i + 1\n"
    "    }\n"
    "    return i\n"
    "}\n"
    "\n"
    "fn site_mono_list(o: any, n: int) -> int {\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        o.apply(i)\n"
    "        i = i + 1\n"
    "    }\n"
    "    return i\n"
    "}\n"
    "\n"
    "fn site_poly(a: any, b: any, n: int) -> int {\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        var o = a\n"
    "        if i % 2 == 1 { o = b }\n"
    "        o.apply(i)\n"
    "        i = i + 1\n"
    "    }\n"
    "    return i\n"
    "}\n"
    "\n"
    "fn site_unstable(o: any, n: int) -> int {\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        o.apply(i)\n"
    "        i = i + 1\n"
    "    }\n"
    "    return i\n"
    "}\n"
    "\n"
    "fn site_builtin_stable(xs: list, n: int) -> int {\n"
    "    var acc = 0\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        acc = acc + xs.len()\n"
    "        i = i + 1\n"
    "    }\n"
    "    return acc\n"
    "}\n"
    "\n"
    "fn site_builtin_unstable(d: dict, n: int) -> int {\n"
    "    var i = 0\n"
    "    while i < n {\n"
    "        d.get(i % 2)\n"
    "        i = i + 1\n"
    "    }\n"
    "    return i\n"
    "}\n"
    "\n"
    "fn probe() -> int {\n"
    "    site_mono_int(IntOp(), 200)\n"
    "    site_mono_list(NativeInitOp(), 200)\n"
    "    site_poly(IntOp(), StrOp(), 200)\n"
    "    site_unstable(FlipOp(), 200)\n"
    "    site_builtin_stable([1, 2, 3], 200)\n"
    "    var d: dict = {0: 1, 1: \"s\"}\n"
    "    site_builtin_unstable(d, 200)\n"
    "    return 0\n"
    "}\n"
    "\n"
    "probe()\n";

#define FB_INT   (1u + (unsigned)VAL_INT)
#define FB_STR   (JAI_FB_OBJ + (unsigned)OBJ_STRING)
#define FB_LIST  (JAI_FB_OBJ + (unsigned)OBJ_LIST)

typedef struct {
    const char *fnName;
    int         wantWays;
    uint8_t     wantKinds[JAI_IC_WAYS];  /* sorted ascending */
    const char *why;
} Expect;

static const Expect kExpect[] = {
    {"site_mono_int", 1, {FB_INT},
     "one class, always an int"},
    {"site_mono_list", 1, {FB_LIST},
     "an object result is recorded by ObjType, not merely as 'an object'"},
    {"site_poly", 2, {FB_INT, FB_STR},
     "two classes returning two kinds keep two answers: PER WAY is the point"},
    {"site_unstable", 1, {JAI_FB_MIXED},
     "one class alternating two kinds is unstable, and says so rather than "
     "reporting whichever it saw first"},
    {"site_builtin_stable", 1, {FB_INT},
     "the builtin record still works"},
    {"site_builtin_unstable", 1, {JAI_FB_MIXED},
     "a builtin whose result kind changes after the first call -- this is the "
     "one roadmap §6 records as recorded-on-first-observation"},
};

static int cmpU8(const void *a, const void *b) {
    return (int)*(const uint8_t *)a - (int)*(const uint8_t *)b;
}

static const char *kindName(uint8_t k) {
    static char buf[4][32];
    static int turn = 0;
    char *out = buf[turn++ & 3];
    if (k == JAI_FB_NONE)  { snprintf(out, 32, "NONE");  return out; }
    if (k == JAI_FB_MIXED) { snprintf(out, 32, "MIXED"); return out; }
    if (k >= JAI_FB_OBJ) {
        snprintf(out, 32, "obj:%s", jaiObjTypeName((ObjType)(k - JAI_FB_OBJ)));
    } else {
        snprintf(out, 32, "val:%u", (unsigned)(k - 1u));
    }
    return out;
}

/* The one OP_INVOKE in `fn`, or NULL with the failure already reported. */
static const InlineCache *soleInvokeCache(const ObjFunction *fn,
                                          const char *where) {
    const Chunk *chunk = &fn->chunk;
    const InlineCache *found = NULL;
    int seen = 0;
    for (int off = 0; off < chunk->count;) {
        uint8_t raw = chunk->code[off];
        if (raw >= OP_COUNT) {
            fail(where, "the chunk walk lost alignment");
            return NULL;
        }
        OpCode op = (OpCode)raw;
        /* The same walk rebuildCaches does, OP_CLOSURE's variable length and
         * all -- a walk that lands mid-instruction reads an operand byte as an
         * opcode, which roadmap §4a records as the symptom of `off += 5`. */
        int size = jaiOpOperandSize(op);
        if (op == OP_CLOSURE) {
            uint32_t k = jaiReadU24(chunk->code + off + 1);
            if (k >= (uint32_t)chunk->constants.count ||
                !IS_FUNCTION(chunk->constants.data[k])) {
                fail(where, "OP_CLOSURE does not name a function");
                return NULL;
            }
            size = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
        }
        if (size < 0 || off > chunk->count - 1 - size) {
            fail(where, "the chunk walk ran off the end");
            return NULL;
        }
        if (op == OP_INVOKE) {
            int at = jaiOpCacheOperand(op);
            if (at < 0) {
                fail(where, "OP_INVOKE has no cache operand");
                return NULL;
            }
            uint16_t idx = jaiReadU16(chunk->code + off + 1 + at);
            if ((int)idx >= chunk->cacheCount) {
                fail(where, "OP_INVOKE names a cache index the chunk lacks");
                return NULL;
            }
            found = &chunk->caches[idx];
            seen++;
        }
        off += 1 + size;
    }
    gChecks++;
    if (seen != 1) {
        char msg[128];
        snprintf(msg, sizeof msg,
                 "expected exactly one OP_INVOKE, found %d -- this probe is no "
                 "longer testing what it names", seen);
        fail(where, msg);
        return NULL;
    }
    return found;
}

/* Probe functions and class methods are all constants of the module body. */
static const ObjFunction *findFunction(const ObjFunction *body,
                                       const char *name) {
    for (int i = 0; i < body->chunk.constants.count; i++) {
        Value v = body->chunk.constants.data[i];
        if (!IS_FUNCTION(v)) continue;
        ObjFunction *fn = AS_FUNCTION(v);
        if (fn->name != NULL && strcmp(fn->name->chars, name) == 0) return fn;
    }
    return NULL;
}

static void check(const ObjFunction *body, const Expect *e) {
    const ObjFunction *fn = findFunction(body, e->fnName);
    gChecks++;
    if (fn == NULL) {
        fail(e->fnName, "no such function in the compiled module");
        return;
    }
    const InlineCache *ic = soleInvokeCache(fn, e->fnName);
    if (ic == NULL) return;

    gChecks++;
    if (ic->count != e->wantWays) {
        char msg[128];
        snprintf(msg, sizeof msg, "filled %d ways, expected %d",
                 (int)ic->count, e->wantWays);
        fail(e->fnName, msg);
        return;
    }

    /* A site that stopped observing before its budget ran out would still pass
     * the kind checks whenever the first call happened to be representative --
     * which is the failure this whole mechanism exists to remove. */
    gChecks++;
    if (ic->obsBudget != 0) {
        char msg[128];
        snprintf(msg, sizeof msg,
                 "%u observations of its budget went unspent after 200 calls",
                 (unsigned)ic->obsBudget);
        fail(e->fnName, msg);
    }

    uint8_t got[JAI_IC_WAYS];
    for (int w = 0; w < ic->count; w++) got[w] = ic->resultKind[w];
    qsort(got, (size_t)ic->count, 1, cmpU8);

    gChecks++;
    for (int w = 0; w < ic->count; w++) {
        if (got[w] == e->wantKinds[w]) continue;
        char msg[256];
        snprintf(msg, sizeof msg, "way %d recorded %s, expected %s (%s)", w,
                 kindName(got[w]), kindName(e->wantKinds[w]), e->why);
        fail(e->fnName, msg);
        return;
    }
}

int main(void) {
    /* The record is the INTERPRETER's observation, and a compiled loop stops
     * reaching OP_INVOKE at all. With the tier on, what the probes observe
     * depends on when the sampler fires, which is roadmap §1's "do not read a
     * single sample" in a form that would make this test flaky rather than
     * wrong. The mechanism under test is the same either way. */
    setenv("JAITHON_NO_JIT", "1", 1);

    jaiDiagInit(&gDiags);
    gDiags.colorOutput = false;
    vm.optLevel = 2;
    jaiVMInit();
    jaiModulePathInit(".");
    if (!jaiLoadPrelude()) {
        fprintf(stderr, "FATAL: prelude failed to load\n");
        return 2;
    }

    ObjModule *module = jaiModuleNew(jaiStringInternC("__invoke_result__"),
                                     jaiStringInternC("<test>"));
    if (module == NULL) {
        fprintf(stderr, "FATAL: module\n");
        return 2;
    }
    jaiPushRoot(OBJ_VAL(module));

    CodegenOptions opts = jaiCodegenDefaults();
    opts.optLevel = 2;
    ObjFunction *body = jaiCompileSource(kSource, sizeof kSource - 1,
                                         "<invoke_result_kind>", module, &opts);
    if (body == NULL) {
        jaiDiagFlush(&gDiags, stderr);
        fprintf(stderr, "FATAL: the probe source did not compile\n");
        return 2;
    }
    jaiPushRoot(OBJ_VAL(body));

    if (jaiVMRunModule(module, body) != JAI_RUN_OK) {
        jaiDiagFlush(&gDiags, stderr);
        fprintf(stderr, "FATAL: the probe source raised\n");
        return 2;
    }

    printf("invoke result-kind records\n");
    for (size_t i = 0; i < sizeof kExpect / sizeof kExpect[0]; i++) {
        check(body, &kExpect[i]);
    }

    printf("%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
