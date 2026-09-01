/* vm_lifecycle.c — bringing the VM up and taking it down, the SIGINT handler
 * it installs while it is up, and the instrumentation the dispatch loop feeds.
 */
#include <inttypes.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

#include "vm/vm_internal.h"
#include "vm/trace/trace.h"
#include "runtime/runtime.h"

static bool sSignalInstalled;

static void interruptHandler(int signum) {
    (void)signum;
    jaiInterrupted = 1;
}

static void installInterruptHandler(void) {
    /* Only claim SIGINT when nothing else has: a host embedding the VM (or the
     * CLI's own REPL handler) must keep priority. */
    void (*previous)(int) = signal(SIGINT, interruptHandler);
    if (previous == SIG_DFL) {
        sSignalInstalled = true;
        return;
    }
    (void)signal(SIGINT, previous);
}

static void removeInterruptHandler(void) {
    if (!sSignalInstalled) return;
    (void)signal(SIGINT, SIG_DFL);
    sSignalInstalled = false;
}

/* Charge one interpreted instruction to the function whose frame is running.
 *
 * Exact rather than sampled: `sum(fn->interpCount) == vm.instructionCount` is
 * the invariant, and scripts/jit_report.py leans on it, so a frame shape this
 * misses would show up as a shortfall rather than as a quietly wrong ranking.
 *
 * The first-touch list is what makes the dump possible at all -- there is no
 * global registry of ObjFunctions, and walking the GC heap for one would be a
 * much larger hammer than this needs. */
void attributeInstruction(CallFrame *frame) {
    if (frame == NULL || frame->closure == NULL) return;
    ObjFunction *fn = frame->closure->fn;
    if (fn == NULL) return;
    if (fn->interpCount == 0) {
        if (vm.attributedCount == vm.attributedCap) {
            unsigned grown = vm.attributedCap < 64 ? 64 : vm.attributedCap * 2;
            ObjFunction **wider = (ObjFunction **)realloc(
                vm.attributed, grown * sizeof *wider);
            /* Out of room simply stops registering: the counts stay exact for
             * everything already listed, and the dump says so. */
            if (wider == NULL) { fn->interpCount++; return; }
            vm.attributed = wider;
            vm.attributedCap = grown;
        }
        vm.attributed[vm.attributedCount++] = fn;
    }
    fn->interpCount++;
}

void traceInstruction(CallFrame *frame, const uint8_t *ip) {
    jaiVMPrintStack(stdout);
    Chunk *chunk = frameChunk(frame);
    (void)jaiDisassembleInstruction(stdout, chunk, (int)(ip - chunk->code));
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                            */
/* ------------------------------------------------------------------ */

void jaiVMResetStack(void) {
    vm.stackTop = vm.stack;
    vm.frameCount = 0;
    sResultSite.ic = NULL;
    vm.openUpvalues = NULL;
    vm.handlers.count = 0;
    vm.defers.count = 0;
    sFinallyPending = 0;
    sRunDepth = 0;
}

/* The dunder and well-known names the VM itself dispatches on. Interning them
 * once turns every later lookup into a pointer comparison. */
static void internWellKnownNames(void) {
    vm.strInit     = jaiStringInternC("init");
    vm.strItems    = jaiStringInternC("items");
    vm.strStr      = jaiStringInternC("__str__");
    vm.strRepr     = jaiStringInternC("__repr__");
    vm.strEq       = jaiStringInternC("__eq__");
    vm.strLt       = jaiStringInternC("__lt__");
    vm.strHash     = jaiStringInternC("__hash__");
    vm.strLen      = jaiStringInternC("__len__");
    vm.strGetItem  = jaiStringInternC("__getitem__");
    vm.strSetItem  = jaiStringInternC("__setitem__");
    vm.strContains = jaiStringInternC("__contains__");
    vm.strIter     = jaiStringInternC("__iter__");
    vm.strNext     = jaiStringInternC("__next__");
    vm.strCall     = jaiStringInternC("__call__");
    vm.strAdd      = jaiStringInternC("__add__");
    vm.strSub      = jaiStringInternC("__sub__");
    vm.strMul      = jaiStringInternC("__mul__");
    vm.strDiv      = jaiStringInternC("__div__");
    vm.strMod      = jaiStringInternC("__mod__");
    vm.strPow      = jaiStringInternC("__pow__");
    vm.strNeg      = jaiStringInternC("__neg__");
    vm.strMain     = jaiStringInternC("main");
    vm.strSelf     = jaiStringInternC("self");
    vm.strMessage  = jaiStringInternC("message");
}

void jaiVMInit(void) {
    /* Flags may have been set from the command line before the VM came up, so
     * only the fields this function owns are reset. */
    vm.stack = JAI_ALLOC(Value, JAI_STACK_MAX);
    vm.frames = JAI_ALLOC(CallFrame, JAI_FRAMES_MAX);
    vm.frameCapacity = JAI_FRAMES_MAX;
    vm.frameCount = 0;
    vm.stackTop = vm.stack;
    vm.openUpvalues = NULL;

    JAI_VEC_INIT(&vm.handlers);
    JAI_VEC_INIT(&vm.defers);
    JAI_VEC_INIT(&vm.modulePath);

    jaiMethodCacheInit();
    memset(sMegaCache, 0, sizeof sMegaCache);
    jaiTraceInit();

    vm.pendingException = NULL_VAL;
    vm.hasException = false;
    vm.mainModule = NULL;
    vm.builtins = NULL;

    vm.instructionCount = 0;
    vm.callCount = 0;
    vm.allocCount = 0;
    vm.icHits = 0;
    vm.icMisses = 0;
    vm.jitPicAdmits = 0;
    vm.jitPicRefusals = 0;

    sRunDepth = 0;
    sFinallyPending = 0;
    jaiInterrupted = 0;

    /* The collector has to exist before the first allocation below. */
    GCState *gc = JAI_ALLOC(GCState, 1);
    jaiGCInit(gc);
    gc->stress = vm.gcStress;
    gc->stressEvery = vm.gcStressEvery;
    gc->verbose = vm.debugGC;
    /* stress is one of jaiGCLimit's four inputs and this is the only place
     * outside gc.c that writes one. Without this, --gc-stress kept the
     * threshold it was initialised with and stressed 11% less. */
    jaiGCSyncLimit();

    jaiInternTableInit();
    jaiTableInit(&vm.modules);

    /* Before anything else interns: from here on every reader of the one-byte
     * ASCII table may assume a slot is populated, which is a branch each of
     * them used to carry. See jaiAsciiCharsFill. */
    jaiAsciiCharsFill();

    internWellKnownNames();

    /* The intern table is a weak root, so `builtinsName` would be swept by the
     * collection the *next* allocation triggers. Root it across that one. */
    ObjString *builtinsName = jaiStringInternC("__builtins__");
    jaiGCPushRoot(OBJ_VAL(builtinsName));
    ObjString *builtinsPath = jaiStringInternC("");
    vm.builtins = jaiModuleNew(builtinsName, builtinsPath);
    jaiGCPopRoot();
    vm.builtins->state = MOD_LOADED;

    jaiRegisterErrorClasses();
    jaiRegisterAllBuiltins();

    installInterruptHandler();
}

void jaiVMFree(void) {
    removeInterruptHandler();
    freeSavedTraceback();

    jaiVMResetStack();
    vm.pendingException = NULL_VAL;
    vm.hasException = false;
    vm.mainModule = NULL;
    vm.builtins = NULL;

    jaiTableFree(&vm.modules);
    JAI_VEC_FREE(ExcHandler, &vm.handlers);
    JAI_VEC_FREE(Value, &vm.defers);
    JAI_VEC_FREE(ObjString *, &vm.modulePath);

    vm.strInit = vm.strStr = vm.strRepr = vm.strEq = vm.strLt = NULL;
    vm.strHash = vm.strLen = vm.strGetItem = vm.strSetItem = NULL;
    vm.strContains = vm.strIter = vm.strNext = vm.strCall = NULL;
    vm.strAdd = vm.strSub = vm.strMul = vm.strDiv = vm.strMod = NULL;
    vm.strPow = vm.strNeg = vm.strMain = vm.strSelf = vm.strMessage = NULL;

    vm.cError = vm.cTypeError = vm.cValueError = vm.cNameError = NULL;
    vm.cIndexError = vm.cKeyError = vm.cAttributeError = NULL;
    vm.cArithmeticError = vm.cDivisionByZeroError = vm.cOverflowError = NULL;
    vm.cIOError = vm.cOSError = vm.cRuntimeError = vm.cRecursionError = NULL;
    vm.cStopIteration = vm.cAssertionError = vm.cImportError = NULL;
    vm.cFileNotFoundError = vm.cPermissionError = vm.cParseError = NULL;
    vm.cLookupError = NULL;

    /* Objects first (the sweep needs the intern table intact), then the
     * table that weakly referenced them. */
    GCState *gc = vm.gc;
    if (gc != NULL) {
        jaiGCFree(gc);
        JAI_FREE(GCState, gc);
    }
    vm.gc = NULL;
    /* After the sweep the 128 slots point at freed objects. */
    jaiAsciiCharsReset();
    jaiInternTableFree();

    JAI_FREE_ARRAY(Value, vm.stack, JAI_STACK_MAX);
    JAI_FREE_ARRAY(CallFrame, vm.frames, vm.frameCapacity);
    vm.stack = NULL;
    vm.stackTop = NULL;
    vm.frames = NULL;
    vm.frameCapacity = 0;
}

JaiRunResult jaiVMRunModule(ObjModule *module, ObjFunction *body) {
    if (module == NULL || body == NULL) return JAI_RUN_COMPILE_ERROR;

    jaiVMResetStack();
    jaiClearException();

    body->module = module;
    ObjClosure *closure = jaiClosureNew(body);
    module->body = closure;
    if (vm.mainModule == NULL) vm.mainModule = module;

    Value *base = vm.stackTop;
    *vm.stackTop++ = OBJ_VAL(closure);
    int window = frameWindowSize(body);
    if (base + window + 1 > vm.stack + JAI_STACK_MAX) {
        (void)jaiThrow(vm.cRuntimeError, "module body needs too many slots");
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return JAI_RUN_RUNTIME_ERROR;
    }
    for (int i = 1; i < window; i++) *vm.stackTop++ = NULL_VAL;

    if (!pushFrame(closure, base)) {
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        return JAI_RUN_RUNTIME_ERROR;
    }
    vm.frames[vm.frameCount - 1].module = module;

    JaiRunResult result = run(0);
    if (result != JAI_RUN_OK) {
        module->state = MOD_FAILED;
        jaiReportUncaught(vm.pendingException);
        jaiClearException();
        jaiVMResetStack();
        return result;
    }

    module->state = MOD_LOADED;
    jaiVMResetStack();
    return JAI_RUN_OK;
}

/* ------------------------------------------------------------------ */
/* Debug output                                                         */
/* ------------------------------------------------------------------ */

void jaiVMPrintStack(FILE *out) {
    if (out == NULL || vm.stack == NULL) return;
    fputs("          ", out);
    for (Value *slot = vm.stack; slot < vm.stackTop; slot++) {
        fputs("[ ", out);
        jaiPrintValue(out, *slot, true);
        fputs(" ]", out);
    }
    fputc('\n', out);
}

void jaiVMPrintStats(FILE *out) {
    if (out == NULL) return;
    uint64_t lookups = vm.icHits + vm.icMisses;
    double hitRate = lookups > 0 ? (100.0 * (double)vm.icHits / (double)lookups)
                                 : 0.0;
#ifdef JAI_OPCODE_STATS
    {
        uint64_t tot = 0;
        for (int i = 0; i < OP_COUNT; i++) tot += jaiOpCounts[i];
        /* Every non-zero opcode, not a top slice: a filtered histogram cannot
         * be summed, and summing it against vm.instructionCount is the only
         * check that no dispatch path skips the census. */
        for (int i = 0; i < OP_COUNT; i++)
            if (jaiOpCounts[i] > 0)
                fprintf(out, "op %3d %-24s %10" PRIu64 "  %5.2f%%\n", i,
                        jaiOpName((OpCode)i), jaiOpCounts[i],
                        100.0 * (double)jaiOpCounts[i] / (double)tot);
        for (int a = 0; a < OP_COUNT; a++) {
            uint64_t from = 0;
            for (int b = 0; b < OP_COUNT; b++) from += jaiOpPairs[a][b];
            if (from == 0) continue;
            for (int b = 0; b < OP_COUNT; b++)
                if (jaiOpPairs[a][b] > 0)
                    fprintf(out, "pair %-24s %-24s %10" PRIu64 "  %5.1f%%\n",
                            jaiOpName((OpCode)a), jaiOpName((OpCode)b),
                            jaiOpPairs[a][b],
                            100.0 * (double)jaiOpPairs[a][b] / (double)from);
        }
    }
#endif
#ifdef JAI_PROP_STATS
    for (int i = 0; i < 32; i++)
        if (jaiPropRecv[i] > 0)
            fprintf(out, "getProperty recv type %d: %" PRIu64 "\n", i, jaiPropRecv[i]);
#endif
#ifdef JAI_ALLOC_CENSUS
    jaiAllocPrintCensus(out);
#endif
    if (vm.attributeInstructions && vm.attributedCount > 0) {
        /* Sorted here rather than in the reader, because the reader would have
         * to parse it back; and reported with the SHORTFALL, because the whole
         * value of this table is that it can be summed. A frame shape the
         * dispatch path misses shows up as a gap rather than as a ranking that
         * is quietly wrong. */
        unsigned n = vm.attributedCount;
        for (unsigned i = 1; i < n; i++) {
            ObjFunction *key = vm.attributed[i];
            unsigned j = i;
            while (j > 0 && vm.attributed[j - 1]->interpCount < key->interpCount) {
                vm.attributed[j] = vm.attributed[j - 1];
                j--;
            }
            vm.attributed[j] = key;
        }
        uint64_t summed = 0;
        for (unsigned i = 0; i < n; i++) summed += vm.attributed[i]->interpCount;
        for (unsigned i = 0; i < n; i++) {
            ObjFunction *fn = vm.attributed[i];
            if (fn->interpCount == 0) continue;
            /* Qualified by the DEFINING MODULE, not by fn->qualifiedName.
             * The dump is read to decide what to optimise, and `init`
             * appearing three times with three different totals is not a
             * reading anyone can act on -- but qualifiedName does not fix it:
             * serialize_read.c sets it equal to `name` for every function that
             * came from a cached image ("§5 stores no qualified name"), which
             * is almost all of them. The module is on the function either way. */
            char label[160];
            const char *base = fn->name != NULL ? fn->name->chars : "<anon>";
            if (fn->module != NULL && fn->module->name != NULL) {
                snprintf(label, sizeof label, "%s.%s",
                         fn->module->name->chars, base);
            } else {
                snprintf(label, sizeof label, "%s", base);
            }
            fprintf(out, "attrib %-52s %12" PRIu64 "  %6.2f%%\n",
                    label,
                    fn->interpCount,
                    vm.instructionCount > 0
                        ? 100.0 * (double)fn->interpCount
                              / (double)vm.instructionCount
                        : 0.0);
        }
        fprintf(out, "attrib-total %" PRIu64 " of %" PRIu64 " (%" PRIu64
                     " unattributed)\n",
                summed, vm.instructionCount,
                vm.instructionCount > summed ? vm.instructionCount - summed : 0);
    }
    fprintf(out, "vm: %" PRIu64 " instructions, %" PRIu64 " calls, %" PRIu64
                 " allocations\n",
            vm.instructionCount, vm.callCount, vm.allocCount);
    fprintf(out, "inline caches: %" PRIu64 " hits, %" PRIu64 " misses (%.1f%%)\n",
            vm.icHits, vm.icMisses, hitRate);
    uint64_t picTries = vm.jitPicAdmits + vm.jitPicRefusals;
    double picRate = picTries > 0
        ? (100.0 * (double)vm.jitPicAdmits / (double)picTries) : 0.0;
    fprintf(out, "jit pic: %" PRIu64 " admitted, %" PRIu64 " refused (%.1f%%)\n",
            vm.jitPicAdmits, vm.jitPicRefusals, picRate);
    jaiGCPrintStats(out);
}
