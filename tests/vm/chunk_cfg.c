/* chunk_cfg.c -- jaiChunkCfg's basic blocks, over every function in the tree.
 *
 * The CFG has no consumer, so nothing it gets wrong would show up as a wrong
 * answer anywhere: it can only be wrong quietly. That makes the corpus the
 * whole test. Every .jai under lib/, tests/ and packages/ is compiled, every
 * function object in every constant pool is walked -- nested functions,
 * methods, closures, defer thunks -- and each one's graph is checked against
 * properties that hold for any chunk whatsoever.
 *
 * The last group is the one that matters. jaiChunkStackDepths is pass 4 of the
 * verifier, which is trusted because nothing unverified is ever compiled;
 * running it beside the new pass ties the block boundaries to a walk that is
 * already known to be right. A block the CFG invented, an edge it dropped, or
 * a leader it missed all show up there as a block whose depth regime does not
 * match its neighbours'.
 *
 * The verifier's own answer is a fixture here, not a subject: a chunk that
 * fails to verify is counted and skipped, since jaiChunkCfg declines those by
 * contract.
 *
 * Every check below was landed by breaking the pass on purpose and watching
 * this fail. Two facts came out of that and are worth keeping:
 *
 *   - Deleting every OP_LOOP back edge passed EVERY property here until the
 *     edge-set check was added. A loop head is still reachable by its
 *     fall-through, still has a predecessor the verifier reached, and the
 *     smaller graph is still ordered correctly. Reachability cannot see a
 *     missing edge; only counting them can.
 *   - Removing the exception-handler and default-thunk leader rules changed
 *     nothing at all -- 98555 blocks either way. On everything this front end
 *     emits, a handler and a thunk already follow a jump or a return, so they
 *     are leaders twice over. Those two rules are in jaiChunkCfg because they
 *     are part of the definition, not because this corpus tests them.
 *
 * Built by `make chunk-cfg-test`; part of `make test`.
 */
#include "common/common.h"
#include "common/diag.h"
#include "runtime/runtime.h"
#include "vm/bytecode/chunk.h"
#include "vm/bytecode/verify.h"
#include "vm/object/object.h"
#include "vm/vm.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static long gChecks = 0;
static long gFailures = 0;
static long gFunctions = 0;
static long gBlocks = 0;
static long gFiles = 0;
static long gUncompilable = 0;
static long gUnverifiable = 0;

/* Failures are reported by property, not by occurrence: one wrong edge rule
 * fires on thousands of functions, and a page of identical lines hides
 * whatever else broke. */
#define MAX_REPORTS 12
static int gReports = 0;

static void check(bool cond, const char *where, const char *what) {
    gChecks++;
    if (cond) return;
    gFailures++;
    if (gReports++ < MAX_REPORTS) {
        printf("  FAIL %s: %s\n", where, what);
    } else if (gReports == MAX_REPORTS + 1) {
        printf("  ... further failures not printed\n");
    }
}

/* ------------------------------------------------------- the corpus */

static char **gPaths = NULL;
static int gPathCount = 0, gPathCap = 0;

static void addPath(const char *path) {
    if (gPathCount == gPathCap) {
        gPathCap = gPathCap == 0 ? 256 : gPathCap * 2;
        gPaths = realloc(gPaths, (size_t)gPathCap * sizeof *gPaths);
        if (gPaths == NULL) { fprintf(stderr, "FATAL: out of memory\n"); exit(2); }
    }
    gPaths[gPathCount++] = strdup(path);
}

/* __jaicache__ holds .jaic images, not source, and a dot directory holds no
 * corpus at all. Neither belongs in a list of things the front end compiles. */
static bool skipDirectory(const char *name) {
    return name[0] == '.' || strcmp(name, "__jaicache__") == 0;
}

static void collect(const char *dir) {
    DIR *d = opendir(dir);
    if (d == NULL) return;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        char path[4096];
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        snprintf(path, sizeof path, "%s/%s", dir, entry->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            if (!skipDirectory(entry->d_name)) collect(path);
            continue;
        }
        size_t n = strlen(entry->d_name);
        if (n > 4 && strcmp(entry->d_name + n - 4, ".jai") == 0) addPath(path);
    }
    closedir(d);
}

static int comparePaths(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* ------------------------------------------------- an independent decode */

static int instructionWidth(const Chunk *chunk, int offset) {
    uint8_t op = chunk->code[offset];
    int operands = jaiOpOperandSize((OpCode)op);
    if (op == OP_CLOSURE) {
        uint32_t k = jaiReadU24(chunk->code + offset + 1);
        operands = 3 + 3 * (int)AS_FUNCTION(chunk->constants.data[k])->upvalueCount;
    }
    return 1 + operands;
}

/* ------------------------------------------------------------ the checks */

/* Can `from` reach `to` along successor edges? Asked only of a retreating
 * edge's head, which is what separates a real back edge from an order the
 * search produced by accident. */
static bool reaches(const JaiChunkCfg *cfg, uint32_t from, uint32_t to,
                    uint8_t *mark, uint32_t *stack) {
    memset(mark, 0, cfg->blockCount);
    uint32_t top = 0;
    stack[top++] = from;
    mark[from] = 1;
    while (top > 0) {
        uint32_t b = stack[--top];
        for (uint8_t i = 0; i < cfg->blocks[b].nsucc; i++) {
            uint32_t s = cfg->blocks[b].succ[i];
            if (s == to) return true;
            if (mark[s]) continue;
            mark[s] = 1;
            stack[top++] = s;
        }
    }
    return false;
}

static void checkFunction(const ObjFunction *fn, const char *where) {
    const Chunk *chunk = &fn->chunk;
    int n = chunk->count;
    if (chunk->code == NULL || n <= 0) return;
    gFunctions++;

    int *depths = malloc(sizeof(int) * (size_t)(n + 1));
    if (!jaiChunkStackDepths(fn, depths)) {
        gUnverifiable++;
        free(depths);
        return;
    }

    size_t heapBefore = jaiAllocatedBytes();
    JaiChunkCfg *cfg = jaiChunkCfg(fn);
    check(cfg != NULL, where, "jaiChunkCfg declined a chunk that verifies");
    if (cfg == NULL) { free(depths); return; }
    gBlocks += cfg->blockCount;

    uint32_t bc = cfg->blockCount;
    check(bc >= 1, where, "no blocks");
    check(cfg->codeCount == (uint32_t)n, where, "codeCount is not the chunk size");

    /* Tiling: starts and ends chain, with no gap and no overlap. */
    check(cfg->blocks[0].start == 0, where, "the first block does not start at 0");
    check(cfg->blocks[bc - 1].end == (uint32_t)n, where,
          "the last block does not end at the chunk size");
    for (uint32_t b = 0; b < bc; b++) {
        check(cfg->blocks[b].start < cfg->blocks[b].end, where, "an empty block");
        if (b + 1 < bc) {
            check(cfg->blocks[b].end == cfg->blocks[b + 1].start, where,
                  "a gap or an overlap between consecutive blocks");
        }
    }

    /* Every offset maps to exactly one block, and it is the one that contains
     * it. Walked byte by byte rather than by block, so a blockAt that skipped
     * a byte fails here rather than agreeing with itself. */
    for (int offset = 0; offset < n; offset++) {
        uint32_t b = cfg->blockAt[offset];
        bool inside = b < bc && cfg->blocks[b].start <= (uint32_t)offset &&
                      (uint32_t)offset < cfg->blocks[b].end;
        check(inside, where, "blockAt names a block that does not contain the offset");
        if (!inside) break;
    }

    /* Leaders, derived here from the definition rather than read back from the
     * CFG: offset 0, every code address named, the offset after every
     * instruction that names one or does not fall through, every
     * exception-table handler, every default thunk. Both directions are
     * checked, so an extra block start fails as loudly as a missing one. */
    uint8_t *boundary = calloc((size_t)n + 1, 1);
    uint8_t *leader = calloc((size_t)n, 1);
    leader[0] = 1;
    for (int offset = 0; offset < n;) {
        boundary[offset] = 1;
        int width = instructionWidth(chunk, offset);
        int next = offset + width;
        uint8_t op = chunk->code[offset];
        int at = jaiOpBranchOperandAt(op);
        if (at >= 0) {
            int target = (int)((long)offset + width +
                               jaiReadI16(chunk->code + offset + 1 + at));
            if (target >= 0 && target < n) leader[target] = 1;
            if (next < n) leader[next] = 1;
        } else if (!jaiOpFallsThrough(op) && next < n) {
            leader[next] = 1;
        }
        offset = next;
    }
    boundary[n] = 1;
    for (int e = 0; e < (int)fn->exceptionCount; e++) {
        leader[fn->exceptions[e].handler] = 1;
    }
    if (fn->defaultOffsets != NULL) {
        for (int d = 0; d < (int)fn->defaultCount; d++) {
            leader[fn->defaultOffsets[d]] = 1;
        }
    }
    for (uint32_t b = 0; b < bc; b++) {
        check(boundary[cfg->blocks[b].start] != 0, where,
              "a block starts off an instruction boundary");
        check(boundary[cfg->blocks[b].end] != 0, where,
              "a block ends off an instruction boundary");
        check(leader[cfg->blocks[b].start] != 0, where,
              "a block starts at an offset that is not a leader");
    }
    for (int offset = 0; offset < n; offset++) {
        if (!leader[offset]) continue;
        check(cfg->blocks[cfg->blockAt[offset]].start == (uint32_t)offset, where,
              "a leader is not the start of its block");
    }
    for (int offset = 0; offset < n;) {
        int width = instructionWidth(chunk, offset);
        check(cfg->blockAt[offset] == cfg->blockAt[offset + width - 1], where,
              "an instruction straddles a block boundary");
        offset += width;
    }

    /* Successors and predecessors are one relation written down twice. */
    uint32_t edges = 0;
    uint32_t at = 0;
    for (uint32_t b = 0; b < bc; b++) {
        const JaiBlock *block = &cfg->blocks[b];
        check(block->nsucc <= 2, where, "more than two successors");
        check(block->nsucc != 2 || block->succ[0] != block->succ[1], where,
              "the same successor recorded twice");
        for (uint8_t i = 0; i < block->nsucc; i++) {
            check(block->succ[i] < bc, where, "a successor index out of range");
        }
        edges += block->nsucc;
        check(block->predFirst == at, where,
              "predecessor lists are not laid out in block order");
        at += block->predCount;
    }
    check(at == cfg->predCount, where,
          "the predecessor lists do not fill the flattened array");
    check(edges == cfg->predCount, where,
          "the edge count and the predecessor count disagree");
    for (uint32_t b = 0; b < bc; b++) {
        const JaiBlock *block = &cfg->blocks[b];
        for (uint8_t i = 0; i < block->nsucc; i++) {
            uint32_t s = block->succ[i];
            if (s >= bc) continue;
            int found = 0;
            for (uint32_t p = 0; p < cfg->blocks[s].predCount; p++) {
                if (cfg->preds[cfg->blocks[s].predFirst + p] == b) found++;
            }
            check(found == 1, where,
                  "b is a successor of a, but a is not exactly once a predecessor of b");
        }
        for (uint32_t p = 0; p < block->predCount; p++) {
            uint32_t a = cfg->preds[block->predFirst + p];
            check(a < bc, where, "a predecessor index out of range");
            if (a >= bc) continue;
            bool found = false;
            for (uint8_t i = 0; i < cfg->blocks[a].nsucc; i++) {
                if (cfg->blocks[a].succ[i] == b) found = true;
            }
            check(found, where, "a is a predecessor of b, but b is not a successor of a");
        }
    }

    /* The edge set, derived here from the instructions rather than read back
     * from the blocks. This half is a second statement of the same rule, which
     * is usually worth nothing -- it is here because nothing else in this file
     * can see a MISSING edge. Delete every OP_LOOP back edge and the loop head
     * is still reachable by its fall-through, still has a predecessor the
     * verifier reached, and still leaves an order that is topological, so every
     * other check in this file passes; that was measured, not assumed. Only
     * counting the edges catches it. */
    uint32_t *expected = malloc(sizeof(uint32_t) * 2 * bc);
    uint8_t *expectedCount = calloc(bc, 1);
    for (int offset = 0; offset < n;) {
        int width = instructionWidth(chunk, offset);
        int next = offset + width;
        uint8_t op = chunk->code[offset];
        uint32_t b = cfg->blockAt[offset];
        int branchAt = jaiOpBranchOperandAt(op);
        if ((uint32_t)next < cfg->blocks[b].end) {
            check(branchAt < 0 && jaiOpFallsThrough(op), where,
                  "an instruction that ends control flow is not the end of its block");
            offset = next;
            continue;
        }
        if (jaiOpFallsThrough(op) && next < n) {
            expected[2 * b + expectedCount[b]++] = cfg->blockAt[next];
        }
        if (branchAt >= 0) {
            int target = (int)((long)offset + width +
                               jaiReadI16(chunk->code + offset + 1 + branchAt));
            uint32_t to = cfg->blockAt[target];
            if (expectedCount[b] == 0 || expected[2 * b] != to) {
                expected[2 * b + expectedCount[b]++] = to;
            }
        }
        offset = next;
    }
    for (uint32_t b = 0; b < bc; b++) {
        check(expectedCount[b] == cfg->blocks[b].nsucc, where,
              "the block has a different number of successors than its terminator names");
        if (expectedCount[b] != cfg->blocks[b].nsucc) continue;
        for (uint8_t i = 0; i < cfg->blocks[b].nsucc; i++) {
            bool found = false;
            for (uint8_t j = 0; j < expectedCount[b]; j++) {
                if (expected[2 * b + j] == cfg->blocks[b].succ[i]) found = true;
            }
            check(found, where, "a successor the terminator does not name");
        }
    }

    /* Entries, derived here the way the header says the pass derives them: a
     * handler and a default thunk have no in-edge anywhere in the code, so a
     * single-entry search would call most of a `try` body dead. */
    uint8_t *entry = calloc(bc, 1);
    entry[0] = 1;
    for (int e = 0; e < (int)fn->exceptionCount; e++) {
        entry[cfg->blockAt[fn->exceptions[e].handler]] = 1;
    }
    if (fn->defaultOffsets != NULL) {
        for (int d = 0; d < (int)fn->defaultCount; d++) {
            entry[cfg->blockAt[fn->defaultOffsets[d]]] = 1;
        }
    }

    uint8_t *reachable = calloc(bc, 1);
    uint32_t *stack = malloc(sizeof(uint32_t) * bc);
    uint32_t top = 0;
    for (uint32_t b = 0; b < bc; b++) {
        if (entry[b] && !reachable[b]) { reachable[b] = 1; stack[top++] = b; }
    }
    while (top > 0) {
        uint32_t b = stack[--top];
        for (uint8_t i = 0; i < cfg->blocks[b].nsucc; i++) {
            uint32_t s = cfg->blocks[b].succ[i];
            if (reachable[s]) continue;
            reachable[s] = 1;
            stack[top++] = s;
        }
    }

    /* The reverse postorder holds every reachable block exactly once, and
     * nothing else. */
    uint8_t *inRpo = calloc(bc, 1);
    check(cfg->rpoCount <= bc, where,
          "the reverse postorder is longer than the block list");
    for (uint32_t i = 0; i < cfg->rpoCount && i < bc; i++) {
        uint32_t b = cfg->rpo[i];
        check(b < bc, where, "the reverse postorder names a block out of range");
        if (b >= bc) continue;
        check(!inRpo[b], where, "a block appears twice in the reverse postorder");
        inRpo[b] = 1;
        check(cfg->blocks[b].rpoIndex == i, where,
              "rpoIndex disagrees with the block's position in the order");
    }
    for (uint32_t b = 0; b < bc; b++) {
        check(inRpo[b] == reachable[b], where,
              "the reverse postorder and reachability disagree about a block");
        check(inRpo[b] || cfg->blocks[b].rpoIndex == JAI_BLOCK_UNREACHED, where,
              "an unreachable block carries an rpoIndex");
    }

    /* A topological order of the non-back edges. Retreating edges are the ones
     * the order cannot respect; each must be a genuine back edge, which is
     * exactly the head being able to reach the tail again. Without that second
     * half the first is a tautology -- any order at all satisfies "every edge
     * runs forward except the ones that do not". */
    uint8_t *mark = malloc(bc);
    for (uint32_t b = 0; b < bc; b++) {
        if (!reachable[b]) continue;
        for (uint8_t i = 0; i < cfg->blocks[b].nsucc; i++) {
            uint32_t s = cfg->blocks[b].succ[i];
            if (cfg->blocks[b].rpoIndex < cfg->blocks[s].rpoIndex) continue;
            check(reaches(cfg, s, b, mark, stack), where,
                  "an edge runs backwards in the order without lying on a cycle");
        }
    }

    /* --------------------------------- against the verifier's own walk */

    /* jaiChunkStackDepths writes -1 both where nothing reaches an offset and
     * where the depth came from an imprecise seed, and a block is a straight
     * line with one entry, so within a block the two cases cannot mix: every
     * boundary in it is defined, or none is. This is what a leader the pass
     * missed looks like -- a second entry part way in, arriving at a depth the
     * rest of the block does not share. */
    for (uint32_t b = 0; b < bc; b++) {
        bool defined = depths[cfg->blocks[b].start] >= 0;
        for (uint32_t offset = cfg->blocks[b].start; offset < cfg->blocks[b].end;) {
            check((depths[offset] >= 0) == defined, where,
                  "a block's entry depth does not hold across the block");
            offset += (uint32_t)instructionWidth(chunk, (int)offset);
        }
    }

    /* The verifier only ever writes a depth at an offset it walked an edge to,
     * so an offset it reached must be a block this graph reaches too, from a
     * predecessor it also reached. A dropped edge -- the failure
     * OP_GET_ITER_ITEMS shipped with -- shows up here and nowhere else. */
    for (uint32_t b = 0; b < bc; b++) {
        if (depths[cfg->blocks[b].start] < 0) continue;
        check(reachable[b], where,
              "the verifier reached a block this graph calls unreachable");
        if (entry[b]) continue;
        bool fromSomewhere = false;
        for (uint32_t p = 0; p < cfg->blocks[b].predCount; p++) {
            uint32_t a = cfg->preds[cfg->blocks[b].predFirst + p];
            if (depths[cfg->blocks[a].start] >= 0) fromSomewhere = true;
        }
        check(fromSomewhere, where,
              "a block the verifier reached has no predecessor the verifier reached");
    }

    /* And the other direction, which is what catches an INVENTED edge: the
     * walk pushes every successor of an offset it reaches, and all of its
     * precise assignments happen before any imprecise seed, so a block it
     * reached cannot have a successor it did not. */
    for (uint32_t b = 0; b < bc; b++) {
        if (depths[cfg->blocks[b].start] < 0) continue;
        for (uint8_t i = 0; i < cfg->blocks[b].nsucc; i++) {
            check(depths[cfg->blocks[cfg->blocks[b].succ[i]].start] >= 0, where,
                  "a successor of a block the verifier reached was never reached");
        }
    }

    free(mark);
    free(expectedCount);
    free(expected);
    free(inRpo);
    free(stack);
    free(reachable);
    free(entry);
    free(leader);
    free(boundary);
    jaiChunkCfgFree(cfg);
    check(jaiAllocatedBytes() == heapBefore, where,
          "jaiChunkCfgFree did not return every byte jaiChunkCfg took");
    free(depths);
}

/* Every function object in the pool, and every function object in theirs.
 * Nested bodies are where the shapes are -- a module body is mostly straight
 * line, and the loops, matches and try blocks live one level down. */
static void walkFunctions(const ObjFunction *fn, const char *where, int depth) {
    if (fn == NULL || depth > 64) return;
    checkFunction(fn, where);
    for (int i = 0; i < fn->chunk.constants.count; i++) {
        Value v = fn->chunk.constants.data[i];
        if (IS_FUNCTION(v)) walkFunctions(AS_FUNCTION(v), where, depth + 1);
    }
}

static char *readFile(const char *path, size_t *outLength) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 0) { fclose(f); return NULL; }
    char *buffer = malloc((size_t)size + 1);
    if (buffer == NULL) { fclose(f); return NULL; }
    size_t got = fread(buffer, 1, (size_t)size, f);
    fclose(f);
    buffer[got] = '\0';
    *outLength = got;
    return buffer;
}

/* A file the front end rejects is a fixture, not a failure: tests/ holds
 * sources whose whole purpose is to be rejected. */
static void checkFile(const char *path) {
    size_t length = 0;
    char *source = readFile(path, &length);
    if (source == NULL) return;
    gFiles++;

    char name[64];
    snprintf(name, sizeof name, "corpus%ld", gFiles);
    ObjModule *module = jaiModuleNew(jaiStringInternC(name), jaiStringInternC(path));
    jaiPushRoot(OBJ_VAL(module));

    CodegenOptions opts = jaiCodegenDefaults();
    ObjFunction *body = jaiCompileSource(source, length, path, module, &opts);
    if (body == NULL) {
        gUncompilable++;
    } else {
        jaiPushRoot(OBJ_VAL(body));
        walkFunctions(body, path, 0);
        jaiPopRoot();
    }
    jaiPopRoot();
    jaiDiagReset(&gDiags);
    free(source);
}

int main(int argc, char **argv) {
    jaiDiagInit(&gDiags);
    gDiags.colorOutput = false;
    /* The corpus deliberately contains sources the front end must reject, and
     * a good many warnings besides. Rendering them would bury the one line
     * this gate is here to print. */
    gDiags.quiet = true;
    vm.optLevel = 2;
    jaiVMInit();
    jaiModulePathInit(".");
    if (!jaiLoadPrelude()) {
        fprintf(stderr, "FATAL: prelude failed to load\n");
        return 2;
    }

    if (argc > 1) {
        for (int i = 1; i < argc; i++) collect(argv[i]);
    } else {
        collect("lib");
        collect("tests");
        collect("packages");
    }
    qsort(gPaths, (size_t)gPathCount, sizeof *gPaths, comparePaths);

    printf("jaiChunkCfg over %d source files\n", gPathCount);
    for (int i = 0; i < gPathCount; i++) checkFile(gPaths[i]);

    printf("%ld compiled (%ld rejected by the front end), %ld functions, "
           "%ld blocks, %ld unverifiable chunks\n",
           gFiles - gUncompilable, gUncompilable, gFunctions, gBlocks,
           gUnverifiable);
    printf("%ld checks, %ld failures\n", gChecks, gFailures);
    /* A corpus that silently stopped compiling would report zero failures over
     * nothing at all, which is the one way this gate can lie. */
    if (gFunctions < 1000) {
        printf("  FAIL the corpus is too small to be a gate\n");
        return 1;
    }
    return gFailures == 0 ? 0 : 1;
}
