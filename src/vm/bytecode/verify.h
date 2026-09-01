#ifndef JAI_VM_VERIFY_H
#define JAI_VM_VERIFY_H

#include "vm/object/object.h"

/* Check `fn`'s chunk. False writes the first problem into `errBuf`. Applies
 * equally to a .jaic image and a chunk the compiler just built. */
bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize);

/* The operand-stack depth at every instruction boundary of `fn`, written into
 * `out` (which must hold fn->chunk.count + 1 ints); -1 where no path reaches
 * that offset, or where an unmodelled opcode stopped the walk. False means the
 * chunk did not verify and nothing was written.
 *
 * This is pass 4 of jaiVerifyChunk, which computes it and throws it away. The
 * JIT needs it as an INDEPENDENT oracle for its own operand model: the model is
 * maintained by ~140 hand-written arms, and an arm that leaves it at the wrong
 * depth writes deopt records naming an operand stack the interpreter does not
 * have. Exported rather than re-derived so the two cannot drift -- the same
 * argument jaiOpBranchOperandAt is exported under. */
bool jaiChunkStackDepths(const ObjFunction *fn, int *out);

/* Byte index of the i16 branch operand inside `op`'s operand run, or -1 when
 * the instruction carries no code address. Exported because the JIT needs the
 * same fact; a second, drifted copy would miscompile rather than just fail. */
int jaiOpBranchOperandAt(uint8_t op);

/* False when control cannot reach the NEXT instruction from this one. Exported
 * for the same reason as the two above: the JIT's linear walk has to know where
 * its fall-through edge ends, and a second copy of the list would drift. */
bool jaiOpFallsThrough(uint8_t op);


/* ------------------------------------------------------------------ */
/* The control-flow graph                                              */
/* ------------------------------------------------------------------ */

/* Basic blocks of one ObjFunction's chunk, with edges both ways and a reverse
 * postorder. Built by an INDEPENDENT leader scan: pass 4 above already walks
 * this graph, but it walks it fused with a depth fixpoint whose orderings are
 * safety-critical, so it computes the edges and throws them away rather than
 * being restructured to keep them. The two passes share `jaiOpBranchOperandAt`,
 * `jaiOpFallsThrough` and the OP_CLOSURE operand width, which is where drift
 * between them would come from.
 *
 * Deliberately NOT here, because nothing consumes this yet and an unused
 * approximation is a liability rather than a feature: dominators, loop
 * detection, an edge from every instruction of a protected region to its
 * handler (the exception-table handlers are entry points instead), and any
 * notion of the operand stack -- for that, ask jaiChunkStackDepths, which is
 * the pass that is already trusted. */
typedef struct {
    uint32_t start, end;    /* [start, end) byte offsets */
    /* At most two: the fall-through and the one code address an instruction
     * can name. Deduplicated, so a conditional branch onto its own
     * fall-through has one successor, not the same block twice. */
    uint32_t succ[2];
    uint8_t  nsucc;
    uint32_t predFirst, predCount;   /* into JaiChunkCfg::preds */
    /* Position in `rpo`, or JAI_BLOCK_UNREACHED for a block no entry reaches.
     * Unreachable blocks exist: dead code still decodes, and still gets a
     * block, because the offset-to-block map has to be total. */
    uint32_t rpoIndex;
} JaiBlock;

#define JAI_BLOCK_UNREACHED UINT32_MAX

typedef struct {
    JaiBlock *blocks;
    uint32_t  blockCount;
    /* Predecessor lists, concatenated; block i owns
     * preds[predFirst .. predFirst + predCount). */
    uint32_t *preds;
    uint32_t  predCount;
    /* Reverse postorder of a depth-first search from every entry: block 0
     * first, then each exception-table handler, then each default thunk, in
     * table order. Holds only the blocks that search reaches, so
     * rpoCount <= blockCount, so the two are NOT interchangeable and the
     * storage length is carried explicitly rather than inferred. Freeing this
     * with rpoCount looks right, passes the whole corpus gate -- every function
     * this front end emits has all its blocks reachable, so the two are equal
     * in practice -- and hands jaiRealloc a size class the block did not come
     * from. That is heap corruption, not a wrong number, and no amount of
     * running the gate would find it. */
    uint32_t *rpo;
    uint32_t  rpoCount;
    uint32_t  rpoCapacity;
    /* Total: blockAt[offset] is the block containing `offset` for every byte
     * of the code, not only instruction boundaries. */
    uint32_t *blockAt;
    uint32_t  codeCount;
} JaiChunkCfg;

/* NULL when `fn`'s chunk does not verify -- every offset this walks is one
 * jaiVerifyChunk has already proved in range and on a boundary, so the scan
 * itself carries no bounds checks. The caller owns the result. */
JaiChunkCfg *jaiChunkCfg(const ObjFunction *fn);

void jaiChunkCfgFree(JaiChunkCfg *cfg);

#endif /* JAI_VM_VERIFY_H */
