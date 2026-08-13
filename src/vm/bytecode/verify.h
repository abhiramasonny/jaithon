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

#endif /* JAI_VM_VERIFY_H */
