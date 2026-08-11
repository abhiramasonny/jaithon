#ifndef JAI_VM_VERIFY_H
#define JAI_VM_VERIFY_H

#include "object.h"

/* Check `fn`'s chunk. False writes the first problem into `errBuf`.
 *
 * Part of the bytecode contract, not of any front end: a .jaic image loaded
 * from disk gets the same check as a chunk the compiler just built. */
bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize);

/* Byte index of the i16 branch operand inside `op`'s operand run, or -1 when
 * the instruction carries no code address.
 *
 * Exported because the JIT needs the same fact and a second copy of it would be
 * a miscompile rather than a decline: a jump this list forgot lands on an
 * offset the tier believed nothing could reach, and the tier would then fold a
 * constant that only one of the two paths pushed. */
int jaiOpBranchOperandAt(uint8_t op);

#endif /* JAI_VM_VERIFY_H */
