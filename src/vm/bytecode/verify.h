#ifndef JAI_VM_VERIFY_H
#define JAI_VM_VERIFY_H

#include "vm/object/object.h"

/* Check `fn`'s chunk. False writes the first problem into `errBuf`. Applies
 * equally to a .jaic image and a chunk the compiler just built. */
bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize);

/* Byte index of the i16 branch operand inside `op`'s operand run, or -1 when
 * the instruction carries no code address. Exported because the JIT needs the
 * same fact; a second, drifted copy would miscompile rather than just fail. */
int jaiOpBranchOperandAt(uint8_t op);

#endif /* JAI_VM_VERIFY_H */
