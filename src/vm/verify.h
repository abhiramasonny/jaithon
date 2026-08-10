#ifndef JAI_VM_VERIFY_H
#define JAI_VM_VERIFY_H

#include "object.h"

/* Check `fn`'s chunk. False writes the first problem into `errBuf`.
 *
 * Part of the bytecode contract, not of any front end: a .jaic image loaded
 * from disk gets the same check as a chunk the compiler just built. */
bool jaiVerifyChunk(const ObjFunction *fn, char *errBuf, size_t errBufSize);

#endif /* JAI_VM_VERIFY_H */
