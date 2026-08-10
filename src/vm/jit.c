#include "jit.h"

#include <stdlib.h>
#include <string.h>

bool jaiJitEnabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *off = getenv("JAITHON_NO_JIT");
        cached = (off != NULL && off[0] != '\0' && strcmp(off, "0") != 0) ? 0 : 1;
    }
    return cached != 0;
}

/* Always declines. The interpreter then runs the function, which is what makes
 * this safe to land: the counter and the threshold are exercised by the whole
 * test suite while the answer is still produced entirely by the interpreter. */
bool jaiJitEnter(ObjClosure *closure, Value *slotBase) {
    (void)closure;
    (void)slotBase;
    return false;
}
