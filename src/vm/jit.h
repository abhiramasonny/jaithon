#ifndef JAI_VM_JIT_H
#define JAI_VM_JIT_H

#include "object.h"

/* The compiled tier.
 *
 * The interpreter stays authoritative. The JIT is an accelerator that may
 * always decline: `jaiJitEnter` answers false and the interpreter runs the
 * function exactly as it would have. Every stage of this is built that way, so
 * a tier that is wrong is slow rather than fatal, and so the suite can stay
 * green while the compiler behind it is still a stub.
 *
 * Nothing here compiles anything yet. This is the plumbing -- the entry count,
 * the threshold, the decline path -- proved on its own before a byte of machine
 * code exists. */

/* How many entries before a function is considered hot. Arbitrary for now; the
 * only requirement is that it is high enough that the counter itself does not
 * cost anything on cold code and low enough that a benchmark reaches it. */
#define JAI_JIT_THRESHOLD 64

/* Called on entry to a Jaithon function once it has crossed the threshold.
 *
 * Returns false when there is no compiled form, which is always, today. A true
 * answer will mean the function ran to completion in compiled code and left its
 * result where the interpreter expects it. */
bool jaiJitEnter(ObjClosure *closure, Value *slotBase);

/* Whether the tier is enabled at all. JAITHON_NO_JIT=1 turns it off, so a
 * measurement can be taken against the interpreter without rebuilding. */
bool jaiJitEnabled(void);

#endif /* JAI_VM_JIT_H */
