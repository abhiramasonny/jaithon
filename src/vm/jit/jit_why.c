/* jit_why.c -- saying why a body declined: the named reason, the local whose
 * kind would not settle, and the chain of refusals behind the first one. */
#include "vm/jit/jit.h"

#include "vm/jit/jit_arm64.h"
#include "vm/jit/jit_field_read.h"
#include "vm/gc.h"
#include "runtime/runtime.h"
#include "vm/bytecode/verify.h"
#include "vm/vm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "vm/jit/jit_internal.h"

#if (defined(__aarch64__) || defined(__arm64__))

/* JAI_JIT_CHAIN=1: print the whole chain of refusals a body would hit, not just
 * the first one.
 *
 * "What would this body stop at NEXT?" is the question that decides whether an
 * arm is worth building, and until now it was answered by BUILDING the arm and
 * re-running -- a day per link, and how three separate changes came to measure
 * exactly zero after clearing one link of a longer chain.
 *
 * The mechanism is deliberately dumb: recompile the body with the offending
 * offset forced onto the unarmed path, and see what it says next. That reuses a
 * path the tier already exercises constantly, rather than continuing a walk
 * whose model has gone inconsistent -- which was tried, and segfaults.
 *
 * Diagnostic only. Each link costs one extra compile of one body, and nothing
 * here runs unless the env var is set. */
bool jitChainOn(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAI_JIT_CHAIN");
        cached = (v != NULL && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

void reportChain(const Emit *proto, Emit *first, ObjClosure *closure,
                        ObjFunction *fn) {
    static Emit probe;
    uint32_t skips[JIT_MAX_CHAIN];
    unsigned n = 0;
    const char *name = fn->name != NULL ? fn->name->chars : "<anon>";

    fprintf(stderr, "[jit] chain %s:\n", name);
    /* Link 1 is a fact. Everything below it is a PROBE, and the probe is not
     * the same thing as a fix.
     *
     * Stepping over an instruction takes the unarmed path, which abandons the
     * rest of that straight-line block and can only resume at a later
     * independently-reachable jump target. If a bind lived in the abandoned
     * part, the walk reaches the next link with that local never bound at all
     * -- and then reports a refusal ("local N has kind int, not instance")
     * that a genuinely fixed link 1 would never have produced. One night's
     * `.len()` chain read that way and the link 2 it named was an artefact.
     *
     * So: chase link 1. Treat the rest as a hint about where to look next,
     * never as a list of things that must all be cleared. */
    fprintf(stderr, "[jit]   (link 1 is measured; the links below are probed "
                    "by forcing it unarmed,\n[jit]    which skips the rest of "
                    "its block -- treat them as hints, not facts)\n");
    fprintf(stderr, "[jit]   1. %s  (at %u)\n", declineReason(first),
            first->curOffset);
    skips[n++] = first->curOffset;

    for (unsigned link = 2; link <= JIT_MAX_CHAIN; link++) {
        memcpy(&probe, proto, sizeof probe);
        memcpy(probe.chainSkip, skips, n * sizeof skips[0]);
        probe.chainSkipCount = n;
        if (compileBody(&probe, closure)) {
            fprintf(stderr, "[jit]   %u. compiles, once the %u above %s "
                            "cleared\n", link, n, n == 1 ? "is" : "are");
            return;
        }
        /* Refusing again at the SAME offset means the unarmed path cannot step
         * over that instruction: deoptSite has nowhere to resume, which is a
         * real property of the instruction and not an artefact of this probe.
         * Say so rather than numbering it as the next link, because it is not
         * one -- it is where the walk stops being able to look. */
        if (probe.curOffset == skips[n - 1]) {
            fprintf(stderr,
                    "[jit]   ... cannot look past link %u: stepping over it "
                    "gives \"%s\"\n", n, declineReason(&probe));
            return;
        }
        fprintf(stderr, "[jit]   %u. %s  (at %u)\n", link,
                declineReason(&probe), probe.curOffset);
        if (n >= JIT_MAX_CHAIN) {
            fprintf(stderr, "[jit]   ... and the chain runs longer than %u\n",
                    JIT_MAX_CHAIN);
            return;
        }
        skips[n++] = probe.curOffset;
    }
}

/* Which local the tier could not settle on one kind for.
 *
 * The reason on its own says a body has such a local but not which, and a
 * body with twenty of them then has to be read line by line to find it. The
 * slot number is what the disassembly labels its locals with, so the two can
 * be put side by side.
 */
/* Record which unnamed refusal an arm took, and return false so the call sites
 * read as `return subWhy(e, "...")`. See Emit::whySub. */
bool subWhy(Emit *e, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->whySub, sizeof e->whySub, fmt, ap);
    va_end(ap);
    return false;
}

/* What `JAI_JIT_WHY` prints: the named reason when there is one, otherwise the
 * opcode with whatever the arm noted about it. */
const char *declineReason(Emit *e) {
    if (e->whyNot != NULL) return e->whyNot;
    const char *name = jaiOpName((OpCode)e->lastOp);
    if (e->whySub[0] == '\0') return name;
    snprintf(e->whyBuf, sizeof e->whyBuf, "%s: %s", name, e->whySub);
    return e->whyBuf;
}

bool jitCollectClashes(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("JAITHON_JIT_COLLECT_CLASHES");
        cached = (v != NULL && v[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

const char *kindClash(Emit *e, unsigned slot) {
    /* WHICH two kinds, not just which slot. The slot number says where to look
     * and the pair says what to do about it: an int meeting a float is a
     * widening the tier could learn, an instance meeting a list is a genuinely
     * polymorphic local and nothing will help it. Without the pair, ~290 of
     * these across four compiler files were one undifferentiated heap. */
    snprintf(e->whyBuf, sizeof e->whyBuf,
             "local %u was given two kinds, %s and %s", slot,
             slotKindName(e->localKind[slot]), slotKindName(e->clashKind));
    return e->whyBuf;
}

/* What an unarmed opcode was ABOUT, when the opcode alone does not say.
 *
 * `walked only to OP_GET_GLOBAL at 53` names the instruction and not the
 * question. The question is WHICH global, because that is what decides whether
 * the stop is the deliberate one on a cold `throw` path or a callee that could
 * have compiled: `sqrt` and `floor` each stop at one, and together they are
 * 4.2% of the jaicv benchmark's interpreted work. */
const char *unarmedDetail(const ObjFunction *fn, uint8_t op,
                                 uint32_t at) {
    if (op != OP_GET_GLOBAL) return "";
    if ((size_t)at + 4 > (size_t)fn->chunk.count) return "";
    uint32_t idx = jaiReadU24(fn->chunk.code + at + 1);
    if (idx >= (uint32_t)fn->chunk.constants.count) return "";
    Value name = fn->chunk.constants.data[idx];
    if (!IS_STRING(name)) return "";
    static char buf[96];
    snprintf(buf, sizeof buf, " (`%s`)", AS_STRING(name)->chars);
    return buf;
}

#endif /* __aarch64__ || __arm64__ */
