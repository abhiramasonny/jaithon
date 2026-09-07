/* snapshot.c -- can the initialised heap be written out as an image?
 *
 * Every `check`, `fmt`, `ast`, `doc` and `test`, and every edit-then-run, pays
 * ~15-17ms building the self-hosted front end: 98 modules inflated from
 * boot/seed.bin into ~11.4MB of live objects, used for a few milliseconds and
 * thrown away. Mapping that graph back instead of rebuilding it is the one
 * large multiplier found for the developer loop -- 20ms -> ~6ms. The design,
 * its measurements and its corrections are in
 * docs/research/PLAN-startup-snapshot.md.
 *
 * This file is the FIRST step of that work and deliberately writes nothing. It
 * answers the question the design cannot answer by reading: at the moment the
 * front end is built, does the live heap actually satisfy the preconditions a
 * heap image needs? Those preconditions are not guesses -- each one below is a
 * pointer that would be copied into the image and be WRONG on the way back:
 *
 *   - an ObjFile holding a live FILE *, which cannot be serialised at all;
 *   - ObjFunction::jitCode / jitFunc / jitLoop and JaiOsrForm::code, raw
 *     addresses in the JIT's per-process executable arena with no by-name
 *     rebind path (unlike ObjNative::fn, which has one). blackenFunction
 *     deliberately does not mark through these, so a writer that walks
 *     "whatever the GC walks" would copy them silently;
 *   - Chunk::constIndex, which markChunk also skips, for the same reason;
 *   - an OPEN ObjUpvalue, whose `location` points into the VM's value stack;
 *   - an ObjNative with no name, which the restore path could not rebind.
 *
 * Enumeration is the collector's own live list rather than a second traversal:
 * one forced collection leaves `objects` holding exactly the survivors, so this
 * cannot disagree with the GC about what is live -- which a hand-written walk
 * eventually would.
 *
 * Nothing here runs unless JAITHON_SNAPSHOT_AUDIT is set. */

#include <stdio.h>
#include <stdlib.h>

#include "vm/vm.h"
#include "vm/gc.h"
#include "vm/object/object.h"

void jaiSnapshotAudit(const char *when);

void jaiSnapshotAudit(const char *when) {
    if (getenv("JAITHON_SNAPSHOT_AUDIT") == NULL) return;
    if (vm.gc == NULL) return;

    /* Survivors only: a sweep has just freed everything unreachable, so what is
     * left on the list is precisely the set an image would have to carry. */
    jaiGCCollect();

    unsigned long long byType[OBJ_TYPE_COUNT];
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) byType[i] = 0;

    unsigned long long total = 0;
    unsigned long long openFiles = 0, closedFiles = 0;
    unsigned long long jitCode = 0, jitLoops = 0, osrForms = 0;
    unsigned long long constIndex = 0, openUpvalues = 0, namelessNatives = 0;

    for (Obj *o = vm.gc->objects; o != NULL; o = o->next) {
        total++;
        if (o->type < OBJ_TYPE_COUNT) byType[o->type]++;

        switch (o->type) {
        case OBJ_FILE: {
            ObjFile *f = (ObjFile *)o;
            if (f->handle != NULL) openFiles++; else closedFiles++;
            break;
        }
        case OBJ_FUNCTION: {
            ObjFunction *fn = (ObjFunction *)o;
            if (fn->jitCode != NULL || fn->jitFunc != NULL) jitCode++;
            if (fn->jitLoop != NULL) jitLoops++;
            if (fn->osrForms != NULL) osrForms++;
            if (fn->chunk.constIndex != NULL) constIndex++;
            break;
        }
        case OBJ_UPVALUE: {
            ObjUpvalue *uv = (ObjUpvalue *)o;
            /* An upvalue is CLOSED when its location points at its own `closed`
             * field; anything else is a pointer into the value stack. */
            if (uv->location != &uv->closed) openUpvalues++;
            break;
        }
        case OBJ_NATIVE: {
            ObjNative *n = (ObjNative *)o;
            if (n->name == NULL) namelessNatives++;
            break;
        }
        default:
            break;
        }
    }

    static const char *const kindName[OBJ_TYPE_COUNT] = {
        "string", "bytes", "list", "dict", "set", "tuple", "range",
        "function", "closure", "upvalue", "native", "bound",
        "class", "trait", "instance", "module", "enum", "enumval",
        "iter", "file", "enumctor", "strbuf",
    };

    fprintf(stderr, "[snapshot] audit at %s: %llu live objects\n", when, total);
    for (int i = 0; i < OBJ_TYPE_COUNT; i++) {
        if (byType[i] != 0)
            fprintf(stderr, "[snapshot]   %-10s %llu\n",
                    kindName[i] != NULL ? kindName[i] : "?", byType[i]);
    }
    fprintf(stderr,
            "[snapshot] blockers: open-file=%llu closed-file=%llu "
            "jit-code=%llu jit-loop=%llu osr-forms=%llu const-index=%llu "
            "open-upvalue=%llu nameless-native=%llu\n",
            openFiles, closedFiles, jitCode, jitLoops, osrForms, constIndex,
            openUpvalues, namelessNatives);
    fprintf(stderr, "[snapshot] verdict: %s\n",
            (openFiles == 0 && jitCode == 0 && jitLoops == 0 &&
             osrForms == 0 && constIndex == 0 && openUpvalues == 0 &&
             namelessNatives == 0)
                ? "writable -- no unserialisable pointer is reachable"
                : "NOT writable as-is; the counts above say what must be "
                  "reset or refused");
}
