/* field_natives.c — the compiled tier reads these builtins' fields directly.
 *
 * `src/vm/jit/jit_field_read.h` lets the JIT replace a whole call with one
 * load: `s.len()` becomes `ldr w, [recv, #offsetof(ObjString, scalars)]`.
 * Nothing in the C type system connects that offset to `strLen`, and nothing
 * connects the declared result kind to what the native actually returns —
 * roadmap.md §6 records "every builtin named len returns an int" as an
 * invariant that was true everywhere and checked nowhere. This is the check.
 *
 * For every entry: build a real receiver of that type, resolve the method the
 * way the JIT resolves it (jaiBuiltinMethod on the observed receiver), call it
 * the way the interpreter calls it, and require that
 *
 *   - the Value it returns carries the tag the table declares, and
 *   - its payload equals the field the table names, loaded exactly as the
 *     emitted code loads it (zero-extending, since that is what `ldr w` does).
 *
 * A `lazy` entry is checked twice: once with the memo empty, where the field
 * must hold the sentinel and the emitted code must therefore call out, and
 * once after, where the load must now answer. A sentinel that stopped being
 * reachable would make the JIT's slow path dead and is worth knowing about,
 * so an empty-memo receiver that does NOT hold the sentinel is a failure.
 *
 * Exhaustive by construction: the table is walked, not a list of cases, and a
 * type with no receiver builder here fails rather than being skipped.
 *
 * Built by `make field-natives-test`; part of `make test`.
 */
#include "common/common.h"
#include "common/diag.h"
#include "runtime/runtime.h"
#include "vm/jit/jit_field_read.h"
#include "vm/object/object.h"
#include "vm/vm.h"

#include <stdio.h>
#include <string.h>

static int gFailures = 0;
static int gChecks = 0;

static void fail(const JaiJitFieldRead *r, const char *what) {
    fprintf(stderr, "FAIL: %s.%s: %s\n", jaiObjTypeName(r->type), r->name,
            what);
    gFailures++;
}

/* The same load the invoke arm emits: `ldr w` zero-extends, `ldr x` does not
 * sign-extend either. Every field named in the table is a count. */
static int64_t fieldLoad(const JaiJitFieldRead *r, Value receiver) {
    const char *base = (const char *)AS_OBJ(receiver);
    if (r->width == 4) {
        uint32_t v;
        memcpy(&v, base + r->offset, sizeof v);
        return (int64_t)(uint64_t)v;
    }
    uint64_t v;
    memcpy(&v, base + r->offset, sizeof v);
    return (int64_t)v;
}

/* Resolve and call exactly as the runtime does, so a native that changed its
 * receiver handling is caught here rather than in a benchmark. */
static bool callIt(const JaiJitFieldRead *r, Value receiver, Value *out) {
    Value bound;
    if (!jaiBuiltinMethod(receiver, jaiStringInternC(r->name), &bound)) {
        fail(r, "jaiBuiltinMethod does not resolve this name on this type");
        return false;
    }
    Value native = IS_BOUND(bound) ? AS_BOUND(bound)->method : bound;
    if (!IS_NATIVE(native)) {
        fail(r, "resolves to something that is not a native");
        return false;
    }
    Value args[1] = {receiver};
    if (!jaiInvokeNativeWithReceiver(native, args, 1, out)) {
        fail(r, "the native raised on a plain receiver");
        return false;
    }
    return true;
}

static void checkAgrees(const JaiJitFieldRead *r, Value receiver,
                        const char *what) {
    gChecks++;
    Value got;
    if (!callIt(r, receiver, &got)) return;

    if (got.type != r->tag) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "%s: the table declares tag %d, the native returned %d", what,
                 (int)r->tag, (int)got.type);
        fail(r, msg);
        return;
    }
    int64_t field = fieldLoad(r, receiver);
    if (field != AS_INT(got)) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "%s: the field at +%u reads %lld, the native returned %lld",
                 what, (unsigned)r->offset, (long long)field,
                 (long long)AS_INT(got));
        fail(r, msg);
    }
}

/* A receiver of `type` holding `n` things, built so that **no other word in
 * its header holds the same number** -- a list whose capacity equals its count
 * cannot tell `count` from `capacity`, and the first version of this test
 * passed with the table pointing at the wrong one of the two. `headerBytes` is
 * how far the fixed part of the object reaches; the check below scans it. */
static bool buildReceiver(ObjType type, int n, Value *out,
                          size_t *headerBytes) {
    switch (type) {
    case OBJ_STRING: {
        /* Two bytes per scalar, so `length` and `scalars` cannot be confused.
         * jaiStringNew rather than the interning form: an interned literal may
         * already have been asked for its scalar count elsewhere, and the
         * empty memo is half of what this test is checking. */
        char buf[128];
        size_t at = 0;
        for (int i = 0; i < n && at + 2 < sizeof buf; i++) {
            buf[at++] = (char)0xC3;      /* U+00E9, two bytes */
            buf[at++] = (char)0xA9;
        }
        *out = OBJ_VAL(jaiStringNew(buf, at));
        *headerBytes = sizeof(ObjString);
        return true;
    }
    case OBJ_LIST: {
        /* Capacity deliberately past the count. */
        ObjList *l = jaiListNew(n + 3);
        for (int i = 0; i < n; i++) jaiListSetRaw(l, i, INT_VAL(1000 + i));
        l->count = n;
        *out = OBJ_VAL(l);
        *headerBytes = sizeof(ObjList);
        return true;
    }
    case OBJ_DICT: {
        /* Two extra keys, then deleted: count, orderCount and tombstones all
         * end up different numbers. */
        ObjDict *d = jaiDictNew();
        jaiPushRoot(OBJ_VAL(d));
        for (int i = 0; i < n + 2; i++) jaiDictSet(d, INT_VAL(i), INT_VAL(i));
        for (int i = n; i < n + 2; i++) jaiDictDelete(d, INT_VAL(i));
        jaiPopRoot();
        *out = OBJ_VAL(d);
        *headerBytes = sizeof(ObjDict);
        return true;
    }
    case OBJ_SET: {
        ObjSet *s = jaiSetNew();
        jaiPushRoot(OBJ_VAL(s));
        for (int i = 0; i < n + 2; i++) jaiSetAdd(s, INT_VAL(i));
        for (int i = n; i < n + 2; i++) jaiSetDelete(s, INT_VAL(i));
        jaiPopRoot();
        *out = OBJ_VAL(s);
        *headerBytes = sizeof(ObjSet);
        return true;
    }
    case OBJ_TUPLE: {
        Value items[64];
        for (int i = 0; i < n && i < 64; i++) items[i] = INT_VAL(1000 + i);
        *out = OBJ_VAL(jaiTupleNew(items, n < 64 ? n : 64));
        *headerBytes = sizeof(ObjTuple);
        return true;
    }
    case OBJ_BYTES: {
        uint8_t data[64];
        for (int i = 0; i < n && i < 64; i++) data[i] = (uint8_t)(200 + i);
        *out = OBJ_VAL(jaiBytesNew(data, (size_t)(n < 64 ? n : 64)));
        *headerBytes = sizeof(ObjBytes);
        return true;
    }
    default:
        return false;
    }
}

/* True when no OTHER aligned word of the header holds the same number as the
 * field the table names -- i.e. this receiver can actually tell them apart. At
 * least one of the sizes tried must manage it, or the test proves nothing. */
static bool discriminates(const JaiJitFieldRead *r, Value receiver,
                          size_t headerBytes, size_t *aliasAt) {
    const char *base = (const char *)AS_OBJ(receiver);
    int64_t want = fieldLoad(r, receiver);
    for (size_t at = 0; at + r->width <= headerBytes; at += 4) {
        if (at == r->offset) continue;
        int64_t other;
        if (r->width == 4) {
            uint32_t v;
            memcpy(&v, base + at, sizeof v);
            other = (int64_t)(uint64_t)v;
        } else {
            uint64_t v;
            memcpy(&v, base + at, sizeof v);
            other = (int64_t)v;
        }
        if (other == want) { *aliasAt = at; return false; }
    }
    return true;
}

int main(void) {
    jaiDiagInit(&gDiags);
    gDiags.colorOutput = false;
    vm.optLevel = 2;
    jaiVMInit();
    jaiModulePathInit(".");
    if (!jaiLoadPrelude()) {
        fprintf(stderr, "FATAL: prelude failed to load\n");
        return 2;
    }

    printf("jit field-reading builtins\n");
    for (int i = 0; i < JAI_JIT_FIELD_READ_COUNT; i++) {
        const JaiJitFieldRead *r = &JAI_JIT_FIELD_READS[i];

        gChecks++;
        if (r->width != 4 && r->width != 8) {
            fail(r, "width is neither 4 nor 8");
            continue;
        }
        if (r->argc != 0) {
            /* callIt passes the receiver alone; an entry taking arguments
             * needs this test widened before it can be trusted. */
            fail(r, "this test only covers a zero-argument builtin");
            continue;
        }

        /* Three sizes, so an entry pointing at a neighbouring field that
         * happens to agree on one of them still fails. 0 is included because
         * an empty container is where a stray field reads as a plausible
         * count. */
        const int sizes[] = {0, 1, 7, 19, 23};
        bool discriminated = false;
        bool sawUnset = false;
        size_t aliasAt = 0;
        for (unsigned s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
            Value recv;
            size_t headerBytes = 0;
            if (!buildReceiver(r->type, sizes[s], &recv, &headerBytes)) {
                fail(r, "this test has no receiver builder for that type");
                break;
            }
            jaiPushRoot(recv);
            if (OBJ_TYPE(recv) != r->type) {
                fail(r, "the receiver this test built is not that type");
                jaiPopRoot();
                break;
            }

            /* An INTERNED receiver is not fresh, whatever this test did to
             * build it. jaiStringNew still interns anything short, so at the
             * small sizes below the object that comes back is whichever one
             * the process already had -- and on a COLD __jaicache__ the
             * prelude compile has already asked it for its scalar count, so
             * the memo is full and the sentinel is gone. That made this check
             * pass on a warm cache and fail 5/5 on a cold one, i.e. fail on
             * every fresh clone. The sentinel is only meaningful for a
             * receiver nothing else can be holding. */
            bool freshEnough = OBJ_TYPE(recv) != OBJ_STRING ||
                               !JAI_STR_INTERNED(AS_STRING(recv));
            if (r->lazy && freshEnough) {
                sawUnset = true;
                gChecks++;
                int64_t before = fieldLoad(r, recv);
                if (before != (int64_t)(uint64_t)r->sentinel) {
                    /* Not merely unhelpful: it means the emitted fast path
                     * answers from a field whose "unset" state the table has
                     * stopped describing. */
                    fail(r, "a freshly built receiver does not hold the "
                            "declared sentinel");
                }
            }
            checkAgrees(r, recv, r->lazy ? "with the memo empty" : "");
            if (r->lazy) checkAgrees(r, recv, "with the memo filled");
            if (discriminates(r, recv, headerBytes, &aliasAt)) {
                discriminated = true;
            }
            jaiPopRoot();
        }
        if (r->lazy) {
            /* Skipping every size would retire the sentinel check without
             * anyone noticing -- the same failure mode as the alias check
             * below. At least one size must produce a non-interned receiver. */
            gChecks++;
            if (!sawUnset) {
                fail(r, "every receiver this test built was interned, so the "
                        "empty-memo case was never exercised");
            }
        }
        gChecks++;
        if (!discriminated) {
            /* Without this the test reads as coverage while proving nothing:
             * the first version of it passed with OBJ_LIST pointing at
             * `capacity` instead of `count`, because jaiListNew(n) made the
             * two equal at every size it tried. */
            char msg[160];
            snprintf(msg, sizeof msg,
                     "no receiver this test builds tells the field at +%u "
                     "apart from the word at +%u",
                     (unsigned)r->offset, (unsigned)aliasAt);
            fail(r, msg);
        }
    }

    printf("%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
