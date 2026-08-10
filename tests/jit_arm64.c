/* Every encoder is checked by running the instruction it produces.
 *
 * Inspecting a hand-computed bit pattern proves nothing: a wrong field lands in
 * a neighbouring operand and the instruction still executes, just not the one
 * intended. So each case here assembles a tiny function, runs it, and checks
 * the answer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "src/vm/jit.h"
#include "src/vm/jit_arm64.h"

#if !(defined(__aarch64__) || defined(__arm64__))
int main(void) { printf("jit_arm64: skipped (not arm64)\n"); return 0; }
#else

static JaiCodeArena arena;

/* Build a function from `words` and call it with one pointer argument. */
static int64_t runWith(const uint32_t *words, size_t count, void *arg) {
    JaiCodeArena a;
    if (!jaiCodeArenaInit(&a, 4096)) { fprintf(stderr, "mmap failed\n"); exit(1); }
    if (jaiCodeArenaWrite(&a, words, count * sizeof words[0]) == NULL) exit(1);
    if (!jaiCodeArenaSeal(&a)) exit(1);
    /* mmap returns page-aligned memory, so this is sound; the cast goes
     * through uintptr_t to say so rather than silencing -Wcast-align. */
    int64_t (*fn)(void *) = (int64_t (*)(void *))(uintptr_t)a.code;
    int64_t r = fn(arg);
    /* Deliberately not freed: the mapping outlives the call and the process is
     * about to end. Freeing here would be the only interesting thing that could
     * go wrong in a test that is checking something else. */
    return r;
}

static int failures;

static void check(const char *what, int64_t got, int64_t want) {
    if (got != want) {
        fprintf(stderr, "jit_arm64: %s = %lld, want %lld\n", what,
                (long long)got, (long long)want);
        failures++;
    }
}

int main(void) {
    (void)arena;
    int64_t cell[8] = { 11, 22, 33, 44, 0, 0, 0, 0 };

    /* ldr x0, [x0, #8] ; ret */
    { const uint32_t w[] = { jaiA64LdrX(0, 0, 8), jaiA64Ret() };
      check("ldr x0,[x0,#8]", runWith(w, 2, cell), 22); }

    /* ldr x0, [x0, #24] ; ret  -- a larger scaled offset */
    { const uint32_t w[] = { jaiA64LdrX(0, 0, 24), jaiA64Ret() };
      check("ldr x0,[x0,#24]", runWith(w, 2, cell), 44); }

    /* ldr x1,[x0,#0] ; add x0,x1,#5 ; ret */
    { const uint32_t w[] = { jaiA64LdrX(1, 0, 0), jaiA64AddXImm(0, 1, 5),
                             jaiA64Ret() };
      check("add imm", runWith(w, 3, cell), 16); }

    /* str: write 99 into cell[4], then read it back */
    { const uint32_t w[] = { jaiA64MovzX(1, 99, 0), jaiA64StrX(1, 0, 32),
                             jaiA64LdrX(0, 0, 32), jaiA64Ret() };
      check("movz/str/ldr", runWith(w, 4, cell), 99);
      check("cell written", cell[4], 99); }

    /* movz + movk builds a wide constant: 0x0001_0002 */
    { const uint32_t w[] = { jaiA64MovzX(0, 2, 0), jaiA64MovkX(0, 1, 1),
                             jaiA64Ret() };
      check("movz+movk", runWith(w, 3, cell), 0x10002); }

    /* sdiv + msub is the remainder: 33 % 7 == 5 */
    { const uint32_t w[] = { jaiA64LdrX(1, 0, 16),      /* x1 = 33 */
                             jaiA64MovzX(2, 7, 0),      /* x2 = 7  */
                             jaiA64SdivX(3, 1, 2),      /* x3 = 33/7 */
                             jaiA64MsubX(0, 3, 2, 1),   /* x0 = x1 - x3*x2 */
                             jaiA64Ret() };
      check("sdiv/msub remainder", runWith(w, 5, cell), 33 % 7); }

    /* adds sets V on overflow: INT64_MAX + 1 must take the b.vs branch */
    { const uint32_t w[] = { jaiA64MovzX(1, 0xffff, 0), jaiA64MovkX(1, 0xffff, 1),
                             jaiA64MovkX(1, 0xffff, 2), jaiA64MovkX(1, 0x7fff, 3),
                             jaiA64MovzX(2, 1, 0),
                             jaiA64AddsX(3, 1, 2),
                             /* +3 lands on the overflow arm: the branch counts
                              * from itself, so skipping a movz AND its ret is
                              * three instructions, not two. */
                             jaiA64BCond(JAI_A64_VS, 3),
                             jaiA64MovzX(0, 0, 0),         /* not overflowed */
                             jaiA64Ret(),
                             jaiA64MovzX(0, 1, 0),         /* overflowed */
                             jaiA64Ret() };
      check("adds overflow -> b.vs", runWith(w, 11, cell), 1); }

    /* the same add without overflow must fall through */
    { const uint32_t w[] = { jaiA64MovzX(1, 5, 0), jaiA64MovzX(2, 6, 0),
                             jaiA64AddsX(3, 1, 2),
                             jaiA64BCond(JAI_A64_VS, 3),
                             jaiA64MovX(0, 3),
                             jaiA64Ret(),
                             jaiA64MovzX(0, 999, 0),
                             jaiA64Ret() };
      check("adds no overflow", runWith(w, 8, cell), 11); }

    /* cmp + b.lt: 3 < 10 takes the branch */
    { const uint32_t w[] = { jaiA64MovzX(1, 3, 0),
                             jaiA64SubsXImm(31, 1, 10),    /* cmp x1, #10 */
                             jaiA64BCond(JAI_A64_LT, 3),
                             jaiA64MovzX(0, 0, 0),
                             jaiA64Ret(),
                             jaiA64MovzX(0, 7, 0),
                             jaiA64Ret() };
      check("cmp/b.lt taken", runWith(w, 7, cell), 7); }

    /* smulh: high half of 2^62 * 4 == 2^64, so the high word is 1 */
    { const uint32_t w[] = { jaiA64MovzX(1, 0x4000, 3),      /* x1 = 1<<62 */
                             jaiA64MovzX(2, 4, 0),
                             jaiA64SmulhX(0, 1, 2),
                             jaiA64Ret() };
      check("smulh high half", runWith(w, 4, cell), 1); }

    /* asr keeps the sign: -16 >> 2 == -4 */
    { const uint32_t w[] = { jaiA64MovzX(1, 16, 0),
                             jaiA64SubsX(1, 31, 1),          /* x1 = 0 - 16 */
                             jaiA64AsrX(0, 1, 2),
                             jaiA64Ret() };
      check("asr signed", runWith(w, 4, cell), -4); }

    /* lsr does not: -1 >>> 63 == 1, which is the sign-bit trick the magic
     * division correction uses */
    { const uint32_t w[] = { jaiA64MovzX(1, 1, 0),
                             jaiA64SubsX(1, 31, 1),          /* x1 = -1 */
                             jaiA64LsrX(0, 1, 63),
                             jaiA64Ret() };
      check("lsr sign bit", runWith(w, 4, cell), 1); }

    /* add register */
    { const uint32_t w[] = { jaiA64MovzX(1, 40, 0), jaiA64MovzX(2, 2, 0),
                             jaiA64AddX(0, 1, 2), jaiA64Ret() };
      check("add reg", runWith(w, 4, cell), 42); }

    /* --- the function tier's encoders ------------------------------- */

    /* sub immediate: 50 - 8 */
    { const uint32_t w[] = { jaiA64MovzX(1, 50, 0), jaiA64SubXImm(0, 1, 8),
                             jaiA64Ret() };
      check("sub imm", runWith(w, 3, cell), 42); }

    /* subs register produces the difference as well as the flags */
    { const uint32_t w[] = { jaiA64MovzX(1, 50, 0), jaiA64MovzX(2, 8, 0),
                             jaiA64SubsXReg(0, 1, 2), jaiA64Ret() };
      check("subs reg", runWith(w, 4, cell), 42); }

    /* subs sets V on signed overflow: INT64_MIN - 1. b.vs must be taken, so
     * this returns 1 only if the overflow was actually detected. */
    { const uint32_t w[] = {
          /* 0 */ jaiA64MovzX(1, 0, 0),
          /* 1 */ jaiA64MovkX(1, 0x8000u, 3),      /* x1 = INT64_MIN */
          /* 2 */ jaiA64MovzX(2, 1, 0),
          /* 3 */ jaiA64SubsXReg(3, 1, 2),
          /* 4 */ jaiA64BCond(JAI_A64_VS, 3),      /* -> 7 */
          /* 5 */ jaiA64MovzX(0, 0, 0),
          /* 6 */ jaiA64Ret(),
          /* 7 */ jaiA64MovzX(0, 1, 0),
          /* 8 */ jaiA64Ret() };
      check("subs overflow", runWith(w, 9, cell), 1); }

    /* stp/ldp with writeback: save a pair below sp, clobber, restore. The
     * value survives only if both the pre-index store and the post-index load
     * moved sp by the same amount and back. */
    { const uint32_t w[] = { jaiA64MovzX(1, 7, 0), jaiA64MovzX(2, 9, 0),
                             jaiA64StpPre(1, 2, 31, -16),
                             jaiA64MovzX(1, 0, 0), jaiA64MovzX(2, 0, 0),
                             jaiA64LdpPost(1, 2, 31, 16),
                             jaiA64AddX(0, 1, 2), jaiA64Ret() };
      check("stp pre / ldp post", runWith(w, 8, cell), 16); }

    /* stp/ldp at a plain offset, base untouched: write two words into the
     * caller's cell array and read one back. */
    { int64_t scratch[4] = { 0, 0, 0, 0 };
      const uint32_t w[] = { jaiA64MovzX(1, 3, 0), jaiA64MovzX(2, 4, 0),
                             jaiA64StpOff(1, 2, 0, 16),
                             jaiA64LdpOff(3, 4, 0, 16),
                             jaiA64AddX(0, 3, 4), jaiA64Ret() };
      check("stp/ldp offset", runWith(w, 6, scratch), 7);
      check("stp offset wrote", scratch[2], 3);
      check("stp offset wrote 2", scratch[3], 4); }

    /* bl and ret: call forward past a literal, have the callee add 1. x30 is
     * clobbered by the bl, so the outer return needs it saved. */
    { const uint32_t w[] = {
          /* 0 */ jaiA64StpPre(29, 30, 31, -16),
          /* 1 */ jaiA64MovzX(0, 41, 0),
          /* 2 */ jaiA64Bl(4),                     /* -> 6 */
          /* 3 */ jaiA64LdpPost(29, 30, 31, 16),
          /* 4 */ jaiA64Ret(),
          /* 5 */ jaiA64Ret(),                     /* unreached pad */
          /* 6 */ jaiA64AddXImm(0, 0, 1),          /* callee */
          /* 7 */ jaiA64Ret() };
      check("bl forward", runWith(w, 8, cell), 42); }

    /* ldr literal: the constant sits after the ret, four instructions on from
     * the load, and is read as one 64-bit word. */
    { const uint32_t w[] = { jaiA64LdrLit(0, 4), jaiA64Ret(), jaiA64Ret(),
                             jaiA64Ret(), 0x0000002au, 0x00000000u };
      check("ldr literal", runWith(w, 6, cell), 42); }

    /* ldr literal reaches a full 64-bit value, not just the low word */
    /* The literal sits at instruction 4, so its byte offset from a
     * page-aligned arena is 16 and the 64-bit load is aligned. */
    { const uint32_t w[] = { jaiA64LdrLit(0, 4), jaiA64Ret(), jaiA64Ret(),
                             jaiA64Ret(), 0x00000000u, 0x00000001u };
      check("ldr literal high", runWith(w, 6, cell), (int64_t)1 << 32); }

    /* mul */
    { const uint32_t w[] = { jaiA64MovzX(1, 6, 0), jaiA64MovzX(2, 7, 0),
                             jaiA64MulX(0, 1, 2), jaiA64Ret() };
      check("mul", runWith(w, 4, cell), 42); }

    /* The signed-overflow test for a product: smulh gives the high half, and
     * the product fits exactly when that equals the low half's sign bit
     * replicated. 6 * 7 fits, so this reports 0. */
    { const uint32_t w[] = { jaiA64MovzX(1, 6, 0), jaiA64MovzX(2, 7, 0),
                             jaiA64MulX(3, 1, 2), jaiA64SmulhX(4, 1, 2),
                             jaiA64SubsXAsr(31, 4, 3, 63),
                             jaiA64BCond(JAI_A64_NE, 3),
                             jaiA64MovzX(0, 0, 0), jaiA64Ret(),
                             jaiA64MovzX(0, 1, 0), jaiA64Ret() };
      check("mul fits", runWith(w, 10, cell), 0); }

    /* 2^40 * 2^40 does not fit, so the same sequence reports 1. */
    { const uint32_t w[] = { jaiA64MovzX(1, 1, 0), jaiA64MovkX(1, 0x100u, 2),
                             jaiA64MovzX(1, 0, 0), jaiA64MovkX(1, 0x100u, 2),
                             jaiA64MovX(2, 1),
                             jaiA64MulX(3, 1, 2), jaiA64SmulhX(4, 1, 2),
                             jaiA64SubsXAsr(31, 4, 3, 63),
                             jaiA64BCond(JAI_A64_NE, 3),
                             jaiA64MovzX(0, 0, 0), jaiA64Ret(),
                             jaiA64MovzX(0, 1, 0), jaiA64Ret() };
      check("mul overflows", runWith(w, 13, cell), 1); }

    if (failures != 0) return 1;
    printf("jit_arm64: ok\n");
    return 0;
}
#endif
