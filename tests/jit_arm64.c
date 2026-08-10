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
#include <string.h>
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

/* A double's bits, so the fp cases can go through the same integer check().
 * Comparing the bits rather than the values is the point: two doubles that
 * print alike can differ in the last place, and a division that is off by one
 * ulp is exactly the kind of wrong encoding that would otherwise pass. */
static int64_t dbits(double d) {
    int64_t b;
    memcpy(&b, &d, sizeof b);
    return b;
}

/* fcmp Dn, Dm followed by b.<cond>, returning 1 if the branch was taken. The
 * six comparison cases differ only in the condition and the operand order, so
 * spelling each one out as its own eight-word block would bury that line. */
static int64_t fcmpTakes(unsigned cond, unsigned rn, unsigned rm, double *pair) {
    const uint32_t w[] = {
        /* 0 */ jaiA64LdrD(0, 0, 0),          /* d0 = pair[0] */
        /* 1 */ jaiA64LdrD(1, 0, 8),          /* d1 = pair[1] */
        /* 2 */ jaiA64FcmpD(rn, rm),
        /* 3 */ jaiA64BCond(cond, 3),         /* -> 6 */
        /* 4 */ jaiA64MovzX(0, 0, 0),
        /* 5 */ jaiA64Ret(),
        /* 6 */ jaiA64MovzX(0, 1, 0),
        /* 7 */ jaiA64Ret() };
    return runWith(w, 8, pair);
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

    /* --- the double-precision encoders ------------------------------ */

    /* Doubles reach the fp registers through memory, so ldr Dt and str Dt are
     * under test in every case below as well as on their own account: if the
     * load's offset field were misplaced the arithmetic would be right about
     * the wrong numbers. Hence checking both the returned bits and what
     * landed back in the array. */
    { double d[4] = { 1.5, 0.25, 0.0, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(0, 0, 0), jaiA64LdrD(1, 0, 8),
                             jaiA64FaddD(2, 0, 1), jaiA64FsubD(3, 0, 1),
                             jaiA64StrD(2, 0, 16), jaiA64StrD(3, 0, 24),
                             jaiA64FmovXD(0, 2), jaiA64Ret() };
      check("fadd", runWith(w, 8, d), dbits(1.75));
      check("fadd stored", dbits(d[2]), dbits(1.75));
      check("fsub stored", dbits(d[3]), dbits(1.25)); }

    /* fmul and fdiv on 1 and 3. The quotient is deliberately inexact: it has
     * to match the double C computes bit for bit, which a nearly-right
     * encoding would not. */
    { double d[4] = { 1.0, 3.0, 0.0, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(0, 0, 0), jaiA64LdrD(1, 0, 8),
                             jaiA64FmulD(2, 0, 1), jaiA64FdivD(3, 0, 1),
                             jaiA64StrD(2, 0, 16),
                             jaiA64FmovXD(0, 3), jaiA64Ret() };
      check("fdiv 1/3", runWith(w, 7, d), dbits(1.0 / 3.0));
      check("fmul stored", dbits(d[2]), dbits(3.0)); }

    /* fneg both ways round, which also carries a negative double in and out
     * through ldr/str untouched. */
    { double d[4] = { -2.75, 0.0, 0.0, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(0, 0, 0),
                             jaiA64FnegD(1, 0), jaiA64FnegD(2, 1),
                             jaiA64StrD(1, 0, 8), jaiA64StrD(2, 0, 16),
                             jaiA64FmovXD(0, 0), jaiA64Ret() };
      check("negative survives ldr", runWith(w, 7, d), dbits(-2.75));
      check("fneg of negative", dbits(d[1]), dbits(2.75));
      check("fneg of positive", dbits(d[2]), dbits(-2.75)); }

    /* fsqrt on an exact square and on one that is not: 15129 is 123*123 and
     * lands dead on, while sqrt(2) has to be the correctly rounded double
     * rather than merely close to it. */
    { double d[4] = { 15129.0, 2.0, 0.0, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(0, 0, 0), jaiA64LdrD(1, 0, 8),
                             jaiA64FsqrtD(2, 0), jaiA64FsqrtD(3, 1),
                             jaiA64StrD(2, 0, 16),
                             jaiA64FmovXD(0, 3), jaiA64Ret() };
      check("fsqrt 2", runWith(w, 7, d), dbits(1.4142135623730951));
      check("fsqrt 15129", dbits(d[2]), dbits(123.0)); }

    /* fmov Dd, Dn through registers above 15, so a register field that had
     * lost its top bit would show up as the wrong source. */
    { double d[2] = { 6.25, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(19, 0, 0), jaiA64FmovDD(27, 19),
                             jaiA64StrD(27, 0, 8),
                             jaiA64FmovXD(0, 27), jaiA64Ret() };
      check("fmov d,d", runWith(w, 5, d), dbits(6.25));
      check("fmov d,d stored", dbits(d[1]), dbits(6.25)); }

    /* The pair that tells fmov and scvtf apart, which is the whole reason
     * they are named differently. Both take x1 = 1 and both leave a double in
     * d0. fmov copies the bits, so the double is the denormal 5e-324 whose
     * bit pattern is 1; scvtf converts, so the double is 1.0 and its bit
     * pattern is 0x3ff0000000000000. Swap the two encoders and each of these
     * returns the other's answer. */
    /* Both take their integer from x17, with x1 -- the register a source
     * field one bit too narrow would name instead -- holding a decoy, so a
     * lost top bit gives a wrong answer rather than an accidentally right
     * one. Same reason for x16 and the prior value below. */
    { const uint32_t w[] = { jaiA64MovzX(17, 1, 0), jaiA64MovzX(1, 99, 0),
                             jaiA64FmovDX(0, 17), jaiA64FmovXD(0, 0),
                             jaiA64Ret() };
      check("fmov d,x keeps the bits", runWith(w, 5, cell), 1); }

    { const uint32_t w[] = { jaiA64MovzX(17, 1, 0), jaiA64MovzX(1, 99, 0),
                             jaiA64MovzX(16, 123, 0),   /* must be overwritten */
                             jaiA64ScvtfDX(0, 17),
                             jaiA64FmovXD(16, 0), jaiA64MovX(0, 16),
                             jaiA64Ret() };
      check("scvtf converts", runWith(w, 7, cell), dbits(1.0)); }

    /* And the same distinction from the other end: 1.0's bit pattern put in
     * an x register must arrive as a usable 1.0, which fadd then doubles.
     * scvtf on those same bits would give about 4.6e18. d20 is loaded with
     * 7.0 first so that an fmov writing the wrong register leaves that
     * behind instead of leaving something undefined. */
    { double d[4] = { 0.0, 0.0, 7.0, 0.0 };
      const uint32_t w[] = { jaiA64LdrD(20, 0, 16),
                             jaiA64MovzX(1, 0x3ff0u, 3), jaiA64FmovDX(20, 1),
                             jaiA64FaddD(21, 20, 20),
                             jaiA64StrD(20, 0, 0), jaiA64StrD(21, 0, 8),
                             jaiA64FmovXD(0, 21), jaiA64Ret() };
      check("fmov x->d then fadd", runWith(w, 8, d), dbits(2.0));
      check("fmov x->d is a bit move", dbits(d[0]), dbits(1.0));
      check("fmov x->d result is a number", dbits(d[1]), dbits(2.0)); }

    /* scvtf on a negative, built with movn so no fp constant is involved,
     * and then on a value with nothing in its low 32 bits. The source width
     * is one bit of the encoding, and a 32-bit scvtf would read only the low
     * half of x17 and answer 0.0 while still looking like a conversion. */
    { const uint32_t w[] = { jaiA64MovnX(1, 4), jaiA64ScvtfDX(0, 1),
                             jaiA64FmovXD(0, 0), jaiA64Ret() };
      check("scvtf negative", runWith(w, 4, cell), dbits(-5.0));
      const uint32_t w2[] = { jaiA64MovzX(17, 1, 2),   /* x17 = 1 << 32 */
                              jaiA64ScvtfDX(0, 17),
                              jaiA64FmovXD(0, 0), jaiA64Ret() };
      check("scvtf reads all 64 bits", runWith(w2, 4, cell),
            dbits(4294967296.0)); }

    /* fcvtzs truncates toward zero rather than rounding to nearest, and it
     * truncates toward zero on negatives too -- which is the difference
     * between -3 and -4. */
    { double d[2] = { 3.9, -3.9 };
      const uint32_t w[] = { jaiA64LdrD(7, 0, 0), jaiA64MovzX(17, 55, 0),
                             jaiA64FcvtzsXD(17, 7), jaiA64MovX(0, 17),
                             jaiA64Ret() };
      check("fcvtzs truncates", runWith(w, 5, d), 3);
      const uint32_t w2[] = { jaiA64LdrD(7, 0, 8), jaiA64FcvtzsXD(0, 7),
                              jaiA64Ret() };
      check("fcvtzs truncates negative", runWith(w2, 3, d), -3); }

    /* the round trip closes: 42 -> 42.0 -> 42. d20 rather than a low
     * register because d8-d15 are callee-saved and this code saves nothing.
     * It arrives holding 9.0 and d4 holds 5.0, so if either instruction's
     * register field lost its top bit the answer would be one of those. */
    { double d[2] = { 9.0, 5.0 };
      const uint32_t w[] = { jaiA64LdrD(20, 0, 0), jaiA64LdrD(4, 0, 8),
                             jaiA64MovzX(1, 42, 0), jaiA64ScvtfDX(20, 1),
                             jaiA64FcvtzsXD(0, 20), jaiA64Ret() };
      check("scvtf/fcvtzs round trip", runWith(w, 6, d), 42); }

    /* Every arithmetic form again, but driven through registers above 15 so
     * that the top bit of each 5-bit field has to be in the right place. A
     * field encoded one bit too narrow still names a real register, and with
     * d0-d7 it names the correct one, so the cases above cannot see it. x17
     * carries the base for the same reason -- it is the highest general
     * register a leaf function may clobber here. */
    { double d[8] = { 2.5, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      const uint32_t w[] = { jaiA64MovX(17, 0),
                             jaiA64LdrD(17, 17, 0),      /* d17 = 2.5 */
                             jaiA64LdrD(30, 17, 8),      /* d30 = 4.0 */
                             jaiA64FaddD(21, 17, 30),
                             jaiA64FsubD(22, 17, 30),
                             jaiA64FmulD(23, 17, 30),
                             jaiA64FdivD(24, 17, 30),
                             jaiA64FnegD(25, 22),
                             jaiA64FsqrtD(26, 30),
                             jaiA64StrD(21, 17, 16), jaiA64StrD(22, 17, 24),
                             jaiA64StrD(23, 17, 32), jaiA64StrD(24, 17, 40),
                             jaiA64StrD(25, 17, 48), jaiA64StrD(26, 17, 56),
                             jaiA64FmovXD(0, 24), jaiA64Ret() };
      check("fdiv high regs", runWith(w, 17, d), dbits(0.625));
      check("fadd high regs", dbits(d[2]), dbits(6.5));
      check("fsub high regs", dbits(d[3]), dbits(-1.5));
      check("fmul high regs", dbits(d[4]), dbits(10.0));
      check("fdiv high regs stored", dbits(d[5]), dbits(0.625));
      check("fneg high regs", dbits(d[6]), dbits(1.5));
      check("fsqrt high regs", dbits(d[7]), dbits(2.0)); }

    /* fcmp with both operands above 15. A comparison cannot report which
     * registers it read, so d4 and d7 -- the registers a field one bit too
     * narrow would select instead of d20 and d23 -- are preloaded with the
     * operands the other way round. Then a lost top bit answers 0 rather
     * than reading whatever those registers happened to hold. */
    { double d[2] = { -0.5, 6.0 };
      const uint32_t w[] = {
          /* 0 */ jaiA64LdrD(20, 0, 0), jaiA64LdrD(23, 0, 8),
          /* 2 */ jaiA64LdrD(4, 0, 8),  jaiA64LdrD(7, 0, 0),
          /* 4 */ jaiA64FcmpD(20, 23),
          /* 5 */ jaiA64BCond(JAI_A64_MI, 3),   /* -> 8 */
          /* 6 */ jaiA64MovzX(0, 0, 0), jaiA64Ret(),
          /* 8 */ jaiA64MovzX(0, 1, 0), jaiA64Ret() };
      check("fcmp high regs", runWith(w, 10, d), 1); }

    /* fcmp sets the flags and nothing else, so every case is a branch. mi is
     * the less-than arm and gt the greater-than one; lt would agree here but
     * disagree on a nan, which is why the fp forms use the other pair. */
    { double d[2] = { 1.5, 2.5 };
      check("fcmp 1.5<2.5 b.mi", fcmpTakes(JAI_A64_MI, 0, 1, d), 1);
      check("fcmp 2.5<1.5 b.mi", fcmpTakes(JAI_A64_MI, 1, 0, d), 0);
      check("fcmp 2.5>1.5 b.gt", fcmpTakes(JAI_A64_GT, 1, 0, d), 1);
      check("fcmp 1.5>2.5 b.gt", fcmpTakes(JAI_A64_GT, 0, 1, d), 0);
      check("fcmp 1.5==1.5 b.eq", fcmpTakes(JAI_A64_EQ, 0, 0, d), 1);
      check("fcmp 1.5==2.5 b.eq", fcmpTakes(JAI_A64_EQ, 0, 1, d), 0); }

    /* the same comparisons with a negative operand, where an encoder that
     * had quietly compared magnitudes would part company */
    { double d[2] = { -4.0, 0.5 };
      check("fcmp -4<0.5 b.mi", fcmpTakes(JAI_A64_MI, 0, 1, d), 1);
      check("fcmp 0.5>-4 b.gt", fcmpTakes(JAI_A64_GT, 1, 0, d), 1);
      check("fcmp -4>0.5 b.gt", fcmpTakes(JAI_A64_GT, 0, 1, d), 0); }

    /* ldr Dt / str Dt at the far end of the scaled offset range, with a base
     * that is not x0's original value, so a misplaced imm12 or Rn shows up as
     * a wrong slot rather than as a wrong number. */
    { double d[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.5 };
      const uint32_t w[] = { jaiA64AddXImm(3, 0, 8),   /* x3 = &d[1] */
                             jaiA64LdrD(0, 0, 56),     /* d0 = d[7] */
                             jaiA64StrD(0, 3, 32),     /* d[5] = 7.5 */
                             jaiA64FmovXD(0, 0), jaiA64Ret() };
      check("ldr d at #56", runWith(w, 5, d), dbits(7.5));
      check("str d through a moved base", dbits(d[5]), dbits(7.5));
      check("str d left d[4] alone", dbits(d[4]), dbits(0.0)); }

    if (failures != 0) return 1;
    printf("jit_arm64: ok\n");
    return 0;
}
#endif
