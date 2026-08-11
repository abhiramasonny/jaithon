#include "jit_arm64.h"

#include <string.h>

uint32_t jaiA64LdrX(unsigned rt, unsigned rn, unsigned offset) {
    return 0xf9400000u | ((offset / 8u) << 10) | (rn << 5) | rt;
}

uint32_t jaiA64StrX(unsigned rt, unsigned rn, unsigned offset) {
    return 0xf9000000u | ((offset / 8u) << 10) | (rn << 5) | rt;
}

uint32_t jaiA64LdrW(unsigned rt, unsigned rn, unsigned offset) {
    return 0xb9400000u | ((offset / 4u) << 10) | (rn << 5) | rt;
}

uint32_t jaiA64AddXImm(unsigned rd, unsigned rn, unsigned imm12) {
    return 0x91000000u | (imm12 << 10) | (rn << 5) | rd;
}

uint32_t jaiA64SubsXImm(unsigned rd, unsigned rn, unsigned imm12) {
    return 0xf1000000u | (imm12 << 10) | (rn << 5) | rd;
}

uint32_t jaiA64AddsX(unsigned rd, unsigned rn, unsigned rm) {
    return 0xab000000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64SdivX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x9ac00c00u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64MsubX(unsigned rd, unsigned rn, unsigned rm, unsigned ra) {
    return 0x9b008000u | (rm << 16) | (ra << 10) | (rn << 5) | rd;
}

uint32_t jaiA64MovzX(unsigned rd, unsigned imm16, unsigned shift) {
    return 0xd2800000u | (shift << 21) | (imm16 << 5) | rd;
}

uint32_t jaiA64MovkX(unsigned rd, unsigned imm16, unsigned shift) {
    return 0xf2800000u | (shift << 21) | (imm16 << 5) | rd;
}

uint32_t jaiA64MovX(unsigned rd, unsigned rn) {
    /* orr Xd, xzr, Xn */
    return 0xaa0003e0u | (rn << 16) | rd;
}

uint32_t jaiA64BCond(unsigned cond, int32_t instructions) {
    uint32_t imm19 = (uint32_t)(instructions & 0x7ffff);
    return 0x54000000u | (imm19 << 5) | cond;
}

uint32_t jaiA64B(int32_t instructions) {
    return 0x14000000u | ((uint32_t)instructions & 0x03ffffffu);
}

uint32_t jaiA64SubsX(unsigned rd, unsigned rn, unsigned rm) {
    return 0xeb000000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64SmulhX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x9b407c00u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64AsrX(unsigned rd, unsigned rn, unsigned shift) {
    /* SBFM Xd, Xn, #shift, #63 */
    return 0x9340fc00u | (shift << 16) | (rn << 5) | rd;
}

uint32_t jaiA64LsrX(unsigned rd, unsigned rn, unsigned shift) {
    /* UBFM Xd, Xn, #shift, #63 */
    return 0xd340fc00u | (shift << 16) | (rn << 5) | rd;
}

uint32_t jaiA64AddX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x8b000000u | (rm << 16) | (rn << 5) | rd;
}

/* add Xd, Xn, Xm, lsl #shift -- the shifted-register form. A Value is sixteen
 * bytes, so every element address is base + index*16, and folding the shift
 * into the add is one instruction instead of two on a path that runs eight
 * times per cell in `life`. */
uint32_t jaiA64AddXLsl(unsigned rd, unsigned rn, unsigned rm, unsigned shift) {
    return 0x8b000000u | (rm << 16) | ((shift & 63u) << 10) | (rn << 5) | rd;
}

uint32_t jaiA64Ret(void) {
    return 0xd65f03c0u;
}

uint32_t jaiA64SubsXReg(unsigned rd, unsigned rn, unsigned rm) {
    return 0xeb000000u | (rm << 16) | (rn << 5) | rd;
}

/* The two extended-register forms, which take only the LOW 32 BITS of Xm and
 * zero-extend them.
 *
 * They exist so that an ObjList's `items` and `count` can be fetched by one
 * `ldp`: count and capacity are adjacent int32s, so the second half of the pair
 * arrives as `count | capacity << 32` and every use of it has to ignore the top
 * half. `uxtw` is that, folded into the instruction that was going to run
 * anyway rather than paid as a separate `and`.
 *
 * option = 010 (UXTW) sits at bits 15:13, and imm3 = 0 means no further shift,
 * so the constant below is base | 0x4000. Note rd = 31 is XZR here and NOT the
 * stack pointer, because these are the flag-setting/adding register forms with
 * an extended operand -- SUBS with Rd = 31 is `cmp`, exactly as for SubsXReg. */
uint32_t jaiA64SubsXUxtw(unsigned rd, unsigned rn, unsigned rm) {
    return 0xeb204000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64AddXUxtw(unsigned rd, unsigned rn, unsigned rm) {
    return 0x8b204000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64SubXImm(unsigned rd, unsigned rn, unsigned imm12) {
    return 0xd1000000u | (imm12 << 10) | (rn << 5) | rd;
}

uint32_t jaiA64Bl(int32_t instructions) {
    return 0x94000000u | ((uint32_t)instructions & 0x03ffffffu);
}

/* stp Xt1, Xt2, [Xn, #imm]! -- imm is a byte offset, 8-aligned, -512..504.
 * The pre-index form writes the new base back, which is how a prologue both
 * saves a pair and opens the frame in one instruction. */
uint32_t jaiA64StpPre(unsigned rt, unsigned rt2, unsigned rn, int32_t imm) {
    uint32_t imm7 = (uint32_t)((imm / 8) & 0x7f);
    return 0xa9800000u | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt;
}

/* ldp Xt1, Xt2, [Xn], #imm -- post-index: read, then advance the base. */
uint32_t jaiA64LdpPost(unsigned rt, unsigned rt2, unsigned rn, int32_t imm) {
    uint32_t imm7 = (uint32_t)((imm / 8) & 0x7f);
    return 0xa8c00000u | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt;
}

/* stp Xt1, Xt2, [Xn, #imm] -- plain signed offset, base unchanged. */
uint32_t jaiA64StpOff(unsigned rt, unsigned rt2, unsigned rn, int32_t imm) {
    uint32_t imm7 = (uint32_t)((imm / 8) & 0x7f);
    return 0xa9000000u | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt;
}

/* ldp Xt1, Xt2, [Xn, #imm] */
uint32_t jaiA64LdpOff(unsigned rt, unsigned rt2, unsigned rn, int32_t imm) {
    uint32_t imm7 = (uint32_t)((imm / 8) & 0x7f);
    return 0xa9400000u | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt;
}

/* ldr Xt, <label> -- PC-relative literal load. One instruction to get a full
 * 64-bit constant, against four for a movz/movk chain, which is why the stack
 * limit is a literal after the body rather than materialised at every entry. */
uint32_t jaiA64LdrLit(unsigned rt, int32_t instructions) {
    uint32_t imm19 = (uint32_t)(instructions & 0x7ffff);
    return 0x58000000u | (imm19 << 5) | rt;
}

/* movn Xd, #imm16 -- Xd = ~imm16, which is how a small negative constant
 * becomes one instruction instead of a materialise-and-negate pair. */
uint32_t jaiA64MovnX(unsigned rd, unsigned imm16) {
    return 0x92800000u | (imm16 << 5) | rd;
}

uint32_t jaiA64Nop(void) {
    return 0xd503201fu;
}

/* mul Xd, Xn, Xm -- MADD with the addend as xzr. The low 64 bits only; the
 * overflow test needs smulh alongside it. */
uint32_t jaiA64MulX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x9b007c00u | (rm << 16) | (rn << 5) | rd;
}

/* cmp Xn, Xm, asr #shift -- the shifted-register form of subs, used to test
 * whether a product's high half is the sign extension of its low half. */
uint32_t jaiA64SubsXAsr(unsigned rd, unsigned rn, unsigned rm, unsigned shift) {
    return 0xeb800000u | (shift << 10) | (rm << 16) | (rn << 5) | rd;
}

/* The two-source floating-point forms share one skeleton: 0x1e6.. is "scalar
 * fp, type = double", and a 4-bit opcode at bit 12 is the only thing that
 * separates add from sub from mul from div. Register fields sit exactly where
 * the integer forms put them, which is the one mercy of this corner. */
uint32_t jaiA64FaddD(unsigned rd, unsigned rn, unsigned rm) {
    return 0x1e602800u | (rm << 16) | (rn << 5) | rd;
}

/* fsub is fadd with opcode 3 rather than 2; nothing else moves. */
uint32_t jaiA64FsubD(unsigned rd, unsigned rn, unsigned rm) {
    return 0x1e603800u | (rm << 16) | (rn << 5) | rd;
}

/* fmul is opcode 0, so its word looks suspiciously bare. */
uint32_t jaiA64FmulD(unsigned rd, unsigned rn, unsigned rm) {
    return 0x1e600800u | (rm << 16) | (rn << 5) | rd;
}

/* fdiv is opcode 1, and unlike sdiv it has no divide-by-zero trap: the result
 * is an infinity and the guard has to live in the emitted code. */
uint32_t jaiA64FdivD(unsigned rd, unsigned rn, unsigned rm) {
    return 0x1e601800u | (rm << 16) | (rn << 5) | rd;
}

/* The one-source forms swap the layout: the opcode grows to six bits and moves
 * up to bit 15, taking the space where a two-source form keeps Rm. So there is
 * no Rm field here at all, and only Rn and Rd stay put. */
uint32_t jaiA64FnegD(unsigned rd, unsigned rn) {
    return 0x1e614000u | (rn << 5) | rd;
}

/* fsqrt is opcode 3 in that same one-source group. */
uint32_t jaiA64FsqrtD(unsigned rd, unsigned rn) {
    return 0x1e61c000u | (rn << 5) | rd;
}

/* fcmp Dn, Dm -- the odd one: it has no destination, so the field where Rd
 * would go is a 5-bit opcode instead, and zero there means "compare against a
 * register" rather than against #0.0. Writing rd = 31 out of integer habit
 * would silently assemble fcmpe Dn, #0.0. */
uint32_t jaiA64FcmpD(unsigned rn, unsigned rm) {
    return 0x1e602000u | (rm << 16) | (rn << 5);
}

/* fmov Dd, Dn -- opcode 0 of the one-source group, which makes it the plain
 * register move rather than an alias of anything. */
uint32_t jaiA64FmovDD(unsigned rd, unsigned rn) {
    return 0x1e604000u | (rn << 5) | rd;
}

/* fmov Dd, Xn -- a different instruction class from the one above: this is the
 * fp/integer transfer group, sf = 1 for a 64-bit source and opcode 7 meaning
 * "general to fp". It copies bits. The double it produces is whatever those 64
 * bits already spelled, so feeding it the integer 1 yields a denormal near
 * zero, not 1.0. Use scvtf when a number is meant. */
uint32_t jaiA64FmovDX(unsigned rd, unsigned rn) {
    return 0x9e670000u | (rn << 5) | rd;
}

uint32_t jaiA64FmovDImm(unsigned rd, unsigned imm8) {
    return 0x1e601000u | ((imm8 & 0xffu) << 13) | rd;
}

/* The eight-bit form covers +-(1 + m/16) * 2^e for m in 0..15 and e in -3..4,
 * which is where the constants in float code actually live: 0.5, 2.0, 4.0,
 * 0.25. Zero is not in it -- the format has an implicit leading one -- so
 * `x = 0.0` still goes the long way round, which is right, since it is not in
 * a loop. Rather than reproduce VFPExpandImm's bit surgery backwards, this
 * builds each of the 256 values and compares: it runs at compile time, once
 * per constant, and it cannot be subtly wrong the way an inverse would be. */
bool jaiA64FpImm8(double v, unsigned *imm8) {
    for (unsigned i = 0; i < 256; i++) {
        unsigned s = (i >> 7) & 1u, b = (i >> 6) & 1u;
        unsigned cd = (i >> 4) & 3u, efgh = i & 15u;
        uint64_t exp = ((uint64_t)(b ? 0u : 1u) << 10) |
                       ((uint64_t)(b ? 0xffu : 0u) << 2) | cd;
        uint64_t bits = ((uint64_t)s << 63) | (exp << 52) | ((uint64_t)efgh << 48);
        double d;
        memcpy(&d, &bits, sizeof d);
        if (d == v) { *imm8 = i; return true; }
    }
    return false;
}

/* fmov Xd, Dn -- the same transfer group with opcode 6, the other direction.
 * Also bits, so this is how a test reads a double out exactly. */
uint32_t jaiA64FmovXD(unsigned rd, unsigned rn) {
    return 0x9e660000u | (rn << 5) | rd;
}

/* scvtf Dd, Xn -- opcode 2 of that group, and the real conversion: the integer
 * value becomes the nearest double. One bit of opcode away from fmov, which is
 * why the tests distinguish them by result rather than by inspection. */
uint32_t jaiA64ScvtfDX(unsigned rd, unsigned rn) {
    return 0x9e620000u | (rn << 5) | rd;
}

/* fcvtzs Xd, Dn -- the way back. The rounding mode is in the instruction, not
 * in FPCR: rmode 3 is toward zero, which is the truncation C's cast wants. */
uint32_t jaiA64FcvtzsXD(unsigned rd, unsigned rn) {
    return 0x9e780000u | (rn << 5) | rd;
}

/* ldr Dt, [Xn, #offset] -- the SIMD/FP encoding, distinguished from the
 * integer ldr only by bit 26. Same scaled 12-bit offset, so the same 8-aligned
 * 0..32760 range. */
uint32_t jaiA64LdrD(unsigned rt, unsigned rn, unsigned offset) {
    return 0xfd400000u | ((offset / 8u) << 10) | (rn << 5) | rt;
}

/* str Dt, [Xn, #offset] */
uint32_t jaiA64StrD(unsigned rt, unsigned rn, unsigned offset) {
    return 0xfd000000u | ((offset / 8u) << 10) | (rn << 5) | rt;
}

/* str Wt, [Xn, #offset] -- 4-aligned, 0..16380. The counterpart to jaiA64LdrW:
 * a Value's tag is 32 bits, so writing one takes this and a 64-bit store. */
uint32_t jaiA64StrW(unsigned rt, unsigned rn, unsigned offset) {
    return 0xb9000000u | ((offset / 4u) << 10) | (rn << 5) | rt;
}

/* blr Xn -- call through a register. The absolute address of a C helper does
 * not fit bl's 26-bit displacement, and the arena can land anywhere. */
uint32_t jaiA64Blr(unsigned rn) {
    return 0xd63f0000u | (rn << 5);
}

/* cset Xd, <cond> -- 1 when the condition holds, 0 otherwise. Encoded as
 * CSINC Xd, XZR, XZR with the condition inverted, which is the standard alias:
 * a comparison's answer becomes a value without a branch. */
uint32_t jaiA64CsetX(unsigned rd, unsigned cond) {
    unsigned inv = cond ^ 1u;
    return 0x9a9f07e0u | (inv << 12) | rd;
}

/* lsl Xd, Xn, #shift -- UBFM Xd, Xn, #(-shift mod 64), #(63-shift). Scaling an
 * index to a Value offset is a shift by four. */
uint32_t jaiA64LslX(unsigned rd, unsigned rn, unsigned shift) {
    unsigned immr = (64u - shift) & 63u;
    unsigned imms = 63u - shift;
    return 0xd3400000u | (immr << 16) | (imms << 10) | (rn << 5) | rd;
}

/* eor Xd, Xn, Xm -- the sign test floor division needs: the remainder and the
 * divisor have different signs exactly when their exclusive-or is negative. */
uint32_t jaiA64EorX(unsigned rd, unsigned rn, unsigned rm) {
    return 0xca000000u | (rm << 16) | (rn << 5) | rd;
}

/* ldrb w<rd>, [x<rn>, #imm12] -- one unsigned byte, zero-extended. */
/* csel x<rd>, x<rn>, x<rm>, <cond> -- rn when the condition holds, else rm. */
uint32_t jaiA64CselX(unsigned rd, unsigned rn, unsigned rm, unsigned cond) {
    return 0x9a800000u | (rm << 16) | ((cond & 0xfu) << 12) | (rn << 5) | rd;
}

uint32_t jaiA64LdrByte(unsigned rd, unsigned rn, unsigned offset) {
    return 0x39400000u | ((offset & 0xfffu) << 10) | (rn << 5) | rd;
}

uint32_t jaiA64AndX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x8a000000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64OrrX(unsigned rd, unsigned rn, unsigned rm) {
    return 0xaa000000u | (rm << 16) | (rn << 5) | rd;
}

/* and Xd, Xn, #((1 << ones) - 1) -- the AND (immediate) form, restricted to the
 * one pattern this needs: a run of `ones` set bits at the bottom.
 *
 * arm64's logical immediates are an encoded (N, immr, imms) triple, not a
 * number, and only some 64-bit values have one. A low run of n ones is
 * N=1, immr=0, imms=n-1, which is why the restriction: it makes the encoding a
 * subtraction rather than the general bitmask search, and the general search is
 * exactly the sort of thing that would be wrong in a corner nothing exercises.
 * `ones` must be 1..63 -- 0 is not a legal pattern (all-zero) and 64 is the
 * all-ones one, and neither is reachable from a positive int64 power of two. */
uint32_t jaiA64AndXOnes(unsigned rd, unsigned rn, unsigned ones) {
    return 0x92400000u | (((ones - 1u) & 0x3fu) << 10) | (rn << 5) | rd;
}

/* Shift by a register. Both use only the low six bits of the amount, so a
 * count of 64 or more wraps rather than saturating -- the callers guard for
 * that before emitting these. */
uint32_t jaiA64LslvX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x9ac02000u | (rm << 16) | (rn << 5) | rd;
}

uint32_t jaiA64AsrvX(unsigned rd, unsigned rn, unsigned rm) {
    return 0x9ac02800u | (rm << 16) | (rn << 5) | rd;
}

/* adds Xd, Xn, #imm12 -- the flag-setting add-immediate. The register form
 * already existed; this one exists because `i += 1` was two instructions, a
 * movz of the constant and then the add, in every counted loop in the
 * language. Same encoding family as SubsXImm above with the op bit clear. */
uint32_t jaiA64AddsXImm(unsigned rd, unsigned rn, unsigned imm12) {
    return 0xb1000000u | ((imm12 & 0xfffu) << 10) | (rn << 5) | rd;
}

/* tbz/tbnz Xt, #bit, #offset -- branch on one bit, offset in instructions.
 *
 * The bit index splits across two fields: its top bit is instruction bit 31
 * (which is also what makes it a 64-bit test) and the low five are bits 23:19.
 * Getting that split wrong tests a neighbouring bit and the instruction still
 * runs, which is why this is executed in tests/jit_arm64.c rather than read. */
static uint32_t tbEncode(unsigned op, unsigned rt, unsigned bit,
                         int32_t instructions) {
    return 0x36000000u | ((bit >> 5) << 31) | (op << 24) |
           ((bit & 0x1fu) << 19) | (((uint32_t)instructions & 0x3fffu) << 5) |
           rt;
}

uint32_t jaiA64Tbz(unsigned rt, unsigned bit, int32_t instructions) {
    return tbEncode(0u, rt, bit, instructions);
}

uint32_t jaiA64Tbnz(unsigned rt, unsigned bit, int32_t instructions) {
    return tbEncode(1u, rt, bit, instructions);
}
