#ifndef JAI_VM_JIT_ARM64_H
#define JAI_VM_JIT_ARM64_H

#include <stdint.h>

/* The handful of arm64 instructions the compiled loop needs.
 *
 * Not a general assembler. Each encoder covers exactly the operand forms the
 * loop body uses, and returns the instruction word rather than writing it, so
 * a test can check the bits without an arena and the emitter can buffer.
 *
 * Every one of these is verified by executing it -- see tests/jit_arm64.c.
 * Hand-computed encodings that are merely inspected are how a JIT acquires a
 * bug that reproduces once in a million iterations. */

/* ldr Xt, [Xn, #offset] -- offset is a byte offset, must be 8-aligned, 0..32760 */
uint32_t jaiA64LdrX(unsigned rt, unsigned rn, unsigned offset);
/* str Xt, [Xn, #offset] -- same constraints */
uint32_t jaiA64StrX(unsigned rt, unsigned rn, unsigned offset);
/* ldr Wt, [Xn, #offset] -- 4-aligned, 0..16380; reads a Value's 32-bit tag */
uint32_t jaiA64LdrW(unsigned rt, unsigned rn, unsigned offset);
/* add Xd, Xn, #imm12 */
uint32_t jaiA64AddXImm(unsigned rd, unsigned rn, unsigned imm12);
/* subs Xd, Xn, #imm12 -- with rd = 31 this is `cmp` */
uint32_t jaiA64SubsXImm(unsigned rd, unsigned rn, unsigned imm12);
/* adds Xd, Xn, Xm -- sets V on signed overflow, which is the guard */
uint32_t jaiA64AddsX(unsigned rd, unsigned rn, unsigned rm);
/* sdiv Xd, Xn, Xm */
uint32_t jaiA64SdivX(unsigned rd, unsigned rn, unsigned rm);
/* msub Xd, Xn, Xm, Xa -- Xa - Xn*Xm; with sdiv this is the remainder */
uint32_t jaiA64MsubX(unsigned rd, unsigned rn, unsigned rm, unsigned ra);
/* movz Xd, #imm16, lsl #(16*shift) */
uint32_t jaiA64MovzX(unsigned rd, unsigned imm16, unsigned shift);
/* movk Xd, #imm16, lsl #(16*shift) */
uint32_t jaiA64MovkX(unsigned rd, unsigned imm16, unsigned shift);
/* mov Xd, Xn (an alias for orr Xd, xzr, Xn) */
uint32_t jaiA64MovX(unsigned rd, unsigned rn);
/* b.<cond> #offset -- offset in instructions, signed 19-bit */
uint32_t jaiA64BCond(unsigned cond, int32_t instructions);
/* b #offset -- unconditional, offset in instructions, signed 26-bit */
uint32_t jaiA64B(int32_t instructions);
/* subs Xd, Xn, Xm -- with rd = 31 this is a register-register `cmp` */
uint32_t jaiA64SubsX(unsigned rd, unsigned rn, unsigned rm);
/* smulh Xd, Xn, Xm -- high 64 bits of the signed 128-bit product */
uint32_t jaiA64SmulhX(unsigned rd, unsigned rn, unsigned rm);
/* asr Xd, Xn, #shift -- arithmetic, 0..63 */
uint32_t jaiA64AsrX(unsigned rd, unsigned rn, unsigned shift);
/* lsr Xd, Xn, #shift -- logical, 0..63 */
uint32_t jaiA64LsrX(unsigned rd, unsigned rn, unsigned shift);
/* add Xd, Xn, Xm */
uint32_t jaiA64AddX(unsigned rd, unsigned rn, unsigned rm);
/* subs Xd, Xn, Xm -- result AND flags; V set on signed overflow */
uint32_t jaiA64SubsXReg(unsigned rd, unsigned rn, unsigned rm);
/* sub Xd, Xn, #imm12 */
uint32_t jaiA64SubXImm(unsigned rd, unsigned rn, unsigned imm12);
/* bl #offset -- offset in instructions, signed 26-bit */
uint32_t jaiA64Bl(int32_t instructions);
/* stp Xt1, Xt2, [Xn, #imm]!  (pre-index, writes back) */
uint32_t jaiA64StpPre(unsigned rt, unsigned rt2, unsigned rn, int32_t imm);
/* ldp Xt1, Xt2, [Xn], #imm   (post-index, writes back) */
uint32_t jaiA64LdpPost(unsigned rt, unsigned rt2, unsigned rn, int32_t imm);
/* stp Xt1, Xt2, [Xn, #imm]   (signed offset) */
uint32_t jaiA64StpOff(unsigned rt, unsigned rt2, unsigned rn, int32_t imm);
/* ldp Xt1, Xt2, [Xn, #imm]   (signed offset) */
uint32_t jaiA64LdpOff(unsigned rt, unsigned rt2, unsigned rn, int32_t imm);
/* ldr Xt, <label> -- PC-relative literal load, offset in instructions */
uint32_t jaiA64LdrLit(unsigned rt, int32_t instructions);
/* movn Xd, #imm16 -- Xd = ~imm16 */
uint32_t jaiA64MovnX(unsigned rd, unsigned imm16);
/* nop */
uint32_t jaiA64Nop(void);
/* ret */
uint32_t jaiA64Ret(void);

/* Condition codes, named for the ones this uses. */
#define JAI_A64_EQ 0u
#define JAI_A64_NE 1u
#define JAI_A64_VS 6u   /* overflow set */
#define JAI_A64_LT 11u
#define JAI_A64_GE 10u
#define JAI_A64_LO  3u   /* unsigned lower -- the stack-limit test */
#define JAI_A64_LE 13u
#define JAI_A64_GT 12u

#endif /* JAI_VM_JIT_ARM64_H */
