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
/* ret */
uint32_t jaiA64Ret(void);

/* Condition codes, named for the ones this uses. */
#define JAI_A64_EQ 0u
#define JAI_A64_NE 1u
#define JAI_A64_VS 6u   /* overflow set */
#define JAI_A64_LT 11u
#define JAI_A64_GE 10u

#endif /* JAI_VM_JIT_ARM64_H */
