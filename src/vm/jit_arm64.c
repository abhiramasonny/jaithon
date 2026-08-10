#include "jit_arm64.h"

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

uint32_t jaiA64Ret(void) {
    return 0xd65f03c0u;
}
