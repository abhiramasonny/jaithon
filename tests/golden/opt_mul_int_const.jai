#: gc-stress-every: 50
#: `GET_LOCAL S; <int k>; MUL` fused into `OP_MUL_INT_CONST`, the third
#: sibling of ADD_INT_CONST/SUB_INT_CONST (spec/BYTECODE.md §8.6). Same
#: shape, same overflow contract: `slots[S] * imm` raises `OverflowError`
#: exactly where the unfused `*` would, both interpreted and JIT-compiled
#: (the JIT arm shares plain OP_MUL's smulh-based overflow test and its
#: overflow-stub slot, since it is the same operator).
#:
#: The several `let`s that die right after their own binding, before the
#: loop, are load-bearing: they exist so slot coalescing (spec §8.4) has
#: something to renumber, moving `i`'s own frame slot down. A fused opcode
#: with a `u16` slot operand has to be registered wherever slots get
#: renumbered (`_build_slot_operands` in
#: `lib/jaithon/compile/opt/chunk.jai`) or coalescing silently leaves that
#: one operand pointing at its pre-renumbering slot while every other
#: reference to the same variable moves -- exactly what a trivial function
#: (no dead locals, nothing to coalesce) would never exercise.
fn scale(n: int) -> int {
    let junk1 = n + 1
    let junk2 = junk1 + 1
    let junk3 = junk2 + 1
    let junk4 = junk3 + 1
    var total = 0
    for i in 0..n {
        total = i * 7 + total
    }
    return total
}

fn overflow_case() -> int {
    let big: int = 4611686018427387904
    return big * 3
}

fn main() -> int {
    print(scale(200000))
    var caught = false
    try {
        overflow_case()
    } catch OverflowError {
        caught = true
    }
    print(caught)
    return 0
}
