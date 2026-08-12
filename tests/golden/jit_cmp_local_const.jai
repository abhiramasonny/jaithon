#: OP_CMP_LOCAL_CONST_LT must have a JIT arm.
#:
#: The function tier's opcode switch ends in `default: return false`, which
#: declines the WHOLE function rather than the instruction -- so a fused opcode
#: without an arm evicts every function containing it from the compiled tier.
#: This opcode shipped without one twice; the second time no benchmark happened
#: to contain it, so nothing noticed. Measured on this shape at 30M iterations:
#: 906 ms declining against 26 ms compiled, 34.8x.
#:
#: `let small = i < N` and not `if i < N`: a comparison feeding a branch fuses
#: further into OP_JUMP_IF_CMP_LOCAL_K, which has its own arm. Binding the bool
#: to a name is what leaves OP_CMP_LOCAL_CONST_LT standing.
import std.io

fn count_small(n: int) -> int {
    var hits = 0
    var i = 0
    while i < n {
        let small = i < 100
        hits = hits + (if small { 1 } else { 0 })
        i = i + 1
    }
    return hits
}

print(count_small(1000))
print(count_small(50))
