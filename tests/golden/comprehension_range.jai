#: A comprehension clause whose iterable is a range literal takes the same
#: OP_ITER_RANGE / OP_FOR_RANGE_BIND lowering `_for` gives the statement form
#: (spec §3.4): no ObjRange, no ObjIter, and nothing pushed on the operand
#: stack where the iterator used to sit.
#:
#: The accumulator sits BELOW every clause's iterator, and the final
#: OP_LIST_APPEND / OP_DICT_INSERT / OP_SET_ADD computes its peek depth as
#: `self._depth() - 1 - container_depth`. A range clause pushes nothing, so
#: that arithmetic has to come out right whether zero, one or two of a
#: comprehension's clauses used the fused form — this file is aimed at
#: exactly that, not at the element values.
#:
#: Everything here must print the same at every -O level, under --gc-stress,
#: and under JAITHON_JIT_DEOPT_STRESS=1.

# --- the basic forms, one per container kind --------------------------------
print([i * 2 for i in 0..5])
print([i for i in 0..10 if i % 3 == 0])
print({i: i * i for i in 0..4})
let uniq = {i % 3 for i in 0..10}
print(uniq.to_list().sorted())

print([9 for _ in 0..4])
print([[0 for _ in 0..=2] for _ in 0..=1])

# --- nested clauses, every combination of fused vs not ----------------------
print([[i, j] for i in 0..3 for j in 0..3])

print([[i, s] for i in 0..2 for s in ["a", "b"]])

print([[s, i] for s in ["x", "y"] for i in 0..2])

print([i for i in 0..8 if i % 2 == 0 if i > 2])

# --- what must still take the general lowering ------------------------------
fn via_variable(lo: int, hi: int) -> list[int] {
    let r = lo..hi
    return [i * i for i in r]
}
print(via_variable(0, 4))

print([x * 10 for x in [1, 2, 3]])

fn bad_pattern() -> str {
    try {
        let broken = [a for (a, b) in 0..3]
        return f"{broken}"
    } catch e: TypeError {
        return f"{e}"
    }
}
print(bad_pattern())

# --- the accumulator outliving the loop --------------------------------------
var grown = [i for i in 0..3]
grown.push(99)
grown.push(100)
print(grown)
print(grown.len())

# --- hot enough to compile ----------------------------------------------------
fn poly(n: int) -> int {
    let squares = [i * i for i in 0..n]
    var t = 0
    for s in squares { t += s }
    return t
}
var total = 0
for k in 0..5000 { total = total + poly(5) }
print(total)
