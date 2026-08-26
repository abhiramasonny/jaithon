#: A `list[int]`, `list[float]` or `list[bool]` stores its elements unboxed --
#: eight bytes, eight bytes, one byte -- and every reader has to agree with
#: every writer about which. These are the four ways they stopped agreeing.
#:
#: 1. A SLICE sized its output from the source's `elemKind` but copied at the
#:    source's `stg`. The two disagree the moment a list is de-specialised
#:    (jaiListBox leaves elemKind alone), and the copy then wrote sixteen
#:    bytes per element into an eight-byte-per-element block: garbage for two
#:    elements, a segfault for two million.
#: 2. The COMPILED TIER stamped `elemKind` on a list literal without choosing
#:    the storage, so `var xs: list[float] = []` came out unboxed when it was
#:    interpreted and boxed when it was compiled. A loop form pinned to one
#:    storage was then denied entry for every list built the other way -- how
#:    a stencil over a half-and-half plate ran 2.5 BILLION interpreted
#:    instructions where the boxed build ran 2.9 million.
#: 3. The tier's two-arm storage dispatch tested `stg != BOXED` and then used
#:    the storage the VALUE's kind implies. `xs[i] = 3` on a `list[bool]`
#:    predicts eight bytes over a one-byte array.
#: 4. An int put into a `list[float]` widened to a double, so `[3]` printed
#:    as `[3.0]`. tests/golden/container_elem_kind pins the other answer.
#:
#: Every case is checked against the value the boxed path gives, and the loops
#: are hot enough to reach the compiled tier. Run with JAITHON_LIST_UNBOX=0
#: as well: the answers must not move.
# jit-compiles: sum_ints, sum_floats, count_flags
from std.test import assert_eq, assert_true

let ROUNDS = 400

fn sum_ints(values: list[int], rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < values.len() {
            total += values[index]
            index += 1
        }
        round += 1
    }
    return total
}

fn sum_floats(values: list[float], rounds: int) -> float {
    var total = 0.0
    var round = 0
    while round < rounds {
        var index = 0
        while index < values.len() {
            total = total + values[index]
            index += 1
        }
        round += 1
    }
    return total
}

fn count_flags(flags: list[bool], rounds: int) -> int {
    var seen = 0
    var round = 0
    while round < rounds {
        var index = 0
        while index < flags.len() {
            if flags[index] { seen += 1 }
            index += 1
        }
        round += 1
    }
    return seen
}

fn build_ints(count: int) -> list[int] {
    var values: list[int] = []
    for index in 0..count { values.push(index % 7) }
    return values
}

fn set_through_any(container: any, index: int, value: any) -> void {
    container[index] = value
}

pub fn test_unboxed_reads_survive_the_tier() -> void {
    let values = build_ints(64)
    assert_eq(sum_ints(values, ROUNDS), 189 * ROUNDS)

    var floats: list[float] = []
    for index in 0..64 { floats.push(index % 7 * 1.0) }
    assert_eq(sum_floats(floats, ROUNDS), 189.0 * ROUNDS)

    var flags: list[bool] = []
    for index in 0..64 { flags.push(index % 3 == 0) }
    assert_eq(count_flags(flags, ROUNDS), 22 * ROUNDS)
}

pub fn test_slice_of_a_despecialised_list() -> void {
    #: An int pushed into a `list[float]` de-specialises it rather than
    #: widening, which leaves elemKind saying `float` over a boxed array --
    #: the disagreement the slice used to size one side by and copy the other.
    var mixed: list[float] = []
    mixed.push(3)
    mixed.push(2.5)
    mixed.push(4)
    mixed.push(6.5)
    assert_eq(f"{mixed}", "[3, 2.5, 4, 6.5]")
    assert_eq(f"{mixed[0:2]}", "[3, 2.5]")
    assert_eq(f"{mixed[0:4:2]}", "[3, 4]")
    assert_eq(f"{mixed[2:4]}", "[4, 6.5]")

    #: And a slice of a list that IS specialised keeps its elements intact.
    let plain = build_ints(16)
    assert_eq(f"{plain[0:4]}", "[0, 1, 2, 3]")
    assert_eq(f"{plain[8:16:3]}", "[1, 4, 0]")
}

pub fn test_a_literal_built_by_either_tier_agrees() -> void {
    #: `build_ints` is called often enough that the tier compiles it, so the
    #: early lists are built by the interpreter and the later ones by compiled
    #: code. Every one of them has to slice, read and compare the same.
    var round = 0
    while round < 300 {
        let values = build_ints(8)
        assert_eq(values.len(), 8)
        assert_eq(f"{values}", "[0, 1, 2, 3, 4, 5, 6, 0]")
        assert_eq(f"{values[2:5]}", "[2, 3, 4]")
        assert_eq(sum_ints(values, 1), 21)
        round += 1
    }
}

pub fn test_a_value_the_store_cannot_hold() -> void {
    #: FIELD_KIND_FLOAT accepts an int, so this is a legal push -- and the
    #: element must read back as the int it was written as.
    let floats: list[float] = []
    floats.push(3)
    assert_eq(f"{floats}", "[3]")

    #: Reached through `any`, the tier predicts a storage from the VALUE's
    #: kind rather than the list's. Neither an eight-byte store over a
    #: one-byte array nor an int's bits read back as a double is acceptable.
    var flags: list[bool] = []
    for index in 0..32 { flags.push(index % 2 == 0) }
    var round = 0
    while round < 300 {
        set_through_any(flags, 0, false)
        set_through_any(flags, 31, true)
        round += 1
    }
    assert_true(not flags[0])
    assert_true(flags[31])
    assert_eq(flags.len(), 32)

    var reals: list[float] = []
    for index in 0..32 { reals.push(index * 1.0) }
    round = 0
    while round < 300 {
        set_through_any(reals, 5, 2.5)
        round += 1
    }
    assert_eq(reals[5], 2.5)
    assert_eq(reals.len(), 32)
}
