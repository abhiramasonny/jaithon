#: The enum-variant `match`, in bodies hot enough to reach the compiled tier.
#:
#: `match e { Op.Add => .., Op.Sub => .. }` lowers each alternative to five
#: instructions -- OP_MATCH_TYPE_POP, OP_GET_LOCAL, OP_ENUM_TAG, OP_SWAP_POP,
#: OP_MATCH_CONST_POP -- and the tier had an arm for none of them, so a body
#: built out of them compiled a nine-instruction prefix and then abandoned it:
#: 1,999,936 deopts for 2,000,000 calls on a five-variant probe.
#:
#: Every case here checks the ANSWER, and the ones that matter are the edges:
#:
#:   - every variant of an enum, including the first (whose tag is zero) and
#:     the last (whose failed test lands somewhere different from the others');
#:   - the `|` alternation, whose failed alternatives end in an unconditional
#:     jump the arm has to emit itself before stepping over the OP_POP the
#:     bytecode sends it to;
#:   - a guard arm and a default arm, which are the two ways an arm matches
#:     without a tag test at all;
#:   - a subject that is NOT an enum value. That is the case where the compiled
#:     type test has to MISS and take its branch, and a heap object reaches it:
#:     a SLOT_OBJ parameter is admitted on IS_OBJ alone, so a str or a list
#:     arriving at a body specialised for an enum runs the compiled guard
#:     rather than being turned away at the door.
#:
#: `sized`, `rank_exhaustive` and `maybe_rank` are deliberately NOT on the
#: jit-compiles line: each stops the walk somewhere the tier still cannot
#: speak (OP_ENUM_FIELD, the OP_ASSERT_FAIL an exhaustive match without a
#: default ends in, and a `null` pattern's non-integer OP_MATCH_CONST_POP).
#: They are here because a shape the tier refuses still has to give the right
#: answer, and because the refusal is meant to be soft -- the body still
#: compiles what comes before it.
# jit-compiles: rank, family, first_or_last, wrapped, of_any
from std.test import assert_eq, assert_true

#: Above JAI_JIT_THRESHOLD entries, so every body below compiles.
let CALLS = 300

enum Op { Add, Sub, Mul, Div, Mod, Pow }

enum Shape { Dot, Circle(radius: int), Box(w: int, h: int) }

#: Every variant, one arm each, plus a default that nothing reaches.
fn rank(o: Op) -> int {
    return match o {
        Op.Add => 1,
        Op.Sub => 2,
        Op.Mul => 3,
        Op.Div => 4,
        Op.Mod => 5,
        Op.Pow => 6,
        _ => 0,
    }
}

#: The same answer without a `match` anywhere, as the oracle.
fn rank_if(o: Op) -> int {
    if o is Op.Add { return 1 }
    if o is Op.Sub { return 2 }
    if o is Op.Mul { return 3 }
    if o is Op.Div { return 4 }
    if o is Op.Mod { return 5 }
    return 6
}

#: No default: the fall-through is an OP_ASSERT_FAIL that always raises, and
#: the walk cannot restate the operand stack after it, so the body compiles
#: only its prefix. Correctness, not coverage.
fn rank_exhaustive(o: Op) -> int {
    return match o {
        Op.Add => 1,
        Op.Sub => 2,
        Op.Mul => 3,
        Op.Div => 4,
        Op.Mod => 5,
        Op.Pow => 6,
    }
}

#: `|` alternation plus a default. Each failed alternative of a group ends in
#: an unconditional jump to the arm body, which is the shape whose OP_POP the
#: compiled miss edge has to step over.
fn family(o: Op) -> str {
    return match o {
        Op.Add | Op.Sub => "additive",
        Op.Mul | Op.Div | Op.Mod => "multiplicative",
        _ => "other",
    }
}

#: First and last variant only, so the default carries everything between.
fn first_or_last(o: Op) -> bool {
    return match o {
        Op.Add => true,
        Op.Pow => true,
        _ => false,
    }
}

#: A guard arm sitting between two variant arms: the binding matches anything,
#: so the arms after it are reached only when the guard is false.
fn wrapped(o: Op, n: int) -> int {
    return match o {
        Op.Add => n + 1,
        x if n < 0 => rank(x) * 100,
        Op.Sub => n - 1,
        _ => n,
    }
}

#: Payload variants, which put OP_ENUM_FIELD after the tag test.
fn sized(s: Shape) -> int {
    return match s {
        Shape.Dot => 0,
        Shape.Circle(r) => r * 2,
        Shape.Box(w, h) => w * h,
    }
}

#: A subject the type test must MISS on. Warmed with enum values only, so the
#: compiled form is specialised for one; the misses below then arrive at that
#: form rather than at a fresh compile.
fn of_any(v: any) -> str {
    return match v {
        Op.Add => "add",
        Op.Mod => "mod",
        _ => "not an op",
    }
}

#: A nullable subject. The `null` arm is an OP_MATCH_CONST_POP whose constant
#: is not an integer, which the tier refuses; the answer still has to be right.
fn maybe_rank(o: Op?) -> int {
    return match o {
        null => -1,
        Op.Add => 1,
        _ => 0,
    }
}

fn _ops() -> list[Op] {
    return [Op.Add, Op.Sub, Op.Mul, Op.Div, Op.Mod, Op.Pow]
}

pub fn test_every_variant_agrees_with_the_if_chain() -> void {
    let ops = _ops()
    var call = 0
    while call < CALLS {
        let o = ops[call % 6]
        assert_eq(rank(o), rank_if(o), "rank agrees with the if chain")
        assert_eq(rank_exhaustive(o), rank_if(o), "so does the exhaustive form")
        call += 1
    }
    #: Named one by one too, so a tag compare that is off by one cannot hide
    #: behind an oracle computed the same wrong way.
    assert_eq(rank(Op.Add), 1, "the first variant")
    assert_eq(rank(Op.Sub), 2, "the second")
    assert_eq(rank(Op.Mul), 3, "the third")
    assert_eq(rank(Op.Div), 4, "the fourth")
    assert_eq(rank(Op.Mod), 5, "the fifth")
    assert_eq(rank(Op.Pow), 6, "the last variant")
}

pub fn test_alternation_and_default() -> void {
    var additive = 0
    var multiplicative = 0
    var other = 0
    let ops = _ops()
    var call = 0
    while call < CALLS {
        let f = family(ops[call % 6])
        if f == "additive" { additive += 1 }
        if f == "multiplicative" { multiplicative += 1 }
        if f == "other" { other += 1 }
        call += 1
    }
    assert_eq(additive, CALLS / 3, "two of six variants are additive")
    assert_eq(multiplicative, CALLS / 2, "three of six are multiplicative")
    assert_eq(other, CALLS / 6, "one of six falls to the default")
}

pub fn test_first_and_last_variant_only() -> void {
    var hits = 0
    let ops = _ops()
    var call = 0
    while call < CALLS {
        if first_or_last(ops[call % 6]) { hits += 1 }
        call += 1
    }
    assert_eq(hits, CALLS / 3, "the first and last variant, and nothing else")
}

pub fn test_a_guard_arm_between_two_variant_arms() -> void {
    var call = 0
    while call < CALLS {
        #: The guard is false, so the variant arms below it decide.
        assert_eq(wrapped(Op.Add, 7), 8, "the arm above the guard")
        assert_eq(wrapped(Op.Sub, 7), 6, "an arm below the guard")
        assert_eq(wrapped(Op.Mul, 7), 7, "the default below the guard")
        #: The guard is true, so it takes everything but the arm above it.
        assert_eq(wrapped(Op.Add, -1), 0, "the arm above the guard still wins")
        assert_eq(wrapped(Op.Sub, -1), 200, "the guard takes the rest")
        assert_eq(wrapped(Op.Pow, -1), 600, "the guard takes the default too")
        call += 1
    }
}

pub fn test_payload_variants() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(sized(Shape.Dot), 0, "a payload-less variant")
        assert_eq(sized(Shape.Circle(5)), 10, "a one-field variant")
        assert_eq(sized(Shape.Box(3, 4)), 12, "a two-field variant")
        call += 1
    }
}

pub fn test_a_subject_that_is_not_the_enum() -> void {
    #: Warm on enum values alone, so the compiled form is the one the misses
    #: below have to be answered by.
    var call = 0
    while call < CALLS {
        assert_eq(of_any(Op.Add), "add", "an enum value still matches")
        assert_eq(of_any(Op.Mod), "mod", "a later arm still matches")
        call += 1
    }
    #: Heap objects: admitted by the entry guard (SLOT_OBJ asks only IS_OBJ),
    #: so these run the compiled type test and take its miss branch.
    call = 0
    while call < CALLS {
        assert_eq(of_any("Add"), "not an op", "a string misses every variant")
        assert_eq(of_any([1, 2]), "not an op", "so does a list")
        assert_eq(of_any({"Add": 1}), "not an op", "and a dict")
        assert_eq(of_any(Op.Sub), "not an op", "an unlisted variant misses")
        #: Not a heap object at all, so this one is turned away at the door.
        assert_eq(of_any(3), "not an op", "an int misses too")
        call += 1
    }
}

pub fn test_a_nullable_subject() -> void {
    var call = 0
    var nulls = 0
    while call < CALLS {
        assert_eq(maybe_rank(Op.Add), 1, "a named variant")
        assert_eq(maybe_rank(Op.Sub), 0, "the default")
        if maybe_rank(null) == -1 { nulls += 1 }
        call += 1
    }
    assert_eq(nulls, CALLS, "null matched the null arm every time")
}

pub fn test_the_tag_is_read_sixteen_bits_wide() -> void {
    #: An ObjEnumVal's `tag` is a uint16_t with its `count` byte in the same
    #: word, and `count` is non-zero exactly for a payload variant. A tag read
    #: four bytes wide would drag that byte into the compare, so a payload
    #: variant would match no arm at all -- which is what this asks.
    var call = 0
    while call < CALLS {
        assert_true(sized(Shape.Circle(1)) == 2, "a one-field variant matched")
        assert_true(sized(Shape.Box(1, 1)) == 1, "a two-field variant matched")
        call += 1
    }
}
