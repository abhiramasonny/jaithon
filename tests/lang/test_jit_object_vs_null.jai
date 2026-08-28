#: `x == null` where the model calls x an object -- a string, a list, a dict.
#:
#: The compare arm already mixed SLOT_INST with SLOT_MAYBE_INST, because both
#: are a pointer or a zero in a register and the compare is the same `subs`.
#: SLOT_OBJ is the same shape and stronger: `emitTagFor` gives it VAL_OBJ
#: unconditionally, so its register never holds a zero, and comparing it against
#: the null literal's zero register produces exactly the right answer.
#:
#: Without it, `if v == null` on anything the model had not named more
#: precisely declined the whole body -- 108 declines across four files of the
#: self-hosted compiler, in `_parse_postfix`, `_parse_decorators` and their kin.
#:
#:   a hot `v == null` loop, 3M rounds   39,000,326 -> 784,523   0.08s -> 0.01s   8x
#:   check --no-cache parser.jai/resolve.jai                     0.55s / 0.27s    --
#:
#: The second line is the honest control and it is why the first is worth
#: having: the compiler's own 108 sites are in bodies that decline for other
#: reasons anyway, so clearing this link moves nothing THERE. It moves 8x where
#: the comparison is the hot thing, which in a nullable-heavy language is most
#: loops that walk an optional chain.
#:
#: SLOT_OBJ ONLY, and that is the load-bearing half. A SLOT_INT is also never
#: null, but its register holds a NUMBER -- and zero is a perfectly good int, so
#: `0 == null` would compare equal and answer true. Only a kind whose register
#: holds a pointer may be compared against the null literal's zero.
#:
#: EQUALITY ONLY, and that is load-bearing too: `x < null` is a TypeError in the
#: interpreter, and compiling it as a register compare would answer where it
#: must raise. It still reports
#: `TypeError: '<' is not supported between 'str' and 'null'` from compiled
#: code, the interpreter's own message.
# jit-compiles: count_nulls
from std.test import assert_eq, assert_true, assert_throws

#: Above JAI_JIT_THRESHOLD entries.
let CALLS = 200

#: Read out of a list, so the model names the value an OBJECT. Taking it from a
#: function that returns `any` does not reach this arm at all: the model has the
#: SLOT_INT a zeroed Emit starts on and declines on that instead.
fn count_nulls(texts: list[str], rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        let v = texts[i & 3]
        #: Both senses, because they take different conditions out of the same
        #: compare.
        if v == null { seen += 1 }
        if v != null { seen += 2 }
        i += 1
    }
    return seen
}

fn less(a: any, b: any) -> bool {
    return a < b
}

pub fn test_an_object_compared_against_null() -> void {
    let texts = ["a", "bb", "ccc", "dddd"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += count_nulls(texts, 8)
        call += 1
    }
    #: None of the eight is null, so 2 each.
    assert_eq(total, 16 * CALLS, "sum")
}

pub fn test_every_object_kind_answers() -> void {
    assert_true("text" != null, "a string is not null")
    assert_true([1, 2] != null, "nor a list")
    assert_true({"k": 1} != null, "nor a dict")
    assert_true(null == null, "and null is")
    assert_eq(count_nulls(["a", "b", "c", "d"], 0), 0, "no rounds")
    assert_eq(count_nulls(["a", "b", "c", "d"], 4), 8, "four non-null reads")
}

pub fn test_an_ordering_against_null_still_raises() -> void {
    #: Warm the ordering arm on two strings first, so the body is compiled when
    #: the null arrives.
    var call = 0
    while call < CALLS {
        let _ = less("a", "b")
        call += 1
    }
    assert_true(less("a", "b"), "still answers")
    assert_throws(TypeError, || less("a", null), "an ordering against null must raise, not answer")
}
