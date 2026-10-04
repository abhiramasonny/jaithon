#: f-strings from compiled code, which go through jaiValueFormatLeaf -- a leaf
#: call with no roots -- and fall back to the descriptor call to jitFormat for
#: anything it does not render itself: a float or list part, a result past
#: the intern limit, or a collection due.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1` runs
#: the same file, and every answer is the interpreter's.
# jit-compiles: fmt_key, fmt_mixed, fmt_long, fmt_float, fmt_maybe, fmt_many
from std.test import assert_eq, assert_true

let CALLS = 100

class Box {
    var v: int

    fn init(self, v: int) { self.v = v }
}

fn fmt_key(n: int) -> str {
    return f"k{n % 10}"
}

fn fmt_mixed(flag: bool, n: int, s: str) -> str {
    return f"{s}:{n}:{flag}"
}

fn fmt_long(n: int) -> str {
    return f"a prefix long enough to pass the intern limit {n}"
}

fn fmt_float(x: float) -> str {
    return f"x={x}"
}

fn fmt_maybe(b: Box?) -> str {
    return f"<{b == null}>"
}

fn fmt_many(n: int) -> int {
    #: Thousands of distinct short strings, enough to cross the collector's
    #: threshold inside the loop: the leaf must decline then, not allocate.
    var keep = []
    var total = 0
    var i = 0
    while i < n {
        let s = f"v{i}-{i * 7}"
        total += s.len()
        if i % 100 == 0 { keep.push(s) }
        i += 1
    }
    return total + keep.len()
}

pub fn test_short_keys_are_the_interned_strings() -> void {
    var call = 0
    while call < CALLS {
        let k = fmt_key(call)
        assert_eq(k, f"k{call % 10}")
        call += 1
    }
    #: Equal short results are one object, as the interpreter's are.
    assert_true(fmt_key(3) is fmt_key(13), "interned")
}

pub fn test_ints_bools_and_strings() -> void {
    var call = 0
    while call < CALLS {
        let flag = call % 2 == 0
        assert_eq(fmt_mixed(flag, call - 50, "s"), f"s:{call - 50}:{flag}")
        call += 1
    }
    assert_eq(fmt_mixed(true, -7, "ab"), "ab:-7:true")
    assert_eq(fmt_mixed(false, 0, ""), ":0:false")
}

pub fn test_a_long_result_takes_the_general_path() -> void {
    var call = 0
    var total = 0
    while call < CALLS {
        total += fmt_long(call).len()
        call += 1
    }
    assert_eq(fmt_long(42), "a prefix long enough to pass the intern limit 42")
    assert_eq(total, 46 * CALLS + 10 + 90 * 2)
}

pub fn test_a_float_part_takes_the_general_path() -> void {
    var call = 0
    while call < CALLS {
        let _ = fmt_float(1.5)
        call += 1
    }
    assert_eq(fmt_float(2.25), "x=2.25")
}

pub fn test_a_nullable_part() -> void {
    var call = 0
    while call < CALLS {
        let b: Box? = if call % 2 == 0 { Box(call) } else { null }
        assert_eq(fmt_maybe(b), if call % 2 == 0 { "<false>" } else { "<true>" })
        call += 1
    }
}

pub fn test_many_distinct_strings_across_collections() -> void {
    var call = 0
    var total = 0
    while call < CALLS {
        total += fmt_many(300)
        call += 1
    }
    assert_eq(total, CALLS * fmt_many(300))
}
