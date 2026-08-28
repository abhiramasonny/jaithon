#: A list method whose result is USED, in a loop hot enough to compile.
#:
#: The arm took only two shapes: a field-reading builtin, which states its own
#: result kind (`xs.len()` is the list's count), and a call whose result the
#: next instruction pops. Anything else declined the whole body, and the stated
#: reason was that "a wrong guess has nowhere to go" -- true when it was
#: written, and not since. The SLOT_OBJ arm predicts from the site's own
#: `InlineCache::resultKind` and resumes AFTER the call with the result read
#: out of the descriptor (`branchOnDeoptAt`'s `lastFromDesc`), which is exactly
#: the somewhere. This arm now does the same.
#:
#: `xs.contains(v) / xs.index(v) / xs.count(v)` on a five-element list, two
#: million rounds: 34,156,577 interpreted instructions and 165,300 after,
#: **1.44x** on the clock (0.13s to 0.09s, five runs each, no spread, measured
#: in one binary through `JAITHON_JIT_LIST_RESULT`).
#:
#: Which methods this is worth having for is the same question `sorted` failed:
#: a row buys the enclosing BODY, so the method has to be cheap next to the loop
#: around it. `contains`, `index` and `count` on a short list are. `enumerate`,
#: `map`, `filter` and `sorted` allocate a list per call and are not -- they
#: compile now too, and that is fine, but it is not where the time was.
# jit-compiles: scan_for, first_index, tally
from std.test import assert_eq

let CALLS = 100

fn scan_for(xs: list[int], rounds: int) -> int {
    var seen = 0
    var i = 0
    while i < rounds {
        if xs.contains(30) { seen += 1 }
        i += 1
    }
    return seen
}

fn first_index(xs: list[int], rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        t += xs.index(30)
        i += 1
    }
    return t
}

fn tally(xs: list[int], rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        t += xs.count(20)
        i += 1
    }
    return t
}

pub fn test_a_bool_result_is_one_byte() -> void {
    #: A bool read eight bytes wide picks up whatever sat beside it in the
    #: result slot, and every SLOT_BOOL consumer tests the whole word.
    let xs = [10, 20, 30, 40, 20]
    var seen = 0
    var call = 0
    while call < CALLS {
        seen += scan_for(xs, 10)
        call += 1
    }
    assert_eq(seen, 10 * CALLS)
}

pub fn test_an_int_result() -> void {
    let xs = [10, 20, 30, 40, 20]
    var total = 0
    var call = 0
    while call < CALLS {
        total += first_index(xs, 10)
        call += 1
    }
    assert_eq(total, 2 * 10 * CALLS)
}

pub fn test_a_counted_result() -> void {
    let xs = [10, 20, 30, 40, 20]
    var total = 0
    var call = 0
    while call < CALLS {
        total += tally(xs, 10)
        call += 1
    }
    assert_eq(total, 2 * 10 * CALLS)
}

pub fn test_a_list_result_still_answers() -> void {
    #: `sorted` allocates, so this is not where the time is -- but it compiles
    #: now, and it has to still be right.
    let xs = [30, 10, 20]
    var total = 0
    var call = 0
    while call < CALLS {
        total += xs.sorted().len()
        call += 1
    }
    assert_eq(total, 3 * CALLS)
}
