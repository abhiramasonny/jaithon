#: A dict the compiled body builds itself -- `var d = {}` -- has no live
#: sample when the body compiles, only the type OP_BUILD_DICT records. The
#: dict arms compile against a probe of that type (jitDictProbe) and guard the
#: real receiver at run time; before, any use of such a local declined the
#: whole function.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1`
#: runs the same file, and every answer is the interpreter's.
# jit-compiles: built_count, built_pop, built_tuple_keys, built_members
from std.test import assert_eq, assert_true

let CALLS = 100

fn built_count(words: list[str]) -> dict[str, int] {
    var counts = {}
    for w in words {
        counts[w] = counts.get(w, 0) + 1
    }
    return counts
}

fn built_pop(words: list[str]) -> int {
    var d = {}
    for w in words { d[w] = w.len() }
    var t = 0
    for w in words {
        if w in d { t += d.pop(w) }
    }
    return t + d.len()
}

fn built_tuple_keys(n: int) -> int {
    var d = {}
    var i = 0
    while i < n {
        let k = (i % 7, i % 3)
        d[k] = d.get(k, 0) + 1
        i += 1
    }
    return d.len() * 1000 + d.get((0, 0), 0)
}

fn built_members(xs: list[int]) -> int {
    var seen = {}
    var dups = 0
    for x in xs {
        if x in seen {
            dups += 1
        } else {
            seen[x] = true
        }
    }
    return dups
}

pub fn test_counting_into_a_fresh_dict() -> void {
    var call = 0
    while call < CALLS {
        let c = built_count(["a", "b", "a", "c", "a"])
        assert_eq(c["a"], 3)
        assert_eq(c.len(), 3)
        call += 1
    }
}

pub fn test_pop_from_a_fresh_dict() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(built_pop(["ab", "cde", "ab"]), 5)
        call += 1
    }
}

pub fn test_tuple_keys_in_a_fresh_dict() -> void {
    var call = 0
    while call < CALLS {
        #: 21 distinct (i % 7, i % 3) pairs over 0..63; (0, 0) at i = 0, 21, 42.
        assert_eq(built_tuple_keys(63), 21 * 1000 + 3)
        call += 1
    }
}

pub fn test_membership_in_a_fresh_dict() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(built_members([1, 2, 1, 3, 2, 1]), 3)
        call += 1
    }
    assert_true(built_members([]) == 0, "empty")
}
