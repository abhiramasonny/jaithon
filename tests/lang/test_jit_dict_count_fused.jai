#: `d[k] = d.get(k, n) + c` -- the counting idiom -- from compiled code, which
#: runs as one leaf (jitDictAddStr) when the dict and the key are the same
#: locals both times, and falls into the ordinary get, add and store for
#: everything that leaf does not settle: a key that is not a string, a value
#: that is not an int, an add that overflows, a typed dict that refuses.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1`
#: runs the same file, and every answer is the interpreter's.
# jit-compiles: count_into, count_by, count_any, count_typed, count_two_dicts, bump_aug, count_key_rebound, count_dict_rebound
from std.test import assert_eq, assert_throws, assert_true

let CALLS = 100

fn count_into(counts: dict[str, int], words: list[str]) -> void {
    for w in words {
        counts[w] = counts.get(w, 0) + 1
    }
}

fn count_by(counts: dict[str, int], words: list[str]) -> void {
    for w in words {
        counts[w] = counts.get(w, 10) + 7
    }
}

fn count_any(counts: dict[any, any], keys: list[any]) -> void {
    for k in keys {
        counts[k] = counts.get(k, 0) + 1
    }
}

fn count_typed(counts: dict[str, int], words: list[str]) -> void {
    for w in words {
        counts[w] = counts.get(w, 0) + 2
    }
}

fn count_two_dicts(a: dict[str, int], b: dict[str, int], words: list[str]) -> void {
    for w in words {
        a[w] = b.get(w, 0) + 1
    }
}

pub fn test_counting() -> void {
    var counts = {}
    var call = 0
    while call < CALLS {
        count_into(counts, ["a", "bb", "a", "ccc", "a"])
        call += 1
    }
    assert_eq(counts.len(), 3)
    assert_eq(counts["a"], 3 * CALLS)
    assert_eq(counts["bb"], CALLS)
    assert_eq(counts["ccc"], CALLS)
    #: Insertion order is the first time each key was counted.
    var order = []
    for (k, _v) in counts.items() { order.push(k) }
    assert_eq(order, ["a", "bb", "ccc"])
}

pub fn test_a_default_and_a_step() -> void {
    var counts = {}
    var call = 0
    while call < CALLS {
        count_by(counts, ["x", "y", "x"])
        call += 1
    }
    #: First time: 10 + 7, then +7 per later sighting.
    assert_eq(counts["x"], 10 + 7 * 2 * CALLS)
    assert_eq(counts["y"], 10 + 7 * CALLS)
}

pub fn test_long_keys_and_growth() -> void {
    let prefix = "a-key-long-enough-to-skip-interning-"
    var counts = {}
    var call = 0
    while call < CALLS {
        var words = []
        var i = 0
        while i < 30 {
            words.push(f"{prefix}{(call * 30 + i) % 1000}")
            i += 1
        }
        count_into(counts, words)
        call += 1
    }
    assert_eq(counts.len(), 1000)
    var total = 0
    for (_k, v) in counts.items() { total += v }
    assert_eq(total, 30 * CALLS)
    assert_eq(counts[f"{prefix}7"], 3)
}

pub fn test_values_and_keys_the_leaf_leaves_alone() -> void {
    var counts: dict[any, any] = {}
    counts["f"] = 1.5
    counts["s"] = "text"
    var call = 0
    while call < CALLS {
        count_any(counts, ["n", 4, (1, 2)])
        call += 1
    }
    assert_eq(counts["n"], CALLS)
    assert_eq(counts[4], CALLS)
    assert_eq(counts[1, 2], CALLS)
    #: A float value takes the ordinary add, which keeps it a float.
    count_any(counts, ["f"])
    assert_eq(counts["f"], 2.5)
    #: A string value takes the ordinary add, which raises what it raises.
    assert_throws(TypeError, fn() -> void {
        count_any(counts, ["s"])
    }, "str + int from a compiled body")
}

pub fn test_an_overflow_still_raises_and_writes_nothing() -> void {
    var counts = {}
    var call = 0
    while call < CALLS {
        count_into(counts, ["k"])
        call += 1
    }
    counts["big"] = 9223372036854775807
    assert_throws(OverflowError, fn() -> void {
        count_into(counts, ["big"])
    }, "the count past the largest int")
    assert_eq(counts["big"], 9223372036854775807)
}

pub fn test_a_typed_dict() -> void {
    var counts: dict[str, int] = {}
    var call = 0
    while call < CALLS {
        count_typed(counts, ["p", "q", "p"])
        call += 1
    }
    assert_eq(counts["p"], 4 * CALLS)
    assert_eq(counts["q"], 2 * CALLS)
}

pub fn test_two_different_dicts_are_not_fused() -> void {
    var a = {}
    var b = {"w": 5}
    var call = 0
    while call < CALLS {
        count_two_dicts(a, b, ["w", "z"])
        call += 1
    }
    assert_eq(a["w"], 6)
    assert_eq(a["z"], 1)
    assert_true("w" not in a or b["w"] == 5, "b is untouched")
    assert_eq(b.len(), 1)
}

fn bump_aug(c: any, keys: list[any]) -> void {
    for k in keys {
        c[k] += 5
    }
}

pub fn test_augmented_add() -> void {
    #: `d[k] += c`: the same leaf with no default.
    var counts: dict[any, any] = {"a": 0, "b": 1}
    var call = 0
    while call < CALLS {
        bump_aug(counts, ["a", "b", "a"])
        call += 1
    }
    assert_eq(counts["a"], 10 * CALLS)
    assert_eq(counts["b"], 1 + 5 * CALLS)
    #: An absent key is the read's KeyError, and nothing is inserted.
    assert_throws(KeyError, fn() -> void { bump_aug(counts, ["nope"]) }, "`+=` on a missing key")
    assert_true("nope" not in counts, "nothing inserted")
    #: A float value, a non-string key and a list all take the unfused code.
    counts["f"] = 0.5
    counts[3] = 1
    bump_aug(counts, ["f", 3])
    assert_eq(counts["f"], 5.5)
    assert_eq(counts[3], 6)
    var xs = [1, 2, 3]
    bump_aug(xs, [0, 2])
    assert_eq(xs, [6, 2, 8])
    counts["big"] = 9223372036854775805
    assert_throws(OverflowError, fn() -> void {
        bump_aug(counts, ["big"])
    }, "`+=` past the largest int")
    assert_eq(counts["big"], 9223372036854775805)
}

fn count_while_walking(counts: dict[str, int]) -> void {
    for (k, _v) in counts.items() {
        counts[k] = counts.get(k, 0) + 1
    }
}

pub fn test_a_count_during_iteration_is_still_seen() -> void {
    var raised = 0
    var call = 0
    while call < CALLS {
        let d = {"a": 1, "b": 2}
        try {
            count_while_walking(d)
        } catch _e: Error {
            raised += 1
        }
        call += 1
    }
    #: A store during iteration bumps the dict's version, so every call
    #: raises -- compiled or not.
    assert_eq(raised, CALLS)
}

fn nonempty(s: str) -> bool {
    return s.len() > 0
}

#: The store's dict and key are read before its value, and a block inside the
#: value can rebind either local after a call has settled the stack: the
#: second pair of loads then reads something else, and the leaf, which uses
#: the second pair for both, would store to the wrong place.
fn count_key_rebound(d: dict, k: str) -> void {
    d[k] = (if nonempty(k) {
        k = "zz"
        d
    } else {
        d
    }).get(k, 0) + 1
}

fn count_dict_rebound(d: dict, other: dict, k: str) -> void {
    d[k] = (if nonempty(k) {
        d = other
        d
    } else {
        d
    }).get(k, 0) + 1
}

pub fn test_a_local_rebound_between_the_loads_is_not_fused() -> void {
    var d: dict[str, int] = {}
    var call = 0
    while call < CALLS {
        count_key_rebound(d, "a")
        call += 1
    }
    #: Stored under "a" every time, from a read of "zz", which is never set.
    assert_eq(d.len(), 1)
    assert_eq(d["a"], 1)
    var a: dict[str, int] = {}
    var b: dict[str, int] = {}
    call = 0
    while call < CALLS {
        count_dict_rebound(a, b, "x")
        call += 1
    }
    assert_eq(a.len(), 1)
    assert_eq(a["x"], 1)
    assert_eq(b.len(), 0)
}
