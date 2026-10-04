#: String-keyed `d.get(k)`, `d.get(k, default)`, `d[k]` and `d[k] = v` from
#: compiled code, which answer through a leaf call (jitDictGetStr /
#: jitDictSetStr) and fall back to the descriptor call for anything the leaf
#: cannot settle: a key that is not a string, a miss under `d[k]`, a stored key
#: of another kind whose hash matches, a value a typed dict refuses.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times, so the
#: whole-function tier compiles it on call count alone, and every answer is
#: one the interpreter gives too -- `JAITHON_NO_JIT=1` runs the same file.
# jit-compiles: leaf_bump_words, leaf_sum_present, leaf_total_get, leaf_read_floats, leaf_get_each, leaf_mixed_keys, leaf_bump_typed, leaf_store_any, leaf_count_members
from std.test import assert_eq, assert_throws, assert_true

let CALLS = 100

fn leaf_bump_words(counts: dict[str, int], words: list[str]) -> void {
    for w in words {
        counts[w] = counts.get(w, 0) + 1
    }
}

fn leaf_sum_present(d: dict[str, int], keys: list[str]) -> int {
    var total = 0
    for k in keys {
        total += d[k]
    }
    return total
}

#: Every key present: a site that has seen `null` come back has a mixed
#: result kind and is refused, so the no-default form compiles only here.
fn leaf_total_get(d: dict[str, int], keys: list[str]) -> int {
    var total = 0
    for k in keys {
        total += d.get(k)
    }
    return total
}

fn total_get_or_null(d: dict[str, int], keys: list[str]) -> int {
    var total = 0
    for k in keys {
        let v = d.get(k)
        if v != null { total += v }
    }
    return total
}

fn leaf_read_floats(d: dict[str, float], keys: list[str]) -> float {
    var total = 0.0
    for k in keys {
        total += d.get(k, 0.5)
    }
    return total
}

pub fn test_counting_inserts_then_updates() -> void {
    let words = ["a", "bb", "a", "ccc", "bb", "a"]
    var counts = {}
    var call = 0
    while call < CALLS {
        leaf_bump_words(counts, words)
        call += 1
    }
    assert_eq(counts.len(), 3)
    assert_eq(counts["a"], 3 * CALLS)
    assert_eq(counts["bb"], 2 * CALLS)
    assert_eq(counts["ccc"], CALLS)
}

pub fn test_growth_inside_a_compiled_loop() -> void {
    var counts = {}
    var call = 0
    while call < CALLS {
        var keys = []
        var i = 0
        while i < 50 {
            keys.push(f"key{call * 50 + i}")
            i += 1
        }
        leaf_bump_words(counts, keys)
        leaf_bump_words(counts, keys)
        call += 1
    }
    assert_eq(counts.len(), 50 * CALLS)
    assert_eq(counts["key0"], 2)
    assert_eq(counts[f"key{50 * CALLS - 1}"], 2)
    #: Insertion order survives the table growing under the leaf.
    var n = 0
    for (k, _v) in counts.items() {
        if n == 0 { assert_eq(k, "key0") }
        if n == 4999 { assert_eq(k, "key4999") }
        n += 1
    }
    assert_eq(n, 50 * CALLS)
}

pub fn test_long_keys_that_are_equal_but_not_identical() -> void {
    #: Past the intern limit two equal strings are two objects, so the leaf
    #: has to compare bytes rather than pointers.
    let prefix = "a-key-long-enough-to-skip-interning-"
    var counts = {}
    var call = 0
    while call < CALLS {
        var keys = []
        var i = 0
        while i < 14 {
            keys.push(f"{prefix}{i % 7}")
            i += 1
        }
        leaf_bump_words(counts, keys)
        call += 1
    }
    assert_eq(counts.len(), 7)
    assert_eq(counts[f"{prefix}3"], 2 * CALLS)
}

pub fn test_get_without_default() -> void {
    let d = {"x": 5, "y": 7}
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_total_get(d, ["x", "y", "y"])
        total += total_get_or_null(d, ["x", "zz"])
        call += 1
    }
    assert_eq(total, 24 * CALLS)
}

pub fn test_float_values_and_a_float_default() -> void {
    let d = {"p": 1.25, "q": 2.5}
    let keys = ["p", "missing", "q"]
    var total = 0.0
    var call = 0
    while call < CALLS {
        total += leaf_read_floats(d, keys)
        call += 1
    }
    assert_eq(total, 4.25 * 100.0)
}

pub fn test_present_reads_through_index() -> void {
    let d = {"a": 1, "b": 2}
    let keys = ["a", "b", "b"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_sum_present(d, keys)
        call += 1
    }
    assert_eq(total, 5 * CALLS)
}

pub fn test_a_miss_under_index_still_raises() -> void {
    let d = {"a": 1, "b": 2}
    var call = 0
    while call < CALLS {
        let _ = leaf_sum_present(d, ["a", "b"])
        call += 1
    }
    let err = assert_throws(KeyError, fn() -> void {
        let _ = leaf_sum_present(d, ["a", "nope"])
    }, "a missing key read from a compiled body")
    assert_true(err.message.contains("nope"), "the KeyError names the key")
}

class Lookalike {
    var name: str

    fn init(self, name: str) { self.name = name }

    pub fn __hash__(self) -> int { return hash(self.name) }

    pub fn __eq__(self, other: any) -> bool {
        return other == self.name
    }
}

fn leaf_get_each(d: dict[any, int], keys: list[str]) -> int {
    var total = 0
    for k in keys {
        total += d.get(k, 100)
    }
    return total
}

pub fn test_a_dict_that_also_holds_an_instance_key() -> void {
    #: A user `__hash__` is mixed before the table sees it, so the instance
    #: never shares a string's slot hash and "twin" is a miss in both tiers.
    #: What matters is that the leaf walks past the instance entry, and that
    #: the plain string key beside it still answers.
    var d: dict[any, int] = {}
    d[Lookalike("twin")] = 1
    d["plain"] = 2
    let keys = ["twin", "plain", "none"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_get_each(d, keys)
        call += 1
    }
    assert_eq(total, 202 * CALLS)
}

fn leaf_mixed_keys(d: dict[any, int], keys: list[any]) -> int {
    var total = 0
    for k in keys {
        total += d.get(k, 1000)
        d[k] = d.get(k, 0) + 1
    }
    return total
}

pub fn test_object_keys_that_are_not_strings_take_the_slow_path() -> void {
    #: The key's slot says "object", and the leaf's string test sends the
    #: tuple and the instance -- whose __eq__ runs user code -- back through
    #: the descriptor call.
    var d: dict[any, int] = {}
    let twin = Lookalike("twin")
    let keys: list[any] = ["s", (1, 2), twin, "s"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_mixed_keys(d, keys)
        call += 1
    }
    assert_eq(d.len(), 3)
    assert_eq(d["s"], 2 * CALLS)
    assert_eq(d[1, 2], CALLS)
    assert_eq(d[twin], CALLS)
    #: First call: 1000 + 1000 + 1000 + 1 (the second "s" sees the first's
    #: store). Call c after that: (2c-2) + (c-1) + (c-1) + (2c-1) = 6c - 5.
    var expect = 3001
    var c = 2
    while c <= CALLS {
        expect += 6 * c - 5
        c += 1
    }
    assert_eq(total, expect)
}

fn leaf_bump_typed(d: dict[str, int], keys: list[str]) -> void {
    for k in keys {
        d[k] = d.get(k, 0) + 1
    }
}

pub fn test_a_typed_dict() -> void {
    var d: dict[str, int] = {}
    let keys = ["t0", "t1", "t0"]
    var call = 0
    while call < CALLS {
        leaf_bump_typed(d, keys)
        call += 1
    }
    assert_eq(d.len(), 2)
    assert_eq(d["t0"], 2 * CALLS)
    assert_eq(d["t1"], CALLS)
}

fn leaf_store_any(d: dict[str, int], k: str, v: any) -> void {
    d[k] = v
}

pub fn test_a_typed_dict_still_refuses_a_wrong_value() -> void {
    #: The leaf checks the dict's declared kinds itself and hands a refusal
    #: back to the descriptor call, which raises the interpreter's TypeError.
    var d: dict[str, int] = {}
    var call = 0
    while call < CALLS {
        leaf_store_any(d, f"n{call % 3}", call)
        call += 1
    }
    assert_eq(d["n0"], 99)
    let err = assert_throws(TypeError, fn() -> void {
        leaf_store_any(d, "n1", "not an int")
    }, "a str stored into a dict[str, int] from a compiled body")
    assert_true(err.message.contains("int"), "the message names the kind")
    assert_eq(d["n1"], 97)
}

fn leaf_touch_while_walking(d: dict[str, int]) -> void {
    for (k, v) in d.items() {
        d[k] = v + 1
    }
}

pub fn test_a_store_during_iteration_is_still_seen() -> void {
    var raised = 0
    var call = 0
    while call < CALLS {
        let d = {"a": 1, "b": 2, "c": 3}
        try {
            leaf_touch_while_walking(d)
        } catch _e: Error {
            raised += 1
        }
        call += 1
    }
    #: A store into a dict mid-walk bumps its version, so every round
    #: raises -- in compiled code as in the interpreter.
    assert_eq(raised, CALLS)
}

fn leaf_count_members(c: any, keys: list[str]) -> int {
    var n = 0
    for k in keys {
        if k in c { n += 1 }
        if k not in c { n += 100 }
    }
    return n
}

pub fn test_membership() -> void {
    let d = {"a": 1, "bb": 2}
    let long = "a-key-long-enough-to-skip-interning-" + "x"
    var ld = {}
    ld[long] = 1
    let keys = ["a", "zz", "bb", f"{long}"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_count_members(d, keys)
        call += 1
    }
    #: a and bb present, zz and the long key absent.
    assert_eq(total, (2 + 200) * CALLS)
    #: A long key equal to the stored one but not the same object.
    assert_eq(leaf_count_members(ld, [f"{long}", "a"]), 1 + 100)
    #: The same compiled site, handed containers that are not dicts.
    assert_eq(leaf_count_members(["a", "q"], ["a", "zz"]), 1 + 100)
    assert_eq(leaf_count_members({"bb", "c"}, ["bb", "a"]), 1 + 100)
    assert_eq(leaf_count_members("abc", ["b", "z"]), 1 + 100)
}

fn leaf_int_keys(d: dict[int, int], n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        total += d.get(i % 4, 10)
        i += 1
    }
    return total
}

pub fn test_non_string_keys_are_untouched() -> void {
    let d = {0: 1, 1: 2}
    var total = 0
    var call = 0
    while call < CALLS {
        total += leaf_int_keys(d, 4)
        call += 1
    }
    assert_eq(total, 23 * CALLS)
}
