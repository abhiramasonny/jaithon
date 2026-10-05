#: The inline slot hit in front of the string-keyed dict get and store leaves:
#: a present key in the first slot its hash names is read or updated with no
#: call. These check that each kind of value goes in and comes out whole, that
#: removed and re-added keys and colliding keys are still found, and that a
#: typed dict's store is left to the leaf.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it; `JAITHON_NO_JIT=1` runs the same file, and
#: every answer is the interpreter's.
# jit-compiles: read_all, store_all, store_maybe, read_or, int_count, int_read, int_store
from std.test import assert_eq, assert_true

class Box {
    pub var v: int

    fn init(self, v: int) { self.v = v }
}

fn read_all(d: dict, keys: list[str]) -> int {
    var total = 0
    for k in keys {
        let v: int = d[k]
        total += v
    }
    return total
}

fn read_or(d: dict, keys: list[str]) -> int {
    var total = 0
    for k in keys {
        let v: int = d.get(k, -1)
        total += v
    }
    return total
}

fn store_all(d: dict, keys: list[str], v: float) -> void {
    for k in keys {
        d[k] = v
    }
}

fn store_maybe(d: dict, keys: list[str], b: Box?) -> void {
    for k in keys {
        d[k] = b
    }
}

fn int_count(d: dict, keys: list[int]) -> void {
    for k in keys {
        d[k] = d.get(k, 0) + 1
    }
}

fn int_read(d: dict, keys: list[int]) -> float {
    var total = 0.0
    for k in keys {
        let v: float = d[k]
        total += v
    }
    return total
}

fn int_store(d: dict, keys: list[int], v: float) -> void {
    for k in keys {
        d[k] = v
    }
}

fn keys_of(n: int) -> list[str] {
    var out = []
    var i = 0
    while i < n {
        out.push(f"slot-{i}")
        i += 1
    }
    return out
}

pub fn test_reads_with_collisions_and_removals() -> void {
    let keys = keys_of(3000)
    var d = {}
    var i = 0
    for k in keys {
        d[k] = i
        i += 1
    }
    let want = 3000 * 2999 // 2
    var round = 0
    while round < 70 {
        assert_eq(read_all(d, keys), want)
        assert_eq(read_or(d, keys), want)
        round += 1
    }
    var j = 0
    while j < keys.len() {
        d.remove(keys[j])
        j += 5
    }
    #: Removed keys answer the default; re-added ones their new value.
    var removed = 0
    j = 0
    while j < keys.len() {
        removed += j
        j += 5
    }
    assert_eq(read_or(d, keys), want - removed - 600)
    j = 0
    while j < keys.len() {
        d[keys[j]] = 0
        j += 5
    }
    assert_eq(read_all(d, keys), want - removed)
}

pub fn test_stores_of_every_kind() -> void {
    let keys = keys_of(40)
    var d = {}
    for k in keys { d[k] = 0 }
    var round = 0
    while round < 70 {
        store_all(d, keys, 1.5)
        assert_eq(d["slot-7"], 1.5)
        store_maybe(d, keys, Box(round))
        let b: Box? = d["slot-9"]
        assert_true(b != null)
        assert_eq(b.v, round)
        store_maybe(d, keys, null)
        assert_true(d["slot-3"] == null)
        round += 1
    }
    assert_eq(d.len(), 40)
}

pub fn test_a_typed_dict_store_is_the_leafs() -> void {
    let keys = keys_of(10)
    var round = 0
    while round < 70 {
        var t: dict[str, float] = {}
        for k in keys { t[k] = 0.0 }
        store_all(t, keys, 2.5)
        assert_eq(t["slot-1"], 2.5)
        round += 1
    }
}

pub fn test_int_keys() -> void {
    var keys = []
    var i = -300
    while i < 300 {
        keys.push(i * 1_000_003)
        i += 1
    }
    keys.push(9223372036854775807)
    keys.push(-9223372036854775807 - 1)
    var d = {}
    var round = 0
    while round < 70 {
        int_count(d, keys)
        round += 1
    }
    assert_eq(d.len(), keys.len())
    for k in keys { assert_eq(d[k], 70) }
    var f = {}
    for k in keys { f[k] = 0.0 }
    round = 0
    while round < 70 {
        int_store(f, keys, 0.25)
        assert_eq(int_read(f, keys), 0.25 * keys.len())
        round += 1
    }
}

pub fn test_an_equal_float_key_is_the_leafs() -> void {
    #: 2.0 and 2 are one key; the inline test answers only an int stored as
    #: an int, and the leaf finds the float.
    var keys = [1, 2, 3]
    var round = 0
    while round < 70 {
        var d = {}
        d[2.0] = 0.5
        d[1] = 0.5
        d[3] = 0.5
        assert_eq(int_read(d, keys), 1.5)
        int_store(d, keys, 1.0)
        assert_eq(d.len(), 3)
        assert_eq(d[2.0], 1.0)
        var c = {}
        c[2.0] = 5
        int_count(c, keys)
        assert_eq(c[2], 6)
        assert_eq(c.len(), 3)
        round += 1
    }
}
