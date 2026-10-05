#: The counting arms' inline update (emitDictAddInline): a string key found in
#: the first slot its hash names, holding an int, gets the step added with no
#: call; anything else goes to jitDictAddStr. These drive the cases where the
#: inline path must hand over instead of answering: keys that collide out of
#: their first slot, slots left as tombstones by a removal, a negative step,
#: a value that is not an int, and an add that would overflow.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it; `JAITHON_NO_JIT=1` runs the same file, and
#: every answer is the interpreter's.
# jit-compiles: up_by_one, down_by_two, aug_up
from std.test import assert_eq, assert_true

fn up_by_one(d: dict, keys: list[str]) -> void {
    for k in keys {
        d[k] = d.get(k, 0) + 1
    }
}

fn down_by_two(d: dict, keys: list[str]) -> void {
    for k in keys {
        d[k] = d.get(k, 10) + -2
    }
}

fn aug_up(d: dict, keys: list[str]) -> void {
    for k in keys {
        d[k] += 3
    }
}

fn keys_of(n: int) -> list[str] {
    var out = []
    var i = 0
    while i < n {
        out.push(f"key-{i}")
        i += 1
    }
    return out
}

pub fn test_many_keys_with_collisions() -> void {
    let keys = keys_of(5000)
    var d = {}
    var round = 0
    while round < 70 {
        up_by_one(d, keys)
        round += 1
    }
    assert_eq(d.len(), 5000)
    for k in keys { assert_eq(d[k], 70) }
}

pub fn test_tombstones_and_reinserts() -> void {
    let keys = keys_of(400)
    var d = {}
    var round = 0
    while round < 70 {
        up_by_one(d, keys)
        if round % 7 == 3 {
            var i = 0
            while i < keys.len() {
                d.remove(keys[i])
                i += 3
            }
        }
        round += 1
    }
    #: Removed on rounds 3, 10, ..., 66; counted again after each.
    assert_eq(d["key-0"], 3)
    assert_eq(d["key-1"], 70)
    assert_eq(d["key-3"], 3)
    assert_eq(d.len(), 400)
}

pub fn test_negative_step() -> void {
    let keys = keys_of(50)
    var d = {}
    var round = 0
    while round < 80 {
        down_by_two(d, keys)
        round += 1
    }
    for k in keys { assert_eq(d[k], 10 - 2 * 80) }
}

pub fn test_values_that_are_not_ints() -> void {
    let keys = keys_of(20)
    var d = {}
    for k in keys { d[k] = 0.5 }
    var round = 0
    while round < 70 {
        up_by_one(d, keys)
        aug_up(d, keys)
        round += 1
    }
    for k in keys { assert_eq(d[k], 0.5 + 4.0 * 70.0) }
}

pub fn test_overflow_raises_from_the_inline_path() -> void {
    let keys = keys_of(10)
    var d = {}
    var round = 0
    while round < 70 {
        up_by_one(d, keys)
        round += 1
    }
    d["key-4"] = 9223372036854775807
    var raised = false
    try {
        up_by_one(d, keys)
    } catch _e: Error {
        raised = true
    }
    assert_true(raised)
    assert_eq(d["key-4"], 9223372036854775807)
    assert_eq(d["key-3"], 71)
}

pub fn test_augmented_add_of_an_absent_key_raises() -> void {
    let keys = keys_of(10)
    var round = 0
    while round < 70 {
        var e = {}
        for k in keys { e[k] = 0 }
        aug_up(e, keys)
        assert_eq(e["key-9"], 3)
        var raised = false
        var f = {"key-0": 1}
        try {
            aug_up(f, keys)
        } catch _e: Error {
            raised = true
        }
        assert_true(raised)
        assert_eq(f["key-0"], 4)
        round += 1
    }
}
