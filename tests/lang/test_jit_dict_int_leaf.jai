#: Int-keyed dict reads, `get`, stores, `in` and the two counting idioms from
#: compiled code, through the int leaves (jitDictGetInt and its kin). They
#: answer only from an entry holding that very int; a hash-equal key of
#: another kind -- the float equal to it, which hashes the same and compares
#: equal -- is the general path's to judge, through the descriptor call.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1`
#: runs the same file, and every answer is the interpreter's.
# jit-compiles: int_count, int_get_sum, int_index_sum, int_members, int_store, int_aug
from std.test import assert_eq, assert_throws, assert_true

let CALLS = 100

fn int_count(counts: dict[int, int], xs: list[int]) -> void {
    for x in xs {
        counts[x] = counts.get(x, 0) + 1
    }
}

fn int_get_sum(d: dict[int, int], xs: list[int]) -> int {
    var t = 0
    for x in xs {
        t += d.get(x, 100)
    }
    return t
}

fn int_index_sum(d: dict[int, int], xs: list[int]) -> int {
    var t = 0
    for x in xs {
        t += d[x]
    }
    return t
}

fn int_members(d: dict[int, int], xs: list[int]) -> int {
    var n = 0
    for x in xs {
        if x in d { n += 1 }
        if x not in d { n += 10 }
    }
    return n
}

fn int_store(d: dict[int, int], xs: list[int]) -> void {
    for x in xs {
        d[x] = x * 2
    }
}

fn int_aug(d: dict[int, int], xs: list[int]) -> void {
    for x in xs {
        d[x] += 3
    }
}

pub fn test_counting_by_int() -> void {
    var counts = {}
    var call = 0
    while call < CALLS {
        int_count(counts, [1, 2, 1, -5, 1000000007])
        call += 1
    }
    assert_eq(counts.len(), 4)
    assert_eq(counts[1], 2 * CALLS)
    assert_eq(counts[-5], CALLS)
    assert_eq(counts[1000000007], CALLS)
    var order = []
    for (k, _v) in counts.items() { order.push(k) }
    assert_eq(order, [1, 2, -5, 1000000007])
}

pub fn test_reads() -> void {
    let d = {1: 10, 2: 20}
    var g = 0
    var i = 0
    var m = 0
    var call = 0
    while call < CALLS {
        g += int_get_sum(d, [1, 3])
        i += int_index_sum(d, [2, 2, 1])
        m += int_members(d, [1, 7])
        call += 1
    }
    assert_eq(g, 110 * CALLS)
    assert_eq(i, 50 * CALLS)
    assert_eq(m, 11 * CALLS)
    assert_throws(KeyError, fn() -> void {
        let _ = int_index_sum(d, [9])
    }, "a missing int key read from a compiled body")
}

pub fn test_stores_and_growth() -> void {
    var d = {}
    var call = 0
    while call < CALLS {
        var xs = []
        var j = 0
        while j < 40 {
            xs.push(call * 40 + j)
            j += 1
        }
        int_store(d, xs)
        call += 1
    }
    assert_eq(d.len(), 40 * CALLS)
    assert_eq(d[123], 246)
    assert_eq(d[40 * CALLS - 1], 2 * (40 * CALLS - 1))
}

pub fn test_augmented_add() -> void {
    var d = {5: 0, 6: 1}
    var call = 0
    while call < CALLS {
        int_aug(d, [5, 6, 5])
        call += 1
    }
    assert_eq(d[5], 6 * CALLS)
    assert_eq(d[6], 1 + 3 * CALLS)
    assert_throws(KeyError, fn() -> void {
        int_aug(d, [7])
    }, "`+=` on a missing int key")
}

pub fn test_a_float_key_equal_to_the_int() -> void {
    #: 2.0 == 2 and the two hash alike, so a dict keyed by the float answers
    #: for the int -- through the general path, which owns that equality.
    var d: dict[any, int] = {}
    d[2.0] = 7
    d[3] = 1
    var call = 0
    while call < CALLS {
        let _ = int_get_sum(d, [3])
        call += 1
    }
    assert_eq(int_get_sum(d, [2]), 7)
    assert_eq(int_index_sum(d, [2]), 7)
    assert_eq(int_members(d, [2]), 1)
    int_count(d, [2])
    assert_eq(d.len(), 2)
    assert_eq(d[2.0], 8)
    int_store(d, [2])
    assert_eq(d.len(), 2)
    assert_eq(d[2], 4)
}

pub fn test_a_typed_dict() -> void {
    var d: dict[int, int] = {}
    var call = 0
    while call < CALLS {
        int_count(d, [4, 4, 9])
        call += 1
    }
    assert_eq(d[4], 2 * CALLS)
    assert_eq(d[9], CALLS)
}
