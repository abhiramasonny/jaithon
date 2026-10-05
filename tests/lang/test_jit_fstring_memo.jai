#: The f-string memo: compiled code probes jaiFmtMemo for (pre, n, post)
#: before it calls jaiValueFormatIntLeafMemo, and a hit is the string the leaf
#: answered last time. The memo is weak -- every collection empties it -- and
#: keys its runs by identity, so these check the cases where a stale or
#: confused entry would show: the all-absent key of `f"{n}"` against an empty
#: slot, two sites sharing one int, a run that is a different string on each
#: call, runs freed and their addresses reused across collections, and the
#: intern table's soft cap, past which every call is a new string.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it; `JAITHON_NO_JIT=1` runs the same file, and
#: every answer is the interpreter's.
# jit-compiles: bare, stringify, two_sites, with_runs, keyed_count, label, plain_label, same_label_count, same_plain_count
from std.test import assert_eq, assert_true

fn bare(n: int) -> str {
    return f"{n}"
}

fn stringify(n: int) -> str {
    return str(n)
}

fn two_sites(n: int) -> str {
    let a = f"a{n}"
    let b = f"b{n}"
    return a + "|" + b
}

fn with_runs(pre: str, n: int, post: str) -> str {
    return f"{pre}{n}{post}"
}

fn keyed_count(counts: dict, n: int, reps: int) -> void {
    var i = 0
    while i < reps {
        let key = f"k{i % n}"
        counts[key] = counts.get(key, 0) + 1
        i += 1
    }
}

fn label(n: int) -> str {
    return f"item-{n}"
}

fn plain_label(n: int) -> str {
    return str(n + 1000)
}

fn same_label_count(reps: int) -> int {
    var i = 0
    var t = 0
    while i < reps {
        let a = label(i % 20)
        let b = label(i % 20)
        if a is b { t += 1 }
        i += 1
    }
    return t
}

fn same_plain_count(reps: int) -> int {
    var i = 0
    var t = 0
    while i < reps {
        let a = plain_label(i % 20)
        let b = plain_label(i % 20)
        if a is b { t += 1 }
        i += 1
    }
    return t
}

pub fn test_bare_int_zero_is_not_an_empty_slot() -> void {
    var round = 0
    while round < 3 {
        var i = -5
        while i < 200 {
            assert_eq(bare(i), str(i))
            assert_eq(bare(0), "0")
            i += 1
        }
        round += 1
    }
}

pub fn test_str_of_an_int_shares_the_memo() -> void {
    var round = 0
    while round < 3 {
        var i = -50
        while i < 250 {
            let a = stringify(i)
            let b = bare(i)
            assert_eq(a, b)
            assert_eq(stringify(0), "0")
            i += 1
        }
        __prim__.gc_collect()
        round += 1
    }
}

pub fn test_two_sites_on_one_int_stay_apart() -> void {
    var round = 0
    while round < 4 {
        var i = 0
        while i < 300 {
            assert_eq(two_sites(i), "a" + str(i) + "|b" + str(i))
            i += 1
        }
        round += 1
    }
}

pub fn test_runs_are_keys_by_identity() -> void {
    let pres = ["x", "y", "xy", ""]
    let posts = ["", "z", "-"]
    var round = 0
    while round < 3 {
        var i = 0
        while i < 240 {
            let pre = pres[i % pres.len()]
            let post = posts[i % posts.len()]
            let n = i % 17
            assert_eq(with_runs(pre, n, post), pre + str(n) + post)
            i += 1
        }
        round += 1
    }
}

pub fn test_freed_runs_do_not_answer_for_new_ones() -> void {
    #: A run built at run time, used until the memo holds it, then dropped
    #: and collected; the next run is built the same way and may land at the
    #: same address. A memo that outlived the collection would answer the
    #: old run's string.
    var round = 0
    while round < 30 {
        let pre = "p" + str(round) + "_"
        var i = 0
        while i < 150 {
            assert_eq(with_runs(pre, i % 5, ""), pre + str(i % 5))
            i += 1
        }
        __prim__.gc_collect()
        var junk = []
        var j = 0
        while j < 200 {
            junk.push("q" + str(round * 1000 + j) + "_")
            j += 1
        }
        round += 1
    }
}

pub fn test_counting_loop_counts() -> void {
    var counts = {}
    keyed_count(counts, 37, 5000)
    __prim__.gc_collect()
    keyed_count(counts, 37, 5000)
    assert_eq(counts.len(), 37)
    var total = 0
    for (_k, v) in counts.items() { total += v }
    assert_eq(total, 10000)
    assert_eq(counts["k0"], 272)
    assert_eq(counts["k36"], 270)
}

pub fn test_past_the_intern_soft_cap() -> void {
    #: More distinct short strings than the intern table keeps; the memo must
    #: still answer only equal strings, and a dict keyed by them must count.
    var keep = []
    var i = 0
    while i < 40000 {
        keep.push("s" + str(i))
        i += 1
    }
    var counts = {}
    keyed_count(counts, 50, 3000)
    assert_eq(counts.len(), 50)
    var total = 0
    for (_k, v) in counts.items() { total += v }
    assert_eq(total, 3000)
    assert_true(keep.len() == 40000)
    var r = 0
    while r < 200 {
        assert_eq(bare(r + 100000), str(r + 100000))
        r += 1
    }
}

pub fn test_past_the_cap_each_call_is_a_new_string() -> void {
    #: Once the intern table is past its soft cap the leaf builds a fresh
    #: string on every call, so two calls are two objects -- as the
    #: interpreter has them. A memo entry filed before the cap must not keep
    #: answering the old one. The ballast lifts the collector's threshold,
    #: since a collection in between would empty the memo and hide it.
    var live = []
    var j = 0
    while j < 300000 {
        live.push([j, j + 1])
        j += 1
    }
    __prim__.gc_collect()
    same_label_count(3000)
    same_plain_count(3000)
    var keep = []
    var k = 0
    while k < 40000 {
        keep.push(f"name{k}")
        k += 1
    }
    assert_eq(same_label_count(3000), 0)
    assert_eq(same_plain_count(3000), 0)
    assert_true(keep.len() == 40000 and live.len() == 300000)
}
