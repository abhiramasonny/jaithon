#: OP_DUP and OP_DUP2 in compiled code. Every augmented subscript assignment
#: -- `xs[i] += v`, `d[k] += v`, `grid[i][j] -= v` -- reads its container and
#: subscript once and uses them twice, and before these were armed the walk
#: stopped at the copy and everything after it in the loop ran interpreted.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1`
#: runs the same file, and every answer is the interpreter's.
# jit-compiles: dup_hist, dup_floats, dup_dict, dup_grid, dup_field
from std.test import assert_eq, assert_throws

let CALLS = 100

fn dup_hist(hist: list[int], xs: list[int]) -> void {
    for x in xs {
        hist[x % 4] += 1
    }
}

fn dup_floats(acc: list[float], xs: list[float]) -> void {
    var i = 0
    while i < xs.len() {
        acc[i % 2] += xs[i] * 0.5
        i += 1
    }
}

fn dup_dict(counts: dict[str, int], words: list[str]) -> void {
    for w in words {
        counts[w] += 3
    }
}

fn dup_grid(grid: list[list[int]], n: int) -> void {
    var i = 0
    while i < n {
        grid[i % 3][i % 2] -= i
        i += 1
    }
}

class Tally {
    pub var n: int

    fn init(self) { self.n = 0 }
}

fn dup_field(t: Tally, k: int) -> void {
    var i = 0
    while i < k {
        t.n += i
        i += 1
    }
}

pub fn test_a_histogram() -> void {
    var hist = [0, 0, 0, 0]
    var call = 0
    while call < CALLS {
        dup_hist(hist, [1, 2, 3, 4, 5, 6, 7])
        call += 1
    }
    assert_eq(hist, [1 * CALLS, 2 * CALLS, 2 * CALLS, 2 * CALLS])
}

pub fn test_floats() -> void {
    var acc = [0.0, 0.0]
    var call = 0
    while call < CALLS {
        dup_floats(acc, [1.0, 2.0, 3.0])
        call += 1
    }
    assert_eq(acc, [2.0 * CALLS, 1.0 * CALLS])
}

pub fn test_a_dict() -> void {
    var counts = {"a": 0, "b": 0}
    var call = 0
    while call < CALLS {
        dup_dict(counts, ["a", "b", "a"])
        call += 1
    }
    assert_eq(counts["a"], 6 * CALLS)
    assert_eq(counts["b"], 3 * CALLS)
    #: An absent key still raises the KeyError a read owes.
    assert_throws(KeyError, fn() -> void {
        dup_dict(counts, ["zz"])
    }, "`+=` on a missing key from a compiled body")
}

pub fn test_a_grid() -> void {
    var grid = [[0, 0], [0, 0], [0, 0]]
    var call = 0
    while call < CALLS {
        dup_grid(grid, 6)
        call += 1
    }
    #: i = 0..5 lands on [i % 3][i % 2]: [0][0] 0, [1][1] 1, [2][0] 2,
    #: [0][1] 3, [1][0] 4, [2][1] 5.
    assert_eq(grid, [[0, -3 * CALLS], [-4 * CALLS, -1 * CALLS], [-2 * CALLS, -5 * CALLS]])
}

pub fn test_a_field() -> void {
    let t = Tally()
    var call = 0
    while call < CALLS {
        dup_field(t, 5)
        call += 1
    }
    assert_eq(t.n, 10 * CALLS)
}

pub fn test_an_overflow_still_raises() -> void {
    var hist = [9223372036854775807, 0, 0, 0]
    var call = 0
    while call < CALLS {
        dup_hist(hist, [1, 2])
        call += 1
    }
    assert_throws(OverflowError, fn() -> void { dup_hist(hist, [4]) }, "`+=` past the largest int")
    assert_eq(hist[0], 9223372036854775807)
}
