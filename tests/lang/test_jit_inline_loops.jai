#: A direct call to a small body that branches and loops is inlined where the
#: call is (JAITHON_JIT_INLINE_LOOPS): the callee's locals get homes in the
#: caller's frame, every `return` branches to one exit, and a guard anywhere
#: inside re-runs the whole call in the interpreter. Each case below is a
#: shape that inlining could get wrong -- an early return from inside the
#: loop, a guard that fails half-way through, a raise, a parameter the callee
#: reassigns, two inlines in one caller -- or one that must NOT inline at all
#: (a store into the caller's list, recursion), checked against the answer
#: the interpreter gives. Every call site sits in a small driver function the
#: test calls hundreds of times, because that is the tier that inlines them;
#: the drivers call out too, so their own locals share the registers with the
#: homes.
from std.test import assert_eq, assert_true

fn safe(cols: list[int], row: int, col: int) -> bool {
    var r = 0
    while r < row {
        let c = cols[r]
        if c == col { return false }
        if c - r == col - row { return false }
        if c + r == col + row { return false }
        r += 1
    }
    return true
}

fn place(cols: list[int], row: int, n: int) -> int {
    if row == n { return 1 }
    var found = 0
    var col = 0
    while col < n {
        if safe(cols, row, col) {
            cols[row] = col
            found += place(cols, row + 1, n)
        }
        col += 1
    }
    return found
}

fn count_small(xs: list[any], n: int, k: int) -> int {
    var i = 0
    var hits = 0
    while i < n {
        if xs[i] < k { hits += 1 }
        i += 1
    }
    return hits
}

fn count_below(xs: list[int], n: int, k: int) -> int {
    var i = 0
    var hits = 0
    while i < n {
        if xs[i] < k { hits += 1 }
        i += 1
    }
    return hits
}

fn first_at_least(xs: list[int], k: int) -> int {
    var i = 0
    while i < 1000000 {
        if xs[i] >= k { return i }
        i += 1
    }
    return -1
}

fn sum_to(xs: list[float], n: int) -> float {
    var i = 0
    var s = 0.0
    while i < n {
        s += xs[i]
        i += 1
    }
    return s
}

fn countdown(n: int) -> int {
    while n > 3 { n -= 3 }
    return n
}

fn nested(n: int) -> int {
    var i = 0
    var t = 0
    while i < n {
        var j = 0
        while j < i {
            t += j
            j += 1
        }
        i += 1
    }
    return t
}

fn fill(xs: list[int], n: int) -> int {
    var i = 0
    while i < n {
        xs[i] = i * 2
        i += 1
    }
    return i
}

fn depth(n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t += 1
        i += 1
    }
    if n > 0 { return t + depth(n - 1) }
    return t
}

fn noop(x: int) -> int { return x }

fn wrap_sum(a: int, b: int) -> int {
    var t = a
    for q in 0..b {
        t = t +% (q *% 3)
        if t % 7 == 0 { return t }
    }
    return t +% a
}

fn halving(a: int, b: int) -> float {
    var t = float(a)
    for q in 0..b {
        t = t * 0.5 + float(q)
        if t > 100.0 { return t }
    }
    return t
}

fn drive_ranges(i: int) -> float {
    return float(wrap_sum(i, 5) + wrap_sum(i + 1, 9) + noop(0)) + halving(i, i % 13)
}

fn drive_first(xs: list[int], k: int) -> int {
    return first_at_least(xs, k) + noop(1)
}

fn drive_counts(xs: list[int], fs: list[float], round: int) -> float {
    let hits = count_below(xs, 64, round % 64) + count_below(xs, 32, 16)
    let s = sum_to(fs, round % 64)
    return float(hits + noop(0)) + s
}

fn drive_small(xs: list[any], n: int) -> int {
    return count_small(xs, n, 20) + noop(0)
}

fn drive_raise(xs: list[int], n: int) -> int {
    var got = 0
    try {
        got = count_below(xs, n, 100)
    } catch e: IndexError {
        got = -1
    }
    return got + noop(0)
}

fn drive_missing(xs: list[int], k: int) -> int {
    var got = 0
    try {
        got = first_at_least(xs, k)
    } catch e: IndexError {
        got = -1
    }
    return got
}

fn drive_countdown(n: int) -> int {
    let r = countdown(n)
    return r * 1000 + n + noop(0)
}

fn drive_nested(n: int) -> int { return nested(n) + noop(0) }

fn drive_fill(xs: list[int], n: int) -> int { return fill(xs, n) + noop(0) }

fn test_queens_with_an_inlined_safe() -> void {
    for _round in 0..6 {
        var cols: list[int] = []
        for _i in 0..8 { cols.push(0) }
        assert_eq(place(cols, 0, 8), 92)
    }
}

fn test_an_early_return_from_inside_the_loop() -> void {
    var xs: list[int] = []
    for k in 0..50 { xs.push(k * 3) }
    for round in 0..400 {
        let k = round % 140
        assert_eq(drive_first(xs, k), (k + 2) // 3 + 1)
    }
}

fn test_two_counts_and_a_sum_in_one_caller() -> void {
    var xs: list[int] = []
    var fs: list[float] = []
    for k in 0..64 {
        xs.push((k * 37) % 64)
        fs.push(float(k) * 0.5)
    }
    for round in 0..300 {
        var want = 0
        for k in 0..64 { if xs[k] < round % 64 { want += 1 } }
        for k in 0..32 { if xs[k] < 16 { want += 1 } }
        var ws = 0.0
        for k in 0..(round % 64) { ws += fs[k] }
        assert_eq(drive_counts(xs, fs, round), float(want) + ws)
    }
}

fn test_a_guard_that_fails_half_way_reruns_the_call() -> void {
    var xs: list[any] = []
    for k in 0..40 { xs.push(k) }
    for _round in 0..300 { assert_eq(drive_small(xs, 40), 20) }
    # A float past the first thirty elements: the element guard fails inside
    # the inlined loop after thirty iterations have already run.
    xs[30] = 1.5
    for _round in 0..100 { assert_eq(drive_small(xs, 40), 21) }
    xs[30] = 30
    for _round in 0..100 { assert_eq(drive_small(xs, 40), 20) }
}

fn test_a_raise_inside_the_inlined_loop() -> void {
    var xs: list[int] = []
    for k in 0..20 { xs.push(k) }
    for round in 0..300 {
        let n = 19 + round % 3
        assert_eq(drive_raise(xs, n), if n > 20 { -1 } else { n })
    }
    for round in 0..200 {
        assert_eq(drive_missing(xs, if round % 2 == 0 { 1000 } else { 7 }),
                  if round % 2 == 0 { -1 } else { 7 })
    }
}

fn test_a_parameter_the_callee_reassigns() -> void {
    for n in 0..500 {
        var m = n
        while m > 3 { m -= 3 }
        assert_eq(drive_countdown(n), m * 1000 + n)
    }
}

fn test_nested_loops_in_the_callee() -> void {
    for round in 0..300 {
        let n = round % 20
        assert_eq(drive_nested(n), n * (n - 1) * (n - 2) // 6)
    }
}

fn test_a_callee_that_stores_into_the_callers_list() -> void {
    var xs: list[int] = []
    for _k in 0..16 { xs.push(-1) }
    for round in 0..300 {
        let n = round % 16
        assert_eq(drive_fill(xs, n), n)
        if n > 0 { assert_eq(xs[n - 1], (n - 1) * 2) }
    }
    assert_eq(xs[3], 6)
}

fn test_a_recursive_body_with_a_loop() -> void {
    for round in 0..200 {
        let n = round % 12
        assert_eq(depth(n), n * (n + 1) // 2)
    }
}

fn test_range_loops_with_returns_inside() -> void {
    for i in 0..300 {
        var want = 0
        for p in 0..2 {
            let a = i + p
            let b = if p == 0 { 5 } else { 9 }
            var t = a
            var done = false
            for q in 0..b {
                t = t +% (q *% 3)
                if t % 7 == 0 {
                    done = true
                    break
                }
            }
            want += if done { t } else { t +% a }
        }
        var f = float(i)
        for q in 0..(i % 13) {
            f = f * 0.5 + float(q)
            if f > 100.0 { break }
        }
        assert_eq(drive_ranges(i), float(want) + f)
    }
}
