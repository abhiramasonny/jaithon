#: `xs[a:b]` on a list from compiled code, which goes through
#: jaiListSliceLeaf -- a leaf call that allocates the slice with no descriptor
#: and no roots, and declines when a collection is due -- and falls back to the
#: descriptor call to jitGetSlice for that and for any slice with a step.
#:
#: Every storage width is cut (ints and floats at eight bytes, bools at one,
#: boxed values at sixteen), short and long, so both copy paths in the leaf
#: run. Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; every answer is the
#: interpreter's, which `JAITHON_NO_JIT=1` checks on the same file.
# jit-compiles: cut_ints, cut_floats, cut_bools, cut_strs, cut_head, cut_tail, halves
from std.test import assert_eq, assert_true

let CALLS = 100

fn cut_ints(xs: list[int], a: int, b: int) -> list[int] {
    return xs[a:b]
}

fn cut_floats(xs: list[float], a: int, b: int) -> list[float] {
    return xs[a:b]
}

fn cut_bools(xs: list[bool], a: int, b: int) -> list[bool] {
    return xs[a:b]
}

fn cut_strs(xs: list[str], a: int, b: int) -> list[str] {
    return xs[a:b]
}

fn cut_head(xs: list[int], b: int) -> list[int] {
    return xs[:b]
}

fn cut_tail(xs: list[int], a: int) -> list[int] {
    return xs[a:]
}

#: The divide-and-conquer shape: many small slices, each kept only briefly,
#: so collections land between them and the leaf has to decline some.
fn halves(xs: list[int]) -> int {
    if xs.len() <= 1 {
        if xs.len() == 1 { return xs[0] }
        return 0
    }
    let mid = xs.len() // 2
    return halves(xs[0:mid]) + halves(xs[mid:xs.len()])
}

fn ints(n: int) -> list[int] {
    var out: list[int] = []
    for i in 0..n { out.push(i * 3) }
    return out
}

pub fn test_int_bounds_clamp_like_the_interpreter() -> void {
    let xs = ints(10)
    var call = 0
    while call < CALLS {
        assert_eq(cut_ints(xs, 2, 5), [6, 9, 12])
        assert_eq(cut_ints(xs, -3, -1), [21, 24])
        assert_eq(cut_ints(xs, -100, 2), [0, 3])
        assert_eq(cut_ints(xs, 8, 100), [24, 27])
        assert_eq(cut_ints(xs, 5, 5), [])
        assert_eq(cut_ints(xs, 7, 3), [])
        assert_eq(cut_ints(xs, 100, 200), [])
        assert_eq(cut_ints(xs, 0, 10), xs)
        assert_eq(cut_ints([], 0, 3), [])
        call += 1
    }
}

pub fn test_missing_bounds() -> void {
    let xs = ints(10)
    var call = 0
    while call < CALLS {
        assert_eq(cut_head(xs, 3), [0, 3, 6])
        assert_eq(cut_head(xs, -8), [0, 3])
        assert_eq(cut_tail(xs, 7), [21, 24, 27])
        assert_eq(cut_tail(xs, -2), [24, 27])
        assert_eq(cut_tail(xs, 20), [])
        call += 1
    }
}

pub fn test_a_slice_is_a_copy_that_keeps_its_type() -> void {
    let xs = ints(40)
    var call = 0
    while call < CALLS {
        var s = cut_ints(xs, 10, 30)
        assert_eq(s.len(), 20)
        s[0] = -1
        s.push(99)
        assert_eq(xs[10], 30)
        assert_eq(xs.len(), 40)
        assert_eq(s[20], 99)
        assert_eq(s[19], 87)
        call += 1
    }
}

pub fn test_every_storage_width() -> void {
    var fs: list[float] = []
    var bs: list[bool] = []
    var ss: list[str] = []
    for i in 0..37 {
        fs.push(i * 0.5)
        bs.push(i % 3 == 0)
        ss.push(f"s{i}")
    }
    var call = 0
    while call < CALLS {
        assert_eq(cut_floats(fs, 1, 3), [0.5, 1.0])
        assert_eq(cut_floats(fs, 30, 37).len(), 7)
        assert_eq(cut_floats(fs, 30, 37)[6], 18.0)
        assert_eq(cut_bools(bs, 0, 4), [true, false, false, true])
        #: Thirty-three one-byte elements: not a whole number of words.
        let bl = cut_bools(bs, 2, 35)
        assert_eq(bl.len(), 33)
        assert_eq(bl[1], true)
        assert_eq(bl[32], false)
        assert_eq(bl[31], true)
        assert_eq(cut_strs(ss, 3, 5), ["s3", "s4"])
        assert_eq(cut_strs(ss, -2, 100), ["s35", "s36"])
        assert_eq(cut_strs(ss, 0, 37).len(), 37)
        call += 1
    }
}

pub fn test_many_short_lived_slices() -> void {
    let xs = ints(5000)
    var want = 0
    for x in xs { want += x }
    var call = 0
    while call < 20 {
        assert_eq(halves(xs), want)
        call += 1
    }
}

pub fn test_a_step_still_slices() -> void {
    let xs = ints(10)
    var call = 0
    while call < CALLS {
        assert_eq(xs[1:8:3], [3, 12, 21])
        assert_eq(xs[::-4], [27, 15, 3])
        call += 1
    }
    assert_true(true, "reached")
}
