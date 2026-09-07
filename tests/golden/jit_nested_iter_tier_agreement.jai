#: A nested `for x in <list>` over every list STORAGE, taken inside JIT-warmed
#: bodies. The two tiers must agree exactly.
#:
#: Fourth of the tier-agreement set. The other three cover arithmetic and
#: indexing, who CATCHES a raise from compiled code, and the container surface;
#: this one covers emitForIterBind's NESTED arm -- the branch taken when the
#: loop head is not the OSR entry point, which builds its own ObjIter and steps
#: it inline.
#:
#: Why it needs its own file: that arm used to emit `emitListBoxedGuard`, so a
#: `push`-built `list[int]` (LIST_STORE_I64) deoptimised on every single loop
#: entry and the inner loop ran interpreted -- correct output, 3.6x slower, and
#: invisible to every existing test. It now dispatches on storage like
#: jit_list.c's other sites, which means a SECOND arm with no tag check and an
#: 8-byte (or 1-byte) stride. Nothing else in the suite reaches that arm.
#:
#: The edges chosen are the ones whose two arms could plausibly disagree: each
#: unboxed storage against the boxed literal holding the same values, an empty
#: inner list, a single-element one, a list of LISTS (which is boxed and must
#: keep the object-type guard), a list that is DE-SPECIALISED to boxed before
#: the loop runs again, and mutation under the loop, which must raise the same
#: way in both tiers.
#:
#: Every body is called 400 times before the edge arrives, so the edge is taken
#: in compiled code. Output must be byte-identical under JAITHON_NO_JIT=1; the
#: golden runner also re-runs it at JAITHON_JIT_THRESHOLD=1 and under
#: DEOPT_STRESS.

fn sum_ints(rows: list[list[int]]) -> int {
    var t = 0
    for r in rows {
        for v in r { t += v }
    }
    return t
}

fn sum_floats(rows: list[list[float]]) -> float {
    var t = 0.0
    for r in rows {
        for v in r { t += v }
    }
    return t
}

fn count_bools(rows: list[list[bool]]) -> int {
    var n = 0
    for r in rows {
        for v in r { if v { n += 1 } }
    }
    return n
}

fn concat_strs(rows: list[list[str]]) -> str {
    var out = ""
    for r in rows {
        for v in r { out = out + v }
    }
    return out
}

fn depth2(rows: list[list[list[int]]]) -> int {
    var t = 0
    for r in rows {
        for inner in r {
            for v in inner { t += v }
        }
    }
    return t
}

fn mutate_under(rows: list[list[int]]) -> int {
    var t = 0
    for r in rows {
        for v in r {
            t += v
            r.push(v)
        }
    }
    return t
}

fn main() {
    # push-built rows: LIST_STORE_I64, the storage that used to deoptimise.
    var pushed: list[list[int]] = []
    for i in 0..8 {
        var row: list[int] = []
        for j in 0..4 { row.push(i * 4 + j) }
        pushed.push(row)
    }

    # The same values as boxed literals, for the arm that always worked.
    var boxed: list[list[int]] = [[0, 1, 2, 3], [4, 5, 6, 7]]

    var floats: list[list[float]] = []
    for i in 0..8 {
        var row: list[float] = []
        for j in 0..4 { row.push(float(i) + float(j) * 0.5) }
        floats.push(row)
    }

    var bools: list[list[bool]] = []
    for i in 0..8 {
        var row: list[bool] = []
        for j in 0..4 { row.push((i + j) % 3 == 0) }
        bools.push(row)
    }

    var strs: list[list[str]] = []
    for i in 0..4 {
        var row: list[str] = []
        for j in 0..3 { row.push(str(i) + "-" + str(j) + ";") }
        strs.push(row)
    }

    var nested: list[list[list[int]]] = []
    for i in 0..4 {
        var mid: list[list[int]] = []
        for j in 0..3 {
            var leaf: list[int] = []
            for k in 0..3 { leaf.push(i + j + k) }
            mid.push(leaf)
        }
        nested.push(mid)
    }

    # Warm every body well past the tier's threshold.
    var warm = 0
    for _i in 0..400 {
        warm += sum_ints(pushed)
        warm += sum_ints(boxed)
        warm += count_bools(bools)
        warm += depth2(nested)
    }
    var warmf = 0.0
    for _i in 0..400 { warmf += sum_floats(floats) }
    var warms = ""
    for _i in 0..400 { warms = concat_strs(strs) }

    print("ints pushed " + str(sum_ints(pushed)))
    print("ints boxed " + str(sum_ints(boxed)))
    print("floats " + str(sum_floats(floats)))
    print("bools " + str(count_bools(bools)))
    print("strs " + warms)
    print("depth2 " + str(depth2(nested)))
    print("warm " + str(warm))
    print("warmf " + str(warmf))

    # Empty and single-element inner lists, after the loop is already compiled.
    var edges: list[list[int]] = [[], [7], [], [1, 2]]
    print("edges " + str(sum_ints(edges)))

    # A row de-specialised to boxed: an int list that gains a non-int.
    var mixed: list[list[any]] = []
    var mrow: list[any] = []
    mrow.push(1)
    mrow.push(2)
    mixed.push(mrow)
    var mt = 0
    for r in mixed {
        for v in r { mt += 1 }
    }
    print("mixed " + str(mt))

    # Mutation under the loop must raise, and must raise the same way in both
    # tiers: the version guard is what says so.
    var grow: list[list[int]] = []
    var growRow: list[int] = []
    growRow.push(1)
    growRow.push(2)
    grow.push(growRow)
    try {
        let _ = mutate_under(grow)
        print("mutate: no raise")
    } catch e {
        print("mutate raised " + str(e))
    }
}
