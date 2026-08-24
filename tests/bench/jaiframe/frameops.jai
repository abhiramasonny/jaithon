#: The frame operations a pandas program spends its time in.
#:
#: `jaiframe` had 281 tests and no benchmark, so nothing said whether any of
#: pandas' surface was actually faster. Every row here is one call on one frame,
#: chosen for how often a real script makes it and how much data it moves.
#:
#: pandas runs on the CPU and this runs on the GPU, so a row that loses is a
#: real loss. Sizes are large on purpose: a frame small enough to fit a cache
#: line measures the command-buffer round trip, which is 0.15 ms here and says
#: nothing about the operation.
import std.gpu as gpu
import jaiframe as jf
from jaiframe import Column, DataFrame
from jaiframe.groupby import group_by
from jaiframe.keys import drop_duplicates, nunique, sort_values, unique, value_counts
from jaiframe.access import filter_rows
from jaiframe.arith import add, greater, multiply
from jaiframe.join import merge
from jaiframe.reduce import mean, sum as frame_sum
from std.os import argv, env_or
from std.time import Instant

let TARGET_SECONDS = 0.08
let MIN_REPEATS = 12
let MAX_REPEATS = 512
let WARMUP = 6

#: One row a process, the two sides turn about; see tests/bench/run_suite.py.
var _only = ""
var _listing = false

fn wanted(name: str) -> bool {
    if _listing {
        print(name)
        return false
    }
    return _only == "" or _only == name
}

fn repeats_for(work: fn() -> void) -> int {
    for _warm in 0..3 { work() }
    gpu.synchronize()
    let started = Instant.now()
    for _probe in 0..3 { work() }
    gpu.synchronize()
    let each = started.elapsed().to_seconds() / 3.0
    if each <= 0.0 { return MAX_REPEATS }
    let wanted_count = int(TARGET_SECONDS / each)
    if wanted_count < MIN_REPEATS { return MIN_REPEATS }
    if wanted_count > MAX_REPEATS { return MAX_REPEATS }
    return wanted_count
}

fn run(name: str, rows_moved: int, work: fn() -> void) -> void {
    if not wanted(name) { return }
    let counted = repeats_for(work)
    for _warm in 0..WARMUP { work() }
    gpu.synchronize()
    let started = Instant.now()
    for _r in 0..counted { work() }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let scaled = seconds / float(counted) * float(MIN_REPEATS)
    print(f"jtb\t{name}\t{scaled:.9f}\t{rows_moved * MIN_REPEATS}\t0\tok\trows")
}

fn main() -> int {
    let given = argv()
    if given.len() > 1 {
        if given[1] == "list" { _listing = true } else { _only = given[1] }
    }
    if not _listing and not gpu.is_available() { throw RuntimeError("bench needs a GPU") }
    let level = env_or("BENCH_LEVEL", "hard")
    let scale = if level == "easy" { 8 } elif level == "medium" { 4 } else { 1 }
    let rows = 2000000 // scale
    let groups = 5000

    var keys: list[float] = []
    var values: list[float] = []
    var other: list[float] = []
    for index in 0..rows {
        keys.push(float(index * 7919 % groups))
        values.push(float(index % 1000) * 0.5)
        other.push(float(index % 97) - 48.0)
    }
    let frame = DataFrame(
        ["key", "value", "other"],
        [Column.from_floats(keys), Column.from_floats(values), Column.from_floats(other)]
    )

    #: A small right-hand frame, which is the shape a lookup join has.
    var right_keys: list[float] = []
    var right_values: list[float] = []
    for index in 0..groups {
        right_keys.push(float(index))
        right_values.push(float(index) * 1.5)
    }
    let lookup = DataFrame(
        ["key", "tag"],
        [Column.from_floats(right_keys), Column.from_floats(right_values)]
    )

    #: The handle is freed as well as the answer. A probe that drops it
    #: measures the allocator, which has caught this repository out before.
    run("groupby-sum", rows, fn() -> void {
        let handle = group_by(frame, by: "key")
        let out = handle.sum()
        out.free()
        handle.free()
    })
    run("groupby-mean", rows, fn() -> void {
        let handle = group_by(frame, by: "key")
        let out = handle.mean()
        out.free()
        handle.free()
    })
    run("sort-values", rows, fn() -> void {
        let out = sort_values(frame, by: "value")
        out.free()
    })
    run("filter-rows", rows, fn() -> void {
        let mask = greater(frame.column("other"), 0.0)
        let out = filter_rows(frame, mask)
        mask.free()
        out.free()
    })
    run("column-add", rows, fn() -> void {
        let out = add(frame.column("value"), frame.column("other"))
        out.free()
    })
    run("column-multiply", rows, fn() -> void {
        let out = multiply(frame.column("value"), 2.0)
        out.free()
    })
    run("frame-sum", rows, fn() -> void {
        let out = frame_sum(frame)
        out.free()
    })
    run("frame-mean", rows, fn() -> void {
        let out = mean(frame)
        out.free()
    })
    run("unique", rows, fn() -> void {
        let out = unique(frame.column("key"))
        out.free()
    })
    run("value-counts", rows, fn() -> void {
        let out = value_counts(frame.column("key"))
        out.free()
    })
    run("drop-duplicates", rows, fn() -> void {
        let out = drop_duplicates(frame, subset: ["key"])
        out.free()
    })
    run("merge-on-key", rows, fn() -> void {
        let out = merge(frame, lookup, on: "key")
        out.free()
    })
    frame.free()
    lookup.free()
    return 0
}
