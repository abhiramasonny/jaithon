#: `jainum.stats`, which had no test file at all until 2026-08-24.
#:
#: That is how `digitize` shipped unable to answer anything. It called `diff`,
#: which did not exist, so it raised `undefined name`; and behind that its
#: monotonic test answered false for every input, so it rejected bins as
#: "neither rising nor falling" whatever they were. `make check` reported the
#: first and nothing read the output; nothing at all reported the second.
#:
#: Expected values here come from numpy, run on the same inputs.
import std.gpu as gpu
from std.test import assert_eq, assert_throws
from jainum.array import FLOAT32, NDArray
from jainum.stats import digitize, histogramdd
from jainum.reduce import sum as reduce_sum

fn have_gpu() -> bool { return gpu.is_available() }

fn vector(values: list[float]) -> NDArray {
    let a = NDArray([values.len()], FLOAT32)
    a.upload(values)
    return a
}

fn test_digitize_places_values_in_rising_bins() -> void {
    if not have_gpu() { return }
    let bins = vector([0.0, 1.0, 2.0, 3.0])
    defer { bins.free() }
    let x = vector([0.5, 1.5, 2.5])
    defer { x.free() }

    let placed = digitize(x, bins)
    defer { placed.free() }
    assert_eq(placed.to_list(), [1.0, 2.0, 3.0])

    let closed = digitize(x, bins, true)
    defer { closed.free() }
    assert_eq(closed.to_list(), [1.0, 2.0, 3.0])
}

#: Bins that fall are the branch that never ran, because the monotonic test
#: refused every input before reaching it.
fn test_digitize_places_values_in_falling_bins() -> void {
    if not have_gpu() { return }
    let bins = vector([3.0, 2.0, 1.0, 0.0])
    defer { bins.free() }
    let x = vector([0.5, 1.5, 2.5])
    defer { x.free() }

    let placed = digitize(x, bins)
    defer { placed.free() }
    assert_eq(placed.to_list(), [3.0, 2.0, 1.0])

    let closed = digitize(x, bins, true)
    defer { closed.free() }
    assert_eq(closed.to_list(), [3.0, 2.0, 1.0])
}

fn test_digitize_on_the_edges_and_on_bins_that_do_neither() -> void {
    if not have_gpu() { return }
    let bins = vector([0.0, 1.0, 2.0, 3.0])
    defer { bins.free() }

    #: A value on an edge belongs to the bin above it, and one past the last
    #: edge is one past the last bin.
    let edges = vector([0.0, 1.0, 3.0])
    defer { edges.free() }
    let placed = digitize(edges, bins)
    defer { placed.free() }
    assert_eq(placed.to_list(), [1.0, 2.0, 4.0])

    let wandering = vector([0.0, 2.0, 1.0])
    defer { wandering.free() }
    assert_throws(ValueError, || digitize(edges, wandering))
}

fn test_histogramdd_counts_every_point_once() -> void {
    if not have_gpu() { return }
    let points = NDArray([3, 2], FLOAT32)
    points.upload([0.5, 0.5, 1.5, 1.5, 2.5, 0.5])
    defer { points.free() }

    let (counts, edges) = histogramdd(points, 2)
    defer { counts.free() }
    for edge in edges { edge.free() }
    assert_eq(counts.shape, [2, 2])

    #: Every point lands in exactly one cell, so the counts sum to the count.
    let total = reduce_sum(counts)
    defer { total.free() }
    assert_eq(total.item(), 3.0)
}
