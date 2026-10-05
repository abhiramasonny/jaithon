#: Connected components against a flood fill on the host.
#:
#: The labels are numbered on the device, by ranking where each component is
#: first met in OpenCV's scan order: raster order for four-connectivity, and
#: two-by-two blocks for eight. The recorded OpenCV cases cover a few frames;
#: these cover odd sides, where the block order has positions no pixel owns,
#: a single row and column, empty and full frames, and noise dense enough that
#: there are thousands of components.
import std.gpu as gpu
from std.test import assert_eq
import jaicv as cv
from jaicv import Mat
from jaicv.imgproc.segmentation import connected_components

fn noise(rows: int, cols: int, seed: int, density: int) -> list[float] {
    var values: list[float] = []
    var state = seed
    for _index in 0..rows * cols {
        state = (state * 1103515245 + 12345) % 2147483648
        values.push(if state // 65536 % 100 < density { 255.0 } else { 0.0 })
    }
    return values
}

fn scan_position(y: int, x: int, cols: int, connectivity: int) -> int {
    if connectivity == 4 { return y * cols + x }
    let blocks = (cols + 1) // 2
    return (y // 2 * blocks + x // 2) * 4 + y % 2 * 2 + x % 2
}

#: Label by flood fill, then number the components by the smallest scan
#: position each one covers.
fn reference(
    values: list[float],
    rows: int,
    cols: int,
    connectivity: int
) -> tuple[int, list[int]] {
    var parts = [0 for _index in 0..rows * cols]
    var firsts: list[int] = []
    var part = 0
    for start in 0..rows * cols {
        if values[start] == 0.0 or parts[start] != 0 { continue }
        part += 1
        var first = scan_position(start // cols, start % cols, cols, connectivity)
        var stack = [start]
        parts[start] = part
        while stack.len() > 0 {
            let cell = stack.pop()
            let y = cell // cols
            let x = cell % cols
            let here = scan_position(y, x, cols, connectivity)
            if here < first { first = here }
            for dy in -1..2 {
                for dx in -1..2 {
                    if dy == 0 and dx == 0 { continue }
                    if connectivity == 4 and dy != 0 and dx != 0 { continue }
                    let ny = y + dy
                    let nx = x + dx
                    if ny < 0 or nx < 0 or ny >= rows or nx >= cols { continue }
                    let next = ny * cols + nx
                    if values[next] == 0.0 or parts[next] != 0 { continue }
                    parts[next] = part
                    stack.push(next)
                }
            }
        }
        firsts.push(first)
    }
    var order = [index for index in 0..part]
    let ranked = order.sort(key: |index| firsts[index])
    var number = [0 for _index in 0..part + 1]
    for rank in 0..part { number[ranked[rank] + 1] = rank + 1 }
    return (part + 1, [number[value] for value in parts])
}

fn check(values: list[float], rows: int, cols: int) -> void {
    let image = Mat.from_list(rows, cols, cv.CV_8UC1, values)
    for connectivity in [4, 8] {
        let (count, labels) = connected_components(image, connectivity)
        let (want_count, want) = reference(values, rows, cols, connectivity)
        assert_eq(count, want_count)
        assert_eq([int(value) for value in labels.to_list()], want)
        labels.release()
    }
    image.release()
}

fn test_numbering_matches_a_flood_fill_on_odd_and_even_sides() -> void {
    if not gpu.is_available() { return }
    for (rows, cols) in [(1, 1), (1, 9), (9, 1), (2, 2), (5, 7), (8, 8), (13, 17), (31, 2), (40, 63)] {
        for density in [15, 50, 75] {
            check(noise(rows, cols, rows * 131 + cols * 7 + density, density), rows, cols)
        }
    }
}

fn test_numbering_handles_empty_and_full_frames() -> void {
    if not gpu.is_available() { return }
    check([0.0 for _index in 0..35], 5, 7)
    check([255.0 for _index in 0..35], 5, 7)
}

fn test_numbering_holds_with_thousands_of_components() -> void {
    if not gpu.is_available() { return }
    #: Enough positions that the ranking spans many groups of them.
    check(noise(97, 211, 7, 40), 97, 211)
}
