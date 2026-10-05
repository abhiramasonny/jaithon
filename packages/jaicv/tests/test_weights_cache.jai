#: Shared weights stay alive while the call that fetched them still uses them.
#:
#: Filter taps, derivative rows and warp matrices are kept on the device and
#: shared between calls, in a cache that turns over when it fills. A filter
#: fetches its row taps and then its column taps; when the second fetch is the
#: one that fills the cache, turning it over must not free the first. When it
#: did, a run of blurs with changing sigmas failed at its 127th call with "this
#: Buffer has already been freed" -- or ran clean, depending on whether earlier
#: calls had left an odd or an even number of entries. Each case here runs at
#: both parities, so it does not depend on what the cache held when it began.
import std.gpu as gpu
from std.test import assert_eq, assert_true
import jaicv as cv
from jaicv import Mat, Size
from jaicv.imgproc.common import cached_weights

fn fetch_pairs(start: float) -> void {
    for pair in 0..300 {
        let value = start + float(pair) * 2.0
        let first = cached_weights(1, 1, [value])
        let second = cached_weights(1, 1, [value + 1.0])
        assert_true(not first.storage.is_freed())
        assert_eq(first.to_list()[0], value)
        assert_eq(second.to_list()[0], value + 1.0)
    }
}

fn test_a_fetched_matrix_outlives_the_next_fetch() -> void {
    if not gpu.is_available() { return }
    fetch_pairs(1000.0)
    #: One fetch alone moves which fetch of a pair fills the cache.
    let _ = cached_weights(1, 1, [-1.0])
    fetch_pairs(5000.0)
}

fn picture(rows: int, cols: int, kind: int) -> Mat {
    var values: list[float] = []
    for index in 0..rows * cols { values.push(float(index * 37 % 251)) }
    return Mat.from_list(rows, cols, kind, values)
}

fn blurs(image: Mat, start: float) -> void {
    for index in 0..300 {
        let sigma = start + float(index) * 0.01
        cv.gaussian_blur(image, Size(5, 7), sigma, sigma_y: sigma + 0.005).release()
    }
}

fn test_blurs_with_changing_sigmas_run_through_a_turnover() -> void {
    if not gpu.is_available() { return }
    let image = picture(40, 56, cv.CV_32FC1)
    blurs(image, 1.0)
    #: A blur whose two sigmas agree takes one entry, not two.
    cv.gaussian_blur(image, Size(0, 0), 0.75, sigma_y: 0.75).release()
    blurs(image, 5.0)
    image.release()
}

fn boxes(image: Mat, start: int) -> void {
    for index in 0..150 {
        cv.box_filter(image, -1, Size(3, start + 2 * index)).release()
    }
}

fn test_box_filters_with_changing_sizes_run_through_a_turnover() -> void {
    if not gpu.is_available() { return }
    let image = picture(48, 64, cv.CV_8UC1)
    boxes(image, 5)
    #: Unnormalised and square, the row and the column are the same taps.
    cv.box_filter(image, -1, Size(3, 3), normalize: false).release()
    boxes(image, 305)
    image.release()
}
