#: The fast JPEG and PNG decoders against the reference decoders they replace.
#:
#: Both fast paths promise the same bytes as the code they stand in front of,
#: not merely a close picture, so every case here compares the two outputs
#: exactly. The inputs are made here with jaicv's own encoders: photographs
#: and datasets are checked by hash outside the suite (packages/jaicv/README.md
#: says how), and images do not go in the repo.
import std.gpu as gpu
import jaicv.imgcodecs.jpeg as jpeg

from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_8UC3, CV_8UC4

fn have_gpu() -> bool { return gpu.is_available() }

#: Deterministic texture: a gradient for the low frequencies, a hash for the
#: high ones, so the JPEG has long codes and large coefficients as well as
#: short ones.
fn texture(rows: int, cols: int, type_code: int, seed: int) -> Mat {
    let channels = if type_code == CV_8UC1 { 1 } elif type_code == CV_8UC4 { 4 } else { 3 }
    var values: list[float] = []
    var state = seed * 2654435761 % 4294967296
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..channels {
                state = (state * 1103515245 + 12345) % 2147483648
                let noise = state >> 16 & 63
                let smooth = (x * 7 + y * 5 + c * 40) % 192
                values.push(float(smooth + noise))
            }
        }
    }
    return Mat.from_list(rows, cols, type_code, values)
}

fn same_jpeg(data: bytes) -> void {
    let fast = jpeg.decode_with(data, true)
    let reference = jpeg.decode_with(data, false)
    assert_eq(fast.rows, reference.rows)
    assert_eq(fast.cols, reference.cols)
    assert_eq(fast.channels(), reference.channels())
    assert_true(fast.to_bytes() == reference.to_bytes())
    fast.release()
    reference.release()
}

#: Whatever the reference does with `data` -- a picture or an error -- the
#: fast path does too.
fn same_outcome(data: bytes) -> void {
    var reference_error = ""
    var fast_error = ""
    var reference: Mat? = null
    var fast: Mat? = null
    try {
        reference = jpeg.decode_with(data, false)
    } catch error {
        reference_error = f"{error}"
    }
    try {
        fast = jpeg.decode_with(data, true)
    } catch error {
        fast_error = f"{error}"
    }
    assert_eq(fast_error, reference_error)
    if reference is not null and fast is not null {
        assert_true(fast.to_bytes() == reference.to_bytes())
    }
}

fn test_the_fast_jpeg_scan_matches_the_reference_byte_for_byte() -> void {
    if not have_gpu() { return }
    for size in [[1, 1], [5, 7], [16, 16], [17, 33], [48, 64]] {
        for type_code in [CV_8UC1, CV_8UC3] {
            let image = texture(size[0], size[1], type_code, size[0] * 31 + size[1])
            for quality in [100, 75, 15] { same_jpeg(jpeg.encode(image, quality)) }
            image.release()
        }
    }
}

fn test_a_damaged_jpeg_goes_the_way_the_reference_goes() -> void {
    if not have_gpu() { return }
    let image = texture(24, 40, CV_8UC3, 3)
    let data = jpeg.encode(image, 90)
    image.release()
    let whole = data.to_list()
    #: Cut short: the fast scan reads past the end of the data it has and
    #: hands the stream back rather than make up the bits.
    for keep in [whole.len() * 9 // 10, whole.len() * 3 // 4] {
        var cut: list[int] = []
        for index in 0..keep { cut.push(whole[index]) }
        same_outcome(bytes(cut))
    }
    #: Bit flips in the entropy-coded data.
    for at in [whole.len() // 2, whole.len() * 2 // 3, whole.len() - 20] {
        var flipped = whole.slice(0, whole.len())
        flipped[at] = flipped[at] ^ 0x5A
        same_outcome(bytes(flipped))
    }
}
