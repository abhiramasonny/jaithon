#: The fast JPEG and PNG decoders against the reference decoders they replace.
#:
#: Both fast paths promise the same bytes as the code they stand in front of,
#: not merely a close picture, so every case here compares the two outputs
#: exactly. The inputs are made here with jaicv's own encoders: photographs
#: and datasets are checked by hash outside the suite (packages/jaicv/README.md
#: says how), and images do not go in the repo.
import std.gpu as gpu
import jaicv.imgcodecs.jpeg as jpeg
import jaicv.imgcodecs.png as png

from std.test import assert_eq, assert_true
from jaicv.core.mat import Mat
from jaicv.core.types import CV_8UC1, CV_8UC3, CV_8UC4, CV_16UC3

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
    #: A stretch cut out of the middle with the end-of-image marker kept: the
    #: scan runs into the marker, which the reference reads as data and the
    #: fast path must not guess at.
    for gap in [40, 200] {
        let start = whole.len() // 2
        var holed = whole.slice(0, start)
        for index in start + gap..whole.len() { holed.push(whole[index]) }
        same_outcome(bytes(holed))
    }
}

#: A scan header whose length field runs past the end of the file. The
#: reference reads the missing data as zeros and still gives a picture; the
#: fast path has nothing to read and must hand the stream back, not fail.
fn test_a_scan_header_running_past_the_end_goes_the_way_the_reference_goes() -> void {
    if not have_gpu() { return }
    for type_code in [CV_8UC1, CV_8UC3] {
        let image = texture(15, 17, type_code, 7)
        let whole = jpeg.encode(image, 90).to_list()
        image.release()
        var sos = -1
        for at in 0..whole.len() - 1 {
            if whole[at] == 0xFF and whole[at + 1] == 0xDA { sos = at }
        }
        assert_true(sos > 0)
        let length_at = sos + 2
        #: Where the header says the scan starts, against the end of the data:
        #: inside it, on it, within the eight bytes the fast path pads with,
        #: just beyond those, and as far as the field can reach.
        var lengths: list[int] = [0xFFFF]
        for past in [-1, 0, 1, 8, 9, 71] { lengths.push(whole.len() - length_at + past) }
        for length in lengths {
            var patched = whole.slice(0, whole.len())
            patched[length_at] = length >> 8 & 255
            patched[length_at + 1] = length & 255
            same_outcome(bytes(patched))
        }
    }
}

fn be(value: int) -> list[int] {
    return [value >> 24 & 255, value >> 16 & 255, value >> 8 & 255, value & 255]
}

fn png_chunk(out: list[int], name: str, payload: list[int]) -> void {
    var tagged: list[int] = []
    for character in name.chars() { tagged.push(ord(character)) }
    for byte in payload { tagged.push(byte) }
    for byte in be(payload.len()) { out.push(byte) }
    for byte in tagged { out.push(byte) }
    for byte in be(__prim__.crc32(bytes(tagged))) { out.push(byte) }
}

#: A PNG put together by hand, so every filter type turns up: row y uses
#: filter `filters[y % filters.len()]` over arbitrary bytes, which any filter
#: turns into some picture. The data goes in `pieces` IDAT chunks.
fn handmade_png(
    width: int,
    height: int,
    colour: int,
    filters: list[int],
    pieces: int,
    seed: int
) -> bytes {
    let channels = if colour == 0 { 1 } elif colour == 2 { 3 } elif colour == 4 { 2 } else { 4 }
    var raw: list[int] = []
    var state = seed
    for y in 0..height {
        raw.push(filters[y % filters.len()])
        for _index in 0..width * channels {
            state = (state * 1103515245 + 12345) % 2147483648
            raw.push(state >> 16 & 255)
        }
    }
    let packed = __prim__.deflate(bytes(raw), 6).to_list()
    var out: list[int] = [137, 80, 78, 71, 13, 10, 26, 10]
    var header = be(width)
    for byte in be(height) { header.push(byte) }
    for byte in [8, colour, 0, 0, 0] { header.push(byte) }
    png_chunk(out, "IHDR", header)
    let step = (packed.len() + pieces - 1) // pieces
    var at = 0
    while at < packed.len() {
        let end = min(at + step, packed.len())
        png_chunk(out, "IDAT", packed.slice(at, end))
        at = end
    }
    png_chunk(out, "IEND", [])
    return bytes(out)
}

fn same_png(data: bytes) -> void {
    var reference_error = ""
    var fast_error = ""
    var reference: Mat? = null
    var fast: Mat? = null
    try {
        reference = png.decode_with(data, false)
    } catch error {
        reference_error = f"{error}"
    }
    try {
        fast = png.decode_with(data, true)
    } catch error {
        fast_error = f"{error}"
    }
    assert_eq(fast_error, reference_error)
    if reference is not null and fast is not null {
        assert_eq(fast.rows, reference.rows)
        assert_eq(fast.cols, reference.cols)
        assert_eq(fast.channels(), reference.channels())
        assert_eq(fast.depth(), reference.depth())
        assert_true(fast.to_bytes() == reference.to_bytes())
    }
}

fn test_the_fast_png_path_matches_the_reference_for_every_filter() -> void {
    if not have_gpu() { return }
    for colour in [0, 2, 4, 6] {
        for size in [[1, 1], [3, 5], [32, 32], [17, 9]] {
            for filters in [[0], [1], [2], [3], [4], [4, 3, 2, 1, 0]] {
                for pieces in [1, 3] {
                    same_png(
                        handmade_png(size[0], size[1], colour, filters, pieces, size[0] + colour)
                    )
                }
            }
        }
    }
}

fn test_png_files_the_fast_path_does_not_cover_still_decode() -> void {
    if not have_gpu() { return }
    #: Sixteen bits goes to the reference whichever path is asked for.
    let deep = texture(6, 7, CV_8UC3, 5).convert_to(CV_16UC3, 257.0, 0.0)
    same_png(png.encode(deep))
    #: An undefined filter type, a header promising more rows than the data
    #: holds, and a chunk running off the end: refused the same way.
    same_png(handmade_png(4, 4, 2, [0, 5], 1, 9))
    let whole = handmade_png(8, 8, 6, [4], 2, 11).to_list()
    var short: list[int] = []
    for index in 0..whole.len() - 30 { short.push(whole[index]) }
    same_png(bytes(short))
    var taller = whole.slice(0, whole.len())
    taller[23] = 40
    same_png(bytes(taller))
}

#: A header whose size cannot be multiplied out, with and without image data
#: behind it: the fast path must not be the one to find that out, or it fails
#: with a different error from the reference's.
fn test_a_png_too_large_to_size_fails_the_way_the_reference_fails() -> void {
    if not have_gpu() { return }
    let data = __prim__.deflate(bytes([0, 1, 2, 3]), 6).to_list()
    for size in [0xFFFFFFFF, 0x80000000] {
        for colour in [0, 2, 6] {
            for with_data in [false, true] {
                var out: list[int] = [137, 80, 78, 71, 13, 10, 26, 10]
                var header = be(size)
                for byte in be(size) { header.push(byte) }
                for byte in [8, colour, 0, 0, 0] { header.push(byte) }
                png_chunk(out, "IHDR", header)
                if with_data { png_chunk(out, "IDAT", data) }
                png_chunk(out, "IEND", [])
                same_png(bytes(out))
            }
        }
    }
}
