#: PNG, both directions.
#:
#: Decoding covers every colour type and both bit depths the format defines,
#: including a palette and a transparency chunk, but not interlacing — Adam7 is
#: rare enough that refusing it is better than half-supporting it.
#:
#: The deflate stream itself is zlib's, through `__prim__.inflate`; everything
#: above it — chunk walk, filter reversal, bit unpacking — is here.
import std.math
from std.os import env_or
from ..core.mat import Mat
from ..core.types import CV_8U, CV_16U, CV_MAKETYPE

#: Read once. On, an eight-bit grey, grey-alpha, RGB or RGBA PNG goes through
#: `decode_fast`, which gives the same bytes as the reference decoder; off,
#: and for palette, sub-byte, sixteen-bit and interlaced files either way, it
#: goes through the reference.
let _FAST_PNG = env_or("JAICV_FAST_PNG", "1") != "0"

#: The eight bytes every PNG starts with.
pub let SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10]

#: Greyscale.
pub const COLOR_GRAY = 0
#: Red, green, blue.
pub const COLOR_RGB = 2
#: Indices into a palette.
pub const COLOR_PALETTE = 3
#: Greyscale with alpha.
pub const COLOR_GRAY_ALPHA = 4
#: Red, green, blue, alpha.
pub const COLOR_RGBA = 6

fn channels_for(colour_type: int) -> int {
    if colour_type == COLOR_GRAY { return 1 }
    if colour_type == COLOR_RGB { return 3 }
    if colour_type == COLOR_PALETTE { return 1 }
    if colour_type == COLOR_GRAY_ALPHA { return 2 }
    if colour_type == COLOR_RGBA { return 4 }
    throw ValueError(f"PNG colour type {colour_type} is not one the format defines")
}

fn be32(data: bytes, at: int) -> int {
    return data[at] << 24 | data[at + 1] << 16 | data[at + 2] << 8 | data[at + 3]
}

#: What a PNG header says about the image, before any pixels are read.
pub class Header {
    pub let width: int
    pub let height: int
    pub let bit_depth: int
    pub let colour_type: int
    pub let interlaced: bool

    pub fn init(self, width: int, height: int, bit_depth: int, colour_type: int, interlaced: bool) {
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.colour_type = colour_type
        self.interlaced = interlaced
    }
}

#: Read the header without decoding the pixels.
pub fn read_header(data: bytes) -> Header {
    if data.len() < 8 { throw ValueError("this is too short to be a PNG") }
    for index in 0..8 {
        if data[index] != SIGNATURE[index] { throw ValueError("this is not a PNG") }
    }
    var at = 8
    while at + 8 <= data.len() {
        let length = be32(data, at)
        let kind = data[at + 4] << 24 | data[at + 5] << 16 | data[at + 6] << 8 | data[at + 7]
        if kind == 0x49484452 {
            let body = at + 8
            return Header(
                be32(data, body),
                be32(data, body + 4),
                data[body + 8],
                data[body + 9],
                data[body + 12] != 0
            )
        }
        at += 12 + length
    }
    throw ValueError("this PNG has no header chunk")
}

#: Undo one scanline's filter, in place, given the row above it.
fn unfilter_row(row: list[int], previous: list[int], filter: int, stride: int) -> void {
    let width = row.len()
    if filter == 0 { return }
    if filter == 1 {
        for index in stride..width { row[index] = row[index] + row[index - stride] & 255 }
        return
    }
    if filter == 2 {
        for index in 0..width { row[index] = row[index] + previous[index] & 255 }
        return
    }
    if filter == 3 {
        for index in 0..width {
            let left = if index >= stride { row[index - stride] } else { 0 }
            row[index] = row[index] + (left + previous[index]) // 2 & 255
        }
        return
    }
    if filter == 4 {
        for index in 0..width {
            let left = if index >= stride { row[index - stride] } else { 0 }
            let up = previous[index]
            let corner = if index >= stride { previous[index - stride] } else { 0 }
            #: Paeth predicts from whichever of the three neighbours is closest
            #: to their linear combination.
            let estimate = left + up - corner
            let dl = abs(estimate - left)
            let du = abs(estimate - up)
            let dc = abs(estimate - corner)
            let predicted = if dl <= du and dl <= dc { left } elif du <= dc { up } else { corner }
            row[index] = row[index] + predicted & 255
        }
        return
    }
    throw ValueError(f"PNG filter type {filter} is not one the format defines")
}

fn gather_chunks(data: bytes) -> tuple[Header, list[int], list[int], list[int]] {
    let header = read_header(data)
    var compressed: list[int] = []
    var palette: list[int] = []
    var transparency: list[int] = []
    var at = 8
    while at + 8 <= data.len() {
        let length = be32(data, at)
        let kind = data[at + 4] << 24 | data[at + 5] << 16 | data[at + 6] << 8 | data[at + 7]
        let body = at + 8
        if kind == 0x49444154 {
            for index in 0..length { compressed.push(data[body + index]) }
        } elif kind == 0x504C5445 {
            for index in 0..length { palette.push(data[body + index]) }
        } elif kind == 0x74524E53 {
            for index in 0..length { transparency.push(data[body + index]) }
        } elif kind == 0x49454E44 {
            break
        }
        at += 12 + length
    }
    return (header, compressed, palette, transparency)
}

#: Decode a PNG into a `Mat`.
#:
#: The result is `CV_8U` or `CV_16U` to match the file, with channels in BGR or
#: BGRA order — the order the rest of jaicv works in, as OpenCV's `imread`
#: also produces.
pub fn decode(data: bytes) -> Mat { return decode_with(data, _FAST_PNG) }

#: The eight-bit, non-palette, non-interlaced PNG -- nearly every PNG a
#: dataset holds -- decoded straight into bytes, or null for anything else.
#:
#: The reference below copies each row into a list, builds a list of floats a
#: pixel at a time (with a fresh list per pixel for the channels), and hands
#: the floats to `Mat.from_list`. Here the rows are unfiltered in one int list
#: in place, the channels swapped to BGR in it, and the list packed into bytes
#: for `Mat.from_bytes`. Eight-bit samples pass through the reference's
#: `float(sample) * 257.0 / 257.0` unchanged, so the result is the same bytes.
#: Whatever the reference would refuse -- a short chunk, a missing IDAT, an
#: unknown filter -- comes back null, and the reference refuses it.
fn decode_fast(data: bytes) -> Mat? {
    let header = read_header(data)
    if header.interlaced or header.bit_depth != 8 { return null }
    if header.width <= 0 or header.height <= 0 { return null }
    let colour = header.colour_type
    if colour != COLOR_GRAY and
        colour != COLOR_RGB and
        colour != COLOR_GRAY_ALPHA and
        colour != COLOR_RGBA {
        return null
    }
    let channels = channels_for(colour)
    let width = header.width
    let height = header.height
    let row_bytes = width * channels

    #: `gather_chunks`' walk, keeping the IDAT bodies as slices.
    var compressed = bytes([])
    var any = false
    var short = false
    var at = 8
    var going = true
    while going and at + 8 <= data.len() {
        let length = be32(data, at)
        let kind = data[at + 4] << 24 | data[at + 5] << 16 | data[at + 6] << 8 | data[at + 7]
        let body = at + 8
        if kind == 0x49444154 {
            if body + length > data.len() {
                short = true
                going = false
            } else {
                compressed = if any {
                    __prim__.bytes_concat(compressed, data.slice(body, body + length))
                } else {
                    data.slice(body, body + length)
                }
                any = any or length > 0
            }
        } elif kind == 0x49454E44 {
            going = false
        } elif kind == 0x504C5445 or kind == 0x74524E53 {
            #: Read, and so range-checked, by the reference; not needed here.
            if body + length > data.len() {
                short = true
                going = false
            }
        }
        at += 12 + length
    }
    if short or not any { return null }
    #: A size that does not fit in an int is the reference's to refuse,
    #: and it does so only after it has looked for image data.
    if row_bytes + 1 > (1 << 62) // height { return null }
    let expected = (row_bytes + 1) * height
    let raw = __prim__.inflate(compressed, min(expected, compressed.len() * 1032 + 1024))
    if raw.len() < expected { return null }

    var out: list[int] = __prim__.list_filled(row_bytes * height, 0)
    if not unfilter_into(raw, out, height, row_bytes, channels) { return null }
    if channels >= 3 { swap_red_blue(out, width * height, channels) }
    return Mat.from_bytes(height, width, CV_MAKETYPE(CV_8U, channels), __prim__.bytes_new(out))
}

#: `unfilter_row` over the whole image, from `raw` (each row behind its
#: filter byte) into `out` (the rows end to end, the row above read back out
#: of `out`). False on a filter type the format does not define.
fn unfilter_into(raw: bytes, out: list[int], height: int, row_bytes: int, stride: int) -> bool {
    var ok = true
    var y = 0
    var source = 0
    var row = 0
    var above = 0
    var index = 0
    var filter = 0
    var left = 0
    var up = 0
    var corner = 0
    var estimate = 0
    var dl = 0
    var du = 0
    var dc = 0
    while ok and y < height {
        filter = raw[source]
        source += 1
        row = y * row_bytes
        above = row - row_bytes
        index = 0
        if filter == 0 {
            while index < row_bytes {
                out[row + index] = raw[source + index]
                index += 1
            }
        } elif filter == 1 {
            while index < row_bytes {
                left = if index >= stride { out[row + index - stride] } else { 0 }
                out[row + index] = raw[source + index] + left & 255
                index += 1
            }
        } elif filter == 2 {
            while index < row_bytes {
                up = if y > 0 { out[above + index] } else { 0 }
                out[row + index] = raw[source + index] + up & 255
                index += 1
            }
        } elif filter == 3 {
            while index < row_bytes {
                left = if index >= stride { out[row + index - stride] } else { 0 }
                up = if y > 0 { out[above + index] } else { 0 }
                out[row + index] = raw[source + index] + (left + up) // 2 & 255
                index += 1
            }
        } elif filter == 4 {
            while index < row_bytes {
                left = if index >= stride { out[row + index - stride] } else { 0 }
                up = if y > 0 { out[above + index] } else { 0 }
                corner = if index >= stride and y > 0 { out[above + index - stride] } else { 0 }
                estimate = left + up - corner
                dl = abs(estimate - left)
                du = abs(estimate - up)
                dc = abs(estimate - corner)
                corner = if dl <= du and dl <= dc { left } elif du <= dc { up } else { corner }
                out[row + index] = raw[source + index] + corner & 255
                index += 1
            }
        } else {
            ok = false
        }
        source += row_bytes
        y += 1
    }
    return ok
}

#: RGB(A) to BGR(A), in place: the order the reference emits.
fn swap_red_blue(out: list[int], pixels: int, channels: int) -> void {
    var at = 0
    var red = 0
    let end = pixels * channels
    while at < end {
        red = out[at]
        out[at] = out[at + 2]
        out[at + 2] = red
        at += channels
    }
}

#: `decode`, choosing the path: `fast` takes `decode_fast` for the files it
#: covers, and false always takes the reference. The two give the same bytes;
#: this is how a test holds them to it.
pub fn decode_with(data: bytes, fast: bool) -> Mat {
    if fast {
        let quick = decode_fast(data)
        if quick is not null { return quick }
    }
    let (header, compressed, palette, transparency) = gather_chunks(data)
    if header.interlaced { throw ValueError("interlaced PNG is not supported") }
    if compressed.len() == 0 { throw ValueError("this PNG carries no image data") }

    let source_channels = channels_for(header.colour_type)
    let depth = header.bit_depth
    if depth != 1 and depth != 2 and depth != 4 and depth != 8 and depth != 16 {
        throw ValueError(f"PNG bit depth {depth} is not one the format defines")
    }
    let bits_per_pixel = source_channels * depth
    let row_bytes = (header.width * bits_per_pixel + 7) // 8
    let stride = max(1, bits_per_pixel // 8)
    let expected = (row_bytes + 1) * header.height
    #: The size is only where inflate's buffer starts, and deflate cannot
    #: expand data more than 1032 times. A header promising more is short
    #: whatever the data says; asking for all of it up front aborted the
    #: process on a hostile header instead of refusing the file.
    let raw = __prim__.inflate(bytes(compressed), min(expected, compressed.len() * 1032 + 1024))
    if raw.len() < expected {
        throw ValueError("this PNG's image data is shorter than its header promises")
    }

    var previous: list[int] = []
    for _index in 0..row_bytes { previous.push(0) }
    #: Pixels are gathered as they come out of the filter, then handed to the
    #: device in one upload rather than a row at a time.
    var out_channels = source_channels
    if header.colour_type == COLOR_PALETTE {
        out_channels = if transparency.len() > 0 { 4 } else { 3 }
    }
    var values: list[float] = []
    let maximum = float((1 << depth) - 1)
    let scale_16 = if depth == 16 { 1.0 } else { 65535.0 / max(maximum, 1.0) }

    var offset = 0
    for _y in 0..header.height {
        let filter = raw[offset]
        var row: list[int] = []
        for index in 0..row_bytes { row.push(raw[offset + 1 + index]) }
        offset += row_bytes + 1
        unfilter_row(row, previous, filter, stride)
        emit_row(
            row,
            values,
            header,
            source_channels,
            out_channels,
            depth,
            palette,
            transparency,
            scale_16
        )
        previous = row
    }

    let type_code = CV_MAKETYPE(if depth == 16 { CV_16U } else { CV_8U }, out_channels)
    return Mat.from_list(header.height, header.width, type_code, values)
}

fn sample_at(row: list[int], index: int, depth: int) -> int {
    if depth == 8 { return row[index] }
    if depth == 16 { return row[index * 2] << 8 | row[index * 2 + 1] }
    let per_byte = 8 // depth
    let byte = row[index // per_byte]
    let shift = (per_byte - 1 - index % per_byte) * depth
    return byte >> shift & (1 << depth) - 1
}

fn emit_row(
    row: list[int],
    values: list[float],
    header: Header,
    source_channels: int,
    out_channels: int,
    depth: int,
    palette: list[int],
    transparency: list[int],
    scale_16: float
) -> void {
    let widen = depth == 16
    for x in 0..header.width {
        if header.colour_type == COLOR_PALETTE {
            let index = sample_at(row, x, depth)
            let base = index * 3
            let red = if base + 2 < palette.len() { palette[base] } else { 0 }
            let green = if base + 2 < palette.len() { palette[base + 1] } else { 0 }
            let blue = if base + 2 < palette.len() { palette[base + 2] } else { 0 }
            values.push(float(blue))
            values.push(float(green))
            values.push(float(red))
            if out_channels == 4 {
                let alpha = if index < transparency.len() { transparency[index] } else { 255 }
                values.push(float(alpha))
            }
            continue
        }
        var channel_values: list[float] = []
        for c in 0..source_channels {
            let sample = sample_at(row, x * source_channels + c, depth)
            channel_values.push(if widen {
                float(sample)
            } else {
                float(sample) * scale_16 / 257.0
            })
        }
        if source_channels == 1 {
            values.push(channel_values[0])
        } elif source_channels == 2 {
            values.push(channel_values[0])
            values.push(channel_values[1])
        } elif source_channels == 3 {
            values.push(channel_values[2])
            values.push(channel_values[1])
            values.push(channel_values[0])
        } else {
            values.push(channel_values[2])
            values.push(channel_values[1])
            values.push(channel_values[0])
            values.push(channel_values[3])
        }
    }
}

fn u32_bytes(value: int) -> list[int] {
    return [value >> 24 & 255, value >> 16 & 255, value >> 8 & 255, value & 255]
}

fn chunk(name: str, payload: list[int]) -> list[int] {
    var tagged: list[int] = []
    for character in name.chars() { tagged.push(ord(character)) }
    for byte in payload { tagged.push(byte) }
    var out = u32_bytes(payload.len())
    for byte in tagged { out.push(byte) }
    for byte in u32_bytes(__prim__.crc32(bytes(tagged))) { out.push(byte) }
    return out
}

#: Encode a `Mat` as a PNG.
#:
#: One, three, and four channels become greyscale, RGB, and RGBA. A `CV_16U`
#: image is written at sixteen bits; everything else is written at eight.
pub fn encode(image: Mat, level: int = 6) -> bytes {
    let channels = image.channels()
    let colour_type = if channels == 1 {
        COLOR_GRAY
    } elif channels == 3 {
        COLOR_RGB
    } elif channels == 4 {
        COLOR_RGBA
    } else {
        throw ValueError(f"PNG cannot hold {channels} channels")
    }
    let deep = image.depth() == CV_16U
    let bit_depth = if deep { 16 } else { 8 }
    let values = image.to_list()

    var raw: list[int] = []
    for y in 0..image.rows {
        #: Filter type zero throughout: the encoder trades a slightly larger
        #: file for not having to score five candidate filters per row.
        raw.push(0)
        for x in 0..image.cols {
            let base = (y * image.cols + x) * channels
            var ordered: list[float] = []
            if channels == 1 {
                ordered.push(values[base])
            } elif channels == 3 {
                ordered.push(values[base + 2])
                ordered.push(values[base + 1])
                ordered.push(values[base])
            } else {
                ordered.push(values[base + 2])
                ordered.push(values[base + 1])
                ordered.push(values[base])
                ordered.push(values[base + 3])
            }
            for value in ordered {
                var whole = int(math.floor(value + 0.5))
                let ceiling = if deep { 65535 } else { 255 }
                if whole < 0 { whole = 0 }
                if whole > ceiling { whole = ceiling }
                if deep {
                    raw.push(whole >> 8 & 255)
                    raw.push(whole & 255)
                } else {
                    raw.push(whole)
                }
            }
        }
    }

    var header: list[int] = []
    for byte in u32_bytes(image.cols) { header.push(byte) }
    for byte in u32_bytes(image.rows) { header.push(byte) }
    header.push(bit_depth)
    header.push(colour_type)
    header.push(0)
    header.push(0)
    header.push(0)

    var out = [byte for byte in SIGNATURE]
    for byte in chunk("IHDR", header) { out.push(byte) }
    for byte in chunk("IDAT", __prim__.deflate(bytes(raw), level).to_list()) { out.push(byte) }
    for byte in chunk("IEND", []) { out.push(byte) }
    return bytes(out)
}
