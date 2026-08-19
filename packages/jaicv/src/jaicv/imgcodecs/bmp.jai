#: Windows bitmaps, both directions.
#:
#: Reads 8-bit palette, 24-bit and 32-bit uncompressed images, top-down or
#: bottom-up. Writes 24-bit or 32-bit, bottom-up, which is what every reader
#: accepts.
import std.math
from ..core.mat import Mat
from ..core.types import CV_8U, CV_MAKETYPE

fn le16(data: bytes, at: int) -> int { return data[at] | data[at + 1] << 8 }

fn le32(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8 | data[at + 2] << 16 | data[at + 3] << 24
}

fn le32_signed(data: bytes, at: int) -> int {
    let value = le32(data, at)
    if value >= 0x80000000 { return value - 0x100000000 }
    return value
}

#: Whether `data` starts with a bitmap header.
pub fn looks_like(data: bytes) -> bool {
    return data.len() >= 2 and data[0] == 0x42 and data[1] == 0x4D
}

#: Decode a bitmap into a `Mat` with channels in BGR or BGRA order.
pub fn decode(data: bytes) -> Mat {
    if not looks_like(data) { throw ValueError("this is not a bitmap") }
    if data.len() < 54 { throw ValueError("this bitmap is truncated") }
    let pixel_offset = le32(data, 10)
    let header_size = le32(data, 14)
    let width = le32_signed(data, 18)
    let signed_height = le32_signed(data, 22)
    let bits = le16(data, 28)
    let compression = le32(data, 30)
    if compression != 0 { throw ValueError("only uncompressed bitmaps are supported") }
    if width <= 0 { throw ValueError("this bitmap has no width") }
    let height = abs(signed_height)
    #: A negative height means the rows are stored top-down; the usual layout
    #: is bottom-up, which is why the row walk below can run either way.
    let top_down = signed_height < 0

    var palette: list[int] = []
    if bits <= 8 {
        var used = le32(data, 46)
        if used == 0 { used = 1 << bits }
        let base = 14 + header_size
        for index in 0..used * 4 {
            if base + index < pixel_offset { palette.push(data[base + index]) }
        }
    }

    #: A palette whose every entry is grey means the file is a greyscale
    #: image, which is how an 8-bit BMP stores one and how it reads back.
    var greyscale_palette = bits <= 8 and palette.len() >= 4
    if greyscale_palette {
        var index = 0
        while index + 2 < palette.len() {
            if palette[index] != palette[index + 1] or palette[index] != palette[index + 2] {
                greyscale_palette = false
                break
            }
            index += 4
        }
    }
    let out_channels = if bits == 32 { 4 } elif greyscale_palette { 1 } else { 3 }
    let row_bytes = (width * bits + 31) // 32 * 4
    var values: list[float] = []
    for y in 0..height {
        let source_row = if top_down { y } else { height - 1 - y }
        let base = pixel_offset + source_row * row_bytes
        for x in 0..width {
            if bits == 24 or bits == 32 {
                let at = base + x * (bits // 8)
                values.push(float(data[at]))
                values.push(float(data[at + 1]))
                values.push(float(data[at + 2]))
                if out_channels == 4 { values.push(float(data[at + 3])) }
                continue
            }
            let per_byte = 8 // bits
            let byte = data[base + x // per_byte]
            let shift = (per_byte - 1 - x % per_byte) * bits
            let index = byte >> shift & (1 << bits) - 1
            let slot = index * 4
            let blue = if slot + 2 < palette.len() { palette[slot] } else { 0 }
            if out_channels == 1 {
                values.push(float(blue))
                continue
            }
            values.push(float(blue))
            values.push(float(if slot + 2 < palette.len() { palette[slot + 1] } else { 0 }))
            values.push(float(if slot + 2 < palette.len() { palette[slot + 2] } else { 0 }))
        }
    }
    return Mat.from_list(height, width, CV_MAKETYPE(CV_8U, out_channels), values)
}

fn push_le16(out: list[int], value: int) -> void {
    out.push(value & 255)
    out.push(value >> 8 & 255)
}

fn push_le32(out: list[int], value: int) -> void {
    out.push(value & 255)
    out.push(value >> 8 & 255)
    out.push(value >> 16 & 255)
    out.push(value >> 24 & 255)
}

#: Encode a `Mat` as an uncompressed bitmap.
pub fn encode(image: Mat) -> bytes {
    let channels = image.channels()
    if channels != 1 and channels != 3 and channels != 4 {
        throw ValueError(f"a bitmap cannot hold {channels} channels")
    }
    #: A single-channel image is written as an 8-bit palette bitmap with a
    #: grey ramp, which is what every other writer produces and what makes the
    #: file read back as one channel rather than three.
    let bits = if channels == 4 { 32 } elif channels == 1 { 8 } else { 24 }
    let palette_entries = if bits == 8 { 256 } else { 0 }
    let row_bytes = (image.cols * bits + 31) // 32 * 4
    let pixel_bytes = row_bytes * image.rows
    let pixel_offset = 54 + palette_entries * 4
    let values = image.to_list()

    var out: list[int] = []
    out.push(0x42)
    out.push(0x4D)
    push_le32(out, pixel_offset + pixel_bytes)
    push_le32(out, 0)
    push_le32(out, pixel_offset)
    push_le32(out, 40)
    push_le32(out, image.cols)
    push_le32(out, image.rows)
    push_le16(out, 1)
    push_le16(out, bits)
    push_le32(out, 0)
    push_le32(out, pixel_bytes)
    push_le32(out, 2835)
    push_le32(out, 2835)
    push_le32(out, palette_entries)
    push_le32(out, 0)
    for index in 0..palette_entries {
        out.push(index)
        out.push(index)
        out.push(index)
        out.push(0)
    }

    for y in 0..image.rows {
        let source_row = image.rows - 1 - y
        var written = 0
        for x in 0..image.cols {
            let base = (source_row * image.cols + x) * channels
            let blue = clamp_byte(values[base])
            if bits == 8 {
                out.push(blue)
                written += 1
                continue
            }
            out.push(blue)
            out.push(clamp_byte(values[base + 1]))
            out.push(clamp_byte(values[base + 2]))
            written += 3
            if bits == 32 {
                out.push(clamp_byte(values[base + 3]))
                written += 1
            }
        }
        while written < row_bytes {
            out.push(0)
            written += 1
        }
    }
    return bytes(out)
}

fn clamp_byte(value: float) -> int {
    let whole = int(math.floor(value + 0.5))
    if whole < 0 { return 0 }
    if whole > 255 { return 255 }
    return whole
}
