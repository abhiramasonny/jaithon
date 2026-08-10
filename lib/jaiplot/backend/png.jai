#: Writing an image out as a PNG.
#:
#: The format is four chunks — signature, IHDR, IDAT, IEND — each length-prefixed
#: and CRC-checked. Only the compression is native: `deflate` produces the zlib
#: stream IDAT carries, because writing a Huffman coder in interpreted Jaithon to
#: run over a megabyte of scanlines would not be worth the cycles. Everything
#: else, including the CRC and the filtering, is here.
#:
#: ```
#: from std.gui.canvas import Image
#: let bytes = encode(Image(4, 4))
#: bytes.len() > 8      # true
#: ```
from std.gui.canvas import Image

#: The eight bytes every PNG begins with.
#:
#: The high bit and the CRLF pair are a deliberate trap: they catch a transfer
#: that mangled line endings or stripped the eighth bit.
pub let SIGNATURE: list[int] = [137, 80, 78, 71, 13, 10, 26, 10]

#: Colour type 6 in the PNG spec: truecolour with an alpha channel.
pub const COLOR_TYPE_RGBA = 6

#: Bits per channel. Eight matches the packed `0xAARRGGBB` a canvas holds.
pub const BIT_DEPTH = 8

#: Encode `image` as a complete PNG file.
pub fn encode(image: Image, level: int = 6) -> bytes {
    let width = image.width()
    let height = image.height()
    let pixels = image.pixels()

    # Each scanline is prefixed with its filter type. Zero -- no filtering --
    # is chosen because deflate already finds the repetition in a chart's flat
    # areas, and a predictor would cost a pass over every pixel to save little.
    #
    # The packing is native: a 900x900 figure is three and a quarter million
    # bytes, and building that as a Jaithon list of boxed integers costs more
    # than compressing the result does.
    let raw = __prim__.canvas_scanlines(pixels, width, height)

    var out = [byte for byte in SIGNATURE]
    _append(out, _chunk("IHDR", _header(width, height)))
    _append(out, _chunk("IDAT", __prim__.deflate(raw, level).to_list()))
    _append(out, _chunk("IEND", []))
    return bytes(out)
}

#: The thirteen bytes of an IHDR chunk.
fn _header(width: int, height: int) -> list[int] {
    var out = []
    _append(out, _u32(width))
    _append(out, _u32(height))
    out.push(BIT_DEPTH)
    out.push(COLOR_TYPE_RGBA)
    out.push(0)  # deflate, the only compression method PNG defines
    out.push(0)  # adaptive filtering, the only filter method
    out.push(0)  # no interlacing
    return out
}

#: One chunk: length, type, payload, and the CRC over type and payload.
#:
#: The CRC is native. A bit-at-a-time loop in Jaithon runs eight iterations per
#: byte, which over a megabyte of image data costs more than the compression it
#: is checking.
fn _chunk(name: str, payload: list[int]) -> list[int] {
    var tagged = []
    for character in name.chars() { tagged.push(ord(character)) }
    _append(tagged, payload)

    var out = _u32(payload.len())
    _append(out, tagged)
    _append(out, _u32(__prim__.crc32(bytes(tagged))))
    return out
}

#: A 32-bit big-endian integer. PNG is big-endian throughout.
fn _u32(value: int) -> list[int] {
    return [value >> 24 & 0xFF, value >> 16 & 0xFF, value >> 8 & 0xFF, value & 0xFF]
}

fn _append(target: list[int], source: list[int]) -> void {
    for byte in source { target.push(byte) }
}
