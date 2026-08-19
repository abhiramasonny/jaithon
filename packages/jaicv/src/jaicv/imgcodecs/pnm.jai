#: Netpbm: PBM, PGM, and PPM, in both the ASCII and binary flavours.
#:
#: The format is the simplest way to get pixels in or out of a program that has
#: no codec library, which is why OpenCV keeps it and why it is here.
import std.math
from ..core.mat import Mat
from ..core.types import CV_8U, CV_16U, CV_MAKETYPE

#: Whether `data` starts with a Netpbm magic number.
pub fn looks_like(data: bytes) -> bool {
    if data.len() < 2 { return false }
    if data[0] != 0x50 { return false }
    return data[1] >= 0x31 and data[1] <= 0x36
}

class Reader {
    pub let data: bytes
    pub var at: int

    fn init(self, data: bytes) {
        self.data = data
        self.at = 0
    }

    #: Next whitespace-separated token, skipping `#` comments to end of line.
    pub fn token(self) -> str {
        while self.at < self.data.len() {
            let byte = self.data[self.at]
            if byte == 0x23 {
                while self.at < self.data.len() and self.data[self.at] != 0x0A { self.at += 1 }
                continue
            }
            if byte == 0x20 or byte == 0x09 or byte == 0x0A or byte == 0x0D {
                self.at += 1
                continue
            }
            break
        }
        var out = ""
        while self.at < self.data.len() {
            let byte = self.data[self.at]
            if byte == 0x20 or byte == 0x09 or byte == 0x0A or byte == 0x0D { break }
            out = out + chr(byte)
            self.at += 1
        }
        if out.len() == 0 { throw ValueError("this Netpbm file ended early") }
        return out
    }

    pub fn number(self) -> int { return int(self.token()) }
}

#: Decode a Netpbm image.
pub fn decode(data: bytes) -> Mat {
    if not looks_like(data) { throw ValueError("this is not a Netpbm image") }
    let reader = Reader(data)
    let magic = reader.token()
    let kind = int(magic[1:2])
    let width = reader.number()
    let height = reader.number()
    var maximum = 1
    if kind != 1 and kind != 4 { maximum = reader.number() }
    if width <= 0 or height <= 0 { throw ValueError("this Netpbm image is empty") }

    let channels = if kind == 3 or kind == 6 { 3 } else { 1 }
    let deep = maximum > 255
    var values: list[float] = []

    if kind <= 3 {
        for _index in 0..width * height * channels {
            let sample = reader.number()
            #: A PBM stores one for black, which is the opposite of every other
            #: format, so it is inverted on the way in.
            values.push(if kind == 1 { float(255 - sample * 255) } else { float(sample) })
        }
        return finish(values, height, width, channels, deep)
    }

    #: Binary data starts one whitespace byte after the header.
    var at = reader.at
    if at < data.len() { at += 1 }
    if kind == 4 {
        let row_bytes = (width + 7) // 8
        for y in 0..height {
            for x in 0..width {
                let byte = data[at + y * row_bytes + x // 8]
                let bit = byte >> 7 - x % 8 & 1
                values.push(float(255 - bit * 255))
            }
        }
        return finish(values, height, width, 1, false)
    }

    let per_sample = if deep { 2 } else { 1 }
    let total = width * height * channels
    for index in 0..total {
        let base = at + index * per_sample
        let sample = if deep { data[base] << 8 | data[base + 1] } else { data[base] }
        values.push(float(sample))
    }
    return finish(values, height, width, channels, deep)
}

fn finish(values: list[float], height: int, width: int, channels: int, deep: bool) -> Mat {
    #: Netpbm stores colour as RGB; the rest of jaicv works in BGR.
    if channels == 3 {
        var swapped: list[float] = []
        var index = 0
        while index < values.len() {
            swapped.push(values[index + 2])
            swapped.push(values[index + 1])
            swapped.push(values[index])
            index += 3
        }
        values = swapped
    }
    let depth = if deep { CV_16U } else { CV_8U }
    return Mat.from_list(height, width, CV_MAKETYPE(depth, channels), values)
}

#: Encode as binary PGM (one channel) or PPM (three).
pub fn encode(image: Mat) -> bytes {
    let channels = image.channels()
    if channels != 1 and channels != 3 {
        throw ValueError(f"Netpbm cannot hold {channels} channels")
    }
    let deep = image.depth() == CV_16U
    let maximum = if deep { 65535 } else { 255 }
    let magic = if channels == 1 { "P5" } else { "P6" }
    var out: list[int] = []
    for character in f"{magic}\n{image.cols} {image.rows}\n{maximum}\n".chars() {
        out.push(ord(character))
    }
    let values = image.to_list()
    for y in 0..image.rows {
        for x in 0..image.cols {
            let base = (y * image.cols + x) * channels
            var ordered: list[float] = []
            if channels == 1 {
                ordered.push(values[base])
            } else {
                ordered.push(values[base + 2])
                ordered.push(values[base + 1])
                ordered.push(values[base])
            }
            for value in ordered {
                var whole = int(math.floor(value + 0.5))
                if whole < 0 { whole = 0 }
                if whole > maximum { whole = maximum }
                if deep {
                    out.push(whole >> 8 & 255)
                    out.push(whole & 255)
                } else {
                    out.push(whole)
                }
            }
        }
    }
    return bytes(out)
}
