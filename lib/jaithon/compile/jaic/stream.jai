#: `Writer` and `Reader`: the two cursors every section of the container is
#: written and read through. Little-endian fixed-width integers, LEB128, raw
#: bytes, and -- on the reading side -- a bounds check before each one.
#:
#: The asymmetry between them is deliberate. `Writer` cannot fail. `Reader`
#: never raises: every accessor checks what is left first and sets the sticky
#: `bad` flag instead, so a truncated file costs one null at the end of
#: `read_module` rather than an exception per field, and a stale cache stays a
#: normal outcome instead of an error.
#:
#: What is NOT here: any knowledge of what the bytes mean. The header, the
#: pools and the function records are `write.jai`'s and `read.jai`'s.
from jaithon.compile.jaic.bits import _SIGN_BIT, bits_to_float, float_bits
from jaithon.compile.jaic.format import VERSION

#: A growable little-endian byte buffer.
pub class Writer {
    #: The bytes written so far.
    pub var data: list[int]

    #: Create an empty buffer.
    pub fn init(self) { self.data = [] }

    #: Append one byte, keeping only its low eight bits.
    pub fn u8(self, value: int) -> void { self.data.push(value & 0xFF) }

    #: Append a little-endian `u16`.
    pub fn u16(self, value: int) -> void {
        self.data.push(value & 0xFF)
        self.data.push(value >> 8 & 0xFF)
    }

    #: Append a little-endian `u32`.
    pub fn u32(self, value: int) -> void {
        for shift in range(4) { self.data.push(value >> shift * 8 & 0xFF) }
    }

    #: Append a little-endian `u64`, sign included.
    pub fn u64(self, value: int) -> void {
        for shift in range(8) { self.data.push(value >> shift * 8 & 0xFF) }
    }

    #: Append the IEEE-754 bits of `value`.
    pub fn f64(self, value: float) -> void { self.u64(float_bits(value)) }

    #: Append raw bytes.
    pub fn bytes(self, values: list[int]) -> void {
        for value in values { self.data.push(value & 0xFF) }
    }

    #: Append an unsigned LEB128. Seven bits per byte, high bit meaning
    #: "continues", so a small value costs one byte instead of four.
    pub fn uleb(self, value: int) -> void {
        var v = value
        loop {
            var byte = v & 0x7F
            v = v >> 7
            if v != 0 { byte = byte | 0x80 }
            self.data.push(byte)
            if v == 0 { break }
        }
    }
}

#: A bounds-checked cursor over a byte list.
#:
#: Every accessor checks bytes remaining first and sets `bad` instead of
#: raising, so a truncated file yields null once, not an exception per field.
pub class Reader {
    #: The buffer being read.
    pub let data: list[int]
    #: Index of the next byte.
    pub var position: int
    #: One past the last readable byte.
    pub var limit: int
    #: Sticky: set by the first read that ran short.
    pub var bad: bool
    #: Container version of the image being read.
    #:
    #: Carried here rather than passed down because exactly one section reads
    #: it — the line table, whose encoding changed at `VERSION_LTV1` — and
    #: threading it through `_read_function`, `_read_constants` and
    #: `_read_tuple` would touch several signatures to reach one branch.
    #: `VERSION` until the container header says otherwise.
    pub var version: int
    #: The module string table, from `VERSION_STRTAB` on: every tag-10 constant
    #: is an index into it. Empty for older containers, which carry their
    #: strings inline.
    pub var strings: list[str]

    #: Read `data` up to `limit`, which defaults to its length.
    pub fn init(self, data: list[int], limit: int = -1) {
        self.data = data
        self.position = 0
        self.limit = if limit < 0 { data.len() } else { limit }
        self.bad = false
        self.version = VERSION
        self.strings = []
    }

    #: Bytes still readable.
    pub fn left(self) -> int {
        if self.bad or self.position > self.limit { return 0 }
        return self.limit - self.position
    }

    #: Skip `count` bytes, or mark the cursor bad.
    pub fn skip(self, count: int) -> bool {
        if self.bad or count < 0 or count > self.left() {
            self.bad = true
            return false
        }
        self.position += count
        return true
    }

    #: True when `count` records of `each` bytes still fit.
    pub fn fits(self, count: int, each: int) -> bool {
        if self.bad or count < 0 { return false }
        return count <= self.left() // (if each > 0 { each } else { 1 })
    }

    #: Read one byte, or 0 with `bad` set.
    pub fn u8(self) -> int {
        if not self.skip(1) { return 0 }
        return self.data[self.position - 1] & 0xFF
    }

    #: Read a little-endian `u16`.
    pub fn u16(self) -> int {
        let low = self.u8()
        return low | self.u8() << 8
    }

    #: Read a little-endian `u32`.
    pub fn u32(self) -> int {
        var value = 0
        for shift in range(4) {
            let byte = self.u8()
            value = value | byte << shift * 8
        }
        return value
    }

    #: Read an unsigned LEB128, or null when the run is malformed or runs past
    #: the limit. Ten bytes is the most a 64-bit value can occupy, so a bad run
    #: refuses the image rather than spinning.
    pub fn uleb(self) -> int? {
        var value = 0
        var shift = 0
        for _step in range(10) {
            if self.position >= self.limit { return null }
            let byte = self.u8()
            if self.bad { return null }
            value = value | (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0 { return value }
        }
        return null
    }

    #: Read a little-endian `u64` as a signed 64-bit integer.
    pub fn u64(self) -> int {
        var value = 0
        for shift in range(7) {
            let byte = self.u8()
            value = value | byte << shift * 8
        }
        let top = self.u8()
        value = value | (top & 0x7F) << 56
        if top >= 128 { value = value | _SIGN_BIT }
        return value
    }

    #: Read an IEEE-754 binary64.
    pub fn f64(self) -> float { return bits_to_float(self.u64()) }

    #: Read `count` raw bytes.
    pub fn bytes(self, count: int) -> list[int] {
        if not self.fits(count, 1) or not self.skip(count) {
            self.bad = true
            return []
        }
        var out: list[int] = []
        for index in range(count) { out.push(self.data[self.position - count + index]) }
        return out
    }
}
