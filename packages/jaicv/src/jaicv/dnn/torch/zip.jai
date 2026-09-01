#: The ZIP container a checkpoint is, read through its central directory.
import std.io as io
from std.io import File
from .bits import le_u16, le_u32, le_u64

#: One member of a ZIP archive, as the central directory describes it.
class ZipEntry {
    pub let name: str
    pub let method: int
    pub let stored_size: int
    pub let size: int
    pub let crc: int
    pub let header_at: int

    pub fn init(
        self,
        name: str,
        method: int,
        stored_size: int,
        size: int,
        crc: int,
        header_at: int
    ) {
        self.name = name
        self.method = method
        self.stored_size = stored_size
        self.size = size
        self.crc = crc
        self.header_at = header_at
    }
}

#: Where an archive's bytes come from.
#:
#: A block already in memory is the easy case and the one a caller who already
#: has the file gets. The other case is why this exists: a checkpoint big
#: enough to need zip64 is a checkpoint that must not be pulled into memory
#: whole and then copied storage by storage back out of itself, which wants
#: twice the file in RAM before a single weight is available. Reading through
#: the handle costs one seek per member instead.
pub class Source {
    pub let block: bytes?
    pub let handle: File?
    pub let total: int

    pub fn init(self, block: bytes?, handle: File?, total: int) {
        self.block = block
        self.handle = handle
        self.total = total
    }

    #: An archive that is already a `bytes`.
    pub static fn of_bytes(data: bytes) -> Source { return Source(data, null, data.len()) }

    #: An archive on disk, read a window at a time.
    pub static fn of_file(path: str) -> Source {
        let handle = io.open(path, "r")
        return Source(null, handle, handle.seek(0, io.SEEK_END))
    }

    #: `count` bytes starting at `offset`.
    pub fn at(self, offset: int, count: int) -> bytes {
        if offset < 0 or count < 0 or offset + count > self.total {
            throw ValueError(
                f"this archive is {self.total} bytes and something in it points at " +
                    f"{offset}..{offset + count}, which is past the end"
            )
        }
        let held = self.block
        if held is not null {
            let data: bytes = held
            return data.slice(offset, offset + count)
        }
        let opened = self.handle
        if opened is not null {
            let file: File = opened
            let _moved = file.seek(offset, io.SEEK_SET)
            var taken = file.read_bytes(count)
            #: A short read is not an error on a stream, only an incomplete
            #: one, so the rest is asked for again rather than handed back as
            #: though it were the answer.
            while taken.len() < count {
                let more = file.read_bytes(count - taken.len())
                if more.len() == 0 {
                    throw ValueError(f"this archive ends {count - taken.len()} bytes early")
                }
                taken = taken.concat(more)
            }
            return taken
        }
        throw ValueError("this archive has neither a block nor a handle to read from")
    }

    #: Release the handle, if this source has one.
    pub fn close(self) -> void {
        let opened = self.handle
        if opened is not null {
            let file: File = opened
            file.close()
        }
    }
}

#: A ZIP archive, read through its central directory.
#:
#: The central directory is the authority on every member's size, method and
#: checksum, and it has to be: torch's writer sets the streaming flag in each
#: local header and leaves the sizes and the CRC there zeroed. The local header
#: is still consulted, but only for its own name and extra-field lengths, which
#: differ from the central directory's — torch pads the local extra field to
#: align the storage that follows it.
pub class ZipArchive {
    pub let source: Source
    pub let names: list[str]
    pub let entries: dict[str, ZipEntry]

    pub fn init(self, source: Source) {
        self.source = source
        self.names = []
        self.entries = {}
        let (end_at, tail, tail_at) = self._find_end_record()
        var count = le_u16(tail, end_at - tail_at + 8)
        var directory_at = le_u32(tail, end_at - tail_at + 16)
        var directory_size = le_u32(tail, end_at - tail_at + 12)
        if count == 0xFFFF or directory_at == 0xFFFFFFFF or directory_size == 0xFFFFFFFF {
            let record = self._read_zip64_end(end_at, tail, tail_at)
            count = record[0]
            directory_size = record[1]
            directory_at = record[2]
        }
        let directory = source.at(directory_at, directory_size)
        var at = 0
        for _index in 0..count {
            at = self._read_directory_entry(directory, at)
        }
    }

    #: Whether the archive holds a member under this exact name.
    pub fn has(self, name: str) -> bool { return self.entries.contains(name) }

    #: One member's bytes, decompressed if it was deflated.
    pub fn read(self, name: str) -> bytes {
        if not self.entries.contains(name) {
            throw ValueError(f"the archive has no member named '{name}'")
        }
        let entry = self.entries[name]
        let header = self.source.at(entry.header_at, 30)
        if le_u32(header, 0) != 0x04034B50 {
            throw ValueError(f"member '{name}' does not start with a local file header")
        }
        let begin = entry.header_at + 30 + le_u16(header, 26) + le_u16(header, 28)
        if entry.method == 0 {
            let payload = self.source.at(begin, entry.size)
            if __prim__.crc32(payload) != entry.crc {
                throw ValueError(f"member '{name}' does not match its recorded checksum")
            }
            return payload
        }
        if entry.method == 8 {
            return inflate_member(
                self.source.at(begin, entry.stored_size),
                entry.size,
                entry.crc,
                name
            )
        }
        throw ValueError(
            f"member '{name}' uses compression method {entry.method}; this reader " +
                "handles stored (0) and deflate (8)"
        )
    }

    #: The end-of-central-directory record: where it starts, the tail it was
    #: found in, and where that tail starts.
    #:
    #: The record is the last thing in the file except for an optional comment
    #: of up to 64 KiB, so the only way to it is backwards from the end looking
    #: for the signature. The tail is handed back with it because everything
    #: else the end of the archive holds is inside the same window.
    fn _find_end_record(self) -> tuple[int, bytes, int] {
        if self.source.total < 22 {
            throw ValueError("this file is too short to be a ZIP archive")
        }
        let span = min(self.source.total, 22 + 65535 + 20)
        let tail_at = self.source.total - span
        let tail = self.source.at(tail_at, span)
        var at = span - 22
        while at >= 0 {
            if le_u32(tail, at) == 0x06054B50 { return (tail_at + at, tail, tail_at) }
            at -= 1
        }
        throw ValueError("this file has no ZIP end-of-central-directory record")
    }

    #: The entry count, directory size and directory offset from the zip64 end
    #: record, which is where an archive past 4 GiB or 65535 members keeps them.
    fn _read_zip64_end(self, end_at: int, tail: bytes, tail_at: int) -> list[int] {
        let locator_at = end_at - 20
        if locator_at < tail_at or le_u32(tail, locator_at - tail_at) != 0x07064B50 {
            throw ValueError("this archive claims zip64 sizes but has no zip64 locator")
        }
        let record_at = le_u64(tail, locator_at - tail_at + 8)
        if record_at + 56 > self.source.total {
            throw ValueError("this archive's zip64 locator points past the end of the file")
        }
        let record = self.source.at(record_at, 56)
        if le_u32(record, 0) != 0x06064B50 {
            throw ValueError("this archive's zip64 locator does not point at a zip64 end record")
        }
        return [le_u64(record, 32), le_u64(record, 40), le_u64(record, 48)]
    }

    #: Read one central directory entry out of the directory block and return
    #: where the next one starts in it.
    fn _read_directory_entry(self, directory: bytes, at: int) -> int {
        if at + 46 > directory.len() or le_u32(directory, at) != 0x02014B50 {
            throw ValueError("this archive's central directory is truncated or misaligned")
        }
        let name_length = le_u16(directory, at + 28)
        let extra_length = le_u16(directory, at + 30)
        let comment_length = le_u16(directory, at + 32)
        if at + 46 + name_length + extra_length + comment_length > directory.len() {
            throw ValueError("this archive's central directory is truncated")
        }
        let name = directory.slice(at + 46, at + 46 + name_length).decode()
        var stored_size = le_u32(directory, at + 20)
        var size = le_u32(directory, at + 24)
        var header_at = le_u32(directory, at + 42)
        #: The zip64 extra field carries replacements only for the fields that
        #: were parked at the sentinel, in this fixed order and no other, so
        #: which ones are present has to be inferred from the 32-bit slots.
        if size == 0xFFFFFFFF or stored_size == 0xFFFFFFFF or header_at == 0xFFFFFFFF {
            var walk = at + 46 + name_length
            let extra_end = walk + extra_length
            while walk + 4 <= extra_end {
                let tag = le_u16(directory, walk)
                let length = le_u16(directory, walk + 2)
                if tag == 0x0001 {
                    var field = walk + 4
                    if size == 0xFFFFFFFF {
                        size = le_u64(directory, field)
                        field += 8
                    }
                    if stored_size == 0xFFFFFFFF {
                        stored_size = le_u64(directory, field)
                        field += 8
                    }
                    if header_at == 0xFFFFFFFF { header_at = le_u64(directory, field) }
                }
                walk += 4 + length
            }
        }
        let entry = ZipEntry(
            name,
            le_u16(directory, at + 10),
            stored_size,
            size,
            le_u32(directory, at + 16),
            header_at
        )
        self.names.push(name)
        self.entries[name] = entry
        return at + 46 + name_length + extra_length + comment_length
    }
}

#: A deflated ZIP member, through zlib.
#:
#: `__prim__.inflate` autodetects a zlib or a gzip header but will not take the
#: bare deflate stream a ZIP member holds, so the stream is dressed as gzip on
#: the way in. That is not a workaround so much as a free integrity check: a
#: gzip trailer is a CRC and a length, the central directory has just supplied
#: both, and zlib refuses the member if either disagrees.
fn inflate_member(payload: bytes, size: int, crc: int, name: str) -> bytes {
    var wrapper = [0x1F, 0x8B, 0x08, 0, 0, 0, 0, 0, 0, 0xFF]
    var trailer: list[int] = []
    for shift in 0..4 { trailer.push(crc >> 8 * shift & 0xFF) }
    for shift in 0..4 { trailer.push(size >> 8 * shift & 0xFF) }
    let expanded = __prim__.inflate(bytes(wrapper).concat(payload).concat(bytes(trailer)), size)
    if expanded.len() != size {
        throw ValueError(
            f"member '{name}' expanded to {expanded.len()} bytes, and the directory says {size}"
        )
    }
    return expanded
}
