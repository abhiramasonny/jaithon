#: Operating-system handles, buffered streams, whole-file transfers and the
#: standard streams.
from std.core import Iterable, Iterator, Printable

#: Seek from the start of the file.
pub const SEEK_SET = 0

#: Seek from the current position.
pub const SEEK_CURRENT = 1

#: Seek backwards from the end of the file.
pub const SEEK_END = 2

#: The buffer size the buffered streams and the whole-file readers use.
pub const DEFAULT_BUFFER_SIZE = 65536

let _EMPTY_BYTES = __prim__.str_encode("")

let _MODES = ["r", "w", "a", "r+", "w+", "a+"]

#: One open operating-system handle, unbuffered.
#:
#: Each `read` and `write` is one system call — fine for whole-file transfers,
#: wasteful for a line-by-line loop (use `BufferedReader` instead). Nothing
#: closes a file for you, so pair `open` with `defer { file.close() }`:
#:
#: ```
#: write_file("/tmp/jaithon-file-doc.txt", "alpha\nbeta\n")
#: fn load(path: str) -> str {
#:     let file = open(path, "r")
#:     defer { file.close() }
#:     return file.read()
#: }
#: load("/tmp/jaithon-file-doc.txt")    # "alpha\nbeta\n"
#: ```
pub class File: Printable {
    #: The name the file was opened under, used in diagnostics.
    pub let name: str

    #: The mode the file was opened in.
    pub let mode: str

    #: The stream this file reads and writes: opaque, only ever handed
    #: straight back to an `io_` primitive.
    let handle: any
    var closed: bool

    #: Adopt an already-open `handle`, such as one end of a pipe.
    #:
    #: `open` is what a caller normally wants; this exists for handles that
    #: did not come from a path (standard streams, spawned-process pipes) and
    #: takes ownership of `handle`.
    pub fn init(self, handle: any, name: str, mode: str) {
        self.handle = handle
        self.name = name
        self.mode = mode
        self.closed = false
    }

    #: Read the rest of the stream and decode it as UTF-8.
    pub fn read(self) -> str { return _decode(self.read_bytes()) }

    #: Read at most `count` bytes, or the rest of the stream when `count` is negative.
    #:
    #: Returns an empty `bytes` at end of stream. A short read is not an error:
    #: a positive `count` is an upper bound, not a demand.
    pub fn read_bytes(self, count: int = -1) -> bytes {
        self._require_open()
        return __prim__.io_read(self.handle, count)
    }

    #: Write `text` as UTF-8 and return the number of bytes written.
    pub fn write(self, text: str) -> int {
        self._require_open()
        return __prim__.io_write(self.handle, text)
    }

    #: Write raw bytes and return the number of bytes written.
    pub fn write_bytes(self, data: bytes) -> int {
        self._require_open()
        return __prim__.io_write(self.handle, data)
    }

    #: Write `text` followed by a newline.
    pub fn write_line(self, text: str = "") -> void {
        let _written = self.write(text + "\n")
    }

    #: Iterate the remaining lines, without their terminators.
    #:
    #: The lines are pulled through a fresh `BufferedReader`, so the iterator
    #: reads ahead: do not mix it with direct reads from the same file.
    #:
    #: ```
    #: write_file("/tmp/jaithon-lines-doc.txt", "one\ntwo\n")
    #: let file = open("/tmp/jaithon-lines-doc.txt")
    #: let collected = file.lines().collect()
    #: file.close()
    #: collected    # ["one", "two"]
    #: ```
    pub fn lines(self) -> Iterator[str] { return BufferedReader(self).lines() }

    #: Move to `offset` bytes from `whence` and return the new absolute position.
    pub fn seek(self, offset: int, whence: int = SEEK_SET) -> int {
        self._require_open()
        if whence != SEEK_SET and whence != SEEK_CURRENT and whence != SEEK_END {
            throw ValueError(f"whence must be SEEK_SET, SEEK_CURRENT or SEEK_END, got {whence}")
        }
        return __prim__.io_seek(self.handle, offset, whence)
    }

    #: Return the current position, in bytes from the start.
    pub fn tell(self) -> int { return self.seek(0, SEEK_CURRENT) }

    #: Move back to the start of the file.
    pub fn rewind(self) -> void {
        let _position = self.seek(0, SEEK_SET)
    }

    #: Return the size of the stream in bytes, leaving the position unchanged.
    #:
    #: Raises `IOError` on a stream that cannot seek, such as a pipe or a
    #: terminal; ask `Path.size` instead when a path is what you have.
    pub fn size(self) -> int {
        let here = self.seek(0, SEEK_CURRENT)
        let end = self.seek(0, SEEK_END)
        let _restored = self.seek(here, SEEK_SET)
        return end
    }

    #: Push any data the operating system is holding out to its destination.
    pub fn flush(self) -> void {
        self._require_open()
        __prim__.io_flush(self.handle)
    }

    #: Close the file. Closing an already closed file does nothing.
    pub fn close(self) -> void {
        if self.closed { return }
        self.closed = true
        __prim__.io_close(self.handle)
    }

    #: Whether `close` has been called.
    pub fn is_closed(self) -> bool { return self.closed }

    #: Render the file as `File(<name>, <mode>)`.
    pub fn to_str(self) -> str { return f"File({self.name}, {self.mode})" }

    fn _require_open(self) -> void {
        if self.closed { throw IOError(f"{self.name} is closed") }
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: A reader that turns many small reads of a `File` into few large ones.
#:
#: Reading a file line by line without buffering costs one system call per
#: character; this pulls `capacity` bytes at a time instead. It decodes UTF-8
#: on character boundaries, holding back a multi-byte character split across
#: two buffers until the rest arrives.
#:
#: ```
#: write_file("/tmp/jaithon-reader-doc.txt", "alpha\nbeta\ngamma\n")
#: let reader = BufferedReader(open("/tmp/jaithon-reader-doc.txt"))
#: reader.read_line()             # "alpha"
#: reader.lines().collect()       # ["beta", "gamma"]
#: reader.close()
#: ```
pub class BufferedReader: Iterable[str] {
    let source: File
    let capacity: int
    var buffer: str
    var position: int
    var carry: bytes
    var finished: bool

    #: Wrap `source`, reading `capacity` bytes at a time.
    pub fn init(self, source: File, capacity: int = DEFAULT_BUFFER_SIZE) {
        if capacity <= 0 { throw ValueError(f"capacity must be positive, got {capacity}") }
        self.source = source
        self.capacity = capacity
        self.buffer = ""
        self.position = 0
        self.carry = _EMPTY_BYTES
        self.finished = false
    }

    #: Read the next line, without its terminator, or null at end of stream.
    #:
    #: A trailing `\r` is removed, so a file with CRLF terminators reads the
    #: same as one with LF terminators. A final line without a terminator is
    #: still returned.
    pub fn read_line(self) -> str? {
        var scanned = 0
        loop {
            let found = _index_of_char(self.buffer, "\n", self.position + scanned)
            if found >= 0 {
                let line = self.buffer[self.position:found]
                self.position = found + 1
                return _drop_carriage(line)
            }
            scanned = self.buffer.len() - self.position
            if not self._fill() {
                if self.position >= self.buffer.len() { return null }
                let rest = self.buffer[self.position:self.buffer.len()]
                self.position = self.buffer.len()
                return _drop_carriage(rest)
            }
        }
    }

    #: Read at most `size` characters, returning "" at end of stream.
    #:
    #: This is the reader to use for a file that is not line-structured, and
    #: the only one that bounds how much memory the read takes.
    #:
    #: ```
    #: write_file("/tmp/jaithon-chunk-doc.txt", "abcdefgh")
    #: let reader = BufferedReader(open("/tmp/jaithon-chunk-doc.txt"))
    #: reader.read_chunk(3)    # "abc"
    #: reader.read_chunk(3)    # "def"
    #: reader.read_chunk(3)    # "gh"
    #: reader.read_chunk(3)    # ""
    #: reader.close()
    #: ```
    pub fn read_chunk(self, size: int) -> str {
        if size <= 0 { throw ValueError(f"chunk size must be positive, got {size}") }
        while self.buffer.len() - self.position < size {
            if not self._fill() { break }
        }
        let taken = min(size, self.buffer.len() - self.position)
        if taken == 0 { return "" }
        let text = self.buffer[self.position:self.position + taken]
        self.position += taken
        return text
    }

    #: Read everything that is left as one string.
    pub fn read_all(self) -> str {
        var parts = [self.buffer[self.position:self.buffer.len()]]
        self._reset_buffer()
        while self._fill() {
            parts.push(self.buffer)
            self._reset_buffer()
        }
        return "".join(parts)
    }

    #: Iterate the remaining lines, without their terminators.
    pub fn lines(self) -> Iterator[str] { return LineIter(self) }

    #: Return an iterator over the remaining lines, so `for line in reader` works.
    pub fn iter(self) -> Iterator[str] { return LineIter(self) }

    #: Close the underlying file. Buffered data that was read ahead is discarded.
    pub fn close(self) -> void { self.source.close() }

    fn _reset_buffer(self) -> void {
        self.buffer = ""
        self.position = 0
    }

    fn _fill(self) -> bool {
        if self.finished { return false }
        if self.position > 0 {
            self.buffer = self.buffer[self.position:self.buffer.len()]
            self.position = 0
        }
        loop {
            let chunk = self.source.read_bytes(self.capacity)
            if chunk.len() == 0 {
                self.finished = true
                if self.carry.len() == 0 { return false }
                self.buffer = self.buffer + _decode(self.carry)
                self.carry = _EMPTY_BYTES
                return true
            }
            let (complete, pending) = _utf8_split(self.carry.concat(chunk))
            self.carry = pending
            if complete.len() > 0 {
                self.buffer = self.buffer + _decode(complete)
                return true
            }
        }
    }
}

#: A writer that collects small writes and pushes them out in full buffers.
#:
#: Nothing reaches the file until the buffer fills or `flush` is called, so a
#: writer that is dropped without being flushed loses whatever it was holding.
#: `close` flushes first; pair the constructor with `defer { writer.close() }`.
#:
#: ```
#: let writer = BufferedWriter(open("/tmp/jaithon-writer-doc.txt", "w"))
#: writer.write_line("alpha")
#: writer.write_line("beta")
#: writer.close()
#: read_lines("/tmp/jaithon-writer-doc.txt")    # ["alpha", "beta"]
#: ```
pub class BufferedWriter {
    let sink: File
    let capacity: int
    var parts: list[str]
    var pending: int

    #: Wrap `sink`, flushing once `capacity` characters have accumulated.
    pub fn init(self, sink: File, capacity: int = DEFAULT_BUFFER_SIZE) {
        if capacity <= 0 { throw ValueError(f"capacity must be positive, got {capacity}") }
        self.sink = sink
        self.capacity = capacity
        self.parts = []
        self.pending = 0
    }

    #: Add `text` to the buffer, flushing if it is now full.
    pub fn write(self, text: str) -> void {
        if text.len() == 0 { return }
        self.parts.push(text)
        self.pending += text.len()
        if self.pending >= self.capacity { self.flush() }
    }

    #: Add `text` and a newline to the buffer.
    pub fn write_line(self, text: str = "") -> void { self.write(text + "\n") }

    #: Write everything buffered so far and push it out of the operating system.
    pub fn flush(self) -> void {
        if self.pending > 0 {
            let _written = self.sink.write("".join(self.parts))
            self.parts = []
            self.pending = 0
        }
        self.sink.flush()
    }

    #: Flush the buffer and close the underlying file.
    pub fn close(self) -> void {
        self.flush()
        self.sink.close()
    }
}

#: Standard input, as an unbuffered file on handle 0.
#:
#: `read_line`, `read_all_stdin` and `prompt` read through one shared buffered
#: reader over this file. Reading from `stdin` directly as well will interleave
#: badly, because the shared reader reads ahead.
pub let stdin: File = File(0, "<stdin>", "r")

#: Standard output, as an unbuffered file on handle 1.
pub let stdout: File = File(1, "<stdout>", "w")

#: Standard error, as an unbuffered file on handle 2.
pub let stderr: File = File(2, "<stderr>", "w")

let _stdin_reader = BufferedReader(stdin)

#: Open the file at `path` and return it.
#:
#: The modes are `r` to read, `w` to truncate and write, `a` to append, and
#: those three with `+` added to allow the other direction as well. Raises
#: `FileNotFoundError`, `PermissionError` or `IOError`.
#:
#: ```
#: let file = open("/tmp/jaithon-open-doc.txt", "w")
#: file.write_line("recorded")
#: file.close()
#: read_file("/tmp/jaithon-open-doc.txt")    # "recorded\n"
#: ```
pub fn open(path: str, mode: str = "r") -> File {
    if not _MODES.contains(mode) {
        throw ValueError(f"mode must be one of {_MODES}, got {repr(mode)}")
    }
    return File(__prim__.io_open(path, mode), path, mode)
}

#: Read the whole file at `path` as UTF-8 text.
#:
#: ```
#: write_file("/tmp/jaithon-read-doc.txt", "content")
#: read_file("/tmp/jaithon-read-doc.txt")    # "content"
#: ```
pub fn read_file(path: str) -> str { return _decode(read_bytes(path)) }

#: Read the whole file at `path` as a list of lines, without their terminators.
#:
#: A trailing newline does not produce a final empty line, and `\r\n` is
#: handled the same as `\n`.
#:
#: ```
#: write_file("/tmp/jaithon-readlines-doc.txt", "one\ntwo\n")
#: read_lines("/tmp/jaithon-readlines-doc.txt")    # ["one", "two"]
#: ```
pub fn read_lines(path: str) -> list[str] { return split_lines(read_file(path)) }

#: Read the whole file at `path` as raw bytes, without decoding it.
pub fn read_bytes(path: str) -> bytes {
    let file = open(path, "r")
    defer { file.close() }
    return file.read_bytes()
}

#: Write `text` to `path` as UTF-8, replacing whatever was there.
pub fn write_file(path: str, text: str) -> void {
    let file = open(path, "w")
    defer { file.close() }
    let _written = file.write(text)
}

#: Write raw `data` to `path`, replacing whatever was there.
pub fn write_bytes(path: str, data: bytes) -> void {
    let file = open(path, "w")
    defer { file.close() }
    let _written = file.write_bytes(data)
}

#: Append `text` to `path`, creating the file when it does not exist.
pub fn append_file(path: str, text: str) -> void {
    let file = open(path, "a")
    defer { file.close() }
    let _written = file.write(text)
}

#: Write `lines` to `path`, each followed by a newline.
#:
#: ```
#: write_lines("/tmp/jaithon-writelines-doc.txt", ["one", "two"])
#: read_file("/tmp/jaithon-writelines-doc.txt")    # "one\ntwo\n"
#: ```
pub fn write_lines(path: str, lines: list[str]) -> void {
    let file = open(path, "w")
    defer { file.close() }
    let writer = BufferedWriter(file)
    for line in lines { writer.write_line(line) }
    writer.flush()
}

#: Split `text` into lines, accepting `\n`, `\r\n` and `\r` as terminators.
#:
#: A terminator at the very end does not produce a trailing empty line, which
#: is the difference between this and `text.split("\n")`.
#:
#: ```
#: split_lines("a\r\nb\n")    # ["a", "b"]
#: split_lines("")            # []
#: ```
pub fn split_lines(text: str) -> list[str] {
    var lines: list[str] = []
    var start = 0
    var index = 0
    while index < text.len() {
        let character = text[index]
        if character != "\n" and character != "\r" {
            index += 1
            continue
        }
        lines.push(text[start:index])
        if character == "\r" and index + 1 < text.len() and text[index + 1] == "\n" { index += 1 }
        index += 1
        start = index
    }
    if start < text.len() { lines.push(text[start:text.len()]) }
    return lines
}

#: Print `values` to standard error, separated by spaces and ended by a newline.
#:
#: Standard error is unbuffered here, so a message written just before a crash
#: has already left the process.
pub fn print_err(...values: any) -> void {
    let _written = stderr.write(" ".join([str(value) for value in values]) + "\n")
    stderr.flush()
}

#: Read one line from standard input, without its terminator, or null at the end.
pub fn read_line() -> str? { return _stdin_reader.read_line() }

#: Read the whole of standard input as one string.
pub fn read_all_stdin() -> str { return _stdin_reader.read_all() }

#: Write `message`, read one line of the reply, and return it without its newline.
#:
#: The message is written without a trailing newline and flushed, so the cursor
#: waits at the end of the prompt. Raises `IOError` when input ends first,
#: because a caller that asked a question has no answer to carry on with.
pub fn prompt(message: str) -> str {
    let _written = stdout.write(message)
    stdout.flush()
    let line = _stdin_reader.read_line()
    if line is null { throw IOError(f"input ended while waiting for a reply to {repr(message)}") }
    return line
}

class LineIter: Iterator[str], Iterable[str] {
    let reader: BufferedReader

    pub fn init(self, reader: BufferedReader) { self.reader = reader }

    pub fn next(self) -> str? { return self.reader.read_line() }

    pub fn iter(self) -> Iterator[str] { return self }
}

fn _index_of_char(text: str, character: str, start: int) -> int {
    var index = start
    while index < text.len() {
        if text[index] == character { return index }
        index += 1
    }
    return -1
}

fn _drop_carriage(line: str) -> str {
    if line.len() > 0 and line[line.len() - 1] == "\r" { return line[0:line.len() - 1] }
    return line
}

fn _utf8_split(data: bytes) -> tuple[bytes, bytes] {
    let length = data.len()
    var back = 0
    while back < 3 and back < length {
        let byte = data[length - 1 - back]
        if byte < 0x80 { break }
        if byte >= 0xC0 {
            var needed = 2
            if byte >= 0xF0 { needed = 4 } elif byte >= 0xE0 { needed = 3 }
            if back + 1 < needed {
                let cut = length - 1 - back
                return (data[0:cut], data[cut:length])
            }
            break
        }
        back += 1
    }
    return (data, _EMPTY_BYTES)
}

fn _decode(data: bytes) -> str { return __prim__.str_decode(data) }
