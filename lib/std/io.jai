#: Files, paths, buffered streams, the standard streams, and delimited text.
#:
#: The whole module is POSIX-shaped: paths are separated by `/`, a file is a
#: stream of bytes with no text mode to opt into, and the three standard streams
#: are ordinary `File` values that can be written to and buffered like any
#: other.
#:
#: Four layers, from the shortest thing that works to the most controlled:
#:
#: - `read_file`, `write_file`, `read_lines` and friends do the whole job in one
#:   call and hold the whole file in memory.
#: - `open` returns a `File`, which is a thin, unbuffered wrapper over one
#:   operating-system handle. Pair it with `defer { file.close() }`.
#: - `BufferedReader` and `BufferedWriter` sit on a `File` and turn many small
#:   operations into few large ones. Reading a large file a line at a time goes
#:   through `BufferedReader`; anything else re-reads the file per line.
#: - `Path` is a path as a value: it joins, normalises, matches globs, walks
#:   directory trees, and asks the filesystem about itself.
#:
#: ```
#: write_file("/tmp/jaithon-io-doc.txt", "alpha\nbeta\n")
#: read_lines("/tmp/jaithon-io-doc.txt")        # ["alpha", "beta"]
#: Path("/tmp/jaithon-io-doc.txt").stem()       # "jaithon-io-doc"
#: ```
#:
#: Text is UTF-8 everywhere. The readers decode on a character boundary even
#: when a buffer boundary falls in the middle of a multi-byte sequence, so a
#: chunked read of a file and a whole-file read produce the same string.
#:
#: ## The primitive contract
#:
#: Everything here reaches the operating system through the `io` primitives of
#: the language specification, and only through the private wrappers at the
#: bottom of this file. Those primitives are:
#:
#: - `io_open(path, mode)` returns an opaque stream handle, or raises
#:   `FileNotFoundError`, `PermissionError` or `IOError`.
#: - `io_read(handle, count) -> bytes` reads at most `count` bytes, or the rest
#:   of the stream when `count` is negative, and returns empty at end of stream.
#: - `io_write(handle, data) -> int` writes a `bytes`, or a `str` encoded as
#:   UTF-8, and returns the number of bytes written.
#: - `io_seek(handle, offset, whence) -> int` returns the new absolute position.
#: - `io_close(handle)`, `io_flush(handle)`.
#: - `io_stat(path, follow = false) -> dict?` is null when nothing is there, and
#:   otherwise has `kind` (`"file"`, `"dir"`, `"link"` or `"other"`), `size` in
#:   bytes, `modified` in nanoseconds since the Unix epoch, and `mode`
#:   permission bits. It reports the link itself unless `follow` asks for the
#:   target, which is what `is_file` and `is_dir` want.
#: - `io_listdir(path) -> list[str]`, `io_mkdir(path)`, `io_remove(path)`, the
#:   last of which removes one file or one empty directory.
from std.core import Comparable, Hashable, Iterable, Iterator, Printable, cmp
from std.time import DateTime

#: Seek from the start of the file.
pub const SEEK_SET = 0

#: Seek from the current position.
pub const SEEK_CURRENT = 1

#: Seek backwards from the end of the file.
pub const SEEK_END = 2

#: The path separator, which is `/` on every platform this module supports.
pub const SEPARATOR = "/"

#: The buffer size the buffered streams and the whole-file readers use.
pub const DEFAULT_BUFFER_SIZE = 65536

let _EMPTY_BYTES = __prim__.str_encode("")

let _MODES = ["r", "w", "a", "r+", "w+", "a+"]

#: One open operating-system handle, unbuffered.
#:
#: A `File` does no buffering of its own: each `read` and `write` is one system
#: call. That is what you want for whole-file transfers and for the standard
#: streams, and what you do not want for a loop over lines — wrap the file in a
#: `BufferedReader` for that.
#:
#: A file is closed by `close`, which is idempotent, and every operation on a
#: closed file raises `IOError`. Nothing closes a file for you, so pair the
#: `open` with a `defer` and the file is closed on both the normal and the
#: exceptional path:
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

    #: The stream this file reads and writes. `io_open` returns one; the
    #: integers 0, 1 and 2 name the standard streams. It is opaque either way,
    #: so it is only ever handed straight back to an `io_` primitive.
    let handle: any
    var closed: bool

    #: Adopt an already-open `handle`, such as one end of a pipe.
    #:
    #: `open` is what a caller normally wants. This constructor exists for
    #: handles that did not come from a path — the standard streams and the
    #: pipes of a spawned process — and takes ownership of the handle: closing
    #: the file closes the handle.
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
#: character. This reader pulls `capacity` bytes at a time and hands out lines
#: and chunks from what it holds, so the cost is one system call per buffer.
#:
#: It decodes UTF-8 on character boundaries: a multi-byte character split across
#: two buffers is held back until the rest of it arrives.
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
        # Counted from `position` rather than from the start of the buffer,
        # because `_fill` drops the consumed prefix and shifts every absolute
        # index left.
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

    # Append at least one character to the buffer, or report end of stream.
    fn _fill(self) -> bool {
        if self.finished { return false }
        # Drop the consumed prefix before growing the buffer. Without this, a
        # file read line by line copies the whole buffer once per line.
        if self.position > 0 {
            self.buffer = self.buffer[self.position:self.buffer.len()]
            self.position = 0
        }
        loop {
            let chunk = self.source.read_bytes(self.capacity)
            if chunk.len() == 0 {
                self.finished = true
                if self.carry.len() == 0 { return false }
                # A truncated sequence at end of stream: hand the bytes to the
                # decoder rather than dropping them without a word.
                self.buffer = self.buffer + _decode(self.carry)
                self.carry = _EMPTY_BYTES
                return true
            }
            # `concat`, not `+`: `+` is not defined between two `bytes`.
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

#: A filesystem path as a value, with the operations a path deserves.
#:
#: A `Path` is immutable and holds nothing but its text, so building one never
#: touches the filesystem and never fails. The methods that do touch the
#: filesystem — `exists`, `list`, `glob`, `walk`, `mkdir` and the rest — say so.
#:
#: Every path operation is lexical: `normalize` resolves `.` and `..` by text,
#: without asking whether the components exist and without following symbolic
#: links. That is the behaviour a path calculation wants, and it is why
#: `Path("a/b/../c").normalize()` is `a/c` even when `a/b` is missing.
#:
#: ```
#: let path = Path("/var/log").join("system", "kernel.log")
#: path.to_str()          # "/var/log/system/kernel.log"
#: path.name()            # "kernel.log"
#: path.stem()            # "kernel"
#: path.extension()       # "log"
#: path.parent().to_str() # "/var/log/system"
#: path.is_absolute()     # true
#: ```
pub class Path: Comparable, Hashable, Printable {
    #: The path as written, before any normalisation.
    pub let text: str

    #: Create a path from `text`, which is stored exactly as given.
    pub fn init(self, text: str) { self.text = text }

    #: Return this path with `parts` appended, one component each.
    #:
    #: An empty part is skipped and an absolute part replaces everything before
    #: it, which is what makes `base.join(user_supplied)` behave predictably
    #: when the caller hands over an absolute path.
    #:
    #: ```
    #: Path("logs").join("2024", "may.txt").to_str()    # "logs/2024/may.txt"
    #: Path("logs").join("/etc/passwd").to_str()        # "/etc/passwd"
    #: ```
    pub fn join(self, ...parts: str) -> Path {
        var text = self.text
        for part in parts {
            if part.len() == 0 { continue }
            if _is_absolute_text(part) {
                text = part
            } elif text.len() == 0 or text[text.len() - 1] == SEPARATOR {
                text = text + part
            } else {
                text = text + SEPARATOR + part
            }
        }
        return Path(text)
    }

    #: Return the directory containing this path.
    #:
    #: The parent of a relative name is `.`, the parent of `.` is `..`, and the
    #: root is its own parent, so repeated `parent` calls terminate.
    #:
    #: ```
    #: Path("/var/log/sys.log").parent().to_str()    # "/var/log"
    #: Path("report.txt").parent().to_str()          # "."
    #: Path("/").parent().to_str()                   # "/"
    #: ```
    pub fn parent(self) -> Path {
        let base = if self.text.len() == 0 { "." } else { self.text }
        return Path(base + SEPARATOR + "..").normalize()
    }

    #: Return the final component, or "" for the root and the empty path.
    pub fn name(self) -> str {
        let (start, end) = _name_bounds(self.text)
        return self.text[start:end]
    }

    #: Return the final component without its extension.
    #:
    #: ```
    #: Path("archive.tar.gz").stem()    # "archive.tar"
    #: Path(".bashrc").stem()           # ".bashrc"
    #: ```
    pub fn stem(self) -> str {
        let name = self.name()
        let dot = _extension_dot(name)
        if dot < 0 { return name }
        return name[0:dot]
    }

    #: Return the extension of the final component, without its dot.
    #:
    #: A leading dot is part of the name rather than an extension, so a dotfile
    #: has no extension, and neither does a name ending in a dot.
    #:
    #: ```
    #: Path("archive.tar.gz").extension()    # "gz"
    #: Path(".bashrc").extension()           # ""
    #: Path("Makefile").extension()          # ""
    #: ```
    pub fn extension(self) -> str {
        let name = self.name()
        let dot = _extension_dot(name)
        if dot < 0 { return "" }
        return name[dot + 1:name.len()]
    }

    #: Return this path with its extension replaced by `extension`.
    #:
    #: A leading dot on `extension` is accepted and ignored; an empty
    #: `extension` removes the extension.
    #:
    #: ```
    #: Path("src/main.jai").with_extension("jaic").to_str()    # "src/main.jaic"
    #: Path("src/main.jai").with_extension("").to_str()        # "src/main"
    #: ```
    pub fn with_extension(self, extension: str) -> Path {
        let name = self.name()
        if name.len() == 0 or name == "." or name == ".." {
            throw ValueError(f"{repr(self.text)} has no file name to change the extension of")
        }
        var suffix = extension
        if suffix.len() > 0 and suffix[0] == "." { suffix = suffix[1:suffix.len()] }
        let renamed = if suffix.len() == 0 { self.stem() } else { self.stem() + "." + suffix }
        let (start, _end) = _name_bounds(self.text)
        return Path(self.text[0:start] + renamed)
    }

    #: Whether the path starts at the root.
    pub fn is_absolute(self) -> bool { return _is_absolute_text(self.text) }

    #: Return the path with `.`, empty components and resolvable `..` removed.
    #:
    #: This is pure text manipulation: it does not consult the filesystem, so it
    #: is wrong in the presence of symbolic links, and it is right in every
    #: other case. A `..` at the root is dropped, and a relative path that
    #: cancels out becomes `.`.
    #:
    #: ```
    #: Path("a/b/../c/./d").normalize().to_str()    # "a/c/d"
    #: Path("/../etc").normalize().to_str()         # "/etc"
    #: Path("a/..").normalize().to_str()            # "."
    #: Path("../../x").normalize().to_str()         # "../../x"
    #: ```
    pub fn normalize(self) -> Path {
        let absolute = self.is_absolute()
        var parts: list[str] = []
        for part in _components(self.text) {
            if part != ".." {
                parts.push(part)
                continue
            }
            if parts.len() > 0 and parts[parts.len() - 1] != ".." {
                let _dropped = parts.pop()
                continue
            }
            # `..` above the root has nowhere to go, so the root swallows it.
            if not absolute { parts.push("..") }
        }
        let joined = SEPARATOR.join(parts)
        if absolute { return Path(SEPARATOR + joined) }
        if joined.len() == 0 { return Path(".") }
        return Path(joined)
    }

    #: Return this path written relative to `base`.
    #:
    #: Both paths are normalised first. Raises `ValueError` when `base` is not a
    #: prefix of this path, because the answer would then need `..` components
    #: whose meaning depends on symbolic links this method cannot see.
    #:
    #: ```
    #: Path("/usr/local/bin/jaithon").relative_to(Path("/usr/local")).to_str()
    #: # "bin/jaithon"
    #: ```
    pub fn relative_to(self, base: Path) -> Path {
        let mine = self.normalize()
        let theirs = base.normalize()
        if mine.is_absolute() != theirs.is_absolute() {
            throw ValueError(f"one of {repr(self.text)} and {repr(base.text)} is absolute")
        }
        let parts = _components(mine.text)
        let prefix = _components(theirs.text)
        if prefix.len() > parts.len() { throw _not_a_prefix(self, base) }
        for index in 0..prefix.len() {
            if parts[index] != prefix[index] { throw _not_a_prefix(self, base) }
        }
        let rest = parts[prefix.len():parts.len()]
        if rest.len() == 0 { return Path(".") }
        return Path(SEPARATOR.join(rest))
    }

    #: Whether anything exists at this path.
    pub fn exists(self) -> bool { return __prim__.io_stat(self.text) is not null }

    #: Whether a regular file exists at this path.
    pub fn is_file(self) -> bool { return self._kind() == "file" }

    #: Whether a directory exists at this path.
    pub fn is_dir(self) -> bool { return self._kind() == "dir" }

    #: Whether a file exists at this path with an execute bit set.
    pub fn is_executable(self) -> bool {
        let info = __prim__.io_stat(self.text)
        if info is null { return false }
        if info["kind"] != "file" { return false }
        return info["mode"] & 0o111 != 0
    }

    #: Return the size of the file in bytes.
    #:
    #: Raises `FileNotFoundError` when nothing is there.
    pub fn size(self) -> int { return self._stat()["size"] }

    #: Return when the path was last modified.
    #:
    #: Raises `FileNotFoundError` when nothing is there.
    pub fn modified_time(self) -> DateTime {
        # `DateTime` normalises an out-of-range nanosecond field into the
        # seconds, so the whole timestamp can go in as nanoseconds.
        return DateTime(0, self._stat()["modified"])
    }

    #: Return the entries of this directory, sorted by name.
    #:
    #: `.` and `..` are not included. Raises `IOError` when this is not a
    #: directory.
    pub fn list(self) -> list[Path] {
        return [self.join(name) for name in __prim__.io_listdir(self.text).sorted()]
    }

    #: Return the paths below this directory matching `pattern`, sorted.
    #:
    #: Within one component, `*` matches any run of characters and `?` matches
    #: exactly one; neither crosses a `/`. A whole component of `**` matches any
    #: number of directories, including none. Hidden entries are matched like
    #: any other. The pattern must be relative — it is always interpreted
    #: against this directory.
    #:
    #: ```
    #: let sandbox = Path("/tmp/jaithon-glob-doc")
    #: sandbox.join("nested").mkdir(true, true)
    #: sandbox.join("keep.txt").write_text("")
    #: sandbox.join("nested", "deep.txt").write_text("")
    #: [entry.name() for entry in sandbox.glob("*.txt")]        # ["keep.txt"]
    #: [entry.name() for entry in sandbox.glob("**/*.txt")]     # ["keep.txt", "deep.txt"]
    #: sandbox.remove(true)
    #: ```
    pub fn glob(self, pattern: str) -> list[Path] {
        if _is_absolute_text(pattern) {
            throw ValueError(f"glob pattern must be relative, got {repr(pattern)}")
        }
        var found: list[Path] = []
        _glob_into(self, _components(pattern), 0, found)
        let names = [entry.text for entry in found]
        return [Path(name) for name in names.sorted()]
    }

    #: Iterate every path below this directory, directories before their contents.
    #:
    #: The walk is depth-first and lazy: nothing is read until the iterator is
    #: advanced, so an early `break` costs only the directories already opened.
    #: This path itself is not yielded.
    pub fn walk(self) -> Iterator[Path] { return PathWalk(self) }

    #: Create this directory.
    #:
    #: With `parents`, the missing ancestors are created too. With `exist_ok`,
    #: an existing directory is accepted instead of raising.
    pub fn mkdir(self, parents: bool = false, exist_ok: bool = false) -> void {
        if self.is_dir() {
            if exist_ok { return }
            throw IOError(f"directory already exists: {self.text}")
        }
        if parents {
            let above = self.parent()
            if above.text != self.text and not above.is_dir() { above.mkdir(true, true) }
        }
        __prim__.io_mkdir(self.text)
    }

    #: Delete this path.
    #:
    #: A directory must be empty unless `recursive` is set, in which case its
    #: contents are deleted first. Raises `FileNotFoundError` when nothing is
    #: there.
    pub fn remove(self, recursive: bool = false) -> void {
        let info = __prim__.io_stat(self.text)
        if info is null { throw FileNotFoundError(f"no such file or directory: {self.text}") }
        if info["kind"] == "dir" and recursive {
            for entry in self.list() { entry.remove(true) }
        }
        __prim__.io_remove(self.text)
    }

    #: Copy this path to `target`, recursively for a directory.
    #:
    #: An existing `target` file is overwritten; an existing `target` directory
    #: is merged into. Permissions and timestamps are not carried over, because
    #: the primitive surface cannot set them.
    pub fn copy(self, target: Path) -> void {
        let info = __prim__.io_stat(self.text)
        if info is null { throw FileNotFoundError(f"no such file or directory: {self.text}") }
        if info["kind"] != "dir" {
            _copy_file(self.text, target.text)
            return
        }
        target.mkdir(true, true)
        for entry in self.list() { entry.copy(target.join(entry.name())) }
    }

    #: Move this path to `target`.
    #:
    #: The primitive surface has no rename, so this copies and then deletes: it
    #: is not atomic, and it costs as much as the data it moves. Raises
    #: `IOError` rather than overwriting an existing `target`.
    pub fn rename(self, target: Path) -> void {
        if target.exists() {
            throw IOError(f"cannot rename onto an existing path: {target.text}")
        }
        self.copy(target)
        self.remove(true)
    }

    #: Read the whole file at this path as UTF-8 text.
    pub fn read_text(self) -> str { return read_file(self.text) }

    #: Write `text` to this path as UTF-8, replacing what was there.
    pub fn write_text(self, text: str) -> void { write_file(self.text, text) }

    #: Order paths by their text, so a sorted list groups a directory together.
    pub fn compare(self, other: Path) -> int { return cmp(self.text, other.text) }

    #: Hash the path by its text, so that two paths that compare equal hash
    #: equal and a `Path` can be a dict key.
    pub fn hash(self) -> int { return hash(self.text) }

    #: Return the path as written.
    pub fn to_str(self) -> str { return self.text }

    # The kind of what this path leads to, following a symlink to its target.
    # "is there a directory here" is a question about the target: `/tmp` is a
    # link to `/private/tmp` on macOS, and answering "link" made every caller
    # that writes under it decide the directory was missing. `io_stat` without
    # the flag still reports the link itself, which is what `remove` needs.
    fn _kind(self) -> str {
        let info = __prim__.io_stat(self.text, true)
        if info is null { return "" }
        return info["kind"]
    }

    fn _stat(self) -> dict[str, any] {
        let info = __prim__.io_stat(self.text)
        if info is null { throw FileNotFoundError(f"no such file or directory: {self.text}") }
        return info
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, Path) { return false }
        return self.text == other.text
    }

    fn __hash__(self) -> int { return hash(self.text) }

    fn __str__(self) -> str { return self.text }
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

#: Parse `text` as delimited values, following RFC 4180 quoting.
#:
#: A field may be wrapped in double quotes, inside which a doubled quote is one
#: literal quote and the delimiter and newlines are ordinary characters.
#: Records end at `\n` or `\r\n`, and a terminator at the end of the input does
#: not produce an extra record. A blank line is one record holding one empty
#: field. An unterminated quoted field raises `ParseError`.
#:
#: ```
#: parse_csv("a,b\n1,2\n")                 # [["a", "b"], ["1", "2"]]
#: parse_csv("\"x,y\",\"he said \"\"hi\"\"\"")
#: # [["x,y", "he said \"hi\""]]
#: ```
pub fn parse_csv(text: str, delimiter: str = ",") -> list[list[str]] {
    _check_delimiter(delimiter)
    var rows: list[list[str]] = []
    var row: list[str] = []
    var field: list[str] = []
    var quoted = false
    var started = false
    var index = 0
    while index < text.len() {
        let character = text[index]
        index += 1
        if quoted {
            if character != "\"" {
                field.push(character)
                continue
            }
            if index < text.len() and text[index] == "\"" {
                field.push("\"")
                index += 1
                continue
            }
            quoted = false
            continue
        }
        if character == "\"" and field.len() == 0 {
            quoted = true
            started = true
            continue
        }
        if character == delimiter {
            row.push("".join(field))
            field = []
            started = true
            continue
        }
        if character == "\n" or character == "\r" {
            if character == "\r" and index < text.len() and text[index] == "\n" { index += 1 }
            row.push("".join(field))
            rows.push(row)
            field = []
            row = []
            started = false
            continue
        }
        field.push(character)
        started = true
    }
    if quoted { throw ParseError("unterminated quoted field at the end of the CSV input") }
    if started {
        row.push("".join(field))
        rows.push(row)
    }
    return rows
}

#: Render `rows` as delimited values, quoting the fields that need it.
#:
#: A field is quoted when it contains the delimiter, a double quote, a newline
#: or a carriage return, or when it begins or ends with a space — the cases
#: where an unquoted field would not read back as itself. Every record,
#: including the last, ends with `\n`.
#:
#: ```
#: format_csv([["a", "b"], ["1", "x,y"]])    # "a,b\n1,\"x,y\"\n"
#: format_csv([])                            # ""
#: ```
pub fn format_csv(rows: list[list[str]], delimiter: str = ",") -> str {
    _check_delimiter(delimiter)
    var out: list[str] = []
    for row in rows {
        out.push(delimiter.join([_quote_field(cell, delimiter) for cell in row]))
        out.push("\n")
    }
    return "".join(out)
}

#: Read the file at `path` as delimited values.
pub fn read_csv(path: str, delimiter: str = ",") -> list[list[str]] {
    return parse_csv(read_file(path), delimiter)
}

#: Write `rows` to `path` as delimited values.
pub fn write_csv(path: str, rows: list[list[str]], delimiter: str = ",") -> void {
    write_file(path, format_csv(rows, delimiter))
}

#: Read the file at `path` as tab-separated values.
#:
#: Quoting works as it does for CSV, which is what spreadsheets emit for TSV
#: and the only way a field containing a tab can survive the round trip.
pub fn read_tsv(path: str) -> list[list[str]] { return read_csv(path, "\t") }

#: Write `rows` to `path` as tab-separated values.
pub fn write_tsv(path: str, rows: list[list[str]]) -> void { write_csv(path, rows, "\t") }

# The lines of a reader, as an iterator. It is private because a caller only
# ever meets it as the `Iterator[str]` that `lines` returns.
class LineIter: Iterator[str], Iterable[str] {
    let reader: BufferedReader

    pub fn init(self, reader: BufferedReader) { self.reader = reader }

    pub fn next(self) -> str? { return self.reader.read_line() }

    pub fn iter(self) -> Iterator[str] { return self }
}

# A depth-first walk of a directory tree, holding one stack of pending entries
# rather than the whole listing.
class PathWalk: Iterator[Path], Iterable[Path] {
    let root: Path
    var pending: list[Path]?

    pub fn init(self, root: Path) {
        self.root = root
        self.pending = null
    }

    pub fn next(self) -> Path? {
        # The root is listed on the first `next`, not in `init`, so building
        # the walk costs nothing and reads no directory the caller skips.
        var stack = self.pending
        if stack is null {
            stack = _reversed_listing(self.root)
            self.pending = stack
        }
        if stack.len() == 0 { return null }
        let entry = stack.pop()
        if entry.is_dir() {
            for child in _reversed_listing(entry) { stack.push(child) }
        }
        return entry
    }

    pub fn iter(self) -> Iterator[Path] { return self }
}

fn _reversed_listing(directory: Path) -> list[Path] {
    let entries = directory.list()
    var stacked: list[Path] = []
    var index = entries.len() - 1
    while index >= 0 {
        stacked.push(entries[index])
        index -= 1
    }
    return stacked
}

# Expand `segments` from `index` onwards against `base`, collecting matches.
fn _glob_into(base: Path, segments: list[str], index: int, found: list[Path]) -> void {
    if index == segments.len() {
        found.push(base)
        return
    }
    let segment = segments[index]
    if segment == "**" {
        # `**` matches no directories at all, and then one more at a time. Each
        # match is reached through exactly one chain, so nothing is duplicated.
        _glob_into(base, segments, index + 1, found)
        if not base.is_dir() { return }
        for entry in base.list() {
            if entry.is_dir() { _glob_into(entry, segments, index, found) }
        }
        return
    }
    if not _has_wildcard(segment) {
        # A literal component is looked up rather than listed, so a deep
        # pattern with one wildcard does not read every directory on the way.
        let child = base.join(segment)
        if child.exists() { _glob_into(child, segments, index + 1, found) }
        return
    }
    if not base.is_dir() { return }
    for entry in base.list() {
        if _match_glob(entry.name(), segment) { _glob_into(entry, segments, index + 1, found) }
    }
}

fn _has_wildcard(segment: str) -> bool {
    for index in 0..segment.len() {
        if segment[index] == "*" or segment[index] == "?" { return true }
    }
    return false
}

# Match one path component against one glob segment. The two-pointer walk
# remembers the last `*` and the text it had consumed, so a mismatch retries
# with that `*` one character longer instead of recursing.
fn _match_glob(name: str, pattern: str) -> bool {
    var text = 0
    var spec = 0
    var star = -1
    var resume = 0
    while text < name.len() {
        if spec < pattern.len() and (pattern[spec] == "?" or pattern[spec] == name[text]) {
            text += 1
            spec += 1
        } elif spec < pattern.len() and pattern[spec] == "*" {
            star = spec
            spec += 1
            resume = text
        } elif star >= 0 {
            spec = star + 1
            resume += 1
            text = resume
        } else {
            return false
        }
    }
    while spec < pattern.len() and pattern[spec] == "*" { spec += 1 }
    return spec == pattern.len()
}

fn _copy_file(source_path: str, target_path: str) -> void {
    let source = open(source_path, "r")
    defer { source.close() }
    let target = open(target_path, "w")
    defer { target.close() }
    loop {
        let chunk = source.read_bytes(DEFAULT_BUFFER_SIZE)
        if chunk.len() == 0 { return }
        let _written = target.write_bytes(chunk)
    }
}

fn _not_a_prefix(path: Path, base: Path) -> ValueError {
    return ValueError(f"{repr(base.text)} is not a prefix of {repr(path.text)}")
}

fn _is_absolute_text(text: str) -> bool {
    return text.len() > 0 and text[0] == SEPARATOR
}

# The components of `text`, with empty ones and `.` dropped.
fn _components(text: str) -> list[str] {
    return [part for part in text.split(SEPARATOR) if part.len() > 0 and part != "."]
}

# The half-open span of the last component of `text`, ignoring trailing slashes.
fn _name_bounds(text: str) -> tuple[int, int] {
    var end = text.len()
    while end > 0 and text[end - 1] == SEPARATOR { end -= 1 }
    var start = end
    while start > 0 and text[start - 1] != SEPARATOR { start -= 1 }
    return (start, end)
}

# The index of the dot that starts the extension of `name`, or -1.
fn _extension_dot(name: str) -> int {
    var index = name.len() - 1
    while index > 0 {
        if name[index] == "." {
            if index == name.len() - 1 { return -1 }
            return index
        }
        index -= 1
    }
    return -1
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

fn _check_delimiter(delimiter: str) -> void {
    if delimiter.len() != 1 {
        throw ValueError(f"delimiter must be one character, got {repr(delimiter)}")
    }
    if delimiter == "\"" or delimiter == "\n" or delimiter == "\r" {
        throw ValueError(f"delimiter may not be a quote or a newline, got {repr(delimiter)}")
    }
}

fn _quote_field(cell: str, delimiter: str) -> str {
    if not _needs_quoting(cell, delimiter) { return cell }
    var parts = ["\""]
    for index in 0..cell.len() {
        parts.push(cell[index] == "\"" ? "\"\"" : cell[index])
    }
    parts.push("\"")
    return "".join(parts)
}

fn _needs_quoting(cell: str, delimiter: str) -> bool {
    if cell.len() == 0 { return false }
    if cell[0] == " " or cell[cell.len() - 1] == " " { return true }
    for index in 0..cell.len() {
        let character = cell[index]
        if character == delimiter or character == "\"" { return true }
        if character == "\n" or character == "\r" { return true }
    }
    return false
}

# Split `data` before a trailing UTF-8 sequence the buffer does not yet hold in
# full, so a chunk boundary never lands inside a character. A continuation byte
# is 0x80..0xBF and a lead byte says how many bytes the sequence takes.
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
