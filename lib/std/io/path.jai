#: A filesystem path as a value: joining, normalising, globbing and walking.
from std.core import Comparable, Hashable, Iterable, Iterator, Printable, cmp
from std.time import DateTime
from .stream import DEFAULT_BUFFER_SIZE, open, read_file, write_file

#: The path separator, which is `/` on every platform this module supports.
pub const SEPARATOR = "/"

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

class PathWalk: Iterator[Path], Iterable[Path] {
    let root: Path
    var pending: list[Path]?

    pub fn init(self, root: Path) {
        self.root = root
        self.pending = null
    }

    pub fn next(self) -> Path? {
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

fn _glob_into(base: Path, segments: list[str], index: int, found: list[Path]) -> void {
    if index == segments.len() {
        found.push(base)
        return
    }
    let segment = segments[index]
    if segment == "**" {
        _glob_into(base, segments, index + 1, found)
        if not base.is_dir() { return }
        for entry in base.list() {
            if entry.is_dir() { _glob_into(entry, segments, index, found) }
        }
        return
    }
    if not _has_wildcard(segment) {
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

fn _components(text: str) -> list[str] {
    return [part for part in text.split(SEPARATOR) if part.len() > 0 and part != "."]
}

fn _name_bounds(text: str) -> tuple[int, int] {
    var end = text.len()
    while end > 0 and text[end - 1] == SEPARATOR { end -= 1 }
    var start = end
    while start > 0 and text[start - 1] != SEPARATOR { start -= 1 }
    return (start, end)
}

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
