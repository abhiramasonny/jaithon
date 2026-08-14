#: An imported module's signatures, for the checker: `src/sema/modsig.c`.
#:
#: Modules resolve at run time (spec §8), so cross-module calls reach the front
#: end unchecked. `SigLoader` parses a module's source for signatures alone; no
#: body is checked. A `.jaic` cannot serve: a warm cache means no AST exists.
from jaithon.ast import Node, NodeKind, Span
from jaithon.compile.parser import parse

#: The extension a module's own file carries: `std.gui` is `std/gui.jai`.
pub const MODULE_EXT: str = ".jai"

#: The file that stands in for a directory-shaped module: `std/gui/mod.jai`.
pub const PACKAGE_FILE: str = "mod.jai"

#: The marker file for a workspace package whose import root is `src`.
pub const PROJECT_MANIFEST: str = "jaithon.package.json"

#: How many re-exports `SigLoader.find` follows. `SIG_MAX_REEXPORT_DEPTH` in C.
pub const MAX_REEXPORT_DEPTH: int = 8

#: One top-level name a module declares, and the node that declares it.
pub class SigDecl {
    #: The name as written.
    pub let name: str
    #: The `FnDecl`, `ClassDecl`, `TraitDecl`, `EnumDecl` or `TypeDecl` node.
    pub let decl: Node

    pub fn init(self, name: str, decl: Node) {
        self.name = name
        self.decl = decl
    }
}

#: A name a module re-exports rather than declares, via a top-level `from`.
pub class SigReexport {
    #: The name this module offers.
    pub let local: str
    #: The name the other module declares, which differs under an alias.
    pub let remote: str
    #: The dotted module name, as written. Spelled `source` because `from` is a
    #: reserved word; the C field is `from`.
    pub let source: str

    pub fn init(self, local: str, remote: str, source: str) {
        self.local = local
        self.remote = remote
        self.source = source
    }
}

#: One module a module imports. `ImportEdge` in `src/runtime/modules/module.c`: a plain
#: `import m` binds no name, so only this records the edge that can close a cycle.
pub class SigImport {
    #: The dotted module name, as written.
    pub let dotted: str
    #: The import statement, for the diagnostic that names the cycle.
    pub let span: Span

    pub fn init(self, dotted: str, span: Span) {
        self.dotted = dotted
        self.span = span
    }
}

#: One module's signatures: what it declares, and what it passes along.
pub class ModuleSig {
    #: The resolved file, absolute and normalised. The loader's cache key.
    pub let path: str
    #: The dotted name it was first loaded under, e.g. `"std.gui"`.
    pub let dotted: str
    #: The id its spans carry, unique within one loader.
    pub let file_id: int
    #: The parsed tree, or null when the file did not parse.
    pub var program: Node?
    #: Every top-level declaration, in source order.
    pub let decls: list[SigDecl]
    #: The class, trait and enum subset of `decls`, in source order. A `TypeDecl`
    #: is deliberately absent: an alias introduces no type of its own.
    pub let types: list[SigDecl]
    #: Names re-exported by a `from ... import ...` at the top level.
    pub let reexports: list[SigReexport]
    #: Every module this one imports, in source order.
    pub let imports: list[SigImport]
    #: Modules re-exported wholesale by a `from ... import *`.
    pub let wildcards: list[str]

    pub fn init(self, path: str, dotted: str, file_id: int) {
        self.path = path
        self.dotted = dotted
        self.file_id = file_id
        self.program = null
        self.decls = []
        self.types = []
        self.reexports = []
        self.imports = []
        self.wildcards = []
    }

    #: `"std.gui.Window"`: the name an imported type is interned under, so two
    #: modules' `Node`s differ. Leading dots say nothing about identity: dropped.
    pub fn qualify(self, name: str) -> str {
        var index = 0
        while index < self.dotted.len() and self.dotted[index] == "." { index += 1 }
        let stripped = self.dotted[index:]
        if stripped == "" { return f"{self.dotted}.{name}" }
        return f"{stripped}.{name}"
    }
}

#: Every module signature one checker loaded, not the C's file-static vector.
pub class SigLoader {
    #: `jaiResolveModulePath`'s `sUserDirs` and `sLibDirs`, concatenated.
    pub let search: list[str]

    var _cache: dict[str, ModuleSig]
    var _by_file: dict[int, ModuleSig]
    var _next_file_id: int
    var _visiting: list[str]

    #: Build a loader. `first_file_id` leaves 0 to the file being checked.
    pub fn init(self, search: list[str] = [], first_file_id: int = 1) {
        self.search = search
        self._cache = {}
        self._by_file = {}
        self._next_file_id = first_file_id
        self._visiting = []
    }

    #: Load, or return the cached, signature of the module `dotted` names, from
    #: the directory of the file that wrote the import. Null, silently, when it
    #: cannot be found. A file that fails to parse is still cached, empty.
    pub fn load(self, dotted: str, from_dir: str = "") -> ModuleSig? {
        if dotted == "" { return null }
        let resolved = resolve_module_path(dotted, from_dir, self.search)
        if resolved is null { return null }
        if resolved in self._cache { return self._cache[resolved] }

        let text = _read_source(resolved)
        if text is null { return null }

        let sig = ModuleSig(resolved, dotted, self._next_file_id)
        self._next_file_id += 1

        let program = parse(text, sig.file_id)
        sig.program = program
        if program != null and program.kind == NodeKind.Program {
            _collect(sig, program.child_list("stmts"))
        }

        self._cache[resolved] = sig
        self._by_file[sig.file_id] = sig
        return sig
    }

    #: The declaration `name` denotes in `sig`, and the module that declares it —
    #: its annotations are read in that namespace, not the re-exporter's.
    pub fn find(self, sig: ModuleSig, name: str) -> tuple[Node?, ModuleSig] {
        let found = self._find_in(sig, name, MAX_REEXPORT_DEPTH)
        let owner = found[1]
        if owner is null { return (null, sig) }
        return (found[0], owner)
    }

    #: The signature parsed from `file_id`, or null when that file is not one.
    #: Spans carry their file, so an indirectly reached declaration traces back.
    pub fn for_file(self, file_id: int) -> ModuleSig? {
        if file_id in self._by_file { return self._by_file[file_id] }
        return null
    }

    #: Every signature loaded so far, in load order.
    pub fn loaded(self) -> list[ModuleSig] {
        return [self._by_file[given] for given in self._by_file.keys().sorted()]
    }

    fn _find_in(self, sig: ModuleSig, name: str, depth: int) -> tuple[Node?, ModuleSig?] {
        if depth <= 0 or sig.path in self._visiting { return (null, null) }
        for entry in sig.decls {
            if entry.name == name { return (entry.decl, sig) }
        }
        self._visiting.push(sig.path)
        let found = self._follow(sig, name, depth)
        let _left = self._visiting.pop()
        return found
    }

    fn _follow(self, sig: ModuleSig, name: str, depth: int) -> tuple[Node?, ModuleSig?] {
        let here = _dirname(sig.path)
        for entry in sig.reexports {
            if entry.local != name { continue }
            let next = self.load(entry.source, here)
            if next is null { continue }
            let found = self._find_in(next, entry.remote, depth - 1)
            if found[0] is not null { return found }
        }
        for wildcard in sig.wildcards {
            let next = self.load(wildcard, here)
            if next is null { continue }
            let found = self._find_in(next, name, depth - 1)
            if found[0] is not null { return found }
        }
        return (null, null)
    }
}

#: Split a dotted module name into its leading dots and a relative path.
#: `splitModuleName` in src/runtime/modules/module.c. `..` and path separators are
#: refused, so a name out of source text cannot escape a search directory.
pub fn split_module_name(dotted: str) -> tuple[int, str]? {
    if dotted == "" { return null }
    var dots = 0
    while dots < dotted.len() and dotted[dots] == "." { dots += 1 }
    if dots >= dotted.len() { return null }

    let parts = dotted[dots:].split(".")
    for part in parts {
        if part == "" { return null }
        for character in part {
            if not _is_name_char(character) { return null }
        }
    }
    return (dots, "/".join(parts))
}

#: Resolve `dotted` to a file the way `jaiResolveModulePath` does, minus its
#: reporting: `<dir>/std/gui.jai` wins over the package `<dir>/std/gui/mod.jai`.
pub fn resolve_module_path(dotted: str, from_dir: str = "", search: list[str] = []) -> str? {
    let split = split_module_name(dotted)
    if split is null { return null }
    let dots = split[0]
    let relative = split[1]

    if dots > 0 {
        let base = _relative_base(from_dir, dots)
        if base is null { return null }
        return _try_directory(base, relative)
    }

    if from_dir != "" {
        let here = _try_directory(from_dir, relative)
        if here is not null { return here }
    }
    for dir in search {
        let found = _try_directory(dir, relative)
        if found is not null { return found }
    }
    return null
}

#: The search path `jaiModulePathInit` would build. `exec_dir` must be supplied,
#: since Jaithon cannot reach the interpreter's path, or `./lib` is missed.
pub fn default_search_path(exec_dir: str? = null) -> list[str] {
    var dirs: list[str] = []

    let configured = __prim__.os_env("JAITHON_PATH")
    if configured != null {
        for entry in configured.split(":") { _add_dir(dirs, entry) }
    }
    if exec_dir != null and exec_dir != "" {
        for suffix in ["lib", "../lib", "../share/jaithon/lib", "../share/jaithon"] {
            _add_lib_dir(dirs, _join(exec_dir, suffix))
        }
        for suffix in ["packages", "../packages", "../share/jaithon/packages"] {
            _add_package_dirs(dirs, _join(exec_dir, suffix))
        }
    }
    for fixed in [
        "/usr/local/share/jaithon/lib",
        "/usr/local/share/jaithon",
        "/opt/homebrew/share/jaithon/lib",
        "/opt/homebrew/share/jaithon",
    ] {
        _add_lib_dir(dirs, fixed)
    }
    for fixed in ["/usr/local/share/jaithon/packages", "/opt/homebrew/share/jaithon/packages"] {
        _add_package_dirs(dirs, fixed)
    }
    return dirs
}

fn _collect(sig: ModuleSig, stmts: list[Node]) -> void {
    for node in stmts {
        match node.kind {
            NodeKind.FnDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.ClassDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TraitDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.EnumDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TypeDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.Import => { _add_edge(sig, node) }
            NodeKind.FromImport => { _add_import(sig, node) }
            _ => { }
        }
    }
}

fn _add_edge(sig: ModuleSig, node: Node) -> void {
    let source = node.opt_text("path")
    if source is null { return }
    sig.imports.push(SigImport(source, node.span))
}

fn _add_decl(sig: ModuleSig, name: str?, decl: Node, is_type: bool) -> void {
    if name is null { return }
    let entry = SigDecl(name, decl)
    sig.decls.push(entry)
    if is_type { sig.types.push(entry) }
}

fn _add_import(sig: ModuleSig, node: Node) -> void {
    let source = node.opt_text("path")
    if source is null { return }
    sig.imports.push(SigImport(source, node.span))
    if node.flag("isWildcard") {
        sig.wildcards.push(source)
        return
    }
    for item in node.records("items") {
        let name: str? = item["name"]
        if name is null { continue }
        let alias: str? = item["alias"]
        sig.reexports.push(SigReexport(alias ?? name, name, source))
    }
}

fn _try_directory(dir: str, relative: str) -> str? {
    let own = _join(dir, relative + MODULE_EXT)
    if _is_regular_file(own) { return _absolute(own) }
    let package = _join(dir, relative + "/" + PACKAGE_FILE)
    if _is_regular_file(package) { return _absolute(package) }
    return null
}

fn _relative_base(from_dir: str, dots: int) -> str? {
    var base = _absolute(if from_dir == "" { "." } else { from_dir })
    var level = 1
    while level < dots {
        let parent = _dirname(base)
        if parent == "" or parent == base { return null }
        base = _absolute(parent)
        level += 1
    }
    return base
}

fn _add_dir(dirs: list[str], dir: str) -> void {
    if dir == "" { return }
    let entry = _absolute(dir)
    if entry in dirs { return }
    dirs.push(entry)
}

fn _add_lib_dir(dirs: list[str], dir: str) -> void {
    if dir == "" or not _is_dir(dir) { return }
    _add_dir(dirs, dir)
}

fn _add_package_dirs(dirs: list[str], packages_dir: str) -> void {
    if not _is_dir(packages_dir) { return }
    for name in __prim__.io_listdir(packages_dir).sorted() {
        if name.starts_with(".") { continue }
        let package = _join(packages_dir, name)
        let manifest = _join(package, PROJECT_MANIFEST)
        let source = _join(package, "src")
        if _is_regular_file(manifest) { _add_lib_dir(dirs, source) }
    }
}

fn _is_name_char(character: str) -> bool {
    let point = ord(character)
    if point >= 0x80 { return true }
    if point >= 0x61 and point <= 0x7a { return true }  # a-z
    if point >= 0x41 and point <= 0x5a { return true }  # A-Z
    if point >= 0x30 and point <= 0x39 { return true }  # 0-9
    return point == 0x5f                                # _
}

fn _join(left: str, right: str) -> str {
    let head = if right.starts_with("/") { "" } else { left }
    var raw = head
    if head != "" and right != "" {
        raw = head + "/" + right
    } elif head == "" {
        raw = right
    }

    var out = ""
    for character in raw {
        if character == "/" and out.ends_with("/") { continue }
        out += character
    }
    while out.len() > 1 and out.ends_with("/") { out = out[:out.len() - 1] }
    return out
}

fn _dirname(path: str) -> str {
    var length = path.len()
    let original = length
    while length > 0 and path[length - 1] == "/" { length -= 1 }
    if length == 0 { return if original > 0 { "/" } else { "." } }
    while length > 0 and path[length - 1] != "/" { length -= 1 }
    if length == 0 { return "." }
    while length > 1 and path[length - 1] == "/" { length -= 1 }
    return path[:length]
}

#: As much of `jaiPathAbsolute` as there is a primitive for: `.` and `..` folded
#: lexically, no symlinks. Public so callers normalise `ModuleSig.path` alike.
pub fn absolute(path: str) -> str {
    return _absolute(path)
}

#: The path without its last component, `.` when there is none. `jaiPathDirname`.
pub fn dirname(path: str) -> str {
    return _dirname(path)
}

fn _absolute(path: str) -> str {
    let rooted = if path.starts_with("/") { path } else { _join(__prim__.os_cwd(), path) }
    var parts: list[str] = []
    for part in rooted.split("/") {
        if part == "" or part == "." { continue }
        if part == ".." {
            if parts.len() > 0 { let _dropped = parts.pop() }
            continue
        }
        parts.push(part)
    }
    return "/" + "/".join(parts)
}

fn _is_regular_file(path: str) -> bool {
    if path == "" { return false }
    let info = __prim__.io_stat(path, true)
    if info is null { return false }
    return info["kind"] != "dir"
}

fn _is_dir(path: str) -> bool {
    if path == "" { return false }
    let info = __prim__.io_stat(path, true)
    if info is null { return false }
    return info["kind"] == "dir"
}

fn _read_source(path: str) -> str? {
    try {
        let handle = __prim__.io_open(path, "rb")
        let raw = __prim__.io_read(handle, -1)
        __prim__.io_close(handle)
        return __prim__.str_decode(raw)
    } catch _failure: Error {
        return null
    }
}
