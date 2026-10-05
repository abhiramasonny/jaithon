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
    #: Whether everything above is filled in. False only between
    #: `SigLoader.load_edges` answering from a kept signature -- `path` and
    #: `imports` are then exact -- and the first `SigLoader.load` of it.
    pub var parsed: bool

    #: Until `parsed`: the source's bytes, and the byte ranges of the bodies
    #: a kept signature says to blank.
    var _raw: any
    var _ranges: list[tuple[int, int]]

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
        self.parsed = true
        self._raw = null
        self._ranges = []
    }

    #: Fill in everything from `raw`, the module's bytes, and `text`, them
    #: decoded: parse, keep the signature on disk, collect, strip.
    pub fn parse_source(self, raw: any, text: str) -> void {
        let program = parse(text, self.file_id)
        if program != null and program.kind == NodeKind.Program and _sig_cache_on() {
            _keep_signature(self.path, raw, program)
        }
        self._fill(program)
    }

    #: Take `path` and `imports` from a kept signature and defer the rest
    #: until `ensure_parsed`. `edges` holds `(start, end, dotted)` per import;
    #: `ranges` the bodies to blank before parsing.
    pub fn defer_parse(
        self,
        raw: any,
        edges: list[tuple[int, int, str]],
        ranges: list[tuple[int, int]]
    ) {
        for (start, end, dotted) in edges {
            self.imports.push(SigImport(dotted, Span(self.file_id, start, end)))
        }
        self._raw = raw
        self._ranges = ranges
        self.parsed = false
    }

    #: Parse what `defer_parse` held back: the source with its bodies blanked,
    #: or the whole source when that does not parse here.
    pub fn ensure_parsed(self) -> void {
        if self.parsed { return }
        let raw = self._raw
        let ranges = self._ranges
        self._raw = null
        self._ranges = []
        self.parsed = true
        while self.imports.len() > 0 { let _dropped = self.imports.pop() }

        var program: Node? = null
        let blanked = _blank(raw, ranges)
        if blanked is not null { program = parse(blanked, self.file_id) }
        if program is not null and program.kind == NodeKind.Program {
            self._fill(program)
            return
        }
        let source = _decode(raw)
        if source is null {
            self._fill(null)
            return
        }
        self.parse_source(raw, source)
    }

    fn _fill(self, program: Node?) -> void {
        self.program = program
        if program != null and program.kind == NodeKind.Program {
            _collect(self, program.child_list("stmts"))
            if not _KEEP_SIG_BODIES { _strip_bodies(program.child_list("stmts")) }
        }
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

#: Whether loaders share the signatures they parse. `JAITHON_SIG_SHARE=0`
#: gives every loader its own parse of every module, as before.
#:
#: A loader is built per search path, and the search path is the checked
#: file's directory and every directory above it, so each directory of a tree
#: got its own loader -- and each loader parsed the whole import closure of
#: that directory again. Checking lib/jaithon (13 directories) lexed and parsed
#: about 3.4 times as much source as the tree holds. A signature depends only
#: on the file it was parsed from and the name it was loaded under, so any
#: loader that resolves the same pair can take the same `ModuleSig`.
let _SHARE_SIGS: bool = (__prim__.os_env("JAITHON_SIG_SHARE") ?? "1") != "0"

#: Signatures parsed by any loader, keyed by `_shared_key(path, dotted)`.
var _shared_sigs: dict[str, ModuleSig] = {}

#: The next file id any loader hands out while signatures are shared. One
#: counter for all of them: a loader that adopts another's signature adopts
#: its file id too, and spans carry those ids, so no two loaders may ever hand
#: out the same one.
var _next_shared_file_id: int = 1

fn _shared_key(path: str, dotted: str) -> str {
    return path + "\n" + dotted
}

#: Forget every shared signature, so the next load re-reads from disk.
pub fn clear_shared_signatures() -> void {
    _shared_sigs = {}
}

#: Every module signature one checker loaded, not the C's file-static vector.
pub class SigLoader {
    #: `jaiResolveModulePath`'s `sUserDirs` and `sLibDirs`, concatenated.
    pub let search: list[str]

    var _cache: dict[str, ModuleSig]
    var _by_file: dict[int, ModuleSig]
    var _resolved: dict[str, dict[str, str]]
    var _next_file_id: int
    var _visiting: list[str]
    var _partial: bool

    #: Build a loader. `first_file_id` leaves 0 to the file being checked.
    pub fn init(self, search: list[str] = [], first_file_id: int = 1) {
        self.search = search
        self._cache = {}
        self._by_file = {}
        self._resolved = {}
        self._next_file_id = first_file_id
        self._visiting = []
        self._partial = false
    }

    #: Load, or return the cached, signature of the module `dotted` names, from
    #: the directory of the file that wrote the import. Null, silently, when it
    #: cannot be found. A file that fails to parse is still cached, empty.
    pub fn load(self, dotted: str, from_dir: str = "") -> ModuleSig? {
        let sig = self.load_edges(dotted, from_dir)
        if sig is not null { sig.ensure_parsed() }
        return sig
    }

    #: `load` for a caller that reads only `path` and `imports` -- the import
    #: walk, which visits a module's whole closure to find a cycle and reads
    #: nothing else. From a kept signature that is two small reads and no
    #: parse; `load` parses it when something asks for more. File ids are
    #: handed out here, so they come out in the same order either way.
    pub fn load_edges(self, dotted: str, from_dir: str = "") -> ModuleSig? {
        if dotted == "" { return null }
        let resolved = self._resolve(dotted, from_dir)
        if resolved is null { return null }
        if resolved in self._cache { return self._cache[resolved] }

        let key = _shared_key(resolved, dotted)
        if _SHARE_SIGS and key in _shared_sigs {
            let shared = _shared_sigs[key]
            self._cache[resolved] = shared
            self._by_file[shared.file_id] = shared
            return shared
        }

        let raw = _read_bytes(resolved)
        if raw is null { return null }
        let kept = if _sig_cache_on() { _read_kept(resolved, raw) } else { null }
        var text: str? = null
        if kept is null {
            text = _decode(raw)
            if text is null { return null }
        }

        let sig = ModuleSig(resolved, dotted, self._take_file_id())
        if kept is not null {
            sig.defer_parse(raw, kept.edges, kept.ranges)
        } else {
            sig.parse_source(raw, text ?? "")
        }

        self._cache[resolved] = sig
        self._by_file[sig.file_id] = sig
        if _SHARE_SIGS { _shared_sigs[key] = sig }
        return sig
    }

    fn _take_file_id(self) -> int {
        var taken = self._next_file_id
        if _SHARE_SIGS {
            if _next_shared_file_id > taken { taken = _next_shared_file_id }
            _next_shared_file_id = taken + 1
        }
        self._next_file_id = taken + 1
        return taken
    }

    #: `resolve_module_path` for this loader's search path, remembered. Every
    #: `load` resolves before it can consult the cache, and resolving walks the
    #: search path with two `stat` calls per directory: those calls were 5% of
    #: a `check`. Only a hit is remembered, so a module that appears later is
    #: still found.
    #:
    #: Keyed by directory and then by name, not by the two joined: `load` runs
    #: for every edge of every import walk, and a joined key was a string built
    #: and thrown away each time.
    fn _resolve(self, dotted: str, from_dir: str) -> str? {
        var known = self._resolved.get(from_dir, null)
        if known is null {
            known = {}
            self._resolved[from_dir] = known
        }
        if dotted in known { return known[dotted] }
        let found = resolve_module_path(dotted, from_dir, self.search)
        if found is not null { known[dotted] = found }
        return found
    }

    #: The declaration `name` denotes in `sig`, and the module that declares it —
    #: its annotations are read in that namespace, not the re-exporter's.
    pub fn find(self, sig: ModuleSig, name: str) -> tuple[Node?, ModuleSig] {
        self._partial = false
        let found = self._find_in(sig, name, MAX_REEXPORT_DEPTH)
        let owner = found[1]
        if owner is null { return (null, sig) }
        return (found[0], owner)
    }

    #: Whether the last `find` searched a surface it could not see all of: a
    #: module that did not parse, a re-export whose source is missing, a cycle,
    #: or the re-export budget running out. A null decl means "absent" only
    #: when this is false; otherwise the loader simply did not get to look.
    pub fn partial(self) -> bool {
        return self._partial
    }

    #: The signature parsed from `file_id`, or null when that file is not one.
    #: Spans carry their file, so an indirectly reached declaration traces back.
    pub fn for_file(self, file_id: int) -> ModuleSig? {
        if file_id not in self._by_file { return null }
        let sig = self._by_file[file_id]
        sig.ensure_parsed()
        return sig
    }

    #: Every signature loaded so far, in load order.
    pub fn loaded(self) -> list[ModuleSig] {
        var out: list[ModuleSig] = []
        for given in self._by_file.keys().sorted() {
            let sig = self._by_file[given]
            sig.ensure_parsed()
            out.push(sig)
        }
        return out
    }

    fn _find_in(self, sig: ModuleSig, name: str, depth: int) -> tuple[Node?, ModuleSig?] {
        if depth <= 0 or sig.path in self._visiting {
            self._partial = true
            return (null, null)
        }
        sig.ensure_parsed()
        if sig.program is null { self._partial = true }
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
            if next is null {
                self._partial = true
                continue
            }
            let found = self._find_in(next, entry.remote, depth - 1)
            if found[0] is not null { return found }
        }
        for wildcard in sig.wildcards {
            let next = self.load(wildcard, here)
            if next is null {
                self._partial = true
                continue
            }
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

#: Whether a loaded signature keeps its functions' bodies. `JAITHON_SIG_BODIES=1`
#: keeps them, as before.
let _KEEP_SIG_BODIES: bool = (__prim__.os_env("JAITHON_SIG_BODIES") ?? "0") != "0"

#: A signature is read for what its functions take and return, never for what
#: they do -- the one question asked of a body is whether there is one, which
#: is what makes a method abstract. So each top-level function's and method's
#: block is emptied once the declarations are collected. The whole file was
#: still parsed, so a file that does not parse answers exactly as before; what
#: changes is that a loaded module no longer keeps every statement it holds
#: alive for the rest of the run, for each collection to trace again. Peak
#: memory of `check lib/jaithon` went from 237 MB to 138 MB.
fn _strip_bodies(stmts: list[Node]) -> void {
    for node in stmts {
        if node.kind == NodeKind.FnDecl {
            _strip_body(node)
        } elif node.kind == NodeKind.ClassDecl {
            for method in node.child_list("methods") { _strip_body(method) }
            for method in node.child_list("getters") { _strip_body(method) }
            for method in node.child_list("setters") { _strip_body(method) }
        } elif node.kind == NodeKind.TraitDecl or node.kind == NodeKind.EnumDecl {
            for method in node.child_list("methods") { _strip_body(method) }
        }
    }
}

fn _strip_body(fn_node: Node) -> void {
    let body = fn_node.child("body")
    if body is null or body.kind != NodeKind.Block { return }
    let empty: list[Node] = []
    body.set("stmts", empty)
}

fn _collect(sig: ModuleSig, stmts: list[Node]) -> void {
    for node in stmts {
        match node.kind {
            NodeKind.FnDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.ClassDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TraitDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.EnumDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TypeDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.VarDecl => { _add_var_decl(sig, node) }
            NodeKind.Import => {
                _add_edge(sig, node)
                _add_import_decl(sig, node)
            }
            NodeKind.FromImport => { _add_import(sig, node) }
            _ => { }
        }
    }
}

#: A top-level `import x.y as z` (or unaliased `import x.y`) binds `z` (or
#: `y`) in the module's own scope same as any other top-level statement, and
#: the runtime module object exposes every one of those -- `import .fft as
#: fft` inside `jainum/mod.jai` is the only reason `nn.fft.rfft(...)` resolves
#: after `import jainum as nn`. Without this, `find()` cannot see `fft` as a
#: member of `jainum` at all, and `_module_member`/`_bind_imported` would
#: report it unknown even though it genuinely is one.
fn _add_import_decl(sig: ModuleSig, node: Node) -> void {
    let path = node.opt_text("path")
    if path is null { return }
    _add_decl(sig, node.opt_text("alias") ?? _tail_of(path), node, false)
}

fn _tail_of(dotted: str) -> str {
    let cut = dotted.rfind(".")
    if cut < 0 { return dotted }
    return dotted[cut + 1:]
}

#: A top-level `let`/`const`/`var`, e.g. `pub const PI: float = 3.14159...`.
#: A destructuring pattern (`pub let (WIDTH, HEIGHT) = (640, 480)`) binds every
#: one of its names at module scope and the runtime exports every one of them,
#: so all of them are recorded: leaving them out makes `find()` miss a name the
#: module genuinely has, which is now an error rather than silence.
fn _add_var_decl(sig: ModuleSig, node: Node) -> void {
    _add_pattern_names(sig, node.child("pattern"), node)
}

fn _add_pattern_names(sig: ModuleSig, pattern: Node?, node: Node) -> void {
    if pattern is null { return }
    if pattern.kind == NodeKind.PatBind {
        _add_decl(sig, pattern.opt_text("name"), node, false)
        return
    }
    if pattern.kind != NodeKind.PatTuple and pattern.kind != NodeKind.PatList { return }
    for element in pattern.child_list("elems") {
        _add_pattern_names(sig, element, node)
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

    #: Collapsing the slashes with `out += character` made this quadratic in
    #: the path length, and it is called for every module path the loader
    #: considers: an attribution run put 73% of a probe in this one function.
    #: Splitting and rejoining is the same answer -- verified over every pair
    #: of sixteen path shapes, zero mismatches -- in one pass.
    var parts: list[str] = []
    for piece in raw.split("/") {
        if piece != "" { parts.push(piece) }
    }
    var out = "/".join(parts)
    if raw.starts_with("/") { out = "/" + out }
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

pub fn read_source(path: str) -> str? {
    let raw = _read_bytes(path)
    if raw is null { return null }
    return _decode(raw)
}

fn _read_bytes(path: str) -> any {
    try {
        let handle = __prim__.io_open(path, "rb")
        let raw = __prim__.io_read(handle, -1)
        __prim__.io_close(handle)
        return raw
    } catch _failure: Error {
        return null
    }
}

fn _decode(raw: any) -> str? {
    try {
        return __prim__.str_decode(raw)
    } catch _failure: Error {
        return null
    }
}

# ------------------------------------------------------------------
# The signature cache
# ------------------------------------------------------------------

#: Whether a signature outlives the process that parsed it.
#: `JAITHON_SIG_CACHE=0` parses every imported module from source, as before;
#: `--no-cache` sets it, because that flag promises everything is compiled from
#: source. `JAITHON_SIG_BODIES=1` turns it off too, since what is kept has no
#: bodies to keep.
#:
#: What is kept is the module's import edges, the byte range of every body
#: `_strip_bodies` would empty, and an exact copy of the source it was made
#: from. The import walk reads the edges and nothing else, so most of a closure
#: is never parsed at all. A module something asks more of is parsed with
#: those bodies overwritten by spaces, newlines left where they were: every
#: byte offset -- so every span -- is unchanged, and the tree is the one
#: `_strip_bodies` leaves, at a fraction of the cost (bodies are 63% of
#: lib/jaithon's bytes). The parser that reads it is the one running, so
#: nothing about the tree's shape is frozen into the file.
#:
#: The key is the source itself, compared byte for byte -- a memcmp, cheaper
#: than any hash and with nothing to collide -- plus the build, the front end
#: (`JAITHON_FRONT_END_ID`, which the runtime derives from the seed: a reseed
#: moves it where the build id does not) and `_SIG_CACHE_VERSION`. With no
#: front-end identity -- the compiler running from anything but the binary's
#: seed -- nothing is read or written. A signature depends on its own module's bytes and
#: nothing else: a re-export is recorded by name and followed through that
#: module's own signature. One is written only for a module that parsed, so
#: whether a module parses is always the full parser's answer: a module with
#: an error anywhere, a body included, is re-read from source every time and
#: still offers nothing. A mismatch, a short or damaged file, a blanked text
#: this parser cannot read, or a directory that cannot be written is a silent
#: miss.
let _SIG_CACHE: bool =
    (__prim__.os_env("JAITHON_SIG_CACHE") ?? "1") != "0" and not _KEEP_SIG_BODIES

#: Bump when what a kept text means changes: which bodies are blanked, or the
#: parser's idea of where a body ends.
const _SIG_CACHE_VERSION: int = 1

const _SIG_CACHE_DIR: str = "__jaicache__"
const _SIG_CACHE_EXT: str = ".jsig"

const _BYTE_LF: int = 10
const _BYTE_LBRACE: int = 123
const _BYTE_RBRACE: int = 125

#: The text a kept signature is parsed from for `source`, or null when
#: `source` does not parse. Parsing it gives the tree `SigLoader.load` keeps.
pub fn signature_text(source: str, file_id: int = 1) -> str? {
    let program = parse(source, file_id)
    if program is null or program.kind != NodeKind.Program { return null }
    return _blank(__prim__.str_encode(source), _body_ranges(program.child_list("stmts")))
}

#: Whether signatures are being kept on disk in this process.
pub fn signature_cache_enabled() -> bool {
    return _sig_cache_on()
}

#: The front end's identity, read the first time it is needed: the runtime
#: exports it once the front end has loaded, which is after this module's own
#: top level ran. "" when there is none to key on.
var _front_end_id: str? = null

fn _front_end() -> str {
    let known = _front_end_id
    if known is not null { return known }
    var found = __prim__.os_env("JAITHON_FRONT_END_ID") ?? ""
    if found == "none" or found.contains(" ") { found = "" }
    _front_end_id = found
    return found
}

fn _sig_cache_on() -> bool {
    return _SIG_CACHE and _front_end() != ""
}

#: `src/__jaicache__/util.jsig` for `src/util.jai`, beside its `.jaic`.
fn _kept_path(path: str) -> str {
    let cut = path.rfind("/")
    let dir = if cut < 0 { "." } else { path[:cut] }
    var base = path[cut + 1:]
    if base.ends_with(MODULE_EXT) { base = base[:base.len() - MODULE_EXT.len()] }
    return f"{dir}/{_SIG_CACHE_DIR}/{base}{_SIG_CACHE_EXT}"
}

#: What a kept signature holds, once it is known to belong to the source.
class _Kept {
    #: `(start, end, dotted)` for each import, in source order.
    pub let edges: list[tuple[int, int, str]]
    #: `(start, end)` of each body to blank, in source order.
    pub let ranges: list[tuple[int, int]]

    pub fn init(self, edges: list[tuple[int, int, str]], ranges: list[tuple[int, int]]) {
        self.edges = edges
        self.ranges = ranges
    }
}

#: The kept signature for `raw`, the bytes just read from `path`, or null.
#:
#: The file is one line `jsig VERSION BUILD FRONT_END LENGTH META_LENGTH META_CRC`, then
#: META_LENGTH bytes of `E start end dotted` and `B start end` lines, then the
#: LENGTH bytes of source it was made from.
fn _read_kept(path: str, raw: any) -> _Kept? {
    let kept = _read_bytes(_kept_path(path))
    if kept is null { return null }
    try {
        let newline = __prim__.bytes_find(kept, _BYTE_LF)
        if newline < 0 { return null }
        let header = _decode(__prim__.bytes_slice(kept, 0, newline))
        if header is null { return null }
        let fields = header.split(" ")
        if fields.len() != 7 or fields[0] != "jsig" { return null }
        if int(fields[1]) != _SIG_CACHE_VERSION { return null }
        if int(fields[2]) != __prim__.jaic_build_id() { return null }
        if fields[3] != _front_end() { return null }
        let length = __prim__.bytes_len(raw)
        if int(fields[4]) != length { return null }
        let meta_length = int(fields[5])
        let meta_start = newline + 1
        let source_start = meta_start + meta_length
        if meta_length < 0 or __prim__.bytes_len(kept) != source_start + length { return null }
        let copy = __prim__.bytes_slice(kept, source_start, source_start + length)
        if __prim__.bytes_cmp(copy, raw) != 0 { return null }
        let meta = __prim__.bytes_slice(kept, meta_start, source_start)
        if __prim__.crc32(meta) != int(fields[6]) { return null }
        let lines = _decode(meta)
        if lines is null { return null }

        var edges: list[tuple[int, int, str]] = []
        var ranges: list[tuple[int, int]] = []
        for line in lines.split("\n") {
            if line == "" { continue }
            let parts = line.split(" ")
            if parts[0] == "E" and parts.len() == 4 {
                edges.push((int(parts[1]), int(parts[2]), parts[3]))
            } elif parts[0] == "B" and parts.len() == 3 {
                ranges.push((int(parts[1]), int(parts[2])))
            } else {
                return null
            }
        }
        return _Kept(edges, ranges)
    } catch _damaged: Error {
        return null
    }
}

#: Write the kept signature for `program`, parsed from `raw`, read from
#: `path`. Best effort: a temporary file renamed into place, so a reader sees
#: a whole file or none, and two processes writing one cannot interleave.
fn _keep_signature(path: str, raw: any, program: Node) -> void {
    let target = _kept_path(path)
    # Named by the clock, not `random_u64`: that would move the program's own
    # random sequence whenever a `run` compiles an import after seeding it.
    let temporary = f"{target}.{__prim__.time_mono()}.tmp"
    try {
        let stmts = program.child_list("stmts")
        var lines: list[str] = []
        for node in stmts {
            if node.kind != NodeKind.Import and node.kind != NodeKind.FromImport { continue }
            let dotted = node.opt_text("path")
            if dotted is null { continue }
            if dotted == "" or dotted.contains(" ") or dotted.contains("\n") { return }
            lines.push(f"E {node.span.start} {node.span.end} {dotted}\n")
        }
        let ranges = _body_ranges(stmts)
        if not _ranges_fit(raw, ranges) { return }
        for (start, end) in ranges { lines.push(f"B {start} {end}\n") }
        let meta = __prim__.str_encode("".join(lines))

        let dir = _dirname(target)
        if not _is_dir(dir) { __prim__.io_mkdir(dir) }
        let handle = __prim__.io_open(temporary, "wb")
        let length = __prim__.bytes_len(raw)
        let build = __prim__.jaic_build_id()
        let meta_length = __prim__.bytes_len(meta)
        let front = _front_end()
        let crc = __prim__.crc32(meta)
        let header = f"jsig {_SIG_CACHE_VERSION} {build} {front} {length} {meta_length} {crc}\n"
        __prim__.io_write(handle, header)
        __prim__.io_write(handle, meta)
        __prim__.io_write(handle, raw)
        __prim__.io_close(handle)
        __prim__.io_rename(temporary, target)
    } catch _failure: Error {
        # A write that failed part way must not leave its temporary behind.
        try {
            if _is_regular_file(temporary) { __prim__.io_remove(temporary) }
        } catch _left: Error {
        }
    }
}

#: The `(start, end)` byte range of every body `_strip_bodies` would empty,
#: in source order. A generator's body is left out: whether a function is one
#: is read off its body.
fn _body_ranges(stmts: list[Node]) -> list[tuple[int, int]] {
    var blocks: list[Node] = []
    for node in stmts {
        if node.kind == NodeKind.FnDecl {
            _blank_target(blocks, node)
        } elif node.kind == NodeKind.ClassDecl {
            for method in node.child_list("methods") { _blank_target(blocks, method) }
            for method in node.child_list("getters") { _blank_target(blocks, method) }
            for method in node.child_list("setters") { _blank_target(blocks, method) }
        } elif node.kind == NodeKind.TraitDecl or node.kind == NodeKind.EnumDecl {
            for method in node.child_list("methods") { _blank_target(blocks, method) }
        }
    }
    var ranges: list[tuple[int, int]] = []
    for block in blocks { ranges.push((block.span.start, block.span.end)) }
    ranges.sort(|range| range[0])
    return ranges
}

#: Whether every range is a `{` ... `}` of `raw`, in order and apart.
fn _ranges_fit(raw: any, ranges: list[tuple[int, int]]) -> bool {
    let total = __prim__.bytes_len(raw)
    var position = 0
    for (start, end) in ranges {
        if start < position or end > total or end - start < 2 { return false }
        if __prim__.bytes_get(raw, start) != _BYTE_LBRACE { return false }
        if __prim__.bytes_get(raw, end - 1) != _BYTE_RBRACE { return false }
        position = end
    }
    return true
}

#: `raw` decoded, with the inside of every range turned to spaces and its
#: newlines kept, so every byte stays where it was. Null when a range is not
#: a `{` ... `}` of `raw`.
fn _blank(raw: any, ranges: list[tuple[int, int]]) -> str? {
    if not _ranges_fit(raw, ranges) { return null }
    let total = __prim__.bytes_len(raw)
    var parts: list[str] = []
    var position = 0
    for (start, end) in ranges {
        let outside = _decode(__prim__.bytes_slice(raw, position, start + 1))
        if outside is null { return null }
        parts.push(outside)
        var at = start + 1
        let stop = end - 1
        while at < stop {
            let newline = __prim__.bytes_find(raw, _BYTE_LF, at, stop)
            if newline < 0 {
                parts.push(" " * (stop - at))
                at = stop
            } else {
                if newline > at { parts.push(" " * (newline - at)) }
                parts.push("\n")
                at = newline + 1
            }
        }
        position = stop
    }
    let rest = _decode(__prim__.bytes_slice(raw, position, total))
    if rest is null { return null }
    parts.push(rest)
    return "".join(parts)
}

fn _blank_target(blocks: list[Node], fn_node: Node) -> void {
    if fn_node.flag("isGenerator") { return }
    let body = fn_node.child("body")
    if body is null or body.kind != NodeKind.Block { return }
    if body.child_list("stmts").len() == 0 { return }
    blocks.push(body)
}
