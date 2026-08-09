#: An imported module's signatures, for the checker: `src/sema/modsig.c`.
#:
#: A Jaithon module is resolved at *run* time (spec §8): `from m import f` emits
#: an import instruction and binds a value, and nothing about `m` reaches the
#: front end. That leaves every call across a module boundary unchecked — arity,
#: keyword-argument names and parameter types all stop at the import.
#:
#: This is the missing half. Given a dotted module name and the directory of the
#: file that names it, `SigLoader` finds the module's source on the same search
#: path the runtime uses, parses it once, and keeps the tree so that the checker
#: can read a declaration's parameter list out of it. Only the *signatures* are
#: used: no body is resolved, no body is checked, and no diagnostic from the
#: parsed file escapes — whatever is wrong with a module is that module's own
#: compilation to report.
#:
#: Parsing rather than reading a `.jaic` is deliberate, and the C says why: a
#: module may have no cache entry, may have one written by a different compiler
#: build, and when it does have a warm one the run time skips the front end
#: entirely — so the cache is exactly the state in which no AST exists.
#:
#: ```
#: split_module_name("std.gui")      # (0, "std/gui")
#: split_module_name(".sibling")     # (1, "sibling")
#: split_module_name("std..gui")     # null
#: ```
#:
#: ## What this port cannot do today
#:
#: The C borrows the runtime's search (`jaiResolveModulePathQuiet`) so that the
#: checker looks a module up exactly where the import instruction will. That
#: function reads three pieces of process state which no primitive exposes to
#: Jaithon, so each one is handed in instead of read:
#:
#: - **The executable's own directory.** `jaiModulePathInit` derives the
#:   installed library from `jaiExecutablePath()` — `<exe>/../lib`,
#:   `<exe>/../share/jaithon/lib` and two more. From Jaithon, `os_argv()[0]` is
#:   the *script* being run, not the interpreter, so `default_search_path` takes
#:   `exec_dir` from its caller. Called without one it contributes only
#:   `JAITHON_PATH` and the fixed system directories, and a build tree's own
#:   `./lib` is not found.
#: - **Directories added by the driver.** `jaiModulePathAdd` is what `-I` and
#:   the CLI's own defaults go through; there is no way to read that list back.
#:   `SigLoader` takes its search path as a constructor argument.
#: - **`realpath`.** `jaiPathAbsolute` collapses symbolic links, which is how
#:   the C guarantees that two spellings of one file share one cache entry.
#:   `_absolute` here can only make a path absolute and fold away `.` and `..`
#:   lexically, so two paths that differ through a symlink are cached twice.
#:   The signatures are equal; the wasted work is one extra parse.
#:
#: Two more differences are simplifications rather than losses. The C swaps the
#: global diagnostic bag for a scratch one around the parse so that a syntax
#: error two modules away cannot surface as a diagnostic of the file being
#: checked; `parse` here takes its bag as an argument, so passing none has that
#: effect by construction. And `jaiModuleSigFreeAll` has no counterpart: a
#: `SigLoader` and everything it parsed are garbage once the checker that owns
#: them is.
from jaithon.ast import Node, NodeKind
from jaithon.compile.parser import parse

#: The extension a module's own file carries: `std.gui` is `std/gui.jai`.
pub const MODULE_EXT: str = ".jai"

#: The file that stands in for a directory-shaped module: `std/gui/mod.jai`.
pub const PACKAGE_FILE: str = "mod.jai"

#: How many re-exports `SigLoader.find` follows before giving up.
#:
#: `SIG_MAX_REEXPORT_DEPTH` in src/sema/modsig.c, and the same reasoning: a
#: chain longer than this is a mistake, not a design.
pub const MAX_REEXPORT_DEPTH: int = 8

#: One top-level name a module declares, and the node that declares it.
pub class SigDecl {
    #: The name as written.
    pub let name: str
    #: The `FnDecl`, `ClassDecl`, `TraitDecl`, `EnumDecl` or `TypeDecl` node.
    pub let decl: Node

    #: Record `name` as declared by `decl`.
    pub fn init(self, name: str, decl: Node) {
        self.name = name
        self.decl = decl
    }
}

#: A name a module re-exports rather than declares.
#:
#: `from other import name as local` at a module's top level: the module does
#: not declare `local`, but importing it from here is legal and lands on
#: `other`'s declaration. The prelude of spec §9 is nothing but these.
pub class SigReexport {
    #: The name this module offers.
    pub let local: str
    #: The name the other module declares, which differs under an alias.
    pub let remote: str
    #: The dotted module name, as written. Spelled `source` because `from` is a
    #: reserved word; the C field is `from`.
    pub let source: str

    #: Record that `local` here is `remote` in `source`.
    pub fn init(self, local: str, remote: str, source: str) {
        self.local = local
        self.remote = remote
        self.source = source
    }
}

#: One module's signatures: what it declares, and what it passes along.
#:
#: Data only, like `ty.jai`'s `Type`. `SigLoader` fills it in and owns the
#: caching, the cycle guard and the re-export walk.
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
    #: The class, trait and enum subset of `decls`, in source order.
    #:
    #: The checker registers all of them, so that one imported declaration's
    #: annotations can name a sibling the importer never asked for. A `TypeDecl`
    #: is deliberately absent: an alias introduces no type of its own, it names
    #: one that is already there. This list is what `jaiModuleSigTypeCount` and
    #: `jaiModuleSigTypeAt` walk between them.
    pub let types: list[SigDecl]
    #: Names re-exported by a `from ... import ...` at the top level.
    pub let reexports: list[SigReexport]
    #: Modules re-exported wholesale by a `from ... import *`.
    #:
    #: At most one today, and only as a file's last statement: the newline after
    #: the `*` is eaten as a line continuation, in both front ends. A list all
    #: the same, because that is a lexer bug rather than a rule.
    pub let wildcards: list[str]

    #: Create an empty signature for the module at `path`.
    pub fn init(self, path: str, dotted: str, file_id: int) {
        self.path = path
        self.dotted = dotted
        self.file_id = file_id
        self.program = null
        self.decls = []
        self.types = []
        self.reexports = []
        self.wildcards = []
    }

    #: `"std.gui.Window"`: the name an imported type is interned under.
    #:
    #: Two modules declaring a `Node` are two types, and neither collides with a
    #: `Node` declared in the importing file. Leading dots are dropped first: a
    #: relative import is written from the importer's point of view (`.thing`),
    #: which says nothing about identity.
    #:
    #: ```
    #: ModuleSig("/lib/std/gui.jai", "std.gui", 1).qualify("Window")
    #: # "std.gui.Window"
    #: ModuleSig("/app/thing.jai", ".thing", 1).qualify("Node")
    #: # "thing.Node"
    #: ```
    pub fn qualify(self, name: str) -> str {
        var index = 0
        while index < self.dotted.len() and self.dotted[index] == "." { index += 1 }
        let stripped = self.dotted[index:]
        if stripped == "" { return f"{self.dotted}.{name}" }
        return f"{stripped}.{name}"
    }
}

#: Every module signature one checker loaded, and the search path it used.
#:
#: The C keeps this in a file-static vector and a process-global module path,
#: freed by `jaiModuleSigFreeAll`. An object instead, for the reason
#: `universe.jai` gives for `Universe`: it makes the loader testable, and it
#: deletes the lifetime hazard of a global that every parsed tree points into.
#:
#: ```
#: let dir = __prim__.os_env("TMPDIR") ?? "/tmp"
#: let handle = __prim__.io_open(dir + "/jaithon_modsig_doc.jai", "wb")
#: __prim__.io_write(handle, "pub fn greet(who: str) -> str { return who }\n")
#: __prim__.io_close(handle)
#: let loader = SigLoader()
#: # `?? ModuleSig(...)` only so the lines below need no `?.`: an empty
#: # signature answers "" to the first of them and the example fails there.
#: let sig = loader.load("jaithon_modsig_doc", dir) ?? ModuleSig("", "", 0)
#: sig.dotted                                     # "jaithon_modsig_doc"
#: sig.decls.len()                                # 1
#: sig.decls[0].name                              # "greet"
#: loader.find(sig, "greet")[0]?.kind             # NodeKind.FnDecl
#: loader.find(sig, "absent")[0] is null          # true
#: loader.load("no.such.module", dir) is null     # true
#: ```
pub class SigLoader {
    #: The directories searched after the importing file's own, in order.
    #:
    #: `jaiResolveModulePath` searches `sUserDirs` and then `sLibDirs`; this is
    #: the two of them concatenated, since nothing here distinguishes them.
    pub let search: list[str]

    # Resolved path to signature: the C's `sSigs` scanned by `strcmp`.
    var _cache: dict[str, ModuleSig]
    # File id to signature, for `for_file`.
    var _by_file: dict[int, ModuleSig]
    # The next file id to hand out.
    var _next_file_id: int
    # Paths on the current re-export walk. The C hangs a `visiting` flag on each
    # signature; keeping the stack here leaves `ModuleSig` pure data, and a
    # module cannot be on the walk twice by any other route.
    var _visiting: list[str]

    #: Build a loader that searches `search` after the importing file's own
    #: directory.
    #:
    #: `first_file_id` is the id given to the first module loaded. The default
    #: leaves 0 to the file being checked, which is what `compile_source` passes
    #: for it.
    pub fn init(self, search: list[str] = [], first_file_id: int = 1) {
        self.search = search
        self._cache = {}
        self._by_file = {}
        self._next_file_id = first_file_id
        self._visiting = []
    }

    #: Load, or return the cached, signature of the module `dotted` names.
    #:
    #: `from_dir` is the directory of the file that wrote the import, which is
    #: where a relative name is resolved from and the first directory a plain
    #: one is looked for in. Null when the module cannot be found or cannot be
    #: read; neither is reported, because the checker is asking rather than
    #: importing and a "no" is an answer.
    #:
    #: A file that does not parse still gets a cached, empty signature. Its
    #: `program` is null and it offers nothing, but re-reading and re-parsing it
    #: once per import would be the whole cost of the feature for no answer.
    pub fn load(self, dotted: str, from_dir: str = "") -> ModuleSig? {
        if dotted == "" { return null }
        let resolved = resolve_module_path(dotted, from_dir, self.search)
        if resolved is null { return null }
        if resolved in self._cache { return self._cache[resolved] }

        let text = _read_source(resolved)
        if text is null { return null }

        let sig = ModuleSig(resolved, dotted, self._next_file_id)
        self._next_file_id += 1

        # No bag is passed, so the parser makes one and it dies with the call.
        let program = parse(text, sig.file_id)
        sig.program = program
        if program != null and program.kind == NodeKind.Program {
            _collect(sig, program.child_list("stmts"))
        }

        self._cache[resolved] = sig
        self._by_file[sig.file_id] = sig
        return sig
    }

    #: The declaration `name` denotes in `sig`, and the module that declares it.
    #:
    #: A name the module itself imported and re-exports is followed to the
    #: module that declares it, and the second half of the answer is that
    #: module — the annotations in the declaration must be read in its
    #: namespace, not in the re-exporter's. When nothing declares `name` the
    #: answer is `(null, sig)`, mirroring the C's `outOwner`, which is only
    #: overwritten by a hit.
    #:
    #: Hand-written re-exports are followed before wildcards, so a name a module
    #: both named and could have got from a `*` import resolves to the one it
    #: named.
    pub fn find(self, sig: ModuleSig, name: str) -> tuple[Node?, ModuleSig] {
        let found = self._find_in(sig, name, MAX_REEXPORT_DEPTH)
        let owner = found[1]
        if owner is null { return (null, sig) }
        return (found[0], owner)
    }

    #: The signature parsed from `file_id`, or null when that file is not one.
    #:
    #: Every span carries the file it was parsed from, so this is how a
    #: declaration reached indirectly — through an imported class's member
    #: table — is traced back to the module whose namespace its annotations must
    #: be read in.
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
#:
#: `splitModuleName` in src/runtime/module.c. A leading dot is an instruction to
#: the resolver rather than part of the name: one means the importing file's own
#: directory and each further one climbs a level. Null for anything that is not
#: `'.'* ident ('.' ident)*` — where an identifier byte is ASCII alphanumeric,
#: `_`, or anything above 0x7f, since spec §2.1 allows non-ASCII identifiers.
#: Everything else, `..` and path separators included, is refused so that a name
#: out of source text cannot reach outside a search directory.
#:
#: The C reports E0804 on the way out; this returns null, because every caller
#: here is `jaiResolveModulePathQuiet`, which throws that report away.
#:
#: ```
#: split_module_name("std.gui")      # (0, "std/gui")
#: split_module_name("..up")         # (2, "up")
#: split_module_name("a/b")          # null
#: split_module_name("a.")           # null
#: split_module_name("...")          # null
#: ```
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

#: Resolve `dotted` to a file, the way the runtime's importer would.
#:
#: `jaiResolveModulePath` in src/runtime/module.c, minus its reporting. A
#: relative name is resolved only against `from_dir` climbed by its extra dots;
#: a plain one is looked for in `from_dir` first and then in `search`, in order.
#: In each directory the module's own file wins over the package form:
#: `<dir>/std/gui.jai` before `<dir>/std/gui/mod.jai`.
#:
#: The answer is absolute and lexically normalised. Unlike the C it is not
#: symlink-resolved — see this module's own doc comment.
#:
#: ```
#: resolve_module_path("no.such.module", "/tmp") is null   # true
#: resolve_module_path("a/b", "/tmp") is null              # true
#: ```
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

#: The search path `jaiModulePathInit` would build, as far as it can be seen.
#:
#: `JAITHON_PATH` split on `:` comes first, then the library directories derived
#: from `exec_dir`, then the fixed system ones. A derived directory that is not
#: a directory is dropped and duplicates are collapsed, exactly as `addLibDir`
#: and `dirListAdd` do.
#:
#: `exec_dir` has to be supplied: the interpreter's own path is not reachable
#: from Jaithon, so without it a build tree's `./lib` — the directory that holds
#: `std` during development — is missed. Pass the directory holding the
#: `jaithon` binary.
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
    }
    for fixed in [
        "/usr/local/share/jaithon/lib",
        "/usr/local/share/jaithon",
        "/opt/homebrew/share/jaithon/lib",
        "/opt/homebrew/share/jaithon",
    ] {
        _add_lib_dir(dirs, fixed)
    }
    return dirs
}

# ------------------------------------------------------------------
# Collecting the top-level declarations
# ------------------------------------------------------------------
fn _collect(sig: ModuleSig, stmts: list[Node]) -> void {
    for node in stmts {
        match node.kind {
            NodeKind.FnDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.ClassDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TraitDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.EnumDecl => { _add_decl(sig, node.opt_text("name"), node, true) }
            NodeKind.TypeDecl => { _add_decl(sig, node.opt_text("name"), node, false) }
            NodeKind.FromImport => { _add_import(sig, node) }
            _ => { }
        }
    }
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

# ------------------------------------------------------------------
# The search rule
# ------------------------------------------------------------------

# `<dir>/std/gui.jai`, then the package form `<dir>/std/gui/mod.jai`.
fn _try_directory(dir: str, relative: str) -> str? {
    let own = _join(dir, relative + MODULE_EXT)
    if _is_regular_file(own) { return _absolute(own) }
    let package = _join(dir, relative + "/" + PACKAGE_FILE)
    if _is_regular_file(package) { return _absolute(package) }
    return null
}

# Base directory of a relative import: one dot is the importer's directory,
# each further dot climbs one level. Null when it climbs past the root.
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

# `dirListAdd`: absolute where possible, and no duplicates.
fn _add_dir(dirs: list[str], dir: str) -> void {
    if dir == "" { return }
    let entry = _absolute(dir)
    if entry in dirs { return }
    dirs.push(entry)
}

# `addLibDir`: the same, but only for something that is there and is a
# directory. A derived path like `<exe>/../share/jaithon/lib` usually is not.
fn _add_lib_dir(dirs: list[str], dir: str) -> void {
    if dir == "" or not _is_dir(dir) { return }
    _add_dir(dirs, dir)
}

# `isNameByte`: spec §2.1 allows non-ASCII identifiers, so everything above the
# ASCII range is let through and everything else — separators, `..`, control
# bytes — is not.
fn _is_name_char(character: str) -> bool {
    let point = ord(character)
    if point >= 0x80 { return true }
    if point >= 0x61 and point <= 0x7a { return true }  # a-z
    if point >= 0x41 and point <= 0x5a { return true }  # A-Z
    if point >= 0x30 and point <= 0x39 { return true }  # 0-9
    return point == 0x5f                                # _
}

# ------------------------------------------------------------------
# Paths and files
# ------------------------------------------------------------------
#
# `jaiPathJoin`, `jaiPathDirname` and `jaiPathAbsolute` from src/common/memory.c,
# rewritten here rather than taken from `std.io.Path`: the search rule is only
# the runtime's rule if it composes paths the same way, and `Path` normalises on
# its own terms. They are private because `std.io.Path` is what a caller wants.

# `jaiPathJoin`: an absolute second half discards the first, runs of `/` collapse
# into one, and no trailing separator survives.
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

# `jaiPathDirname`: the path without its last component, `.` when there is none.
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

# As much of `jaiPathAbsolute` as there is a primitive for: absolute against the
# working directory, with `.` and `..` folded away. `realpath` would also
# collapse symbolic links and would fail on a path that does not exist; this
# does neither, so it is used only on a candidate that was just found to be a
# file, or on a search directory where a stale entry costs nothing.
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

# `isRegularFile`: something is there and it is not a directory. `io_stat`
# follows the link, which is what `stat` does in the C.
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

# `jaiReadFile`: the whole file, or null when it could not be read. A module the
# checker cannot open is a module with no signatures, not an error — it is about
# to be one for the importer, whose diagnostic that is.
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
