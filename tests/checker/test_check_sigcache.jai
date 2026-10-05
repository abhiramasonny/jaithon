#: The signature cache: what `SigLoader` keeps in `__jaicache__` so the next
#: process need not parse an imported module again.
#:
#: What has to hold is that a kept signature changes nothing but the cost: the
#: tree parsed from it is the tree a full parse leaves once its bodies are
#: stripped, spans included; a module that does not parse is never kept and
#: still offers nothing; and a kept signature that no longer matches its
#: source, or is damaged, is ignored rather than believed. Each test writes its
#: own module under `TMPDIR`, and `clear_shared_signatures` stands in for the
#: next process, since it is the in-process share that would otherwise answer.
from std.test import assert_eq, assert_true
from jaithon.ast import Node, NodeKind
from jaithon.ast_encode import to_json
from jaithon.compile.parser import parse
from jaithon.compile.check.modsig import ModuleSig, SigLoader, clear_shared_signatures
from jaithon.compile.check.modsig import signature_cache_enabled, signature_text

#: A module with every shape whose body the cache blanks, and the brace-heavy
#: text a character-level skip would get wrong.
const TRICKY: str = """#: A module for the signature cache tests.
import std.math
from std.str import StringBuilder as Builder

pub const LIMIT: int = 3
pub let (WIDTH, HEIGHT) = (640, 480)

pub fn plain(a: int, b: str = "}{", c: dict[str, int] = {"k": 1}) -> str {
    let nested = {"k": {"n": [1, 2, {"m": 3}]}}
    let s = f"{a} {{literal}} {nested["k"]} }}"
    if a > 0 { return s }
    return "}" + '{' + "\\u{7d}"
}

pub fn spread(xs: list[int]) -> int {
    # a comment with a brace }
    var total = 0
    for x in xs {
        total += x
    }
    return total
}

pub fn counter(n: int) -> int {
    var i = 0
    while i < n {
        yield i
        i += 1
    }
}

pub class Box {
    pub let value: int
    let note: str = "a } in a field default"

    pub fn init(self, value: int) { self.value = value }

    pub get doubled() -> int {
        return self.value * 2
    }

    pub fn café(self) -> str { return "héllo — ✓ {" }
}

pub trait Shape {
    fn area(self) -> float
    fn name(self) -> str {
        return "shape"
    }
}

pub enum Color {
    Red
    Green

    pub fn code(self) -> int {
        if self == Color.Red { return 1 }
        return 2
    }
}

let callback = |x| { x + 1 }

if LIMIT > 2 {
    let inner = 1
}
"""

fn _strip(stmts: list[Node]) -> void {
    for node in stmts {
        if node.kind == NodeKind.FnDecl {
            _strip_one(node)
        } elif node.kind == NodeKind.ClassDecl {
            for method in node.child_list("methods") { _strip_one(method) }
            for method in node.child_list("getters") { _strip_one(method) }
            for method in node.child_list("setters") { _strip_one(method) }
        } elif node.kind == NodeKind.TraitDecl or node.kind == NodeKind.EnumDecl {
            for method in node.child_list("methods") { _strip_one(method) }
        }
    }
}

fn _strip_one(fn_node: Node) -> void {
    let body = fn_node.child("body")
    if body is null or body.kind != NodeKind.Block { return }
    let empty: list[Node] = []
    body.set("stmts", empty)
}

fn _stripped_json(source: str) -> str {
    let program = parse(source, 7)
    if program is null { throw AssertionError("the source did not parse") }
    _strip(program.child_list("stmts"))
    return to_json(program)
}

# ------------------------------------------------------------------
# The kept text
# ------------------------------------------------------------------
fn test_the_fixture_parses() -> void {
    assert_true(parse(TRICKY, 7) is not null, "TRICKY does not parse")
}

fn test_a_kept_text_parses_to_the_stripped_tree_spans_included() -> void {
    let kept = signature_text(TRICKY, 7)
    if kept is null { throw AssertionError("no kept text for a module that parses") }
    assert_eq(_stripped_json(kept), _stripped_json(TRICKY))
}

fn test_a_kept_text_keeps_every_byte_offset_and_line() -> void {
    let kept = signature_text(TRICKY, 7) ?? ""
    assert_eq(kept.bytes().len(), TRICKY.bytes().len())
    assert_eq(kept.count("\n"), TRICKY.count("\n"))
}

fn test_a_kept_text_drops_what_bodies_say() -> void {
    let kept = signature_text(TRICKY, 7) ?? ""
    assert_true(not kept.contains("nested"), "a function body survived")
    assert_true(not kept.contains("self.value * 2"), "a getter body survived")
    assert_true(not kept.contains("Color.Red { return 1 }"), "an enum method body survived")
    assert_true(kept.contains("a } in a field default"), "a field default was blanked")
}

fn test_a_generator_keeps_its_body_so_it_is_still_a_generator() -> void {
    let kept = signature_text(TRICKY, 7) ?? ""
    assert_true(kept.contains("yield i"), "the generator's body was blanked")
}

fn test_a_source_that_does_not_parse_has_no_kept_text() -> void {
    assert_true(signature_text("pub fn f() -> int {\n    let x =\n}\n") is null)
}

# ------------------------------------------------------------------
# Across processes
# ------------------------------------------------------------------
fn test_the_cache_is_on_unless_switched_off() -> void {
    let front_end = __prim__.os_env("JAITHON_FRONT_END_ID") ?? ""
    let wanted = (__prim__.os_env("JAITHON_SIG_CACHE") ?? "1") != "0" and
        (__prim__.os_env("JAITHON_SIG_BODIES") ?? "0") == "0" and
        front_end != "" and
        front_end != "none"
    assert_eq(signature_cache_enabled(), wanted)
}

fn test_the_running_front_end_has_an_identity() -> void {
    # The test runner compiled this file, so the front end is loaded and the
    # runtime has said which one it is: the seed's, in a tree that is seeded.
    let front_end = __prim__.os_env("JAITHON_FRONT_END_ID") ?? ""
    assert_true(front_end != "", "the runtime exported no front-end identity")
}

fn test_a_kept_signature_names_the_front_end_that_wrote_it() -> void {
    if not signature_cache_enabled() { return }
    let dir = _module_dir("identity", TRICKY)
    let _cold = _load_fresh(dir)
    let header = _read(f"{dir}/__jaicache__/tricky.jsig").split("\n")[0].split(" ")
    assert_eq(header[3], __prim__.os_env("JAITHON_FRONT_END_ID") ?? "")
}

fn test_a_second_process_reads_the_edges_without_parsing() -> void {
    if not signature_cache_enabled() { return }
    let dir = _module_dir("second", TRICKY)
    let first = _load_fresh(dir)
    assert_true(first.parsed, "the first load did not parse")

    clear_shared_signatures()
    let edges = SigLoader().load_edges("tricky", dir)
    if edges is null { throw AssertionError("the module did not load") }
    assert_true(not edges.parsed, "the second process parsed instead of reading the kept signature")
    assert_eq(_edge_text(edges), _edge_text(first))
}

fn test_a_kept_signature_loads_the_same_declarations_as_the_source() -> void {
    if not signature_cache_enabled() { return }
    let dir = _module_dir("same", TRICKY)
    let cold = _load_fresh(dir)
    let warm = _load_fresh(dir)
    assert_eq(_decl_text(warm), _decl_text(cold))
    assert_eq(_edge_text(warm), _edge_text(cold))
    assert_eq(warm.reexports.len(), cold.reexports.len())
    assert_eq(_tree_text(warm), _tree_text(cold))
}

fn test_a_module_with_an_error_in_a_body_offers_nothing_and_is_never_kept() -> void {
    let broken = "pub fn good() -> int {\n    return 1\n}\n\npub fn bad() -> int {\n    let x =\n}\n"
    let dir = _module_dir("broken", broken)
    for _round in [1, 2] {
        let sig = _load_fresh(dir)
        assert_true(sig.program is null, "a module with a body error parsed")
        assert_eq(sig.decls.len(), 0)
    }
    assert_true(
        not _exists(f"{dir}/__jaicache__/tricky.jsig"),
        "a module that does not parse was kept"
    )
}

fn test_an_edited_module_is_read_again() -> void {
    let dir = _module_dir("edited", "pub fn before() -> int {\n    return 1\n}\n")
    let _first = _load_fresh(dir)
    _write(f"{dir}/tricky.jai", "pub fn after() -> int {\n    return 2\n}\n")
    let second = _load_fresh(dir)
    assert_eq(_decl_text(second), "after")
}

fn test_an_edit_that_keeps_the_length_is_read_again() -> void {
    let dir = _module_dir("same_length", "pub fn aaaa() -> int {\n    return 1\n}\n")
    let _first = _load_fresh(dir)
    _write(f"{dir}/tricky.jai", "pub fn bbbb() -> int {\n    return 1\n}\n")
    let second = _load_fresh(dir)
    assert_eq(_decl_text(second), "bbbb")
}

fn test_a_damaged_kept_signature_is_ignored() -> void {
    if not signature_cache_enabled() { return }
    let dir = _module_dir("damaged", TRICKY)
    let cold = _load_fresh(dir)
    let kept = f"{dir}/__jaicache__/tricky.jsig"
    let text = _read(kept)
    assert_true(text.starts_with("jsig "), "nothing was kept")

    # An edge changed under a checksum that no longer matches.
    _write(kept, _edit_first_line(text, 0, false))
    assert_eq(_decl_text(_load_fresh(dir)), _decl_text(cold))
    # A body range moved off its brace under a checksum made to match: the
    # range is checked against the source, and the module is parsed in full.
    _write(kept, _edit_first_line(text, 1, true))
    let moved = _load_fresh(dir)
    assert_eq(_decl_text(moved), _decl_text(cold))
    assert_eq(_tree_text(moved), _tree_text(cold))
    # Cut short.
    _write(kept, text[:text.len() // 2])
    assert_eq(_decl_text(_load_fresh(dir)), _decl_text(cold))
    # Garbage.
    _write(kept, "jsig not a header\n")
    assert_eq(_decl_text(_load_fresh(dir)), _decl_text(cold))
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

#: `text`, a kept signature, with the first digit of the first edge line
#: (`section` 0) or body-range line (`section` 1) changed, and the header's
#: checksum recomputed when `fix_crc` is set.
fn _edit_first_line(text: str, section: int, fix_crc: bool) -> str {
    let lines = text.split("\n")
    let fields = lines[0].split(" ")
    let edges_length = int(fields[5])
    let ranges_length = int(fields[6])
    var index = 1
    if section == 1 {
        var used = 0
        while used < edges_length {
            used += lines[index].len() + 1
            index += 1
        }
    }
    let line = lines[index]
    let first = if line[0] == "1" { "2" } else { "1" }
    lines[index] = first + line[1:]
    var meta_lines: list[str] = []
    var used = 0
    var at = 1
    while used < edges_length + ranges_length {
        meta_lines.push(lines[at])
        used += lines[at].len() + 1
        at += 1
    }
    if fix_crc {
        let meta = "\n".join(meta_lines) + "\n"
        fields[7] = f"{__prim__.crc32(__prim__.str_encode(meta))}"
        lines[0] = " ".join(fields)
    }
    return "\n".join(lines)
}

fn _load_fresh(dir: str) -> ModuleSig {
    clear_shared_signatures()
    let sig = SigLoader().load("tricky", dir)
    if sig is null { throw AssertionError(f"tricky did not load from {dir}") }
    return sig
}

fn _decl_text(sig: ModuleSig) -> str {
    return ",".join([entry.name for entry in sig.decls])
}

#: The kept tree as JSON, with this load's file id spelled the same for both.
fn _tree_text(sig: ModuleSig) -> str {
    let program = sig.program
    if program is null { return "null" }
    let own = f"\"file\": {sig.file_id}"
    let packed = f"\"file\":{sig.file_id}"
    return to_json(program).replace(own, "\"file\": F").replace(packed, "\"file\":F")
}

fn _edge_text(sig: ModuleSig) -> str {
    var parts: list[str] = []
    for edge in sig.imports {
        let own = if edge.span.file == sig.file_id { "own" } else { f"{edge.span.file}" }
        parts.push(f"{edge.dotted}@{own}:{edge.span.start}-{edge.span.end}")
    }
    return ",".join(parts)
}

fn _module_dir(name: str, source: str) -> str {
    var base = __prim__.os_env("TMPDIR") ?? "/tmp"
    while base.len() > 1 and base.ends_with("/") { base = base[:base.len() - 1] }
    let dir = f"{base}/jaithon-sigcache-tests-{__prim__.time_mono()}/{name}"
    __prim__.io_mkdir(dir, true)
    _write(f"{dir}/tricky.jai", source)
    return dir
}

fn _write(path: str, text: str) -> void {
    let handle = __prim__.io_open(path, "wb")
    __prim__.io_write(handle, text)
    __prim__.io_close(handle)
}

fn _read(path: str) -> str {
    let handle = __prim__.io_open(path, "rb")
    let raw = __prim__.io_read(handle, -1)
    __prim__.io_close(handle)
    return __prim__.str_decode(raw)
}

fn _exists(path: str) -> bool {
    return __prim__.io_stat(path, true) is not null
}
