#: The signature loader is the one part of the front end that reads a file
#: other than the one being compiled, so almost all of it is the search rule —
#: which file `std.gui` names — and that rule is only correct if it is the
#: runtime's. It is tested against a real directory tree rather than a stubbed
#: one, because the parts that can be wrong are the parts that touch the disk:
#: the package form, the importing file's own directory winning over the search
#: path, and a relative name climbing.
#:
#: The fixture is built under `TMPDIR` and left there. Rebuilding it is
#: idempotent, and a few kilobytes in the temporary directory is cheaper than a
#: teardown that runs whether or not the test that needed it did.
from std.test import assert_eq, assert_true
from jaithon.ast import NodeKind
from jaithon.compile.check.modsig import MAX_REEXPORT_DEPTH, ModuleSig, SigLoader
from jaithon.compile.check.modsig import default_search_path, resolve_module_path
from jaithon.compile.check.modsig import split_module_name

# ------------------------------------------------------------------
# The dotted name
# ------------------------------------------------------------------
fn test_a_dotted_name_becomes_a_relative_path() -> void {
    assert_eq(split_module_name("gui"), (0, "gui"))
    assert_eq(split_module_name("std.gui"), (0, "std/gui"))
    assert_eq(split_module_name("a.b.c.d"), (0, "a/b/c/d"))
}

fn test_leading_dots_are_counted_and_not_spelled() -> void {
    assert_eq(split_module_name(".leaf"), (1, "leaf"))
    assert_eq(split_module_name("..leaf"), (2, "leaf"))
    assert_eq(split_module_name("...pkg.leaf"), (3, "pkg/leaf"))
}

fn test_anything_that_is_not_a_dotted_identifier_is_refused() -> void {
    for refused in ["", ".", "..", "a.", ".a.", "a..b", "a/b", "a-b", "a b", "../etc"] {
        assert_true(split_module_name(refused) is null, f"{repr(refused)} was accepted")
    }
}

fn test_a_non_ascii_identifier_is_a_module_name() -> void {
    assert_eq(split_module_name("café.ünïcode"), (0, "café/ünïcode"))
}

# ------------------------------------------------------------------
# The search rule
# ------------------------------------------------------------------
fn test_a_modules_own_file_wins_over_the_package_form() -> void {
    let root = _fixture()
    assert_eq(resolve_module_path("dual", root), f"{root}/dual.jai")
}

fn test_a_directory_is_a_module_through_its_mod_jai() -> void {
    let root = _fixture()
    assert_eq(resolve_module_path("pkg", root), f"{root}/pkg/mod.jai")
    assert_eq(resolve_module_path("pkg.deep", root), f"{root}/pkg/deep/mod.jai")
}

fn test_the_importing_files_own_directory_is_searched_first() -> void {
    let root = _fixture()
    assert_eq(
        resolve_module_path("leaf", f"{root}/shadow", [f"{root}/pkg"]),
        f"{root}/shadow/leaf.jai"
    )
}

fn test_the_search_path_is_tried_in_order() -> void {
    let root = _fixture()
    let shadow = f"{root}/shadow"
    let pkg = f"{root}/pkg"
    assert_eq(resolve_module_path("leaf", "", [shadow, pkg]), f"{shadow}/leaf.jai")
    assert_eq(resolve_module_path("leaf", "", [pkg, shadow]), f"{pkg}/leaf.jai")
}

fn test_a_relative_name_ignores_the_search_path_entirely() -> void {
    let root = _fixture()
    assert_eq(
        resolve_module_path(".leaf", f"{root}/shadow", [f"{root}/pkg"]),
        f"{root}/shadow/leaf.jai"
    )
    assert_true(resolve_module_path(".dual", f"{root}/shadow", [root]) is null)
}

fn test_each_extra_dot_climbs_one_directory() -> void {
    let root = _fixture()
    let deep = f"{root}/pkg/deep"
    assert_eq(resolve_module_path(".sibling", deep), f"{deep}/sibling.jai")
    assert_eq(resolve_module_path("..leaf", deep), f"{root}/pkg/leaf.jai")
    assert_eq(resolve_module_path("...dual", deep), f"{root}/dual.jai")
}

fn test_climbing_past_the_root_finds_nothing() -> void {
    assert_true(resolve_module_path("..........anything", "/") is null)
}

fn test_a_module_that_is_not_there_resolves_to_nothing() -> void {
    let root = _fixture()
    assert_true(resolve_module_path("no.such.module", root, [root]) is null)
}

fn test_a_resolved_path_is_absolute_and_normalised() -> void {
    let root = _fixture()
    assert_eq(resolve_module_path("leaf", f"{root}/pkg/deep/.."), f"{root}/pkg/leaf.jai")
}

# ------------------------------------------------------------------
# What a signature holds
# ------------------------------------------------------------------
fn test_every_top_level_declaration_is_collected_in_source_order() -> void {
    let sig = _leaf()
    assert_eq([entry.name for entry in sig.decls], ["make", "Widget", "Drawable", "Colour", "Size"])
}

fn test_only_classes_traits_and_enums_count_as_types() -> void {
    let sig = _leaf()
    assert_eq([entry.name for entry in sig.types], ["Widget", "Drawable", "Colour"])
}

fn test_a_declaration_comes_back_as_the_node_that_declares_it() -> void {
    let sig = _leaf()
    let loader = SigLoader()
    let found = loader.find(sig, "Widget")
    assert_eq(found[0]?.kind, NodeKind.ClassDecl)
    assert_eq(found[1].path, sig.path)
}

fn test_a_name_no_one_declares_comes_back_with_the_module_that_was_asked() -> void {
    let sig = _leaf()
    let found = SigLoader().find(sig, "nothing_of_the_sort")
    assert_true(found[0] is null)
    assert_eq(found[1].path, sig.path)
}

fn test_a_declaration_is_collected_whether_or_not_it_is_pub() -> void {
    let sig = _load("private_decl")
    assert_eq([entry.name for entry in sig.decls], ["hidden"])
}

# ------------------------------------------------------------------
# Re-exports
# ------------------------------------------------------------------
fn test_a_re_export_lands_on_the_module_that_declares_the_name() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("pkg", root), "pkg")
    assert_eq([entry.name for entry in sig.decls], [])
    assert_eq(sig.reexports[0].local, "Gadget")
    assert_eq(sig.reexports[0].remote, "Widget")
    assert_eq(sig.reexports[0].source, ".leaf")

    let found = loader.find(sig, "Gadget")
    assert_eq(found[0]?.kind, NodeKind.ClassDecl)
    assert_eq(found[1].path, f"{root}/pkg/leaf.jai")
}

fn test_a_wildcard_import_is_followed_too() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("star", root), "star")
    assert_eq(sig.wildcards, ["pkg.leaf"])
    assert_eq(loader.find(sig, "Colour")[0]?.kind, NodeKind.EnumDecl)
    assert_eq(loader.find(sig, "Colour")[1].path, f"{root}/pkg/leaf.jai")
}

fn test_a_named_re_export_wins_over_a_wildcard() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("both", root), "both")
    assert_eq(loader.find(sig, "Thing")[0]?.kind, NodeKind.FnDecl)
    assert_eq(loader.find(sig, "Thing")[1].path, f"{root}/named_src.jai")
}

fn test_a_re_export_cycle_terminates_instead_of_recurring() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("cycle_a", root), "cycle_a")
    assert_true(loader.find(sig, "Ghost")[0] is null)
}

fn test_a_re_export_chain_is_followed_exactly_eight_deep() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("chain0", root), "chain0")
    assert_eq(MAX_REEXPORT_DEPTH, 8)
    assert_eq(loader.find(sig, "Shallow")[1].path, f"{root}/chain7.jai")
    assert_true(loader.find(sig, "Deep")[0] is null)
}

# ------------------------------------------------------------------
# Caching, file ids and naming
# ------------------------------------------------------------------
fn test_a_module_is_parsed_once_however_often_it_is_asked_for() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let first = _require(loader.load("pkg.leaf", root), "pkg.leaf")
    let again = _require(loader.load("pkg.leaf", root), "pkg.leaf")
    assert_eq(again.file_id, first.file_id)
    assert_eq(loader.loaded().len(), 1)
}

fn test_two_spellings_of_one_file_share_one_signature() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let plain = _require(loader.load("pkg.leaf", root), "pkg.leaf")
    let dotted = _require(loader.load(".pkg.leaf", root), ".pkg.leaf")
    assert_eq(dotted.file_id, plain.file_id)
    assert_eq(dotted.dotted, "pkg.leaf")
}

fn test_a_file_that_does_not_parse_offers_nothing_and_is_still_cached() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let sig = _require(loader.load("broken", root), "broken")
    assert_true(sig.program is null)
    assert_eq(sig.decls.len(), 0)
    assert_eq(_require(loader.load("broken", root), "broken").file_id, sig.file_id)
}

fn test_a_module_that_cannot_be_found_is_not_a_signature() -> void {
    assert_true(SigLoader().load("no.such.module", _fixture()) is null)
    assert_true(SigLoader().load("", _fixture()) is null)
}

fn test_file_ids_are_handed_out_after_the_file_being_checked() -> void {
    let root = _fixture()
    let loader = SigLoader()
    let first = _require(loader.load("pkg.leaf", root), "pkg.leaf")
    let second = _require(loader.load("star", root), "star")
    assert_eq(first.file_id, 1)
    assert_eq(second.file_id, 2)
    assert_eq(loader.for_file(2)?.path, second.path)
    assert_true(loader.for_file(0) is null)
}

fn test_a_qualified_name_carries_the_module_it_came_from() -> void {
    assert_eq(ModuleSig("/lib/std/gui.jai", "std.gui", 1).qualify("Window"), "std.gui.Window")
    assert_eq(ModuleSig("/app/thing.jai", ".thing", 1).qualify("Node"), "thing.Node")
    assert_eq(ModuleSig("/app/thing.jai", "...", 1).qualify("Node"), "....Node")
}

# ------------------------------------------------------------------
# The default search path
# ------------------------------------------------------------------
fn test_both_installed_layouts_derive_the_same_library_directory() -> void {
    let root = _fixture()
    let expected = [f"{root}/lib", f"{root}/packages/demo/src"]
    assert_eq(_under(default_search_path(root), root), expected)
    assert_eq(_under(default_search_path(f"{root}/pkg"), root), expected)
}

fn test_a_workspace_package_source_is_an_import_root() -> void {
    let root = _fixture()
    let search = default_search_path(root)
    assert_eq(resolve_module_path("demo", "", search), f"{root}/packages/demo/src/demo/mod.jai")
    assert_true(f"{root}/packages/unlisted/src" not in search)
}

fn test_a_derived_directory_that_is_not_there_is_dropped() -> void {
    let root = _fixture()
    assert_eq(_under(default_search_path(f"{root}/pkg/deep"), root), [])
}

fn test_jaithon_path_comes_before_the_derived_directories() -> void {
    let root = _fixture()
    let saved = __prim__.os_env("JAITHON_PATH")
    __prim__.os_env("JAITHON_PATH", f"{root}/shadow::{root}/pkg")
    let found = _under(default_search_path(root), root)
    __prim__.os_env("JAITHON_PATH", saved)
    assert_eq(found, [f"{root}/shadow", f"{root}/pkg", f"{root}/lib", f"{root}/packages/demo/src"])
}

fn _under(dirs: list[str], root: str) -> list[str] {
    return [dir for dir in dirs if dir.starts_with(root)]
}

# ------------------------------------------------------------------
# The fixture
# ------------------------------------------------------------------
fn _leaf() -> ModuleSig {
    return _load("pkg.leaf")
}

fn _load(dotted: str) -> ModuleSig {
    let root = _fixture()
    return _require(SigLoader().load(dotted, root), dotted)
}

fn _require(sig: ModuleSig?, dotted: str) -> ModuleSig {
    if sig is null { throw AssertionError(f"the fixture module `{dotted}` did not load") }
    return sig
}

fn _fixture() -> str {
    var base = __prim__.os_env("TMPDIR") ?? "/tmp"
    while base.len() > 1 and base.ends_with("/") { base = base[:base.len() - 1] }
    let root = f"{base}/jaithon-modsig-tests"

    for dir in [
        root,
        f"{root}/pkg",
        f"{root}/pkg/deep",
        f"{root}/shadow",
        f"{root}/dual",
        f"{root}/lib",
        f"{root}/packages",
        f"{root}/packages/demo",
        f"{root}/packages/demo/src",
        f"{root}/packages/demo/src/demo",
        f"{root}/packages/unlisted",
        f"{root}/packages/unlisted/src",
    ] {
        _ensure_dir(dir)
    }

    _write(f"{root}/packages/demo/jaithon.package.json", "{}\n")
    _write(f"{root}/packages/demo/src/demo/mod.jai", "pub fn value() -> int { return 3 }\n")

    _write(f"{root}/pkg/leaf.jai", _LEAF)
    _write(f"{root}/pkg/mod.jai", "from .leaf import Widget as Gadget\n")
    _write(f"{root}/pkg/deep/mod.jai", "from ..leaf import Widget\n")
    _write(f"{root}/pkg/deep/sibling.jai", "pub fn beside() -> int { return 1 }\n")
    _write(f"{root}/shadow/leaf.jai", "pub fn shadowed() -> int { return 0 }\n")
    _write(f"{root}/dual.jai", "pub fn from_the_file() -> int { return 1 }\n")
    _write(f"{root}/dual/mod.jai", "pub fn from_the_package() -> int { return 2 }\n")
    _write(f"{root}/star.jai", "from pkg.leaf import *\n")
    _write(f"{root}/private_decl.jai", "fn hidden() -> int { return 1 }\n")
    _write(f"{root}/named_src.jai", "pub fn Thing() -> int { return 1 }\n")
    _write(f"{root}/wild_src.jai", "pub class Thing { pub fn init(self) {} }\n")
    _write(f"{root}/both.jai", "from named_src import Thing\nfrom wild_src import *\n")
    _write(f"{root}/cycle_a.jai", "from cycle_b import Ghost\n")
    _write(f"{root}/cycle_b.jai", "from cycle_a import Ghost\n")
    _write(f"{root}/broken.jai", "pub fn oops( {\n")

    for step in 0..7 {
        _write(f"{root}/chain{step}.jai", f"from chain{step + 1} import Deep, Shallow\n")
    }
    _write(f"{root}/chain7.jai", "from chain8 import Deep\npub fn Shallow() -> int { return 7 }\n")
    _write(f"{root}/chain8.jai", "pub fn Deep() -> int { return 8 }\n")
    return root
}

let _LEAF: str = "pub fn make(count: int) -> int { return count }\n" +
    "pub class Widget {\n" +
    "    pub let size: int\n" +
    "    pub fn init(self, size: int) { self.size = size }\n" +
    "}\n" +
    "pub trait Drawable {\n" +
    "    fn draw(self) -> str\n" +
    "}\n" +
    "pub enum Colour { Red, Green }\n" +
    "pub type Size = int\n"

fn _ensure_dir(path: str) -> void {
    let info = __prim__.io_stat(path, true)
    if info is not null and info["kind"] == "dir" { return }
    __prim__.io_mkdir(path)
}

fn _write(path: str, text: str) -> void {
    let handle = __prim__.io_open(path, "wb")
    __prim__.io_write(handle, text)
    __prim__.io_close(handle)
}
