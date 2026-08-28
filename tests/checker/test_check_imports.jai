#: `jaithon.compile.check.checker` -- the import pass (`_bind_imported`) and
#: module-member resolution (`_module_member`), i.e. the two call sites that
#: reach `SigLoader.find()` and must turn a miss into a diagnostic rather than
#: swallow it.
#:
#: `find()` returns `(null, sig)` in two situations that look identical at
#: that call site but are not: the module's signature never loaded (a missing
#: file, a typo'd module name -- nothing to check against) and the signature
#: loaded fine but the requested name simply is not among its exports (a typo
#: in the imported name, or a member that was never public). Only the second
#: is a bug in the program under check. These tests pin both the positive case
#: (a name genuinely missing from a loaded module IS reported) and the two
#: cases that must stay silent (a name that exists, and a module that could
#: not be loaded at all).
from std.test import assert_eq, assert_false, assert_true
from jaithon.compile.check.expr import E0410_UNKNOWN_MEMBER
from jaithon.compile.mod import compile_source

fn _fixture() -> str {
    var base = __prim__.os_env("TMPDIR") ?? "/tmp"
    while base.len() > 1 and base.ends_with("/") { base = base[:base.len() - 1] }
    let root = f"{base}/jaithon-checker-import-tests"
    _ensure_dir(root)
    _write(
        f"{root}/lib_one.jai",
        "pub fn real_thing(x: int) -> int { return x + 1 }\n" +
            "fn hidden_thing() -> int { return 1 }\n"
    )
    return root
}

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

#: The error codes `compile_source` reports for `source`, run against a probe
#: file that lives beside the `lib_one` fixture module so `from lib_one import
#: ...` and `import lib_one` both resolve it from `_dir_of(path)`.
fn _error_codes(source: str) -> list[str] {
    let root = _fixture()
    let built = compile_source(source, f"{root}/probe.jai")
    var codes: list[str] = []
    for diagnostic in built.diagnostics.items {
        if diagnostic.is_error() { codes.push(diagnostic.code) }
    }
    return codes
}

# ------------------------------------------------------------------
# `from M import a, b` -- today's bug: `missing_thing` is silently unbound
# ------------------------------------------------------------------
fn test_importing_a_name_a_local_module_does_not_export_is_reported() -> void {
    let codes = _error_codes("from lib_one import real_thing, missing_thing\n")
    assert_true(E0410_UNKNOWN_MEMBER in codes, f"expected E0410, got {codes}")
}

fn test_importing_a_typo_d_stdlib_name_is_reported() -> void {
    #: The exact reduction from the bug report: `sqrtt` for `sqrt`.
    let codes = _error_codes("from std.math import sqrtt\n")
    assert_true(E0410_UNKNOWN_MEMBER in codes, f"expected E0410, got {codes}")
}

# ------------------------------------------------------------------
# `M.name` -- the same miss, reached through member access instead of import
# ------------------------------------------------------------------
fn test_a_member_a_local_module_does_not_have_is_reported() -> void {
    let codes = _error_codes("import lib_one\nprint(lib_one.not_a_thing)\n")
    assert_true(E0410_UNKNOWN_MEMBER in codes, f"expected E0410, got {codes}")
}

fn test_a_member_the_stdlib_module_does_not_have_is_reported() -> void {
    let codes = _error_codes("import std.math as math\nprint(math.not_a_thing)\n")
    assert_true(E0410_UNKNOWN_MEMBER in codes, f"expected E0410, got {codes}")
}

# ------------------------------------------------------------------
# What must stay silent
# ------------------------------------------------------------------
fn test_importing_a_name_that_is_really_exported_is_not_reported() -> void {
    let codes = _error_codes("from lib_one import real_thing\nprint(real_thing(1))\n")
    assert_false(E0410_UNKNOWN_MEMBER in codes, f"got {codes}")
}

fn test_a_member_that_really_exists_is_not_reported() -> void {
    let codes = _error_codes("import lib_one\nprint(lib_one.real_thing(1))\n")
    assert_false(E0410_UNKNOWN_MEMBER in codes, f"got {codes}")
}

fn test_a_module_whose_signature_cannot_load_is_not_reported_as_an_unknown_member() -> void {
    #: `sig` is null here -- there is no such module, so there is nothing to
    #: report a missing member against. A missing dependency must not turn
    #: into an E0410 storm.
    let codes = _error_codes("from no_such_module_at_all import whatever\n")
    assert_false(E0410_UNKNOWN_MEMBER in codes, f"got {codes}")
}

fn test_a_private_name_is_not_reported_as_unknown() -> void {
    #: `hidden_thing` is a real declaration in `lib_one`, just not `pub`. The
    #: signature loader does not export it, so this is indistinguishable at
    #: `find()` from a typo -- pin that it stays silent (the existing, correct
    #: behaviour) rather than start reporting private names as unknown.
    let codes = _error_codes("from lib_one import hidden_thing\n")
    assert_false(E0410_UNKNOWN_MEMBER in codes, f"got {codes}")
}
