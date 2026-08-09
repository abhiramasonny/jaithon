#: The `__prim__` namespace against Appendix C of the language specification.
#:
#: Appendix C says the runtime provides the operations it lists "and nothing
#: else", which is a claim about a set and so can be checked. It had drifted:
#: an audit found thirty-seven names in the namespace that the appendix never
#: mentioned, most of them second spellings of something already there or
#: helpers that `std` writes in Jaithon anyway. This file is the guard that
#: keeps the two in step, in both directions.
#:
#: A primitive added without a line in Appendix C fails here, and so does a
#: line in Appendix C with no primitive behind it. Seven families are written
#: in the appendix as wildcards rather than enumerated, and are treated as
#: such below.
from std.test import assert_eq, assert_len, assert_true

#: Every name Appendix C spells out. The prefixes the appendix uses as
#: shorthand (`io_open read write` for `io_open`, `io_read`, `io_write`) are
#: expanded, because the namespace holds the expanded form.
let SPECIFIED = [
    # arithmetic on int and float
    "add", "sub", "mul", "div", "floordiv", "mod", "pow", "neg",
    # bitwise
    "band", "bor", "bxor", "bnot", "shl", "shr",
    # comparison
    "eq", "ne", "lt", "le", "gt", "ge",
    # collections
    "list_new", "list_get", "list_set", "list_push", "list_pop", "list_insert",
    "list_remove", "list_len", "list_slice", "list_concat",
    "dict_new", "dict_get", "dict_set", "dict_del", "dict_has", "dict_len",
    "dict_keys", "dict_values",
    "set_new", "set_add", "set_del", "set_has", "set_len",
    # text
    "str_len", "str_get", "str_slice", "str_concat", "str_cmp", "str_find",
    "str_from_codepoint", "str_to_codepoint", "str_encode", "str_decode",
    # objects and types
    "type_of", "is_instance", "cast",
    "obj_new", "obj_get_field", "obj_set_field", "obj_get_method",
    "raise", "current_exception", "traceback",
    # clocks and calendars
    "time_mono", "time_mono_seconds", "time_wall", "sleep", "time_format",
    "time_parse",
    # randomness
    "random_u64", "random_seed",
    # files
    "io_open", "io_read", "io_write", "io_close", "io_seek", "io_flush",
    "io_stat", "io_listdir", "io_mkdir", "io_remove", "io_rename",
    # the operating system
    "module_path", "os_env", "os_argv", "os_exit", "os_cwd", "os_chdir", "os_spawn",
    "os_platform",
    # threads
    "thread_spawn", "thread_join", "thread_detach", "cpu_count",
    # the collector
    "gc_collect", "gc_stats", "gc_disable", "gc_enable",
    # the front end
    "compile", "eval", "exec", "parse_ast", "globals",
    "jaic_build_id",
]

#: The families Appendix C writes as `x_*` rather than listing.
let WILDCARDS = ["f64_", "bytes_", "mutex_", "cond_", "atomic_", "gpu_", "gui_"]

#: `time_sleep` is the same native as `sleep` under a second name, kept only
#: because `lib/std/time.jai` still calls it. It is the one name the runtime
#: publishes that Appendix C does not, and it goes when that call is repointed
#: — which is why this is a subset test and not an equality test.
let TOLERATED = ["time_sleep"]

fn covered_by_a_wildcard(name: str) -> bool {
    for prefix in WILDCARDS {
        if name.starts_with(prefix) { return true }
    }
    return false
}

fn test_every_specified_primitive_exists() -> void {
    let live = dir(__prim__)
    var missing: list[str] = []
    for name in SPECIFIED {
        if not live.contains(name) { missing.push(name) }
    }
    assert_len(missing, 0, f"Appendix C names these, the runtime does not provide them: {missing}")
}

fn test_the_runtime_provides_nothing_else() -> void {
    var extra: list[str] = []
    for name in dir(__prim__) {
        if SPECIFIED.contains(name) { continue }
        if TOLERATED.contains(name) { continue }
        if covered_by_a_wildcard(name) { continue }
        extra.push(name)
    }
    assert_len(extra, 0, f"the runtime provides these, Appendix C does not name them: {extra}")
}

#: Each wildcard family has to have something in it, or the appendix is
#: promising a group the runtime dropped.
fn test_every_wildcard_family_is_populated() -> void {
    let live = dir(__prim__)
    for prefix in WILDCARDS {
        var count = 0
        for name in live {
            if name.starts_with(prefix) { count = count + 1 }
        }
        assert_true(count > 0, f"no primitive is named `{prefix}*`")
    }
}

#: The names the audit removed. Each was either a second spelling of a
#: neighbour or something `std` already writes in Jaithon, so each failed
#: Appendix C's own membership rule. Listing them here stops one drifting back
#: in without a line in the appendix to justify it.
fn test_the_removed_duplicates_stay_removed() -> void {
    let live = dir(__prim__)
    let removed = [
        "os_cpu_count", "os_environ",           # cpu_count, os_env()
        "gc_heap_size",                          # gc_stats()["bytes_allocated"]
        "io_tell", "io_eof",                     # io_seek, and the file methods
        "io_read_line", "io_read_lines",         # derived from io_read
        "read_file", "write_file", "append_file", # std.io writes these
        "os_path_join", "os_path_abs", "os_temp_dir", "os_system",
        "int_gcd", "int_lcm", "int_isqrt", "int_bit_count", "int_bit_length",
        "int_abs", "int_pow_mod",                # std.math, and the int methods
        "pi", "e", "tau", "inf", "nan", "float_epsilon", "int_max", "int_min",
        "random_float", "random_int", "random_bytes",  # over random_u64
        "time_wall_seconds", "str_format", "str_format_value", "disassemble",
        "gui_buffers",                           # a GC root, not an operation
    ]
    for name in removed {
        assert_true(not live.contains(name), f"`__prim__.{name}` is back")
    }
}

#: The first of the primitives the appendix gained. `std.os.platform()` still
#: runs `uname -s` and caches the answer, on the strength of a comment saying
#: the primitive surface has no platform query; it has one, and this is it.
fn test_os_platform_answers_without_a_child_process() -> void {
    let name = __prim__.os_platform()
    assert_true(name.len() > 0, "the platform name is empty")
    assert_eq(name, name.lower())
}

fn test_os_spawn_runs_a_program_and_captures_its_output() -> void {
    let result = __prim__.os_spawn(["/bin/echo", "spawned"], {})
    assert_eq(result["exit_code"], 0)
    assert_eq(result["stdout"], "spawned\n")
    assert_eq(result["stderr"], "")
}

fn test_a_condition_variable_can_be_made_and_freed() -> void {
    let cond = __prim__.cond_new()
    assert_eq(__prim__.type_of(cond), "int")
    __prim__.cond_signal(cond)  # no waiter: a no-op, not an error
    __prim__.cond_broadcast(cond)
    __prim__.cond_free(cond)
}

fn test_io_rename_moves_a_file() -> void {
    let dir = __prim__.os_env("TMPDIR") ?? "/tmp"
    let before = dir + "/jaithon-prim-rename-before"
    let after = dir + "/jaithon-prim-rename-after"

    let handle = __prim__.io_open(before, "wb")
    __prim__.io_write(handle, "moved")
    __prim__.io_close(handle)

    __prim__.io_rename(before, after)
    assert_true(__prim__.io_stat(before) == null, "the old name still exists")

    let reopened = __prim__.io_open(after, "rb")
    let content = __prim__.str_decode(__prim__.io_read(reopened, -1))
    __prim__.io_close(reopened)
    assert_eq(content, "moved")

    __prim__.io_remove(after)
}

fn test_time_format_and_time_parse_are_inverses() -> void {
    let text = __prim__.time_format(0, "%Y-%m-%d", true)
    assert_eq(text, "1970-01-01")
    assert_eq(__prim__.time_parse(text, "%Y-%m-%d", true), 0)
}

fn test_jaic_build_id_is_stable_within_a_process() -> void {
    let id = __prim__.jaic_build_id()
    assert_true(id != 0, "the build id is zero")
    assert_eq(__prim__.jaic_build_id(), id)
}

fn test_globals_sees_the_calling_module() -> void {
    let names = __prim__.globals()
    assert_true(names.contains("SPECIFIED"), "globals() cannot see this module")
}
