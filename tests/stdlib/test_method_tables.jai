#: The checker advertises a method name (`kListMethodNames` and its five
#: siblings, `src/runtime/builtins/builtins.c` around line 483) and the
#: runtime dispatches on a separate table (`kListMethods` and its siblings,
#: under `src/runtime/builtins/collections/` and `text/`). Nothing keeps
#: these in step, and a reduction found 46 names where they had already come
#: apart: advertised, and raising `AttributeError` the moment they run.
#:
#: `dir(receiver)` is the one thing in the runtime that reads a candidate
#: table (`candidatesFor` in builtins.c) and keeps only the names that
#: `lookup()` -- the exact function a `.name(...)` call resolves through --
#: actually binds. So `dir(receiver).contains(name)` and "`receiver.name(...)`
#: would not raise AttributeError" are the same fact; this file checks the
#: first to guard the second, which sidesteps ever calling a mutator
#: (`.clear()`, `.shuffle()`, ...) just to find out whether its name exists.
#: (`./jaithon check` was confirmed to not be a second path here -- it exits 0
#: even for a receiver method name that was never advertised at all, e.g.
#: `xs.totally_fake_method_xyz()` -- so the only live consumer of the
#: candidate tables today is `dir()`.)
#:
#: Scope: the six receiver types the reduction actually found drift in --
#: bytes, iter, list, range, str, tuple. dict, set, int, float, file and
#: module were checked by the same reduction and had none.
#:
#: docs/agents/methods-classify.md ruled on each of the 46: eleven get a
#: runtime implementation (below, via ADVERTISED), nineteen get removed from
#: the advertised list entirely (so they are simply absent below and this
#: file has nothing further to say about them), and sixteen are left
#: `UNCERTAIN` -- still advertised, still gaps, pending a reviewer. Both
#: halves are enforced: an ADVERTISED name that is not UNCERTAIN must work,
#: and an UNCERTAIN name must still not work, so a name quietly starting to
#: work (or stopping) is caught either way instead of drifting unnoticed.
from std.test import assert_len, assert_true

#: Mirrors kStrMethodNames (builtins.c). No UNCERTAIN entries for str: its
#: three gap names (is_empty, iter, slice) are all implemented below and its
#: other five (at, code_at, encode, lines, reversed) are simply gone from the
#: advertised list.
let STR_ADVERTISED = [
    "capitalize", "center", "chars", "contains", "count", "ends_with", "find",
    "format", "index", "is_alnum", "is_alpha", "is_digit", "is_empty",
    "is_lower", "is_space", "is_upper", "iter", "join", "len", "lower",
    "lstrip", "pad_left", "pad_right", "repeat", "replace", "rfind", "rstrip",
    "slice", "split", "splitlines", "starts_with", "strip", "title",
    "to_float", "to_int", "to_str", "upper",
]
let STR_UNCERTAIN: list[str] = []

#: Mirrors kListMethodNames. `at`, `set`, `is_empty` and `chunks` (renamed
#: from the runtime's old singular `chunk`) are implemented; `append`, `drop`,
#: `take`, `to_list`, `clone`, `concat`, `find`, `fold` and `position` are gone
#: from the advertised list, each a second name for something that exists.
#: Nothing is UNCERTAIN any more -- every gap got a ruling.
let LIST_ADVERTISED = [
    "all", "any", "at", "chunks", "clear", "contains",
    "copy", "count", "enumerate", "extend", "filter", "first",
    "flatten", "index", "insert", "is_empty", "iter", "join", "last",
    "len", "map", "max", "min", "pop", "push", "reduce", "remove",
    "reverse", "reversed", "set", "shuffle", "slice", "sort", "sorted", "sum",
    "unique", "zip",
]
let LIST_UNCERTAIN: list[str] = []

#: Mirrors kTupleMethodNames. at and is_empty are implemented below;
#: first/last/second/slice are gone from the advertised list. No UNCERTAIN
#: entries for tuple -- all six of its gap names got a ruling.
let TUPLE_ADVERTISED = [
    "at", "contains", "count", "get", "index", "is_empty", "iter", "len",
    "to_list",
]
let TUPLE_UNCERTAIN: list[str] = []

#: Mirrors kRangeMethodNames. is_empty is implemented below; nothing else in
#: range's table changed.
let RANGE_ADVERTISED = [
    "contains", "is_empty", "iter", "len", "reversed", "start", "step",
    "stop", "to_list",
]
let RANGE_UNCERTAIN: list[str] = []

#: Mirrors kBytesMethodNames. at, is_empty and iter are implemented below, and
#: `get` is now an alias of `at` -- the same answer tuple's pair already got,
#: rather than leaving one sequence type inconsistent with the rest.
let BYTES_ADVERTISED = [
    "at", "concat", "contains", "decode", "get", "hex", "is_empty", "iter",
    "len", "slice", "to_list",
]
let BYTES_UNCERTAIN: list[str] = []

#: Mirrors kIterMethodNames. The adapter surface lives in `lib/std/iter`, not
#: on the runtime ObjIter, which has only ever carried these five -- so
#: all/any/enumerate/filter/map/zip AND chain/count/find/fold/last/nth/
#: peekable/to_list are all gone from the advertised list. `iter` goes the
#: other way: implemented, working, and previously missing here, so `dir()`
#: hid it.
let ITER_ADVERTISED = [
    "collect", "drop", "iter", "next", "take",
]
let ITER_UNCERTAIN: list[str] = []

fn assert_advertised_methods_work(type_name: str, receiver: any,
                                  advertised: list[str],
                                  uncertain: list[str]) -> void {
    let live = dir(receiver)
    var broken: list[str] = []
    for name in advertised {
        if uncertain.contains(name) { continue }
        if not live.contains(name) { broken.push(name) }
    }
    assert_len(broken, 0,
        f"{type_name} advertises these but the runtime does not implement " +
            f"them, so calling them raises AttributeError: {broken}")
}

fn assert_uncertain_names_are_still_gaps(type_name: str, receiver: any,
                                         uncertain: list[str]) -> void {
    let live = dir(receiver)
    var resolved: list[str] = []
    for name in uncertain {
        if live.contains(name) { resolved.push(name) }
    }
    assert_len(resolved, 0,
        f"{type_name} lists these as UNCERTAIN (still a gap) in " +
            "docs/agents/methods-classify.md, but the runtime now implements " +
            f"them -- update the classification and this test: {resolved}")
}

fn test_str_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("str", "abc", STR_ADVERTISED, STR_UNCERTAIN)
}

fn test_list_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("list", [1, 2, 3], LIST_ADVERTISED, LIST_UNCERTAIN)
}

fn test_tuple_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("tuple", (1, 2, 3), TUPLE_ADVERTISED, TUPLE_UNCERTAIN)
}

fn test_range_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("range", 0..3, RANGE_ADVERTISED, RANGE_UNCERTAIN)
}

fn test_bytes_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("bytes", bytes("abc"), BYTES_ADVERTISED, BYTES_UNCERTAIN)
}

fn test_iter_advertised_methods_are_all_implemented() -> void {
    assert_advertised_methods_work("iter", [1, 2, 3].iter(), ITER_ADVERTISED, ITER_UNCERTAIN)
}

fn test_uncertain_names_have_not_quietly_started_working() -> void {
    assert_uncertain_names_are_still_gaps("list", [1, 2, 3], LIST_UNCERTAIN)
    assert_uncertain_names_are_still_gaps("bytes", bytes("abc"), BYTES_UNCERTAIN)
    assert_uncertain_names_are_still_gaps("iter", [1, 2, 3].iter(), ITER_UNCERTAIN)
}

#: The reduction's own method: generate a call to every advertised name with
#: zero arguments and confirm only AttributeError -- never TypeError, which is
#: what a wrong arity raises instead -- is what dir() is standing in for
#: above. This is the direct form for the eleven names
#: docs/agents/methods-classify.md actually asked to be implemented, run
#: against real values rather than through dir()'s filter, so a change that
#: made dir() report a name present without actually being callable would
#: still be caught.
fn test_the_eleven_implemented_names_are_directly_callable() -> void {
    let xs = [1, 2, 3]
    xs.is_empty()
    xs.chunks(2)
    let t = (1, 2, 3)
    t.at(0)
    t.is_empty()
    (0..3).is_empty()
    "abc".is_empty()
    "abc".iter()
    "abc".slice(0, 1)
    let b = bytes("abc")
    b.at(0)
    b.is_empty()
    b.iter()
    assert_true(true, "every implemented name above returned without raising")
}

#: `str.slice` with its optional bounds left off, called AFTER a four-argument
#: call.
#:
#: `primStrSlice` has two registrations -- `__prim__.str_slice` at (3, 4) and
#: the `slice` method at (1, 4) -- and `callNativeAt` does not pad a missing
#: optional. It hands the native the raw stack window and a count. So reading
#: `args[1]` without testing `argc` read whatever the previous expression left
#: in that slot, and `"hello".slice(1)` answered "e" instead of "ello".
#:
#: The four-argument call is what makes it fail: in isolation those slots
#: happen to be zero, which is why the suite and the rest of this file missed
#: it. `packages/jaicv/src/jaicv/dnn/tensorflow.jai:648` is a real one-argument
#: `str.slice`, so this was a live wrong answer, not a hypothetical.
fn four_args(a: any, b: any, c: any, d: any) -> str { return "x" }

pub fn test_str_slice_does_not_read_past_the_stack() -> void {
    let _ = four_args(1, "poison", 3, 4)
    assert_eq("hello".slice(1), "ello", "one bound")
    let _2 = four_args(1, "poison", 3, 4)
    assert_eq("hello".slice(), "hello", "no bounds")
    let _3 = four_args(1, "poison", 3, 4)
    assert_eq("hello".slice(1, 3), "el", "both bounds")
}
