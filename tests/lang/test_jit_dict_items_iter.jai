#: `for (k, v) in d.items()` after the compiled tier has specialised it.
#:
#: The tier steps a dict-items view inline -- it walks the table's insertion
#: order array and reads the entry itself, which is jaiTableNext instruction for
#: instruction -- rather than calling out per entry. That buys the loop, and it
#: also means every fact the interpreter checks has to be checked here: that the
#: iterator really is a dict-items one, that its source really is a dict, that
#: the dict has not changed under the loop, and that the key and the value still
#: have the kinds the compile sampled off the dict's first live entry.
#:
#: Component kinds are a SPECIALISATION, not a promise. `dict[str, int]` is
#: enforced on store, but `var d: dict = {}` has no key or value kind at all and
#: LANGUAGE.md is explicit that a container reached through `any` is not
#: element-checked, so a dict that is homogeneous when sampled and mixed
#: afterwards is reachable from ordinary code and must deopt, not guess.
#:
#: Every pair below has a `str` on one side and an `int` on the other. That is
#: deliberate and it is the same reason tests/golden/for_iter_pair.jai gives:
#: the loop pattern's elements are not in source order relative to the slots the
#: resolver assigns, so an arm that emitted its two operands the wrong way round
#: passes silently on `(int, int)` and only a mixed pair catches it.
#:
#: The answers here are the interpreter's, so `JAITHON_NO_JIT=1` proves the
#: expectations and the plain run proves the compiled code agrees. A deopt does
#: not retire compiled code, so one function keeps its teeth across a whole
#: test.
#:
#: scripts/jit_compile_check.py fails if any name below stops reaching the
#: compiled tier under JAI_JIT_WHY=1. `join_keys` and `spell` are NOT in it:
#: both concatenate a `str` (`acc = acc + k`), and OP_ADD_LOCALS on two dynamic
#: operands has no measuring-pass support at all -- "add-bind of a kind that is
#: neither int nor float", every attempt, warm or not. That is a real gap in
#: what this file can pin about the compiled tier's key-side handling; it is
#: not a heat problem this marker (or moving the case) can fix, so it is
#: recorded here rather than silently marked as covered. `bag_total` is
#: correctly never compiled -- `Bag.items()` is a user method, not a dict, and
#: `OP_FOR_ITER_BIND`'s whole reason for existing is to decline exactly that
#: receiver rather than guess, so its absence from the marker is the point, not
#: a gap.
# jit-compiles: sum_values, count, one_call_sum, nested
from std.test import assert_eq, assert_near, assert_throws

#: `any` rather than `int` because the whole point is that a value may widen the
#: accumulator out from under the compiled kind.

#: Above JAI_JIT_THRESHOLD entries. The marker names the FUNCTION tier, which
#: compiles at a count of entries; a body merely looped over many times can only
#: be caught by the OSR tier's 250us sampling timer, which a fast machine does
#: not trip -- and that is why this marker had silently stopped holding.
let WARM_CALLS = 200

fn sum_values(d: dict) -> any {
    var acc = 0
    for (_k, v) in d.items() { acc += v }
    return acc
}

#: Reads the key half only, so a swapped pair binds an int here and throws.
fn join_keys(d: dict) -> any {
    var acc = ""
    for (k, _v) in d.items() { acc = acc + k }
    return acc
}

#: Both halves in one string, which pins the pairing rather than the two
#: multisets: a step that yielded the right keys and the right values but
#: matched up wrongly passes the two tests above and fails this one.
fn spell(d: dict) -> any {
    var acc = ""
    for (k, v) in d.items() { acc = acc + f"{k}={v};" }
    return acc
}

#: Never reads either component, so only the guards and the resume point decide
#: its answer. Under JAITHON_JIT_DEOPT_STRESS=1 every guard fires on every
#: iteration, and a guard placed after the iterator had already advanced would
#: SKIP an entry rather than repeat it -- which no sum can tell apart from a
#: wrong kind, and a count can.
fn count(d: dict) -> any {
    var seen = 0
    for (_k, _v) in d.items() { seen += 1 }
    return seen
}

#: Long enough that the sampling tier compiles these. Hotness is measured in
#: SIGPROF ticks, not iterations, so this is a duration and not a count --
#: generous on purpose. Nothing here asserts that a compile happened: the test
#: is correct under `JAITHON_NO_JIT=1` too, it just has less to say.
fn warm(d: dict) -> void {
    for _r in 0..6000 {
        sum_values(d)
        join_keys(d)
        spell(d)
        count(d)
    }
}

fn scores() -> dict {
    var d: dict = {}
    d["aa"] = 1
    d["bbb"] = 2
    d["cccc"] = 3
    return d
}

fn test_a_str_key_and_an_int_value_keep_their_sides() -> void {
    let d = scores()
    warm(d)
    assert_eq(spell(d), "aa=1;bbb=2;cccc=3;")
    assert_eq(join_keys(d), "aabbbcccc")
    assert_eq(sum_values(d), 6)
    assert_eq(count(d), 3)
}

fn test_a_str_value_after_ints_raises_rather_than_binding_a_pointer() -> void {
    let d = scores()
    warm(d)
    assert_eq(sum_values(d), 6)
    d["z"] = "boom"
    #: The failure this guards: an ObjString's address, added as an integer.
    assert_throws(TypeError, || sum_values(d))
    assert_throws(TypeError, || sum_values(d))
}

fn test_a_float_value_after_ints_widens_rather_than_binding_its_bits() -> void {
    let d = scores()
    warm(d)
    assert_eq(sum_values(d), 6)
    d["z"] = 0.5
    #: int + float is legal and widens, so a missing guard here is a wrong
    #: NUMBER and not a throw.
    assert_near(sum_values(d), 6.5)
    assert_near(sum_values(d), 6.5)
}

fn test_an_int_key_after_str_keys_raises() -> void {
    let d = scores()
    warm(d)
    assert_eq(join_keys(d), "aabbbcccc")
    d[9] = 4
    assert_throws(TypeError, || join_keys(d))
    assert_throws(TypeError, || join_keys(d))
    #: The value half is still an int, so the other loop keeps working: the two
    #: components are guarded apart, not together.
    assert_eq(sum_values(d), 10)
}

fn test_a_null_value_after_ints_raises() -> void {
    let d = scores()
    warm(d)
    assert_eq(sum_values(d), 6)
    d["z"] = null
    assert_throws(TypeError, || sum_values(d))
}

fn test_every_entry_is_still_seen_after_a_guard_fires() -> void {
    let d = scores()
    warm(d)
    assert_eq(count(d), 3)
    d["z"] = "tail"
    assert_eq(count(d), 4)
    assert_eq(count(d), 4)
}

fn test_a_deleted_entry_leaves_a_hole_the_walk_skips() -> void {
    #: A delete writes -1 into the table's order array and leaves it there --
    #: the array is only compacted on a later insert that runs out of room. So
    #: the compiled scan has to skip holes exactly as jaiTableNext does, and a
    #: scan that stopped at the first one, or counted it, is caught here. The
    #: hole is at the FRONT as well as in the middle, because the first live
    #: entry is also where the compile takes its key and value kinds from.
    var d: dict = {}
    d["aa"] = 1
    d["bbb"] = 2
    d["cccc"] = 3
    d["ddddd"] = 4
    d.remove("aa")
    d.remove("cccc")
    warm(d)
    assert_eq(count(d), 2)
    assert_eq(sum_values(d), 6)
    assert_eq(spell(d), "bbb=2;ddddd=4;")
}

fn test_mutating_the_dict_under_a_compiled_loop_raises() -> void {
    let d = scores()
    warm(d)
    assert_eq(sum_values(d), 6)
    #: The version guard, and the message belongs to jaiIterNext -- so the
    #: compiled arm hands the whole instruction back unadvanced rather than
    #: raising anything of its own.
    assert_throws(RuntimeError, || grow_under_loop(d))
    #: An in-place store bumps the version too, which is jaiListTouch's contract
    #: applied to a dict: conservative, and what the interpreter does.
    assert_throws(RuntimeError, || overwrite_under_loop(d))
}

fn grow_under_loop(d: dict) -> int {
    var n = 0
    for (k, _v) in d.items() {
        d[f"{k}!"] = 0
        n += 1
    }
    return n
}

fn overwrite_under_loop(d: dict) -> int {
    var n = 0
    for (k, _v) in d.items() {
        d[k] = 99
        n += 1
    }
    return n
}

fn test_an_empty_dict_yields_nothing() -> void {
    #: Nothing to sample, so the tier declines this loop rather than guessing a
    #: kind -- and it still has to be right.
    var d: dict = {}
    warm(d)
    assert_eq(count(d), 0)
    assert_eq(sum_values(d), 0)
    assert_eq(join_keys(d), "")
}

class Bag {
    pub var made: int

    fn init(self) -> void { self.made = 0 }

    pub fn items(self) -> list[tuple[str, int]] {
        self.made += 1
        return [("u", 10), ("w", 20)]
    }
}

fn bag_total(b: Bag) -> int {
    var t = 0
    for (_k, v) in b.items() { t += v }
    return t
}

#: Everything above reaches the compiled tier through the FUNCTION tier: the
#: bodies are small and called thousands of times. The OSR tier is a different
#: entry with different rules -- the iterator is not on its modelled operand
#: stack at all, it arrives in a reserved register while staying on the
#: interpreter's stack, and the exhausted exit has to tell the interpreter to
#: pop it. Get that wrong and the loop leaves a spare iterator behind, or drops
#: a live value; neither shows up in a sum.
#:
#: These functions are called ONCE, so the function tier never sees them and
#: whatever compiles came in through OSR. `mark` is `any` so that binding the
#: key needs no OP_TYPE_GUARD, which is unarmed and would decline the loop for
#: an unrelated reason.
fn one_call_sum(d: dict[str, int], rounds: int) -> str {
    var acc = 0
    var mark: any = ""
    var r = 0
    while r < rounds {
        for (k, v) in d.items() {
            acc += v
            mark = k
        }
        r += 1
    }
    return f"{mark}/{acc}"
}

#: Nesting is the shape that can SEE the exhausted exit go wrong. The outer
#: loop's iterator sits on the operand stack UNDER the inner one, so an inner
#: exit that forgot to tell the interpreter to drop its iterator leaves the
#: stale inner iterator where the outer one should be, and the outer loop then
#: steps the wrong thing. A single flat loop cannot catch that -- nothing after
#: it looks at the stack, so the spare entry is dead until the frame returns,
#: and the first version of this test passed with the pop deliberately removed.
#: With the pop removed this returns a wrong, run-to-run varying total (43998660
#: and 286545300 were two of them) rather than 939870000 -- varying because how
#: much of the walk the compiled form did depends on when the sampler entered.
fn nested(d: dict[str, int], rounds: int) -> int {
    var acc = 0
    var r = 0
    while r < rounds {
        for (_k1, v1) in d.items() {
            for (_k2, v2) in d.items() { acc += v1 * v2 }
        }
        r += 1
    }
    return acc
}

fn many_keys(n: int) -> dict[str, int] {
    var d: dict[str, int] = {}
    var i = 0
    while i < n {
        d[f"k{i}"] = i
        i += 1
    }
    return d
}

fn test_an_osr_entry_at_a_pair_head_binds_both_halves() -> void {
    let d = many_keys(400)
    var call = 0
    while call < WARM_CALLS {
        assert_eq(one_call_sum(d, 1), "k399/79800")
        call += 1
    }
    assert_eq(one_call_sum(d, 3000), "k399/239400000")
}

fn test_an_osr_pair_loop_leaves_the_stack_where_it_found_it() -> void {
    let d = many_keys(60)
    var call = 0
    while call < WARM_CALLS {
        assert_eq(nested(d, 1), 3132900)
        call += 1
    }
    assert_eq(nested(d, 300), 939870000)
}

fn test_a_user_class_items_method_is_not_hijacked() -> void {
    #: The opcode's whole reason for existing is that the emitter cannot know
    #: the receiver's type, so the compiled form must not assume it either: a
    #: class with its own `items()` still has its method called, once per loop.
    let b = Bag()
    for _r in 0..6000 { bag_total(b) }
    assert_eq(bag_total(b), 30)
    assert_eq(b.made, 6001)
}
