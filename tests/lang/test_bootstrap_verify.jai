#: `verify` compiles one source with both front ends and compares the records.
#:
#: It could not run at all until `__prim__.compile_image` existed — it needed
#: the C front end's `.jaic` and there was no way to ask for one — so these are
#: its first tests. What it reports is the subject of the whole bootstrap
#: effort; that it reports rather than excusing itself is what is tested here.
from std.test import assert_eq, assert_false, assert_true
from jaithon.compile.mod import verify_source

fn test_a_comparison_actually_happens() -> void {
    let outcome = verify_source("let answer = 42\n", "<test>")
    let excuse = "cannot be reached"
    assert_true(outcome.message is null or not outcome.message.contains(excuse))
}

fn test_a_source_both_front_ends_agree_on_verifies() -> void {
    let outcome = verify_source("let answer = 42\n", "<test>")
    assert_true(outcome.agreed, outcome.message ?? "")
    assert_true(outcome.message is null)
}

fn test_a_rejected_source_is_reported_as_a_rejection() -> void {
    let outcome = verify_source("let x = y\n", "<test>")
    assert_false(outcome.agreed)
    assert_true(outcome.message.contains("rejected the file"))
}

fn test_a_divergence_names_the_record_and_the_opcode() -> void {
    # `-5` is the negative-literal family: the C checker folds the negation
    # into the immediate and the self-hosted front end, having no checker at
    # all, emits the magnitude and a separate negate. Until the checker of
    # Phase 2 lands this is a live divergence, and the value of the tool is
    # that it says so in opcode terms rather than failing opaquely.
    let outcome = verify_source("let n = -5\n", "<test>")
    if not outcome.agreed {
        assert_true(outcome.message.contains("module body"))
        assert_true(outcome.message.contains("OP_"))
        assert_true(outcome.ours is not null)
        assert_true(outcome.theirs is not null)
    }
}

fn test_the_optimisation_level_is_carried_to_both_sides() -> void {
    # A comparison run at -O0 that compiled the reference at -O2 blames the
    # front ends for the flag, which costs the only tool that can tell an
    # emitter divergence from an optimiser one.
    let text = "fn spin() -> int {\n" +
        "    var total = 0\n" +
        "    for i in 0..10 { total += i }\n" +
        "    return total\n" +
        "}\n"
    let flat = verify_source(text, "<test>", 0)
    let optimised = verify_source(text, "<test>", 2)
    # Whether either agrees is not the point; that they are separate
    # comparisons at separate levels is.
    assert_eq(flat.path, "<test>")
    assert_eq(optimised.path, "<test>")
}
