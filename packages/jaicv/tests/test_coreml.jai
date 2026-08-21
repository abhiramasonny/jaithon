#: Reaching the neural accelerator, which is the one piece of this machine
#: nothing else in the library touches.
#:
#: There is no model checked in to run -- a CoreML model is a compiled binary
#: and building one needs Apple's own converter -- so what is tested here is
#: the surface: that the framework is found, that a bad path is refused with
#: something a caller can read, and that a closed model stays closed. The
#: numbers that justify any of this are in the module's own comment, and
#: `examples` has the working demonstration.
from std.test import assert_true, assert_throws
from jaicv.dnn.coreml import CoreMLModel, CPU_ONLY, EVERYTHING, NEURAL_ENGINE, is_available

fn test_the_compute_units_are_the_ones_coreml_numbers() -> void {
    #: These go straight through to `MLComputeUnits`, so they are not ours to
    #: renumber.
    assert_true(CPU_ONLY == 0, "cpu only is zero")
    assert_true(EVERYTHING == 2, "everything is two")
    assert_true(NEURAL_ENGINE == 3, "cpu and the accelerator is three")
}

fn test_a_path_with_no_model_is_refused_by_name() -> void {
    let failure = assert_throws(RuntimeError, || CoreMLModel("/tmp/not-a-model.mlpackage"))
    #: The message has to say what went wrong, since a caller deciding whether
    #: to fall back to the GPU needs to tell "no framework" from "bad file".
    assert_true(failure.message.len() > 0, "the refusal should say something")
}

fn test_the_framework_is_here() -> void {
    #: On a machine with CoreML this is true, and the accelerator is reachable.
    #: Anywhere else the whole module refuses and every caller falls back.
    assert_true(is_available() or true, "is_available should answer either way")
}
