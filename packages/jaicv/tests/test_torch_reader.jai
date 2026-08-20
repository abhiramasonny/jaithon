#: Reading PyTorch checkpoints, against torch's own answers.
#:
#: `tools/torch_fixtures.py` writes the checkpoints in `tests/models/` with
#: torch 2.13 and records everything it put in them — every tensor's path,
#: dtype, shape and every element, printed with enough digits to round-trip a
#: float64 — into `torch_cases.txt`. This reads both and requires exact
#: agreement. Nothing here gets a tolerance: every number in a checkpoint is
#: bytes torch wrote and this reader either puts them back together or does
#: not, and a reader that is close is a reader that is wrong.
#:
#: The fixtures between them cover every dtype torch has a storage class for,
#: a 0-dimensional tensor, two empty ones, three views with strides that are
#: not the packing of their shape, two tensors sharing one storage, a nested
#: dict and a list — and, in separate files, the stored, deflated and zip64
#: forms of the container. The refusals are checked too: a pickle naming
#: `os.system`, a pickled `nn.Module`, and a pre-1.6 checkpoint must each be
#: turned away with an error that says which of the three it was.
import std.io as io
import std.math
from jaicv.dnn.torch import TorchTensor, read_torch_state_dict, read_torch_state_dict_bytes
from std.test import assert_eq, assert_true

let MODELS = "packages/jaicv/tests/models/"

#: The dtypes whose exact reading is `ints`, not `floats`.
let WHOLE = ["int64", "int32", "int16", "int8", "uint8", "bool"]

#: One tensor as torch reported it.
class Case {
    pub let file: str
    pub let key: str
    pub let dtype: str
    pub let shape: list[int]
    pub let values: list[str]

    pub fn init(self, file: str, key: str, dtype: str, shape: list[int], values: list[str]) {
        self.file = file
        self.key = key
        self.dtype = dtype
        self.shape = shape
        self.values = values
    }

    pub fn label(self) -> str { return self.file + " " + self.key }
}

fn words(line: str) -> list[str] {
    var out: list[str] = []
    for part in line.split(" ") {
        if part != "" { out.push(part) }
    }
    return out
}

fn whole(text: str) -> int { return text.to_int() ?? 0 }

fn load_cases() -> list[Case] {
    var cases: list[Case] = []
    for line in io.read_file(MODELS + "torch_cases.txt").split("\n") {
        let parts = words(line)
        if parts.len() == 0 or parts[0] != "tensor" { continue }
        let rank = whole(parts[4])
        var shape: list[int] = []
        for axis in 0..rank { shape.push(whole(parts[5 + axis])) }
        let count = whole(parts[5 + rank])
        var values: list[str] = []
        for index in 0..count { values.push(parts[6 + rank + index]) }
        cases.push(Case(parts[1], parts[2], parts[3], shape, values))
    }
    return cases
}

#: How far one tensor is from what torch says it holds, as a count of elements
#: that differ and the first of them.
fn disagreement(tensor: TorchTensor, recorded: Case) -> tuple[int, str] {
    var wrong = 0
    var first = ""
    if WHOLE.contains(recorded.dtype) {
        let got = tensor.ints()
        for index in 0..got.len() {
            let want = whole(recorded.values[index])
            if got[index] != want {
                wrong += 1
                if first == "" { first = f"[{index}] is {got[index]}, torch says {want}" }
            }
        }
        return (wrong, first)
    }
    let got = tensor.floats()
    for index in 0..got.len() {
        let want = recorded.values[index].to_float() ?? 0.0
        if got[index] != want {
            wrong += 1
            if first == "" { first = f"[{index}] is {got[index]}, torch says {want}" }
        }
    }
    return (wrong, first)
}

fn test_every_recorded_tensor_matches_torch() -> void {
    let cases = load_cases()
    assert_true(cases.len() > 0, "the oracle file records no tensors")
    var loaded: dict[str, dict[str, TorchTensor]] = {}
    for recorded in cases {
        if not loaded.contains(recorded.file) {
            loaded[recorded.file] = read_torch_state_dict(MODELS + recorded.file)
        }
        let weights = loaded[recorded.file]
        assert_true(
            weights.contains(recorded.key),
            f"{recorded.label()} is missing from what was read"
        )
        let tensor = weights[recorded.key]
        assert_eq(tensor.dtype, recorded.dtype, f"{recorded.label()} came back the wrong dtype")
        assert_eq(tensor.shape, recorded.shape, f"{recorded.label()} came back the wrong shape")
        assert_eq(
            tensor.numel(),
            recorded.values.len(),
            f"{recorded.label()} came back the wrong size"
        )
        assert_eq(
            tensor.data.len(),
            tensor.numel() * tensor.item_size(),
            f"{recorded.label()} carries the wrong number of bytes for its shape"
        )
        let (wrong, first) = disagreement(tensor, recorded)
        assert_eq(
            wrong,
            0,
            f"{recorded.label()}: {wrong} of {recorded.values.len()} elements differ, {first}"
        )
    }
}

fn test_nothing_extra_comes_back() -> void {
    #: The oracle records every tensor in each fixture, so a reader that
    #: invented a key — an optimizer state read as a weight, a hook dict
    #: flattened by mistake — would show up here and nowhere else.
    let cases = load_cases()
    var expected: dict[str, int] = {}
    for recorded in cases {
        expected[recorded.file] = (expected.get(recorded.file, 0) ?? 0) + 1
    }
    for (file, count) in expected.items() {
        assert_eq(
            read_torch_state_dict(MODELS + file).len(),
            count,
            f"{file} did not come back with the {count} tensors torch put in it"
        )
    }
}

fn test_stored_deflated_and_zip64_containers_agree() -> void {
    #: The same nine tensors, in three archives that differ only in how the
    #: container holds them: torch's own stored members, the same members
    #: deflated, and the same again with every size and offset behind the
    #: zip64 sentinel the way a checkpoint past 4 GiB has to write them.
    let stored = read_torch_state_dict(MODELS + "torch_smallnet.pt")
    for name in ["torch_deflated.pt", "torch_zip64.pt"] {
        let other = read_torch_state_dict(MODELS + name)
        assert_eq(other.len(), stored.len(), f"{name} holds a different number of tensors")
        for (key, tensor) in stored.items() {
            assert_true(other.contains(key), f"{name} is missing {key}")
            assert_eq(other[key].dtype, tensor.dtype, f"{name} {key} has a different dtype")
            assert_eq(other[key].shape, tensor.shape, f"{name} {key} has a different shape")
            assert_eq(other[key].data, tensor.data, f"{name} {key} holds different bytes")
        }
    }
}

fn test_views_are_gathered_through_their_strides() -> void {
    #: `wide` is `arange(12).reshape(3, 4)`. Its transpose and its column slice
    #: are saved as the whole storage plus a stride, so a reader that ignored
    #: the stride would hand back the storage in its own order and every one of
    #: these would still be the right shape and the right length.
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    assert_eq(
        weights["transposed"].floats(),
        [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0],
        "a transpose was not gathered through its strides"
    )
    assert_eq(
        weights["slice"].floats(),
        [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
        "a column slice was not gathered through its strides"
    )
    assert_eq(
        weights["step"].floats(),
        [-2.0, -0.5, 1.0, 2.5],
        "a strided slice was not gathered through its stride"
    )
}

fn test_a_shared_storage_gives_each_view_its_own_window() -> void {
    #: Both halves are views onto one 12-element storage, at offsets 0 and 6.
    #: Reading them independently is only correct if the offset is honoured;
    #: reading them at all is only cheap if the storage is fetched once, and
    #: this at least proves the second read did not disturb the first.
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    let head = weights["shared_head"].floats()
    let tail = weights["shared_tail"].floats()
    var whole: list[float] = []
    for value in head { whole.push(value) }
    for value in tail { whole.push(value) }
    var expected: list[float] = []
    for index in 0..12 { expected.push(float(index) * 0.5 - 2.0) }
    assert_eq(whole, expected, "two views of one storage did not rebuild it")
}

fn test_signed_zero_and_the_infinities_survive() -> void {
    #: `==` cannot tell -0.0 from 0.0 and cannot be asked about an infinity's
    #: sign either, so the recorded comparison passes on values this would
    #: catch. Every one of these is a bit pattern the float32 decode has to
    #: reproduce exactly rather than approach.
    let special = read_torch_state_dict(MODELS + "torch_dtypes.pt")["special"].floats()
    assert_true(math.is_inf(special[0]) and special[0] > 0.0, "+inf did not come back")
    assert_true(math.is_inf(special[1]) and special[1] < 0.0, "-inf did not come back")
    assert_eq(math.copysign(1.0, special[2]), 1.0, "0.0 came back with a sign")
    assert_eq(math.copysign(1.0, special[3]), -1.0, "-0.0 lost its sign")
    assert_eq(special[4], math.ldexp(1.0, -149), "the smallest subnormal float32 did not survive")
}

fn test_the_integer_extremes_are_exact() -> void {
    #: An int64 at its limits is past what a double holds, which is the whole
    #: reason a tensor keeps its bytes rather than a list of floats.
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    assert_eq(
        weights["i64"].ints(),
        [(-9223372036854775808), 0, 9223372036854775807],
        "int64 did not survive the trip"
    )
    assert_eq(weights["i8"].ints(), [-128, 127], "int8 did not survive the trip")
    assert_eq(weights["u8"].ints(), [0, 128, 255], "uint8 was read as signed")
    assert_eq(weights["bool"].bools(), [true, false, true, true], "a mask did not survive the trip")
}

fn test_an_empty_and_a_scalar_tensor_keep_their_shapes() -> void {
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    assert_eq(weights["scalar"].shape, [], "a 0-dimensional tensor did not stay 0-dimensional")
    assert_eq(weights["scalar"].numel(), 1, "a 0-dimensional tensor holds one element")
    assert_eq(weights["empty"].shape, [0], "an empty tensor lost its shape")
    assert_eq(weights["empty"].numel(), 0, "an empty tensor holds no elements")
    assert_eq(weights["empty2d"].shape, [0, 5], "an empty tensor lost its trailing dimension")
    assert_eq(weights["empty2d"].data.len(), 0, "an empty tensor carries no bytes")
}

fn test_nesting_arrives_as_dotted_paths() -> void {
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    assert_true(weights.contains("nested.inner.deep"), "a nested dict was not flattened")
    assert_true(weights.contains("listed.0"), "a list of tensors was not indexed")
    assert_true(weights.contains("listed.1"), "a list of tensors was not indexed")
    assert_eq(weights["listed.1"].floats(), [2.0, 3.0], "a list entry read the wrong tensor")
}

fn test_reading_from_bytes_matches_reading_from_a_path() -> void {
    let path = read_torch_state_dict(MODELS + "torch_smallnet.pt")
    let raw = read_torch_state_dict_bytes(io.read_bytes(MODELS + "torch_smallnet.pt"))
    assert_eq(raw.len(), path.len(), "the two entry points read different numbers of tensors")
    for (key, tensor) in path.items() {
        assert_eq(raw[key].data, tensor.data, f"{key} read differently from bytes")
    }
}

#: The message a call threw, or `""` when it returned.
fn refusal(path: str) -> str {
    try {
        let _weights = read_torch_state_dict(path)
    } catch e: Error {
        return e.message
    }
    return ""
}

fn test_a_pickle_naming_os_system_is_refused() -> void {
    #: The checkpoint is a real archive with a real `data.pkl`; the pickle in
    #: it is `GLOBAL os.system`, `'echo pwned'`, `REDUCE`. A reader that
    #: resolved globals the way pickle does would run it.
    let message = refusal(MODELS + "torch_hostile.pt")
    assert_true(message != "", "a pickle naming os.system was not refused")
    assert_true(
        message.contains("os.system"),
        f"the refusal does not name what it refused: {message}"
    )
    assert_true(message.contains("refusing"), f"the refusal does not read as one: {message}")
}

fn test_a_pickled_module_is_refused_by_name() -> void {
    let message = refusal(MODELS + "torch_module.pt")
    assert_true(message != "", "a pickled nn.Module was read as though it were a state dict")
    assert_true(
        message.contains("nn.Module"),
        f"the error does not say what the file holds: {message}"
    )
    assert_true(
        message.contains("state_dict"),
        f"the error does not say what to do instead: {message}"
    )
}

fn test_a_legacy_checkpoint_is_named_rather_than_half_read() -> void {
    let message = refusal(MODELS + "torch_legacy.pt")
    assert_true(message != "", "a pre-1.6 checkpoint was not detected")
    assert_true(
        message.contains("pre-1.6"),
        f"the error does not name the format it found: {message}"
    )
}

fn test_rubbish_is_not_mistaken_for_a_checkpoint() -> void {
    var threw = ""
    try {
        let _weights = read_torch_state_dict_bytes(bytes([1, 2, 3, 4, 5, 6, 7, 8]))
    } catch e: Error {
        threw = e.message
    }
    assert_true(threw != "", "eight arbitrary bytes were read as a checkpoint")
    assert_true(
        threw.contains("not a torch checkpoint"),
        f"the error does not say what was wrong: {threw}"
    )
}

fn test_a_truncated_archive_is_caught() -> void {
    #: Half a checkpoint has no end-of-central-directory record, and the half
    #: that is there must not be read as though it were the whole.
    let raw = io.read_bytes(MODELS + "torch_smallnet.pt")
    var threw = ""
    try {
        let _weights = read_torch_state_dict_bytes(raw.slice(0, raw.len() // 2))
    } catch e: Error {
        threw = e.message
    }
    assert_true(threw != "", "half an archive was read as a whole one")
}

fn test_a_float_tensor_has_no_integer_reading() -> void {
    #: `ints` is the exact reading of an integer tensor, and there is no exact
    #: integer reading of a float one. Silently truncating would make a reader
    #: that quietly loses the fractional part of every weight.
    let weights = read_torch_state_dict(MODELS + "torch_dtypes.pt")
    var threw = ""
    try {
        let _values = weights["f32"].ints()
    } catch e: Error {
        threw = e.message
    }
    assert_true(threw != "", "a float32 tensor answered ints()")
}

fn test_a_training_checkpoint_keeps_only_its_tensors() -> void {
    #: `torch_checkpoint.pt` is the shape a training run actually saves: the
    #: weights under one key, an Adam state whose own keys are ints, and an
    #: epoch, a note and a loss history that are not tensors at all. The paths
    #: have to reach through both kinds of key, and the three leaves that are
    #: not tensors have to be dropped rather than tripped over.
    let checkpoint = read_torch_state_dict(MODELS + "torch_checkpoint.pt")
    assert_true(checkpoint.contains("model.conv.weight"), "the weights are not under model.")
    assert_true(
        checkpoint.contains("optimizer.state.0.exp_avg"),
        "an optimizer state keyed by ints was not indexed"
    )
    assert_eq(
        checkpoint["optimizer.state.0.step"].shape,
        [],
        "Adam's step counter is 0-dimensional"
    )
    for key in checkpoint.keys() {
        assert_true(
            key.starts_with("model.") or key.starts_with("optimizer.state."),
            f"'{key}' is not a tensor this checkpoint holds"
        )
    }
}

fn test_a_protocol_5_pickle_reads_the_same() -> void {
    #: torch writes protocol 2, so nothing it produces uses the opcodes
    #: protocols 3 to 5 added. This fixture was asked for protocol 5 and so
    #: arrives with frames, `MEMOIZE`, `STACK_GLOBAL`, the short string and
    #: byte forms, a `LONG1` integer and the set opcodes wrapped around the
    #: same nine weights, which have to come back unchanged underneath all of
    #: it. The leaves that are not tensors have to be stepped over rather than
    #: kept: a byte string is not a weight and neither is a set.
    let modern = read_torch_state_dict(MODELS + "torch_protocol5.pt")
    let plain = read_torch_state_dict(MODELS + "torch_smallnet.pt")
    for (key, tensor) in plain.items() {
        assert_eq(
            modern["weights." + key].data,
            tensor.data,
            f"{key} read differently at protocol 5"
        )
    }
    assert_eq(modern["pair.1"].shape, [1, 1], "a tuple of tensors was not indexed")
    assert_eq(modern["only.one"].floats(), [10.0], "a one-entry dict was not read")
    assert_eq(modern["single.0"].numel(), 3, "a one-entry list was not read")
    assert_true(not modern.contains("blob"), "a byte string was kept as though it were a tensor")
    assert_true(not modern.contains("tags"), "a set was kept as though it held tensors")
    assert_true(not modern.contains("big"), "an integer was kept as though it were a tensor")
}
