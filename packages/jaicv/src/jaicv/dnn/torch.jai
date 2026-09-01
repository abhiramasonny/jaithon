#: Reading a PyTorch checkpoint.
#:
#: A `.pt` or `.pth` written by torch 1.6 or later is a ZIP archive holding one
#: pickled object graph in `data.pkl` and every tensor's raw storage as a
#: separate member under `data/`. Reading one therefore needs two things that
#: are not otherwise in this tree — a ZIP reader and a pickle interpreter — and
#: both are here. The deflate itself is zlib's, through `__prim__.inflate`.
#:
#: Reading from a path reads through the handle rather than pulling the file in
#: first: the directory, then each member as it is wanted. A checkpoint big
#: enough to need the zip64 fields is one that must not be held twice over, and
#: `read_torch_state_dict_bytes` — which takes the file already in memory — is
#: for the callers that already have it.
#:
#: The pickle interpreter is deliberately crippled, and that is the whole point
#: of it. A pickle names callables and asks for them to be called, so a general
#: interpreter is arbitrary code execution on a file that came from somewhere
#: else: the standard demonstration is a checkpoint whose pickle names
#: `os.system`. This one resolves globals against a fixed whitelist —
#: `torch._utils._rebuild_tensor`, `_rebuild_tensor_v2` and `_rebuild_parameter`,
#: the ten `torch.*Storage` classes, `collections.OrderedDict`, and the plain
#: container builtins — and throws on anything else, naming what it refused.
#: There is no flag to widen that list. A checkpoint needing more than this is
#: a checkpoint that should be re-saved.
#:
#: What comes back is the object graph flattened onto dotted paths, so a nested
#: `{"model": {"conv": {"weight": t}}}` arrives as `model.conv.weight` and a
#: list of tensors as `0`, `1`, and so on. Leaves that are not tensors — an
#: epoch counter, a learning rate, a `None` — are dropped: this reads weights.
#:
#: A `TorchTensor` is deliberately not the `Tensor` the executor in
#: `dnn.graph` runs on. That one is jaitensor's: on the GPU, float32, and every
#: dimension positive. A checkpoint is none of those things — it holds `int64`
#: counters, `bool` masks, 0-dimensional scalars and empty tensors, and an
#: `int64` past 2^53 does not survive a float32 — so a reader that handed back
#: device tensors would be deciding what to lose before the caller had seen it.
#: Going the other way is one line where a caller actually wants it:
#: `jt.Tensor(weight.shape, weight.floats())`.
#:
#: ```
#: let weights = read_torch_state_dict("resnet18.pth")
#: weights["fc.weight"].shape      # [1000, 512]
#: weights["fc.weight"].floats()   # 512000 values, row-major
#: ```
from .torch.bits import dtype_size
from .torch.read import read_torch_state_dict, read_torch_state_dict_bytes
from .torch.tensor import TorchTensor

export { dtype_size, read_torch_state_dict, read_torch_state_dict_bytes, TorchTensor }
