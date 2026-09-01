#: Reading a frozen TensorFlow graph.
#:
#: The file is a binary `GraphDef`, which is what `freeze_graph` and
#: `convert_variables_to_constants` write and what OpenCV's
#: `readNetFromTensorflow` takes. It is read with the schemaless wire reader in
#: `protobuf.jai` and translated into the shared IR in `graph.jai`, so a
#: TensorFlow `Conv2D` and a Caffe `Convolution` arrive at the executor as the
#: same `Conv` node.
#:
#: The whole difficulty of this importer is layout, and it is worth being
#: precise about the invariant it maintains, because a graph that imports
#: cleanly and computes plausible nonsense is the failure that costs a day.
#:
#: TensorFlow lays images out `NHWC` and convolution weights `HWIO`. Everything
#: downstream of here is ONNX's `NCHW` and `OIHW`. So:
#:
#:   **A rank-four tensor in the imported graph holds TensorFlow's axis
#:   `TF_AXIS[k]` at its own axis `k`. A tensor of any other rank is exactly
#:   TensorFlow's.**
#:
#: That single rule decides everything else. The graph's inputs and outputs are
#: `NCHW`, so a blob from `blob_from_image` feeds straight in. Convolution and
#: pooling and batch normalisation are `NCHW`-native and need no transpose at
#: all -- only the weights are permuted, once, at import. An operator whose
#: meaning is written in terms of axis *numbers* -- `Concat`, `Pad`, `Mean`,
#: `Squeeze`, `Softmax` -- has its axes renumbered instead of its data moved.
#: A `Transpose` composes its permutation with the two layout changes and stays
#: a single node.
#:
#: Only `Reshape` genuinely has to move data, because it reads the buffer in
#: TensorFlow's row-major order; and so does a reduction that drops some but
#: not all of the spatial axes, because dropping `H` alone leaves `NCW` here
#: where TensorFlow is left with `NWC`. Both are wrapped in explicit
#: `Transpose` nodes, and `survives_in_order` is the predicate that decides.
#:
#: The rule generalises past rank four, and every place that changes a
#: tensor's rank has to apply it explicitly rather than trust the number it
#: lands on. A tensor arrives at a node already obeying the invariant --
#: `NCHW` if its rank was four, TensorFlow's own order otherwise -- and an
#: operator that only renumbers or drops axes (`Squeeze`, a reduction,
#: `ArgMax`) never moves data, so its result is automatically correct in
#: whichever of those two regimes its INPUT was in. The one thing that has to
#: be checked by hand is which regime that result rank calls for: dropping
#: axes off a rank-four `NCHW` tensor keeps the result in the same
#: renumbered-axis world (handled by mapping through `NCHW_AXIS` or
#: transposing first, same as `Reshape`), but dropping axes off a tensor of
#: any OTHER rank leaves it in TensorFlow's own order regardless of what rank
#: it lands on -- and if that landing rank is exactly four, the invariant
#: requires `NCHW`, so a `to_nchw` transpose has to run before the result is
#: tagged rank four. Skipping that step is what left `translate_squeeze` and
#: `translate_reduce` tagging a still-`NHWC` tensor as rank four. Broadcasting
#: has the same rule read backwards: `broadcast_input` brings every operand of
#: an elementwise node into whatever regime the node's OWN result rank calls
#: for, so a rank-four operand has to be un-permuted back to TensorFlow's
#: order whenever the node's result is not itself rank four, the same as a
#: `Reshape` would.
#:
#: Renumbering axes needs the rank of a tensor, which is why this walks the
#: graph in topological order carrying a rank for every name. A rank it cannot
#: work out is an error naming the node, never a guess. So is a file this
#: importer will not read: it is turned away whole, with nothing left
#: allocated behind it, rather than half-translated. What it turns away, and
#: why refusing beats guessing in each case -- a graph frozen in `NCHW`, which
#: only a GPU host writes, would break the invariant above; a `Squeeze` with no
#: `squeeze_dims` on a rank-four tensor leaves a rank nothing static can work
#: out; and the `.pbtxt` configuration OpenCV takes beside the model exists to
#: substitute detection layers this executor has no operators for.
#:
#: `Graph.initializers` is the weights the file carried, already on the GPU,
#: and freeing them is the caller's once it has finished with the graph.
#:
#: ```
#: let net = read_net_from_tensorflow("frozen_inference_graph.pb")
#: defer { for (_name, weight) in net.initializers.items() { weight.free() } }
#: let outputs = run_graph(net, {"image": blob})
#: defer { for (_name, tensor) in outputs.items() { tensor.free() } }
#: ```
#:
#: The importer lives in `tensorflow/`; this re-exports its two entry points.
from .tensorflow.build import read_net_from_tensorflow, read_net_from_tensorflow_bytes

export { read_net_from_tensorflow, read_net_from_tensorflow_bytes }
