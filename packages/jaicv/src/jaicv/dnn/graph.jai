#: The graph a network becomes once it has been imported, and the executor
#: that runs it.
#:
#: Every importer -- ONNX, Caffe, TensorFlow, Torch -- lands here, and they all
#: land in the same vocabulary: the `op` string on a `GraphNode` is an ONNX
#: operator name. There is one executor because there is one vocabulary, so a
#: Caffe `Convolution` and a TensorFlow `Conv2D` both arrive as `Conv` and
#: neither importer carries its own arithmetic.
#:
#: Tensors are jaitensor's, so they live on the GPU and hold float32 and
#: nothing else. Shapes, indices and booleans therefore ride in float32 too --
#: exact up to 2^24, which is every index a vision network ever produces. The
#: layout is ONNX's throughout: `NCHW` for images, `MCkHkW` for convolution
#: weights. jaitensor's convolution and pooling are channel-last, so this
#: module transposes around them rather than the other way round; the transpose
#: is a kernel and the convolution is not, which is the right way for that cost
#: to fall.
#:
#: Ownership is simple and worth stating because it is what stops a graph from
#: freeing a tensor it still needs. Every value a node produces is a fresh
#: tensor this module owns. Initializers and the caller's feeds are never
#: freed, and never handed back as an output without being copied first. An
#: intermediate is freed as soon as the last node that reads it has run.
#:
#: Two shapes ONNX has and jaitensor does not, which an importer has to know
#: about because it is the one holding the file:
#:
#: - There is no rank-zero tensor, so a scalar arrives and leaves as `[1]`.
#:   Broadcasting treats the two the same; what differs is that an operator
#:   which would have dropped an axis (a `Gather` by a scalar index) leaves a
#:   width of one behind instead.
#: - There is no empty tensor, so an optional input that a file skips must be
#:   passed as an empty *name* in `GraphNode.inputs` -- position is what
#:   identifies an argument -- and never as a zero-length tensor.
#:
#: ```
#: let graph = load_onnx("mobilenet.onnx")
#: let outputs = run_graph(graph, {"input": blob})
#: defer { for (_name, tensor) in outputs.items() { tensor.free() } }
#: ```
#:
#: The pieces live in `graph/` and are importable on their own; this module is
#: the whole of them, and is what every importer and the plan compiler use.
from .graph.execute import implemented_operators, is_implemented, run_graph
from .graph.nodes import Attr, Graph, GraphNode, attr_float, attr_int, attr_ints, attr_str
from .graph.tensors import normalize_axis

export { Attr, Graph, GraphNode }
export { attr_float, attr_int, attr_ints, attr_str, normalize_axis }
export { implemented_operators, is_implemented, run_graph }
