"""Hand-build the malformed and atypical GraphDefs test_tensorflow.jai reads.

Every graph here is a `GraphDef` protoc would happily emit -- valid wire
format throughout -- but not a shape TensorFlow's own writer produces, because
`make_tensor_proto` always spends `tensor_content` on a numeric array and a
real graph never omits an operator's inputs. Both are still legal on the wire,
and a model file downloaded from somewhere is exactly the kind of untrusted
input this importer has to turn away by name rather than crash on. Building
these with tf.function tracing is not an option: TensorFlow's own type
checking would refuse half of them (Relu has no complex kernel, and there is
no way to trace a node with too few inputs at all), so this goes straight at
the protobuf message classes TensorFlow ships and never runs a graph.

    cd /tmp && uv run --python 3.11 --with tensorflow \\
        python <repo>/packages/jaicv/tools/tensorflow_hostile_fixtures.py <repo>
"""

import os
import sys

from tensorflow.core.framework import graph_pb2, types_pb2


def placeholder(gd, name, dtype, dims):
    node = gd.node.add()
    node.name = name
    node.op = "Placeholder"
    node.attr["dtype"].type = dtype
    for d in dims:
        node.attr["shape"].shape.dim.add(size=d)
    return node


def op_node(gd, name, op, inputs):
    node = gd.node.add()
    node.name = name
    node.op = op
    node.input.extend(inputs)
    return node


def const_node(gd, name, dtype, dims):
    node = gd.node.add()
    node.name = name
    node.op = "Const"
    node.attr["dtype"].type = dtype
    t = node.attr["value"].tensor
    t.dtype = dtype
    for d in dims:
        t.tensor_shape.dim.add(size=d)
    return node


def write(gd, path):
    with open(path, "wb") as out:
        out.write(gd.SerializeToString())
    print(f"{os.path.basename(path)}: {gd.ByteSize()} bytes, ops {sorted({n.op for n in gd.node})}")


def build_uint_consts():
    # Issue 3: `typed_values` sent every WHOLE_DTYPES entry through int_val
    # (field 7), but uint32 and uint64 have their own fields (16 and 17).
    # `tensor_util.make_tensor_proto` never takes this path -- it always
    # writes `tensor_content` for a numeric array -- so this is built by hand,
    # setting only the typed field the way a smaller or older encoder would.
    gd = graph_pb2.GraphDef()
    placeholder(gd, "x", types_pb2.DT_FLOAT, [3])
    c32 = const_node(gd, "c32", types_pb2.DT_UINT32, [3])
    c32.attr["value"].tensor.uint32_val.extend([10, 20, 30])
    op_node(gd, "out32", "AddV2", ["x", "c32"])
    c64 = const_node(gd, "c64", types_pb2.DT_UINT64, [3])
    c64.attr["value"].tensor.uint64_val.extend([100, 200, 300])
    op_node(gd, "out64", "AddV2", ["x", "c64"])
    return gd


def build_huge_shape():
    # Issue 4: a declared shape with nothing behind it. No tensor_content, no
    # typed values -- the "every element is zero" convenience -- but the
    # declared count is four billion, and the whole message is a few dozen
    # bytes. Nothing downstream ever reads this constant; decoding it is
    # supposed to fail before anything is allocated.
    gd = graph_pb2.GraphDef()
    const_node(gd, "huge", types_pb2.DT_INT32, [4_000_000_000])
    return gd


def build_missing_unary():
    # Issue 6: a Relu with no inputs at all.
    gd = graph_pb2.GraphDef()
    op_node(gd, "r", "Relu", [])
    return gd


def build_missing_binary():
    # Issue 6: a MatMul with one input instead of two.
    gd = graph_pb2.GraphDef()
    placeholder(gd, "x", types_pb2.DT_FLOAT, [2, 2])
    op_node(gd, "m", "MatMul", ["x"])
    return gd


def build_complex_const():
    # Issue 7: DT_COMPLEX64 is not a dtype this importer decodes, and
    # `make_tensor_proto` writes a complex array as `scomplex_val`, a
    # repeated-float field this reader's `typed_values` has no case for --
    # so decoding throws for a real reason, not because the constant carries
    # no value. c is consumed by a Relu the way a frozen graph might if a
    # model held a complex intermediate this executor never runs.
    gd = graph_pb2.GraphDef()
    c = const_node(gd, "c", types_pb2.DT_COMPLEX64, [2])
    c.attr["value"].tensor.scomplex_val.extend([1.0, 2.0, 3.0, 4.0])
    op_node(gd, "out", "Relu", ["c"])
    return gd


def main(root):
    models = os.path.join(root, "packages", "jaicv", "tests", "models")
    os.makedirs(models, exist_ok=True)
    fixtures = {
        "tf_uint_consts.pb": build_uint_consts(),
        "tf_huge_shape.pb": build_huge_shape(),
        "tf_missing_unary.pb": build_missing_unary(),
        "tf_missing_binary.pb": build_missing_binary(),
        "tf_complex_const.pb": build_complex_const(),
    }
    for filename, gd in fixtures.items():
        write(gd, os.path.join(models, filename))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
