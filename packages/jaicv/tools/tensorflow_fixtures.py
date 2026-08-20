"""Rebuild the frozen TensorFlow graphs in packages/jaicv/tests/models/.

Run from the repository root with TensorFlow available:

    cd /tmp && uv run --python 3.11 --with tensorflow --with numpy \
        python <repo>/packages/jaicv/tools/tensorflow_fixtures.py <repo>

Every graph is built with tf.function, frozen with
convert_variables_to_constants_v2, and written as a binary GraphDef -- the
same `.pb` OpenCV's readNetFromTensorflow takes.  The graphs are then run by
TensorFlow itself and the inputs and outputs are recorded in
tests/models/tensorflow_cases.txt, permuted into the NCHW layout the importer
hands back so that the test does no reordering of its own.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

RNG = np.random.default_rng(20260819)


def rand(*shape):
    return RNG.standard_normal(shape).astype(np.float32)


def freeze(fn, spec):
    concrete = tf.function(fn).get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(concrete)
    return frozen.graph.as_graph_def(), frozen


def to_nchw(array):
    if array.ndim == 4:
        return np.transpose(array, (0, 3, 1, 2))
    return array


def record(handle, kind, array):
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    parts = [kind, str(array.ndim)]
    parts += [str(d) for d in array.shape]
    parts.append(str(flat.size))
    parts += [repr(float(v)) for v in flat]
    handle.write(" ".join(parts) + "\n")


CASES = []


def emit(name, fn, spec, sample):
    graph_def, frozen = freeze(fn, spec)
    inputs = [t.name.split(":")[0] for t in frozen.inputs]
    outputs = [t.name.split(":")[0] for t in frozen.outputs]
    blob = graph_def.SerializeToString()
    CASES.append((name, blob, inputs, outputs, sample, frozen))
    ops = sorted({node.op for node in graph_def.node})
    print(f"{name}: {len(blob)} bytes, in={inputs} out={outputs}")
    print(f"    ops {ops}")


# --- an MLP, no convolution anywhere -----------------------------------
W1, B1 = rand(7, 11), rand(11)
W2, B2 = rand(11, 4), rand(4)


def mlp(x):
    h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, tf.constant(W1)), tf.constant(B1)))
    return tf.nn.softmax(tf.nn.bias_add(tf.matmul(h, tf.constant(W2)), tf.constant(B2)))


emit(
    "tf_mlp",
    mlp,
    tf.TensorSpec([None, 7], tf.float32, name="x"),
    rand(3, 7),
)


# --- a CNN: conv, batch norm, relu, pooling, flatten, classifier -------
# 9x5 input, 3x2 kernels: every extent differs, so a transposed weight or a
# swapped H and W cannot come out looking plausible.
K1 = rand(3, 2, 3, 5)
GAMMA, BETA = rand(5) * 0.5 + 1.0, rand(5)
MEAN, VAR = rand(5) * 0.1, np.abs(rand(5)) + 0.5
K2 = rand(2, 3, 5, 4)
WD, BD = rand(2 * 1 * 4, 6), rand(6)


def cnn(x):
    h = tf.nn.conv2d(x, tf.constant(K1), strides=[1, 1, 1, 1], padding="SAME")
    h, _, _ = tf.compat.v1.nn.fused_batch_norm(
        h,
        tf.constant(GAMMA),
        tf.constant(BETA),
        mean=tf.constant(MEAN.astype(np.float32)),
        variance=tf.constant(VAR.astype(np.float32)),
        epsilon=1e-3,
        is_training=False,
    )
    h = tf.nn.relu(h)
    h = tf.nn.max_pool2d(h, ksize=2, strides=2, padding="VALID")
    h = tf.nn.conv2d(h, tf.constant(K2), strides=[1, 1, 1, 1], padding="VALID")
    h = tf.nn.relu6(h)
    h = tf.nn.avg_pool2d(h, ksize=2, strides=2, padding="SAME")
    h = tf.reshape(h, [-1, 2 * 1 * 4])
    return tf.nn.softmax(tf.nn.bias_add(tf.matmul(h, tf.constant(WD)), tf.constant(BD)))


emit(
    "tf_cnn",
    cnn,
    tf.TensorSpec([None, 11, 7, 3], tf.float32, name="image"),
    rand(2, 11, 7, 3),
)


# --- a depthwise convolution, then a pointwise one ---------------------
DW = rand(3, 2, 3, 2)
PW = rand(1, 1, 6, 4)
PB = rand(4)


def depthwise(x):
    h = tf.nn.depthwise_conv2d(
        x, tf.constant(DW), strides=[1, 1, 1, 1], padding="SAME"
    )
    h = tf.nn.conv2d(h, tf.constant(PW), strides=[1, 2, 2, 1], padding="VALID")
    return tf.nn.relu6(tf.nn.bias_add(h, tf.constant(PB)))


emit(
    "tf_depthwise",
    depthwise,
    tf.TensorSpec([None, 7, 4, 3], tf.float32, name="image"),
    rand(2, 7, 4, 3),
)


# --- a Mean reduction, in both the keep_dims forms ---------------------
MK = rand(2, 3, 3, 5)
MW, MB = rand(5, 3), rand(3)


def pooled(x):
    h = tf.nn.relu(tf.nn.conv2d(x, tf.constant(MK), strides=[1, 1, 1, 1], padding="SAME"))
    kept = tf.reduce_mean(h, axis=[1, 2], keepdims=True)
    dropped = tf.reduce_mean(h, axis=[1, 2], keepdims=False)
    scaled = tf.multiply(kept, tf.constant(2.0))
    flat = tf.squeeze(scaled, axis=[1, 2])
    return tf.nn.bias_add(tf.matmul(flat + dropped, tf.constant(MW)), tf.constant(MB))


emit(
    "tf_mean",
    pooled,
    tf.TensorSpec([None, 6, 4, 3], tf.float32, name="image"),
    rand(2, 6, 4, 3),
)


# --- the rest of the vocabulary, in one graph --------------------------
# Pad, ConcatV2, Sub, Mul, Identity, LRN, Transpose, ExpandDims, Sum, Max,
# and a constant that TensorFlow stores as a single repeated value.
PK = rand(2, 2, 3, 4)
SPLAT = np.full((4,), 0.25, dtype=np.float32)


def misc(x):
    padded = tf.pad(x, tf.constant([[0, 0], [1, 2], [2, 0], [0, 0]]), mode="CONSTANT")
    h = tf.nn.conv2d(padded, tf.constant(PK), strides=[1, 1, 1, 1], padding="VALID")
    h = tf.nn.local_response_normalization(
        h, depth_radius=2, bias=1.5, alpha=0.0002, beta=0.6
    )
    scaled = tf.multiply(h, tf.constant(SPLAT))
    shifted = tf.subtract(scaled, tf.constant(0.5))
    joined = tf.concat([tf.nn.elu(shifted), tf.sigmoid(h)], axis=3)
    swapped = tf.transpose(joined, [0, 2, 1, 3])
    total = tf.reduce_sum(swapped, axis=[3], keepdims=False)
    largest = tf.reduce_max(swapped, axis=[1, 2, 3], keepdims=False)
    grown = tf.expand_dims(largest, 1)
    return tf.identity(total, name="total"), tf.tanh(grown)


emit(
    "tf_misc",
    misc,
    tf.TensorSpec([None, 5, 4, 3], tf.float32, name="image"),
    rand(2, 5, 4, 3),
)



# --- reductions whose survivors come out in a different order ----------
# Dropping H alone leaves NCW in the imported layout where TensorFlow is
# left with NWC, so each of these forces a transpose rather than a
# renumbering.  Getting that wrong gives a correctly shaped wrong answer.
AK = rand(2, 2, 3, 6)


def axes(x):
    h = tf.nn.relu(tf.nn.conv2d(x, tf.constant(AK), strides=[1, 1, 1, 1], padding="SAME"))
    over_h = tf.reduce_mean(h, axis=[1], keepdims=False)
    over_w = tf.reduce_sum(h, axis=[2], keepdims=False)
    over_n = tf.reduce_max(h, axis=[0], keepdims=False)
    picked = tf.cast(tf.argmax(h, axis=1, output_type=tf.int32), tf.float32)
    squeezed = tf.squeeze(tf.reduce_mean(h, axis=[1], keepdims=True), axis=[1])
    return over_h, over_w, over_n, picked, squeezed


emit(
    "tf_axes",
    axes,
    tf.TensorSpec([None, 5, 4, 3], tf.float32, name="image"),
    rand(2, 5, 4, 3),
)


# --- the arithmetic vocabulary -----------------------------------------
def arith(x):
    squared = tf.square(x)
    inverse = tf.math.rsqrt(squared + 1.0)
    divided = tf.math.divide(squared, inverse)
    capped = tf.minimum(tf.maximum(divided, tf.constant(0.3)), tf.constant(2.0))
    leaky = tf.nn.leaky_relu(capped - 1.0, alpha=0.3)
    soft = tf.nn.softplus(leaky)
    shrunk = tf.exp(tf.negative(tf.abs(soft)))
    logged = tf.math.log(shrunk + 2.0)
    rooted = tf.sqrt(tf.abs(logged))
    summed = tf.add_n([rooted, soft, leaky])
    mirrored = tf.pad(summed, tf.constant([[0, 0], [1, 1], [0, 1], [0, 0]]), mode="REFLECT")
    return tf.nn.log_softmax(tf.reshape(mirrored, [-1, 6 * 4 * 2]))


emit(
    "tf_arith",
    arith,
    tf.TensorSpec([None, 4, 3, 2], tf.float32, name="x"),
    rand(2, 4, 3, 2),
)


# --- filters whose every element says where it came from ---------------
# The weights are 0, 1, 2, ... so the permuted initializer has one right
# answer that the test writes out from the layout rule rather than from a
# forward pass.  A convolution that agrees numerically has already ruled
# this out, but only for the shapes it was run on.
RAMP = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
RAMP_DW = np.arange(2 * 3 * 4 * 2, dtype=np.float32).reshape(2, 3, 4, 2)


def ramp(x):
    plain = tf.nn.conv2d(x, tf.constant(RAMP), strides=[1, 1, 1, 1], padding="VALID")
    deep = tf.nn.depthwise_conv2d(
        x, tf.constant(RAMP_DW), strides=[1, 1, 1, 1], padding="VALID"
    )
    return plain, deep


emit(
    "tf_ramp",
    ramp,
    tf.TensorSpec([None, 4, 5, 4], tf.float32, name="x"),
    rand(1, 4, 5, 4) * 0.01,
)


# --- the layout paths a plain CNN never reaches ------------------------
# A rank-four constant used as data rather than as weights, a rank-three
# constant and a rank-three computed value broadcast against an image, a
# softmax over the channels of a four-dimensional tensor, a concatenation on
# a spatial axis, and a reshape in each direction across rank four.
C4 = rand(1, 4, 3, 2)
C3 = rand(4, 3, 2)


def layout(x):
    shifted = tf.add(x, tf.constant(C4))
    weighted = tf.multiply(shifted, tf.constant(C3))
    spread = tf.nn.softmax(weighted)
    pooled = tf.reduce_mean(spread, axis=[0], keepdims=False)
    scaled = tf.multiply(spread, pooled)
    lifted = tf.expand_dims(pooled, 0)
    mixed = tf.subtract(scaled, lifted)
    joined = tf.concat([spread, mixed], axis=1)
    flat = tf.reshape(joined, [-1, 8 * 3 * 2])
    grid = tf.reshape(flat, [-1, 8, 3, 2])
    return joined, grid, pooled


emit(
    "tf_layout",
    layout,
    tf.TensorSpec([None, 4, 3, 2], tf.float32, name="x"),
    rand(3, 4, 3, 2),
)


# --- landing at rank four out of a rank the invariant never permutes ---
# A tensor that reaches rank five and then has one axis dropped -- by a
# Squeeze or a non-keepdims reduction -- lands back at rank four still in
# TensorFlow's own order, not NCHW, because neither operator moves data.
# Broadcasting hits the same regime split from the other side: a rank-five
# operand stays TensorFlow-ordered while a rank-four constant would normally
# be permuted to NCHW, and mixing the two without correcting for it computes
# a correctly shaped wrong answer.
SQK = rand(3, 2, 3, 4)


def squeeze_lands(x):
    squeezed = tf.squeeze(tf.expand_dims(x, 1), axis=[1])
    return tf.nn.conv2d(squeezed, tf.constant(SQK), strides=[1, 1, 1, 1], padding="SAME")


emit(
    "tf_squeeze_lands",
    squeeze_lands,
    tf.TensorSpec([None, 9, 5, 3], tf.float32, name="x"),
    rand(2, 9, 5, 3),
)


def reduce_lands(x):
    return tf.reduce_sum(tf.expand_dims(x, 2), axis=[2])


emit(
    "tf_reduce_lands",
    reduce_lands,
    tf.TensorSpec([None, 5, 3, 2], tf.float32, name="x"),
    rand(2, 5, 3, 2),
)

BK4 = rand(1, 4, 4, 3)


def broadcast_lands(x):
    return tf.add(tf.expand_dims(x, 1), tf.constant(BK4))


emit(
    "tf_broadcast_lands",
    broadcast_lands,
    tf.TensorSpec([None, 4, 4, 3], tf.float32, name="x"),
    rand(2, 4, 4, 3),
)


# ArgMax/ArgMin drop exactly one axis the same way a non-keepdims reduction
# does, so the same landing-at-rank-four fix belongs there too.
def argmax_lands(x):
    expanded = tf.expand_dims(x, 1)
    return tf.cast(tf.argmax(expanded, axis=4, output_type=tf.int32), tf.float32)


emit(
    "tf_argmax_lands",
    argmax_lands,
    tf.TensorSpec([None, 5, 3, 4], tf.float32, name="x"),
    rand(2, 5, 3, 4),
)


# --- graphs this importer is supposed to turn away ---------------------
REFUSED = []


def refuse(name, fn, spec):
    graph_def, _frozen = freeze(fn, spec)
    REFUSED.append((name, graph_def.SerializeToString()))
    ops = sorted({node.op for node in graph_def.node})
    print(f"{name}: refusal fixture, ops {ops}")


refuse(
    "tf_unsupported",
    lambda x: tf.tile(x, [1, 2, 1, 1]),
    tf.TensorSpec([None, 3, 2, 2], tf.float32, name="x"),
)

TG, TB = rand(3), rand(3)


def training(x):
    h, _, _ = tf.compat.v1.nn.fused_batch_norm(
        x, tf.constant(TG), tf.constant(TB), epsilon=1e-3, is_training=True
    )
    return h


refuse("tf_training", training, tf.TensorSpec([None, 3, 2, 3], tf.float32, name="x"))


# --- writing it all out ------------------------------------------------
def main(root):
    models = os.path.join(root, "packages", "jaicv", "tests", "models")
    os.makedirs(models, exist_ok=True)
    lines = os.path.join(models, "tensorflow_cases.txt")
    with open(lines, "w") as handle:
        handle.write("#: Written by packages/jaicv/tools/tensorflow_fixtures.py.\n")
        handle.write("#: Tensors of rank four are recorded in NCHW, which is the\n")
        handle.write("#: layout the importer's graphs take and give back.\n")
        for name, blob, inputs, outputs, sample, frozen in CASES:
            path = os.path.join(models, name + ".pb")
            with open(path, "wb") as out:
                out.write(blob)
            produced = frozen(tf.constant(sample))
            if not isinstance(produced, (list, tuple)):
                produced = [produced]
            handle.write(f"graph {name} {len(inputs)} {len(outputs)}\n")
            for one in inputs:
                handle.write(f"input {one}\n")
            for one in outputs:
                handle.write(f"output {one}\n")
            record(handle, "feed", to_nchw(sample))
            for value in produced:
                record(handle, "want", to_nchw(np.asarray(value)))
        for name, blob in REFUSED:
            with open(os.path.join(models, name + ".pb"), "wb") as out:
                out.write(blob)
    print(f"wrote {lines}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
