#!/usr/bin/env python3
"""Rebuild the Caffe fixtures in packages/jaicv/tests/models/.

    cd /tmp && uv run --with protobuf --with grpcio-tools --with numpy \
        --with torch python <repo>/packages/jaicv/tools/caffe_fixtures.py --out <repo>

Every .prototxt and .caffemodel here is written by the real protobuf package
from the schema below, so the bytes are a genuine NetParameter and not this
repo's idea of one.  CAFFE_PROTO is upstream caffe.proto with the messages and
fields these fixtures do not touch deleted -- extracted mechanically from
https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto
rather than retyped, so every field number and every default is upstream's.
Pass --proto PATH to compile the full upstream file instead; the fixture bytes
come out identical, which is the check that the trimming changed nothing.

caffe_cases.txt records what the importer has to agree with: the feeds, the raw
blob contents of every layer carrying weights, and a reference forward pass.
That reference is torch in float64.  torch is not an arbitrary choice: its
max_pool2d and avg_pool2d implement Caffe's ceil rounding, Caffe's rule that
the last window must start inside the padded image, and Caffe's pool-size
divisor exactly, and those are the parts of this format most easily got quietly
wrong.  torch's LocalResponseNorm is Caffe's LRN formula for odd sizes.
"""
import argparse
import os
import sys
import tempfile

import numpy as np

CAFFE_PROTO = r"""
syntax = "proto2";

package caffe;

enum Phase {
  TRAIN = 0;
  TEST = 1;
}

message BlobShape {
  repeated int64 dim = 1 [packed = true];
}

message BlobProto {
  optional BlobShape shape = 7;
  repeated float data = 5 [packed = true];
  repeated float diff = 6 [packed = true];
  repeated double double_data = 8 [packed = true];
  repeated double double_diff = 9 [packed = true];
  optional int32 num = 1 [default = 0];
  optional int32 channels = 2 [default = 0];
  optional int32 height = 3 [default = 0];
  optional int32 width = 4 [default = 0];
}

message NetParameter {
  optional string name = 1;
  repeated string input = 3;
  repeated BlobShape input_shape = 8;
  repeated int32 input_dim = 4;
  repeated LayerParameter layer = 100;
}

message LayerParameter {
  optional string name = 1;
  optional string type = 2;
  repeated string bottom = 3;
  repeated string top = 4;
  optional Phase phase = 10;
  repeated ParamSpec param = 6;
  repeated BlobProto blobs = 7;
  repeated NetStateRule include = 8;
  repeated NetStateRule exclude = 9;
  optional TransformationParameter transform_param = 100;
  optional BatchNormParameter batch_norm_param = 139;
  optional BiasParameter bias_param = 141;
  optional ClipParameter clip_param = 148;
  optional ConcatParameter concat_param = 104;
  optional ConvolutionParameter convolution_param = 106;
  optional DropoutParameter dropout_param = 108;
  optional EltwiseParameter eltwise_param = 110;
  optional ELUParameter elu_param = 140;
  optional ExpParameter exp_param = 111;
  optional FlattenParameter flatten_param = 135;
  optional InnerProductParameter inner_product_param = 117;
  optional InputParameter input_param = 143;
  optional LogParameter log_param = 134;
  optional LRNParameter lrn_param = 118;
  optional PoolingParameter pooling_param = 121;
  optional PowerParameter power_param = 122;
  optional PReLUParameter prelu_param = 131;
  optional ReLUParameter relu_param = 123;
  optional ReshapeParameter reshape_param = 133;
  optional ScaleParameter scale_param = 142;
  optional SigmoidParameter sigmoid_param = 124;
  optional SoftmaxParameter softmax_param = 125;
  optional SliceParameter slice_param = 126;
  optional TanHParameter tanh_param = 127;
  optional ThresholdParameter threshold_param = 128;
}

message ConvolutionParameter {
  optional uint32 num_output = 1;
  optional bool bias_term = 2 [default = true];
  repeated uint32 pad = 3;
  repeated uint32 kernel_size = 4;
  repeated uint32 stride = 6;
  repeated uint32 dilation = 18;
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_h = 11;
  optional uint32 kernel_w = 12;
  optional uint32 stride_h = 13;
  optional uint32 stride_w = 14;
  optional uint32 group = 5 [default = 1];
  optional FillerParameter weight_filler = 7;
  optional FillerParameter bias_filler = 8;
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];
  optional int32 axis = 16 [default = 1];
  optional bool force_nd_im2col = 17 [default = false];
}

message PoolingParameter {
  enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional PoolMethod pool = 1 [default = MAX];
  optional uint32 pad = 4 [default = 0];
  optional uint32 pad_h = 9 [default = 0];
  optional uint32 pad_w = 10 [default = 0];
  optional uint32 kernel_size = 2;
  optional uint32 kernel_h = 5;
  optional uint32 kernel_w = 6;
  optional uint32 stride = 3 [default = 1];
  optional uint32 stride_h = 7;
  optional uint32 stride_w = 8;
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 11 [default = DEFAULT];
  optional bool global_pooling = 12 [default = false];
  enum RoundMode {
    CEIL = 0;
    FLOOR = 1;
  }
  optional RoundMode round_mode = 13 [default = CEIL];
}

message InnerProductParameter {
  optional uint32 num_output = 1;
  optional bool bias_term = 2 [default = true];
  optional FillerParameter weight_filler = 3;
  optional FillerParameter bias_filler = 4;
  optional int32 axis = 5 [default = 1];
  optional bool transpose = 6 [default = false];
}

message ReLUParameter {
  optional float negative_slope = 1 [default = 0];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 2 [default = DEFAULT];
}

message LRNParameter {
  optional uint32 local_size = 1 [default = 5];
  optional float alpha = 2 [default = 1.];
  optional float beta = 3 [default = 0.75];
  enum NormRegion {
    ACROSS_CHANNELS = 0;
    WITHIN_CHANNEL = 1;
  }
  optional NormRegion norm_region = 4 [default = ACROSS_CHANNELS];
  optional float k = 5 [default = 1.];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 6 [default = DEFAULT];
}

message BatchNormParameter {
  optional bool use_global_stats = 1;
  optional float moving_average_fraction = 2 [default = .999];
  optional float eps = 3 [default = 1e-5];
}

message ScaleParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 num_axes = 2 [default = 1];
  optional FillerParameter filler = 3;
  optional bool bias_term = 4 [default = false];
  optional FillerParameter bias_filler = 5;
}

message BiasParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 num_axes = 2 [default = 1];
  optional FillerParameter filler = 3;
}

message ConcatParameter {
  optional int32 axis = 2 [default = 1];
  optional uint32 concat_dim = 1 [default = 1];
}

message SoftmaxParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
  optional int32 axis = 2 [default = 1];
}

message EltwiseParameter {
  enum EltwiseOp {
    PROD = 0;
    SUM = 1;
    MAX = 2;
  }
  optional EltwiseOp operation = 1 [default = SUM];
  repeated float coeff = 2;
  optional bool stable_prod_grad = 3 [default = true];
}

message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5];
}

message ReshapeParameter {
  optional BlobShape shape = 1;
  optional int32 axis = 2 [default = 0];
  optional int32 num_axes = 3 [default = -1];
}

message FlattenParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 end_axis = 2 [default = -1];
}

message PowerParameter {
  optional float power = 1 [default = 1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

message SliceParameter {
  optional int32 axis = 3 [default = 1];
  repeated uint32 slice_point = 2;
  optional uint32 slice_dim = 1 [default = 1];
}

message ELUParameter {
  optional float alpha = 1 [default = 1];
}

message ThresholdParameter {
  optional float threshold = 1 [default = 0];
}

message InputParameter {
  repeated BlobShape shape = 1;
}

message PReLUParameter {
  optional FillerParameter filler = 1;
  optional bool channel_shared = 2 [default = false];
}

message ClipParameter {
  required float min = 1;
  required float max = 2;
}

message ExpParameter {
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

message LogParameter {
  optional float base = 1 [default = -1.0];
  optional float scale = 2 [default = 1.0];
  optional float shift = 3 [default = 0.0];
}

message SigmoidParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}

message TanHParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];
}

message ParamSpec {
  optional string name = 1;
  optional DimCheckMode share_mode = 2;
  enum DimCheckMode {
    STRICT = 0;
    PERMISSIVE = 1;
  }
  optional float lr_mult = 3 [default = 1.0];
  optional float decay_mult = 4 [default = 1.0];
}

message NetStateRule {
  optional Phase phase = 1;
  optional int32 min_level = 2;
  optional int32 max_level = 3;
  repeated string stage = 4;
  repeated string not_stage = 5;
}

message FillerParameter {
  optional string type = 1 [default = 'constant'];
  optional float value = 2 [default = 0];
  optional float min = 3 [default = 0];
  optional float max = 4 [default = 1];
  optional float mean = 5 [default = 0];
  optional float std = 6 [default = 1];
  optional int32 sparse = 7 [default = -1];
  enum VarianceNorm {
    FAN_IN = 0;
    FAN_OUT = 1;
    AVERAGE = 2;
  }
  optional VarianceNorm variance_norm = 8 [default = FAN_IN];
}

message TransformationParameter {
  optional float scale = 1 [default = 1];
  optional bool mirror = 2 [default = false];
  optional uint32 crop_size = 3 [default = 0];
  optional string mean_file = 4;
  repeated float mean_value = 5;
  optional bool force_color = 6 [default = false];
  optional bool force_gray = 7 [default = false];
}

"""


def load_schema(proto_path):
    """Compile a .proto and import the module protoc generates for it."""
    from grpc_tools import protoc

    work = tempfile.mkdtemp(prefix="caffe-proto-")
    if proto_path is None:
        proto_path = os.path.join(work, "caffe.proto")
        with open(proto_path, "w") as handle:
            handle.write(CAFFE_PROTO)
    stem = os.path.splitext(os.path.basename(proto_path))[0]
    status = protoc.main(
        [
            "protoc",
            "-I" + os.path.dirname(os.path.abspath(proto_path)),
            "--python_out=" + work,
            os.path.abspath(proto_path),
        ]
    )
    if status != 0:
        sys.exit("protoc failed on " + proto_path)
    sys.path.insert(0, work)
    return __import__(stem + "_pb2")


# --------------------------------------------------------------------------
# building the nets

def layer(net, kind, name, bottoms, tops):
    item = net.layer.add()
    item.type = kind
    item.name = name
    for blob in bottoms:
        item.bottom.append(blob)
    for blob in tops:
        item.top.append(blob)
    return item


def put_blob(item, values, shape):
    blob = item.blobs.add()
    for dimension in shape:
        blob.shape.dim.append(dimension)
    for value in np.asarray(values, dtype=np.float32).reshape(-1):
        blob.data.append(float(value))


def legacy_blob(item, values, shape):
    """A blob in the pre-1.0 num/channels/height/width form, no shape field."""
    blob = item.blobs.add()
    padded = ([1] * (4 - len(shape))) + list(shape)
    blob.num, blob.channels, blob.height, blob.width = padded
    for value in np.asarray(values, dtype=np.float32).reshape(-1):
        blob.data.append(float(value))


def spread(rng, count, scale=1.0):
    return np.asarray(rng.uniform(-scale, scale, count), dtype=np.float32)


def declare_input(net, name, shape):
    """The `input` / `input_shape` pair a deploy prototxt usually opens with."""
    net.input.append(name)
    holder = net.input_shape.add()
    for dimension in shape:
        holder.dim.append(dimension)


def declare_input_dim(net, name, shape):
    """The older `input` / `input_dim` form, four numbers per blob."""
    net.input.append(name)
    for dimension in shape:
        net.input_dim.append(dimension)


def declare_input_layer(net, names, shapes):
    """The modern form: an Input layer with one shape per top."""
    item = layer(net, "Input", "input", [], list(names))
    for shape in shapes:
        holder = item.input_param.shape.add()
        for dimension in shape:
            holder.dim.append(dimension)


def build_smallnet(pb, rng):
    """conv -> in-place relu -> max pool -> inner product -> softmax."""
    net = pb.NetParameter()
    net.name = "smallnet"
    declare_input(net, "data", [2, 3, 8, 8])

    conv = layer(net, "Convolution", "conv1", ["data"], ["conv1"])
    conv.convolution_param.num_output = 4
    conv.convolution_param.kernel_size.append(3)
    conv.convolution_param.stride.append(1)
    conv.convolution_param.pad.append(1)
    put_blob(conv, spread(rng, 4 * 3 * 3 * 3, 0.4), [4, 3, 3, 3])
    put_blob(conv, spread(rng, 4, 0.2), [4])

    layer(net, "ReLU", "relu1", ["conv1"], ["conv1"])

    pool = layer(net, "Pooling", "pool1", ["conv1"], ["pool1"])
    pool.pooling_param.pool = pb.PoolingParameter.MAX
    pool.pooling_param.kernel_size = 2
    pool.pooling_param.stride = 2

    inner = layer(net, "InnerProduct", "ip1", ["pool1"], ["ip1"])
    inner.inner_product_param.num_output = 5
    legacy_blob(inner, spread(rng, 5 * 64, 0.15), [5, 64])
    legacy_blob(inner, spread(rng, 5, 0.1), [5])

    layer(net, "Softmax", "prob", ["ip1"], ["prob"])
    return net, {"data": [2, 3, 8, 8]}


def build_bnnet(pb, rng):
    """BatchNorm with a scale factor, one fused Scale, and two bare ones."""
    net = pb.NetParameter()
    net.name = "bnnet"
    declare_input_dim(net, "data", [2, 3, 4, 4])

    first = layer(net, "BatchNorm", "bn1", ["data"], ["bn1"])
    first.batch_norm_param.eps = 1e-5
    first.batch_norm_param.use_global_stats = True
    factor = 7.5
    put_blob(first, np.asarray([0.3, -0.7, 1.25], np.float32) * factor, [3])
    put_blob(first, np.asarray([2.0, 0.5, 3.25], np.float32) * factor, [3])
    put_blob(first, [factor], [1])

    scale = layer(net, "Scale", "scale1", ["bn1"], ["bn1"])
    scale.scale_param.bias_term = True
    put_blob(scale, spread(rng, 3, 1.5), [3])
    put_blob(scale, spread(rng, 3, 0.5), [3])

    layer(net, "ReLU", "relu1", ["bn1"], ["bn1"])

    #: A BatchNorm with no Scale after it, and a scale factor that is not one.
    second = layer(net, "BatchNorm", "bn2", ["bn1"], ["bn2"])
    second.batch_norm_param.eps = 1e-3
    put_blob(second, np.asarray([-0.1, 0.4, 0.9], np.float32) * 3.0, [3])
    put_blob(second, np.asarray([1.5, 2.5, 0.75], np.float32) * 3.0, [3])
    put_blob(second, [3.0], [1])

    #: Caffe writes a zero scale factor when no statistics were ever
    #: accumulated, and reads it back as "the mean and variance are zero".
    third = layer(net, "BatchNorm", "bn3", ["bn2"], ["bn3"])
    third.batch_norm_param.eps = 1.0
    put_blob(third, [5.0, 5.0, 5.0], [3])
    put_blob(third, [9.0, 9.0, 9.0], [3])
    put_blob(third, [0.0], [1])
    return net, {"data": [2, 3, 4, 4]}


def build_eltnet(pb, rng):
    """Every EltwiseOp, including a SUM with coefficients."""
    net = pb.NetParameter()
    net.name = "eltnet"
    declare_input_layer(net, ("a", "b", "c"), [[2, 2, 3, 3]] * 3)

    plain = layer(net, "Eltwise", "e_sum", ["a", "b", "c"], ["e_sum"])
    plain.eltwise_param.operation = pb.EltwiseParameter.SUM

    weighted = layer(net, "Eltwise", "e_coeff", ["a", "b", "c"], ["e_coeff"])
    weighted.eltwise_param.operation = pb.EltwiseParameter.SUM
    for value in (1.5, -2.0, 0.25):
        weighted.eltwise_param.coeff.append(value)

    product = layer(net, "Eltwise", "e_prod", ["a", "b", "c"], ["e_prod"])
    product.eltwise_param.operation = pb.EltwiseParameter.PROD

    largest = layer(net, "Eltwise", "e_max", ["a", "b", "c"], ["e_max"])
    largest.eltwise_param.operation = pb.EltwiseParameter.MAX

    pair = layer(net, "Eltwise", "e_two", ["a", "b"], ["e_two"])
    pair.eltwise_param.operation = pb.EltwiseParameter.PROD
    return net, {name: [2, 2, 3, 3] for name in ("a", "b", "c")}


def build_opsnet(pb, rng):
    """The rest of the layer table, and the two awkward pooling shapes."""
    net = pb.NetParameter()
    net.name = "opsnet"
    declare_input(net, "data", [2, 4, 6, 6])

    lrn = layer(net, "LRN", "lrn1", ["data"], ["lrn1"])
    lrn.lrn_param.local_size = 3
    lrn.lrn_param.alpha = 2e-4
    lrn.lrn_param.beta = 0.7
    lrn.lrn_param.k = 1.5

    #: 6 -> 4 with pad 1, kernel 3, stride 2.  The last window reaches two past
    #: the input and one past the padding, so Caffe divides it by six and not
    #: by nine.
    avg = layer(net, "Pooling", "pavg", ["lrn1"], ["pavg"])
    avg.pooling_param.pool = pb.PoolingParameter.AVE
    avg.pooling_param.kernel_size = 3
    avg.pooling_param.stride = 2
    avg.pooling_param.pad = 1

    small = layer(net, "Convolution", "conv_s", ["data"], ["conv_s"])
    small.convolution_param.num_output = 4
    small.convolution_param.kernel_h = 2
    small.convolution_param.kernel_w = 2
    small.convolution_param.group = 2
    small.convolution_param.bias_term = False
    put_blob(small, spread(rng, 4 * 2 * 2 * 2, 0.5), [4, 2, 2, 2])

    #: 5 -> 2 with pad 1, kernel 3, stride 3.  Ceil alone would say three, and
    #: Caffe drops the last window because it would start in the padding.
    top = layer(net, "Pooling", "pmax", ["conv_s"], ["pmax"])
    top.pooling_param.pool = pb.PoolingParameter.MAX
    top.pooling_param.kernel_size = 3
    top.pooling_param.stride = 3
    top.pooling_param.pad = 1

    up = layer(net, "Deconvolution", "deconv", ["pmax"], ["deconv"])
    up.convolution_param.num_output = 2
    up.convolution_param.kernel_size.append(2)
    up.convolution_param.stride.append(2)
    put_blob(up, spread(rng, 4 * 2 * 2 * 2, 0.6), [4, 2, 2, 2])
    put_blob(up, spread(rng, 2, 0.3), [2])

    joined = layer(net, "Concat", "cat", ["pavg", "deconv"], ["cat"])
    joined.concat_param.axis = 1

    layer(net, "Sigmoid", "sig", ["cat"], ["sig"])
    layer(net, "TanH", "tan", ["sig"], ["tan"])

    power = layer(net, "Power", "pw", ["tan"], ["pw"])
    power.power_param.power = 2.0
    power.power_param.scale = 0.5
    power.power_param.shift = 0.25

    drop = layer(net, "Dropout", "drop", ["pw"], ["pw"])
    drop.dropout_param.dropout_ratio = 0.4

    standalone = layer(net, "Scale", "scal", ["pw"], ["scal"])
    standalone.scale_param.bias_term = True
    put_blob(standalone, spread(rng, 6, 1.2), [6])
    put_blob(standalone, spread(rng, 6, 0.4), [6])

    bias = layer(net, "Bias", "bi", ["scal"], ["bi"])
    put_blob(bias, spread(rng, 6, 0.3), [6])

    flat = layer(net, "Flatten", "flat", ["bi"], ["flat"])
    flat.flatten_param.axis = 1

    shaped = layer(net, "Reshape", "resh", ["bi"], ["resh"])
    for dimension in (0, 6, -1):
        shaped.reshape_param.shape.dim.append(dimension)

    layer(net, "Split", "fan", ["flat"], ["fan_a", "fan_b"])

    elu = layer(net, "ELU", "elu", ["fan_a"], ["elu"])
    elu.elu_param.alpha = 0.7
    layer(net, "AbsVal", "absv", ["fan_b"], ["absv"])
    layer(net, "BNLL", "bnll", ["resh"], ["bnll"])

    slope = layer(net, "PReLU", "pre", ["tan"], ["pre"])
    put_blob(slope, [0.1, -0.25, 0.5, 0.75, 0.2, 1.5], [6])

    cut = layer(net, "Slice", "cut", ["cat"], ["cut_a", "cut_b"])
    cut.slice_param.axis = 1
    cut.slice_param.slice_point.append(2)

    leaky = layer(net, "ReLU", "leaky", ["cut_a"], ["leaky"])
    leaky.relu_param.negative_slope = 0.125
    return net, {"data": [2, 4, 6, 6]}


def build_extranet(pb, rng):
    """The corners the other nets do not reach: global pooling, floor
    rounding, a Scale whose multiplier is another blob, an InnerProduct that
    is transposed and one that lumps from an inner axis, a Concat through the
    deprecated `concat_dim`, a Flatten that stops short of the last axis, a
    Reshape of part of a shape, and a PReLU with one shared slope."""
    net = pb.NetParameter()
    net.name = "extranet"
    declare_input(net, "data", [1, 4, 4, 4])

    whole = layer(net, "Pooling", "gpool", ["data"], ["gpool"])
    whole.pooling_param.pool = pb.PoolingParameter.AVE
    whole.pooling_param.global_pooling = True

    floored = layer(net, "Pooling", "pfloor", ["data"], ["pfloor"])
    floored.pooling_param.pool = pb.PoolingParameter.MAX
    floored.pooling_param.kernel_size = 3
    floored.pooling_param.stride = 2
    floored.pooling_param.round_mode = pb.PoolingParameter.FLOOR

    vector = layer(net, "Reshape", "gflat", ["gpool"], ["gflat"])
    for dimension in (1, 4):
        vector.reshape_param.shape.dim.append(dimension)

    paired = layer(net, "Scale", "scal2", ["data", "gflat"], ["scal2"])
    paired.scale_param.axis = 0
    paired.scale_param.bias_term = True
    put_blob(paired, spread(rng, 4, 0.5), [1, 4])

    middle = layer(net, "Flatten", "flat2", ["data"], ["flat2"])
    middle.flatten_param.axis = 1
    middle.flatten_param.end_axis = 2

    part = layer(net, "Reshape", "resh2", ["data"], ["resh2"])
    for dimension in (2, 2):
        part.reshape_param.shape.dim.append(dimension)
    part.reshape_param.axis = 1
    part.reshape_param.num_axes = 1

    turned = layer(net, "InnerProduct", "ip2", ["data"], ["ip2"])
    turned.inner_product_param.num_output = 3
    turned.inner_product_param.transpose = True
    turned.inner_product_param.bias_term = False
    put_blob(turned, spread(rng, 64 * 3, 0.2), [64, 3])
    layer(net, "ReLU", "relu_ip", ["ip2"], ["ip2"])

    inner = layer(net, "InnerProduct", "ip3", ["data"], ["ip3"])
    inner.inner_product_param.num_output = 2
    inner.inner_product_param.axis = 2
    put_blob(inner, spread(rng, 2 * 16, 0.4), [2, 16])
    put_blob(inner, spread(rng, 2, 0.1), [2])

    old = layer(net, "Concat", "cat2", ["data", "scal2"], ["cat2"])
    old.concat_param.concat_dim = 1

    shared = layer(net, "PReLU", "pre2", ["data"], ["pre2"])
    shared.prelu_param.channel_shared = True
    put_blob(shared, [-0.3], [1])
    return net, {"data": [1, 4, 4, 4]}


# --------------------------------------------------------------------------
# the reference forward pass

def repeated_pair(param, plain, tall, wide, fallback):
    """(h, w) out of a repeated field, or the _h/_w pair beside it."""
    if tall is not None and (param.HasField(tall) or param.HasField(wide)):
        return (getattr(param, tall), getattr(param, wide))
    values = list(getattr(param, plain))
    if not values:
        return (fallback, fallback)
    if len(values) == 1:
        return (values[0], values[0])
    return (values[0], values[1])


def single_pair(param, plain, tall, wide):
    """(h, w) out of a singular field or the _h/_w pair beside it."""
    if param.HasField(tall) or param.HasField(wide):
        return (getattr(param, tall), getattr(param, wide))
    value = getattr(param, plain)
    return (value, value)


def reference(pb, net, feeds):
    import torch
    import torch.nn.functional as fn

    blobs = {
        name: torch.tensor(value, dtype=torch.float64) for (name, value) in feeds.items()
    }

    def broadcastable(values, rank, axis, width):
        shape = [1] * rank
        for slot in range(width):
            shape[axis + slot] = values.shape[slot]
        return values.reshape(shape)

    def weights(item, index, shape):
        raw = np.asarray(item.blobs[index].data, dtype=np.float32).reshape(shape)
        return torch.tensor(raw, dtype=torch.float64)

    for item in net.layer:
        ins = [blobs[name] for name in item.bottom]
        kind = item.type
        if kind in ("Convolution", "Deconvolution"):
            param = item.convolution_param
            window = repeated_pair(param, "kernel_size", "kernel_h", "kernel_w", 0)
            stride = repeated_pair(param, "stride", "stride_h", "stride_w", 1)
            padding = repeated_pair(param, "pad", "pad_h", "pad_w", 0)
            dilation = repeated_pair(param, "dilation", None, None, 1)
            groups = param.group
            channels = ins[0].shape[1]
            if kind == "Convolution":
                shape = [param.num_output, channels // groups, window[0], window[1]]
            else:
                shape = [channels, param.num_output // groups, window[0], window[1]]
            weight = weights(item, 0, shape)
            bias = weights(item, 1, [param.num_output]) if param.bias_term else None
            if kind == "Convolution":
                out = fn.conv2d(ins[0], weight, bias, stride, padding, dilation, groups)
            else:
                out = fn.conv_transpose2d(
                    ins[0], weight, bias, stride, padding, 0, groups, dilation
                )
        elif kind == "Pooling":
            param = item.pooling_param
            if param.global_pooling:
                window = (ins[0].shape[2], ins[0].shape[3])
                stride, padding = (1, 1), (0, 0)
            else:
                window = single_pair(param, "kernel_size", "kernel_h", "kernel_w")
                stride = single_pair(param, "stride", "stride_h", "stride_w")
                padding = single_pair(param, "pad", "pad_h", "pad_w")
            ceiling = param.round_mode == pb.PoolingParameter.CEIL
            if param.pool == pb.PoolingParameter.MAX:
                out = fn.max_pool2d(ins[0], window, stride, padding, 1, ceil_mode=ceiling)
            else:
                out = fn.avg_pool2d(
                    ins[0], window, stride, padding, ceil_mode=ceiling, count_include_pad=True
                )
        elif kind == "InnerProduct":
            param = item.inner_product_param
            axis = param.axis if param.axis >= 0 else ins[0].dim() + param.axis
            inner = int(np.prod(ins[0].shape[axis:]))
            flat = ins[0].reshape(list(ins[0].shape[:axis]) + [inner])
            shape = [inner, param.num_output] if param.transpose else [param.num_output, inner]
            weight = weights(item, 0, shape)
            out = flat @ (weight if param.transpose else weight.T)
            if param.bias_term:
                out = out + weights(item, 1, [param.num_output])
        elif kind == "ReLU":
            out = fn.leaky_relu(ins[0], item.relu_param.negative_slope)
        elif kind == "PReLU":
            count = 1 if item.prelu_param.channel_shared else ins[0].shape[1]
            slope = weights(item, 0, [count])
            out = torch.where(
                ins[0] > 0, ins[0], ins[0] * broadcastable(slope, ins[0].dim(), 1, 1)
            )
        elif kind == "ELU":
            out = fn.elu(ins[0], item.elu_param.alpha)
        elif kind == "Sigmoid":
            out = torch.sigmoid(ins[0])
        elif kind == "TanH":
            out = torch.tanh(ins[0])
        elif kind == "AbsVal":
            out = torch.abs(ins[0])
        elif kind == "BNLL":
            out = torch.log1p(torch.exp(ins[0]))
        elif kind == "LRN":
            param = item.lrn_param
            out = fn.local_response_norm(
                ins[0], param.local_size, param.alpha, param.beta, param.k
            )
        elif kind == "BatchNorm":
            param = item.batch_norm_param
            factor = float(item.blobs[2].data[0])
            reciprocal = 0.0 if factor == 0.0 else 1.0 / factor
            channels = ins[0].shape[1]
            mean = weights(item, 0, [channels]) * reciprocal
            variance = weights(item, 1, [channels]) * reciprocal
            rank = ins[0].dim()
            out = (ins[0] - broadcastable(mean, rank, 1, 1)) / torch.sqrt(
                broadcastable(variance, rank, 1, 1) + param.eps
            )
        elif kind in ("Scale", "Bias"):
            param = item.scale_param if kind == "Scale" else item.bias_param
            axis = param.axis if param.axis >= 0 else ins[0].dim() + param.axis
            if len(ins) > 1:
                learned = ins[1]
                width = learned.dim()
            else:
                width = param.num_axes if param.num_axes >= 0 else ins[0].dim() - axis
                learned = weights(item, 0, list(ins[0].shape[axis:axis + width]))
            stretched = broadcastable(learned, ins[0].dim(), axis, width)
            if kind == "Bias":
                out = ins[0] + stretched
            else:
                out = ins[0] * stretched
                if param.bias_term:
                    shift = weights(item, 0 if len(ins) > 1 else 1, list(learned.shape))
                    out = out + broadcastable(shift, ins[0].dim(), axis, width)
        elif kind == "Concat":
            param = item.concat_param
            out = torch.cat(ins, dim=param.axis if param.HasField("axis") else param.concat_dim)
        elif kind == "Slice":
            param = item.slice_param
            axis = param.axis if param.HasField("axis") else param.slice_dim
            pieces, start = [], 0
            for stop in list(param.slice_point) + [ins[0].shape[axis]]:
                pieces.append(ins[0].narrow(axis, start, stop - start))
                start = stop
            out = pieces
        elif kind == "Softmax":
            out = torch.softmax(ins[0], dim=item.softmax_param.axis)
        elif kind == "Eltwise":
            param = item.eltwise_param
            if param.operation == pb.EltwiseParameter.PROD:
                out = ins[0]
                for other in ins[1:]:
                    out = out * other
            elif param.operation == pb.EltwiseParameter.MAX:
                out = ins[0]
                for other in ins[1:]:
                    out = torch.maximum(out, other)
            else:
                coefficients = list(param.coeff) or [1.0] * len(ins)
                out = ins[0] * coefficients[0]
                for (other, weight) in zip(ins[1:], coefficients[1:]):
                    out = out + other * weight
        elif kind == "Dropout":
            out = ins[0]
        elif kind == "Input":
            continue
        elif kind == "Flatten":
            param = item.flatten_param
            rank = ins[0].dim()
            first = param.axis if param.axis >= 0 else rank + param.axis
            last = param.end_axis if param.end_axis >= 0 else rank + param.end_axis
            dims = list(ins[0].shape[:first])
            dims.append(int(np.prod(ins[0].shape[first:last + 1])))
            dims.extend(ins[0].shape[last + 1:])
            out = ins[0].reshape(dims)
        elif kind == "Reshape":
            param = item.reshape_param
            rank = ins[0].dim()
            first = param.axis if param.axis >= 0 else rank + param.axis + 1
            span = param.num_axes if param.num_axes >= 0 else rank - first
            dims = list(ins[0].shape[:first])
            for (slot, value) in enumerate(param.shape.dim):
                dims.append(ins[0].shape[first + slot] if value == 0 else value)
            dims.extend(ins[0].shape[first + span:])
            known = int(np.prod([value for value in dims if value != -1]))
            dims = [int(ins[0].numel() // known) if value == -1 else int(value) for value in dims]
            out = ins[0].reshape(dims)
        elif kind == "Split":
            out = [ins[0] for _top in item.top]
        elif kind == "Power":
            param = item.power_param
            out = torch.pow(param.shift + param.scale * ins[0], param.power)
        elif kind == "Threshold":
            out = (ins[0] > item.threshold_param.threshold).to(ins[0].dtype)
        else:
            sys.exit("the reference has no " + kind)
        produced = out if isinstance(out, list) else [out]
        for (name, value) in zip(item.top, produced):
            blobs[name] = value
    return blobs


def graph_outputs(net):
    """The last value written to each blob, when nothing reads it afterwards.

    This is the rule the importer uses, so it has to be the rule here: a blob
    that is written, read, and written again is an output on the strength of
    the second write alone.
    """
    version, written, consumed, counter = {}, [], set(), 0
    for item in net.layer:
        if item.type == "Input":
            continue
        for name in item.bottom:
            if name in version:
                consumed.add(version[name])
        for name in item.top:
            counter += 1
            version[name] = counter
            written.append((name, counter))
    out = []
    for (name, mark) in written:
        if version[name] == mark and mark not in consumed and name not in out:
            out.append(name)
    return out


# --------------------------------------------------------------------------
# writing

def numbers(values):
    return " ".join("%.9g" % float(value) for value in np.asarray(values).reshape(-1))


def exact(values):
    """Shortest text that reads back as the same float64, so a float32 weight
    survives the round trip bit for bit and the test can demand equality."""
    return " ".join(repr(float(value)) for value in np.asarray(values).reshape(-1))


def tensor_line(tag, name, array):
    shape = list(array.shape)
    return "%s %s %d %s %d %s" % (
        tag,
        name,
        len(shape),
        " ".join(str(dimension) for dimension in shape),
        int(np.prod(shape)),
        numbers(array),
    )


AWKWARD = '''\
# A prototxt the text parser has to survive.  Everything in here is accepted by
# protobuf's own text_format parser -- caffe_fixtures.py refuses to write the
# file otherwise.
name: "awk" "ward"   # two literals, one string
input: "data"
input_shape < dim: 1 dim: 3 dim: 4 dim: 4 >   # angle brackets, not braces
input_shape {
  dim: [ 1, 3, 4, 4 ]                          # the bracketed list form
}
layer {
  name: "od\\td\\"d\\\\\\x41\\101"                   # tab, quote, backslash, hex, octal
  type: 'Convolution'                          # single quotes
  bottom: "data";                              # a semicolon separator
  top: "conv",                                 # a comma separator
  param { lr_mult: 1.0 decay_mult: 0 }         # a field this importer skips
  param { lr_mult: 2 }
  include { phase: TEST }
  convolution_param: {                         # a colon before the brace
    num_output: 0x10                           # hex
    kernel_size: 3
    pad: 1
    weight_filler { type: "xavier" }           # skipped, and nested
    bias_filler {
      type: "constant"
      value: -1.5e-2                           # exponent form
    }
  }
}
layer {
  name: "pool" type: "Pooling"                 # two fields on one line
  bottom: "conv" top: "pool"
  pooling_param {
    pool: MAX                                  # a bare enum
    round_mode: FLOOR
    kernel_size: 2 stride: 2
  }
}
layer {
  name: "elt" type: "Eltwise"
  bottom: "pool" bottom: "pool"                # a repeated field, twice
  top: "elt"
  eltwise_param { operation: PROD coeff: 1 coeff: -1 }
}
# a comment with a colon and "quotes" and { braces } in it
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".", help="repository root")
    parser.add_argument("--proto", default=None, help="a caffe.proto to compile instead")
    options = parser.parse_args()
    pb = load_schema(options.proto)
    from google.protobuf import text_format

    models = os.path.join(options.out, "packages", "jaicv", "tests", "models")
    os.makedirs(models, exist_ok=True)
    rng = np.random.default_rng(20240819)
    lines = []

    for builder in (build_smallnet, build_bnnet, build_eltnet, build_opsnet, build_extranet):
        net, inputs = builder(pb, rng)
        name = net.name
        with open(os.path.join(models, "caffe_%s.caffemodel" % name), "wb") as handle:
            handle.write(net.SerializeToString())

        #: The prototxt is the same message with the weights taken out, which
        #: is what a deploy prototxt is.
        skeleton = pb.NetParameter()
        skeleton.CopyFrom(net)
        for item in skeleton.layer:
            del item.blobs[:]
        with open(os.path.join(models, "caffe_%s.prototxt" % name), "w") as handle:
            handle.write(text_format.MessageToString(skeleton))

        feeds = {
            blob: np.asarray(rng.uniform(-1.5, 1.5, shape), dtype=np.float32)
            for (blob, shape) in inputs.items()
        }
        produced = reference(pb, net, feeds)
        lines.append("net %s" % name)
        for (blob, value) in feeds.items():
            lines.append(tensor_line("feed", blob, value))
        for item in net.layer:
            for (index, blob) in enumerate(item.blobs):
                raw = np.asarray(blob.data, dtype=np.float32)
                lines.append("blob %s %d %d %s" % (item.name, index, raw.size, exact(raw)))
        for blob in graph_outputs(net):
            lines.append(tensor_line("want", blob, produced[blob].numpy().astype(np.float64)))

    with open(os.path.join(models, "caffe_cases.txt"), "w") as handle:
        handle.write("\n".join(lines) + "\n")

    #: The awkward one goes through protobuf first, so this cannot ship a file
    #: that the real text parser would reject.
    checked = pb.NetParameter()
    text_format.Parse(AWKWARD, checked)
    with open(os.path.join(models, "caffe_awkward.prototxt"), "w") as handle:
        handle.write(AWKWARD)
    print("wrote fixtures into", models)
    print(text_format.MessageToString(checked))


if __name__ == "__main__":
    main()
