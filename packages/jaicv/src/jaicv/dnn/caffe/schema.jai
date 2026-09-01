#: caffe.proto's field numbers and names, and the enum spellings that go with them.

#: How deeply a message may nest before the file is called malformed. Caffe's
#: own schema goes four deep; the cap is only there so that a hostile run of
#: opening braces cannot walk the parser off the stack.
pub const MAX_DEPTH = 64

#: One field of one caffe.proto message: what a prototxt calls it, what number
#: a caffemodel carries it under, and how to read the value.
#:
#: `kind` is `"int"`, `"bool"`, `"enum"`, `"float"`, `"double"`, `"str"` or
#: `"msg"`. `of` names the enum or the message for the last two kinds.
pub class Spec {
    pub let number: int
    pub let name: str
    pub let kind: str
    pub let of: str
    pub let repeated: bool

    pub fn init(self, number: int, name: str, kind: str, of: str = "", repeated: bool = false) {
        self.number = number
        self.name = name
        self.kind = kind
        self.of = of
        self.repeated = repeated
    }
}

#: One message's fields, indexed the two ways the two file formats need them.
pub class Layout {
    pub var by_name: dict[str, Spec]
    pub var by_number: dict[int, Spec]

    pub fn init(self, fields: list[Spec]) {
        self.by_name = {}
        self.by_number = {}
        for entry in fields {
            self.by_name[entry.name] = entry
            self.by_number[entry.number] = entry
        }
    }
}

fn one(number: int, name: str, kind: str, of: str = "") -> Spec {
    return Spec(number, name, kind, of: of)
}

fn many(number: int, name: str, kind: str, of: str = "") -> Spec {
    return Spec(number, name, kind, of: of, repeated: true)
}

var _schema: dict[str, Layout] = {}

#: caffe.proto, cut down to the messages this importer reads.
#:
#: Every number here is upstream's, checked against
#: https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto
#: rather than against a table in a document. `tools/caffe_fixtures.py` carries
#: the same subset as a real `.proto` and writes the test fixtures through
#: protobuf itself, so a wrong number here fails a test rather than silently
#: reading the wrong field.
fn schema() -> dict[str, Layout] {
    if _schema.len() > 0 { return _schema }
    _schema["BlobShape"] = Layout([many(1, "dim", "int")])
    _schema["BlobProto"] = Layout(
        [
            one(1, "num", "int"),
            one(2, "channels", "int"),
            one(3, "height", "int"),
            one(4, "width", "int"),
            many(5, "data", "float"),
            many(6, "diff", "float"),
            one(7, "shape", "msg", "BlobShape"),
            many(8, "double_data", "double"),
            many(9, "double_diff", "double"),
        ]
    )
    _schema["NetParameter"] = Layout(
        [
            one(1, "name", "str"),
            many(2, "layers", "msg", "V1LayerParameter"),
            many(3, "input", "str"),
            many(4, "input_dim", "int"),
            many(8, "input_shape", "msg", "BlobShape"),
            many(100, "layer", "msg", "LayerParameter"),
        ]
    )
    #: Nothing reads a V1 layer's fields -- its presence alone is refused --
    #: so an empty layout is enough to skip past one in either format.
    _schema["V1LayerParameter"] = Layout([])
    _schema["LayerParameter"] = Layout(
        [
            one(1, "name", "str"),
            one(2, "type", "str"),
            many(3, "bottom", "str"),
            many(4, "top", "str"),
            many(7, "blobs", "msg", "BlobProto"),
            one(10, "phase", "enum", "Phase"),
            one(104, "concat_param", "msg", "ConcatParameter"),
            one(106, "convolution_param", "msg", "ConvolutionParameter"),
            one(108, "dropout_param", "msg", "DropoutParameter"),
            one(110, "eltwise_param", "msg", "EltwiseParameter"),
            one(
                117,
                "inner_product_param",
                "msg",
                "InnerProductParameter"
            ),
            one(118, "lrn_param", "msg", "LRNParameter"),
            one(121, "pooling_param", "msg", "PoolingParameter"),
            one(122, "power_param", "msg", "PowerParameter"),
            one(123, "relu_param", "msg", "ReLUParameter"),
            one(125, "softmax_param", "msg", "SoftmaxParameter"),
            one(126, "slice_param", "msg", "SliceParameter"),
            one(131, "prelu_param", "msg", "PReLUParameter"),
            one(133, "reshape_param", "msg", "ReshapeParameter"),
            one(135, "flatten_param", "msg", "FlattenParameter"),
            one(139, "batch_norm_param", "msg", "BatchNormParameter"),
            one(140, "elu_param", "msg", "ELUParameter"),
            one(141, "bias_param", "msg", "BiasParameter"),
            one(142, "scale_param", "msg", "ScaleParameter"),
            one(143, "input_param", "msg", "InputParameter"),
        ]
    )
    _schema["ConvolutionParameter"] = Layout(
        [
            one(1, "num_output", "int"),
            one(2, "bias_term", "bool"),
            many(3, "pad", "int"),
            many(4, "kernel_size", "int"),
            one(5, "group", "int"),
            many(6, "stride", "int"),
            one(9, "pad_h", "int"),
            one(10, "pad_w", "int"),
            one(11, "kernel_h", "int"),
            one(12, "kernel_w", "int"),
            one(13, "stride_h", "int"),
            one(14, "stride_w", "int"),
            one(16, "axis", "int"),
            many(18, "dilation", "int"),
        ]
    )
    _schema["PoolingParameter"] = Layout(
        [
            one(1, "pool", "enum", "PoolMethod"),
            one(2, "kernel_size", "int"),
            one(3, "stride", "int"),
            one(4, "pad", "int"),
            one(5, "kernel_h", "int"),
            one(6, "kernel_w", "int"),
            one(7, "stride_h", "int"),
            one(8, "stride_w", "int"),
            one(9, "pad_h", "int"),
            one(10, "pad_w", "int"),
            one(12, "global_pooling", "bool"),
            one(13, "round_mode", "enum", "RoundMode"),
        ]
    )
    _schema["InnerProductParameter"] = Layout(
        [
            one(1, "num_output", "int"),
            one(2, "bias_term", "bool"),
            one(5, "axis", "int"),
            one(6, "transpose", "bool"),
        ]
    )
    _schema["ReLUParameter"] = Layout([one(1, "negative_slope", "float")])
    _schema["LRNParameter"] = Layout(
        [
            one(1, "local_size", "int"),
            one(2, "alpha", "float"),
            one(3, "beta", "float"),
            one(4, "norm_region", "enum", "NormRegion"),
            one(5, "k", "float"),
        ]
    )
    _schema["BatchNormParameter"] = Layout(
        [one(1, "use_global_stats", "bool"), one(3, "eps", "float")]
    )
    _schema["ScaleParameter"] = Layout(
        [
            one(1, "axis", "int"),
            one(2, "num_axes", "int"),
            one(4, "bias_term", "bool"),
        ]
    )
    _schema["BiasParameter"] = Layout([one(1, "axis", "int"), one(2, "num_axes", "int")])
    _schema["ConcatParameter"] = Layout([one(1, "concat_dim", "int"), one(2, "axis", "int")])
    _schema["SoftmaxParameter"] = Layout([one(2, "axis", "int")])
    _schema["EltwiseParameter"] = Layout(
        [
            one(1, "operation", "enum", "EltwiseOp"),
            many(2, "coeff", "float"),
        ]
    )
    _schema["DropoutParameter"] = Layout([one(1, "dropout_ratio", "float")])
    _schema["ReshapeParameter"] = Layout(
        [
            one(1, "shape", "msg", "BlobShape"),
            one(2, "axis", "int"),
            one(3, "num_axes", "int"),
        ]
    )
    _schema["FlattenParameter"] = Layout([one(1, "axis", "int"), one(2, "end_axis", "int")])
    _schema["PowerParameter"] = Layout(
        [
            one(1, "power", "float"),
            one(2, "scale", "float"),
            one(3, "shift", "float"),
        ]
    )
    _schema["SliceParameter"] = Layout(
        [
            one(1, "slice_dim", "int"),
            many(2, "slice_point", "int"),
            one(3, "axis", "int"),
        ]
    )
    _schema["ELUParameter"] = Layout([one(1, "alpha", "float")])
    _schema["InputParameter"] = Layout([many(1, "shape", "msg", "BlobShape")])
    _schema["PReLUParameter"] = Layout([one(2, "channel_shared", "bool")])
    return _schema
}

pub fn layout_of(kind: str) -> Layout {
    let table = schema().get(kind, null)
    if table is null { throw ValueError(f"this importer has no schema for a {kind}") }
    return table
}

#: The enum identifiers a prototxt may spell out. A caffemodel writes the
#: numbers, so this is only ever needed on the text side.
pub let ENUMS: dict[str, dict[str, int]] = {
    "Phase": {"TRAIN": 0, "TEST": 1},
    "PoolMethod": {"MAX": 0, "AVE": 1, "STOCHASTIC": 2},
    "RoundMode": {"CEIL": 0, "FLOOR": 1},
    "NormRegion": {"ACROSS_CHANNELS": 0, "WITHIN_CHANNEL": 1},
    "EltwiseOp": {"PROD": 0, "SUM": 1, "MAX": 2},
}
