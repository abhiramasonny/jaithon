#: The operator tables the builder translates against, and the one size limit
#: constant folding answers to.

#: How many elements a value may hold and still be folded at compile time.
#: Shape arithmetic deals in a handful of integers; this is loose enough for
#: any of it and tight enough that no real tensor is ever evaluated twice.
pub let FOLD_LIMIT = 4096

#: The unary operators the builder knows, by the code it takes.
pub let UNARY_CODES = {
    "Relu": 0,
    "Sigmoid": 1,
    "Tanh": 2,
    "Exp": 3,
    "Log": 4,
    "Sqrt": 5,
    "Neg": 6,
    "Abs": 7,
    "Erf": 8,
    "Silu": 9,
    "Floor": 10,
    "Ceil": 11,
    "Reciprocal": 12,
    "Sqr": 13,
}

pub let BINARY_CODES = {
    "Add": 0,
    "Sub": 1,
    "Mul": 2,
    "Div": 3,
    "Max": 4,
    "Min": 5,
    "Pow": 6,
    "Equal": 7,
    "Greater": 8,
    "Less": 9,
    "And": 10,
    "Or": 11,
}

pub let REDUCE_CODES = {"ReduceSum": 0, "ReduceMean": 1, "ReduceMax": 2, "ReduceMin": 3}

#: How a `Pad` fills what it adds, in the builder's numbering.
pub let PAD_MODES = {"constant": 0, "reflect": 1, "edge": 3}

#: The ONNX element types, as the importer numbers them.
pub let INTEGER_TYPES = [2, 3, 4, 5, 6, 7, 12, 13]
pub let FLOAT_TYPES = [1, 10, 11, 16]

#: How a nearest resize picks its source pixel, in the builder's numbering.
pub let ROUNDING_CODES = {"round_prefer_ceil": 0, "round_prefer_floor": 1, "ceil": 2, "floor": 3}

#: Operators whose result depends only on the shapes of things, and which are
#: therefore worth evaluating once at compile time rather than every run.
pub let FOLDABLE = {
    "Shape": true,
    "Constant": true,
    "ConstantOfShape": true,
    "Gather": true,
    "Unsqueeze": true,
    "Squeeze": true,
    "Concat": true,
    "Slice": true,
    "Reshape": true,
    "Cast": true,
    "Add": true,
    "Sub": true,
    "Mul": true,
    "Div": true,
    "Mod": true,
    "Equal": true,
    "Where": true,
    "Range": true,
}
