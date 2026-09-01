#: The op codes the Metal switch reads, and the names they answer to when an
#: operation has to refuse.

#: Op codes for `frame_binary`. The Metal switch reads the same numbers, and
#: nothing but these constants may assign them.
pub const OP_ADD = 0
pub const OP_SUBTRACT = 1
pub const OP_MULTIPLY = 2
pub const OP_DIVIDE = 3
pub const OP_FLOOR_DIVIDE = 4
pub const OP_MODULO = 5
pub const OP_POWER = 6
pub const OP_EQUAL = 7
pub const OP_NOT_EQUAL = 8
pub const OP_LESS = 9
pub const OP_LESS_EQUAL = 10
pub const OP_GREATER = 11
pub const OP_GREATER_EQUAL = 12
pub const OP_AND = 13
pub const OP_OR = 14
pub const OP_XOR = 15

#: Op codes for `frame_unary`, likewise mirrored in the shader.
pub const UN_NEGATE = 0
pub const UN_ABSOLUTE = 1
pub const UN_SIGN = 2
pub const UN_ROUND = 3
pub const UN_FLOOR = 4
pub const UN_CEILING = 5
pub const UN_TRUNCATE = 6
pub const UN_EXP = 7
pub const UN_LOG = 8
pub const UN_LOG2 = 9
pub const UN_LOG10 = 10
pub const UN_SQRT = 11
pub const UN_SQUARE = 12
pub const UN_RECIPROCAL = 13
pub const UN_CLIP = 14
pub const UN_NOT = 15

pub fn binary_name(op: int) -> str {
    let names = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "modulo",
        "power",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "and_",
        "or_",
        "xor",
    ]
    return names[op]
}

pub fn unary_name(op: int) -> str {
    let names = [
        "negate",
        "absolute",
        "sign",
        "round",
        "floor",
        "ceiling",
        "truncate",
        "exp",
        "log",
        "log2",
        "log10",
        "sqrt",
        "square",
        "reciprocal",
        "clip",
        "not_",
    ]
    return names[op]
}
