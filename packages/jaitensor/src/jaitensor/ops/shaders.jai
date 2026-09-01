#: The shader library every kernel in this package is compiled from.
#:
#: `SOURCE` is one string built from the five parts beside this file, so a
#: kernel can be cached by its entry name alone.
from .shader_core import CORE_SOURCE
from .shader_pool import POOL_SOURCE
from .shader_norm import NORM_SOURCE
from .shader_conv import CONV_SOURCE
from .shader_layout import LAYOUT_SOURCE

let BASE_SOURCE = CORE_SOURCE + POOL_SOURCE + NORM_SOURCE + CONV_SOURCE + LAYOUT_SOURCE

#: Each unary operation as the expression it applies to `value`, in the order
#: their operation codes have always had.
#:
#: One kernel is generated per entry instead of the single `unary_op` that
#: switched on a code at runtime. The switch was not the cost -- the shared
#: body was: every branch's maths lives in one function, so the compiler sizes
#: its register use for the worst of them and the GPU runs far fewer threads at
#: once than it could. A specialised copy of SiLU measured 313 GB/s against the
#: shared kernel's 60 on the same buffers.
#:
#: Compiling more entry points is close to free: the Metal library is cached by
#: source text, so each of these costs a pipeline object and no recompilation.
pub let UNARY_FORMS = [
    ("copy", "value"),
    ("relu", "max(value, 0.0f)"),
    ("sigmoid", "1.0f / (1.0f + exp(-value))"),
    ("tanh", "tanh(value)"),
    ("exp", "exp(value)"),
    ("log", "log(value)"),
    ("abs", "fabs(value)"),
    ("neg", "-value"),
    ("sqrt", "sqrt(value)"),
    ("rsqrt", "rsqrt(value)"),
    ("square", "value * value"),
    (
        "gelu",
        "0.5f * value * (1.0f + tanh(0.7978845608f * (value + 0.044715f * value * value * " +
            "value)))"
    ),
    ("silu", "value / (1.0f + exp(-value))"),
    ("leaky_relu", "value > 0.0f ? value : 0.01f * value"),
]

#: The same, for the four elementwise binary operations.
pub let BINARY_FORMS = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]

#: The generated half of the shader library.
fn elementwise_kernels() -> str {
    var parts: list[str] = []
    for (name, expression) in UNARY_FORMS {
        parts.push(
            f"kernel void unary_{name}(device const float* input [[buffer(0)]],\n" +
                "                     device float* output [[buffer(1)]],\n" +
                "                     constant uint& count [[buffer(2)]],\n" +
                "                     uint id [[thread_position_in_grid]]) {\n" +
                "    if (id >= count) return;\n" +
                "    float value = input[id];\n" +
                f"    output[id] = {expression};\n" +
                "}\n"
        )
    }
    for (name, symbol) in BINARY_FORMS {
        parts.push(
            f"kernel void binary_{name}(device const float* a [[buffer(0)]],\n" +
                "                      device const float* b [[buffer(1)]],\n" +
                "                      device float* output [[buffer(2)]],\n" +
                "                      constant uint& count [[buffer(3)]],\n" +
                "                      uint id [[thread_position_in_grid]]) {\n" +
                "    if (id >= count) return;\n" +
                f"    output[id] = a[id] {symbol} b[id];\n" +
                "}\n"
        )
    }
    return "\n".join(parts)
}

pub let SOURCE = BASE_SOURCE + elementwise_kernels()
