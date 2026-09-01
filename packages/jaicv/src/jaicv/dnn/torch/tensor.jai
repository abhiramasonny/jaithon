#: The tensor a checkpoint hands back, and the decoding of its raw elements.
from std.core import Printable
from .bits import dtype_size, f16_from_bits, f32_from_bits, f64_from_halves
from .bits import le_int, le_u16, le_u32

#: One tensor read out of a checkpoint.
#:
#: The elements are kept as the raw little-endian bytes torch wrote rather than
#: as decoded numbers, because nothing narrower holds all of them: an `int64`
#: near its limits does not survive a double, and a `float64` does not survive
#: a float. `floats` and `ints` decode on demand, and `data` is there for a
#: caller that would rather reinterpret the bytes than go round through either.
#:
#: The bytes are this tensor's own row-major order, not the storage's. A
#: transposed or sliced tensor is gathered through its strides when it is read,
#: so nothing downstream has to know it was ever a view.
pub class TorchTensor: Printable {
    #: Dimensions, outermost first. Empty for a 0-dimensional tensor.
    pub let shape: list[int]
    #: One of `float32`, `float64`, `float16`, `bfloat16`, `int64`, `int32`,
    #: `int16`, `int8`, `uint8`, `bool`.
    pub let dtype: str
    #: `numel() * item_size()` bytes, row-major, little-endian.
    pub let data: bytes

    pub fn init(self, shape: list[int], dtype: str, data: bytes) {
        let width = dtype_size(dtype)
        var count = 1
        for dimension in shape {
            if dimension < 0 {
                throw ValueError(f"a tensor cannot have shape {shape}")
            }
            count *= dimension
        }
        if data.len() != count * width {
            throw ValueError(
                f"a {dtype}{shape} tensor holds {count * width} bytes, and {data.len()} arrived"
            )
        }
        self.shape = shape
        self.dtype = dtype
        self.data = data
    }

    #: How many elements the tensor holds. A 0-dimensional tensor holds one.
    pub fn numel(self) -> int {
        var count = 1
        for dimension in self.shape { count *= dimension }
        return count
    }

    #: Bytes per element.
    pub fn item_size(self) -> int { return dtype_size(self.dtype) }

    #: Every element as a float, row-major.
    #:
    #: Integer dtypes widen on the way out, which is exact up to 2^53 and not
    #: past it; `ints` is the exact reading of those.
    pub fn floats(self) -> list[float] {
        let count = self.numel()
        var out: list[float] = []
        if self.dtype == "float32" {
            for index in 0..count { out.push(f32_from_bits(le_u32(self.data, index * 4))) }
            return out
        }
        if self.dtype == "float64" {
            for index in 0..count {
                out.push(
                    f64_from_halves(le_u32(self.data, index * 8 + 4), le_u32(self.data, index * 8))
                )
            }
            return out
        }
        if self.dtype == "float16" {
            for index in 0..count { out.push(f16_from_bits(le_u16(self.data, index * 2))) }
            return out
        }
        if self.dtype == "bfloat16" {
            #: A bfloat16 is the top half of a float32 and nothing else, so
            #: putting the halves back is the whole conversion.
            for index in 0..count { out.push(f32_from_bits(le_u16(self.data, index * 2) << 16)) }
            return out
        }
        for value in self.ints() { out.push(float(value)) }
        return out
    }

    #: Every element as an int, row-major. Booleans come back as 0 and 1.
    pub fn ints(self) -> list[int] {
        let count = self.numel()
        var out: list[int] = []
        if self.dtype == "bool" {
            for index in 0..count { out.push(self.data[index] == 0 ? 0 : 1) }
            return out
        }
        if self.dtype == "uint8" {
            for index in 0..count { out.push(self.data[index]) }
            return out
        }
        if self.dtype == "int8" or
            self.dtype == "int16" or
            self.dtype == "int32" or
            self.dtype == "int64" {
            let width = self.item_size()
            for index in 0..count { out.push(le_int(self.data, index * width, width, true)) }
            return out
        }
        throw ValueError(f"a {self.dtype} tensor has no exact integer reading; call floats()")
    }

    #: Every element of a `bool` tensor.
    pub fn bools(self) -> list[bool] {
        if self.dtype != "bool" {
            throw ValueError(f"a {self.dtype} tensor is not a mask; call floats() or ints()")
        }
        var out: list[bool] = []
        for index in 0..self.numel() { out.push(self.data[index] != 0) }
        return out
    }

    pub fn to_str(self) -> str { return f"TorchTensor({self.dtype}, {self.shape})" }
}
