#: The dtype an operation computes in, and the dtype it produces.
from ..column import Dtype
from .codes import UN_CLIP, UN_EXP, UN_LOG, UN_LOG10, UN_LOG2, UN_NOT
from .codes import UN_RECIPROCAL, UN_SQRT, unary_name

#: The dtype an operation between `a` and `b` computes in.
#:
#: pandas' ladder, and the reason it is a named function rather than a table
#: inside the dispatch: `astype`, a join's key check and a concatenation all
#: want to ask the same question.
#:
#: `Bool` promotes to `Int64` rather than staying `Bool`, so that `mask + mask`
#: counts the way `sum` over a mask counts; numpy would have made it a logical
#: or, which is what `or_` is for. Two dictionary columns share their dtype,
#: which is enough for a comparison and is refused for arithmetic further down.
#: `Datetime` only pairs with itself: a difference of timestamps is a duration,
#: which is the timeseries module's business.
pub fn promote(a: Dtype, b: Dtype) -> Dtype {
    if a.is_dictionary() and b.is_dictionary() { return a }
    if a.is_dictionary() or b.is_dictionary() {
        throw TypeError(f"{a.name()} and {b.name()} have no type in common")
    }
    if a == Dtype.Datetime or b == Dtype.Datetime {
        if a == b { return Dtype.Datetime }
        throw TypeError(f"{a.name()} and {b.name()} have no type in common")
    }
    if a == Dtype.Float64 or b == Dtype.Float64 { return Dtype.Float64 }
    return Dtype.Int64
}

#: The dtype a unary operation over `dtype` produces.
pub fn unary_result(op: int, dtype: Dtype) -> Dtype {
    if op == UN_NOT {
        if dtype != Dtype.Bool {
            throw TypeError(f"not_ is defined on bool, not on {dtype.name()}")
        }
        return Dtype.Bool
    }
    if dtype.is_dictionary() {
        throw TypeError(f"{unary_name(op)} is not defined on {dtype.name()}")
    }
    if dtype == Dtype.Datetime {
        #: Clipping a timestamp to a window is a date; negating one is not.
        if op != UN_CLIP {
            throw TypeError(f"{unary_name(op)} is not defined on {dtype.name()}")
        }
        return Dtype.Datetime
    }
    let transcendental = op == UN_EXP or op == UN_LOG or op == UN_LOG2 or op == UN_LOG10
    if transcendental or op == UN_SQRT or op == UN_RECIPROCAL { return Dtype.Float64 }
    if dtype == Dtype.Bool { return Dtype.Int64 }
    return dtype
}
