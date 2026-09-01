#: The dtype tag, the `Value` cell it describes, and the `Agg` vocabulary.
from std.core import Printable
from std.math import NAN

#: How a column stores its values and what operations mean over it.
pub enum Dtype {
    #: A single float32 lane. Named `Float64` for source compatibility with
    #: pandas, the same compromise jaicv's `CV_64F` makes; the storage is
    #: float32 and so is the precision.
    Float64,
    #: Two lanes, exact to 48 bits.
    Int64,
    #: A single lane holding 0.0 or 1.0.
    Bool,
    #: Dictionary codes in `values`, the strings in `levels`.
    Str,
    #: The same storage as `Str`. What differs is intent: the levels are the
    #: domain, so group-by may report an empty group for a level no row uses.
    Category,
    #: Two lanes of milliseconds since the Unix epoch — datetime64[ms].
    #: 48 bits of milliseconds is +/-8900 years. Nanoseconds, pandas' unit,
    #: would be +/-3 days.
    Datetime,

    #: The pandas spelling of this dtype.
    pub fn name(self) -> str {
        return match self {
            Dtype.Float64 => "float64",
            Dtype.Int64 => "int64",
            Dtype.Bool => "bool",
            Dtype.Str => "str",
            Dtype.Category => "category",
            Dtype.Datetime => "datetime64[ms]",
        }
    }

    #: Whether arithmetic and numeric reductions are defined over it.
    #:
    #: `Bool` counts, as it does in pandas, because summing a mask is the
    #: point of one. `Datetime` does not: differences of timestamps are
    #: durations, which is the timeseries module's business, not arithmetic's.
    pub fn is_numeric(self) -> bool {
        return match self {
            Dtype.Float64 | Dtype.Int64 | Dtype.Bool => true,
            _ => false,
        }
    }

    #: Whether a column of this dtype carries a second `wide` lane.
    pub fn is_wide(self) -> bool {
        return match self {
            Dtype.Int64 | Dtype.Datetime => true,
            _ => false,
        }
    }

    #: Whether `values` holds codes into a `levels` table rather than a payload.
    pub fn is_dictionary(self) -> bool {
        return match self {
            Dtype.Str | Dtype.Category => true,
            _ => false,
        }
    }

    #: The dtype `name` spells, for `astype("int64")` and for CSV headers.
    pub static fn from_name(text: str) -> Dtype {
        return match text {
            "float64" | "float" | "float32" | "f8" => Dtype.Float64,
            "int64" | "int" | "i8" => Dtype.Int64,
            "bool" | "boolean" => Dtype.Bool,
            "str" | "string" | "object" => Dtype.Str,
            "category" => Dtype.Category,
            "datetime64[ms]" | "datetime64[ns]" | "datetime" => Dtype.Datetime,
            _ => throw ValueError(f"unknown dtype {text}"),
        }
    }
}

#: One cell: a dtype tag, the payload for that dtype, and a missing flag.
#:
#: Which field carries the payload follows the dtype, the same way the lanes
#: do. `Float64` and `Bool` use `number`, `Int64` and `Datetime` use `integer`,
#: `Str` and `Category` use `text`. A missing cell leaves all three at their
#: zero and answers `is_na`.
pub class Value: Printable {
    #: What the cell would be if it were not missing.
    pub let dtype: Dtype
    #: Payload for `Float64` and `Bool`.
    pub let number: float
    #: Payload for `Int64` and `Datetime`.
    pub let integer: int
    #: Payload for `Str` and `Category`.
    pub let text: str?
    #: Whether the cell holds nothing.
    pub let missing: bool

    #: Build a cell directly. The `of_*` constructors are the ones to reach for.
    pub fn init(
        self,
        dtype: Dtype,
        number: float = 0.0,
        integer: int = 0,
        text: str? = null,
        missing: bool = false
    ) {
        self.dtype = dtype
        self.number = number
        self.integer = integer
        self.text = text
        self.missing = missing
    }

    #: A `Float64` cell.
    pub static fn of_float(value: float) -> Value {
        return Value(Dtype.Float64, number: value)
    }

    #: An `Int64` cell.
    pub static fn of_int(value: int) -> Value {
        return Value(Dtype.Int64, integer: value)
    }

    #: A `Bool` cell.
    pub static fn of_bool(value: bool) -> Value {
        return Value(Dtype.Bool, number: if value { 1.0 } else { 0.0 })
    }

    #: A `Str` cell, or a `Category` one when `dtype` says so.
    pub static fn of_str(value: str, dtype: Dtype = Dtype.Str) -> Value {
        if not dtype.is_dictionary() {
            throw ValueError(f"{dtype.name()} does not hold text")
        }
        return Value(dtype, text: value)
    }

    #: A `Datetime` cell, milliseconds since the Unix epoch.
    pub static fn of_datetime(millis: int) -> Value {
        return Value(Dtype.Datetime, integer: millis)
    }

    #: The missing cell of `dtype`.
    pub static fn null_of(dtype: Dtype) -> Value {
        return Value(dtype, missing: true)
    }

    #: Whether the cell holds nothing.
    pub fn is_na(self) -> bool { return self.missing }

    #: The payload as a float, whatever lane it came from.
    #:
    #: A missing cell is NaN, which is what a float column stores for one and
    #: what a plot or a mean has to see.
    pub fn to_float(self) -> float {
        if self.missing { return NAN }
        if self.dtype.is_wide() { return float(self.integer) }
        if self.dtype.is_dictionary() {
            throw TypeError(f"a {self.dtype.name()} value has no numeric payload")
        }
        return self.number
    }

    #: The payload as an integer, truncated when it came from a float lane.
    pub fn to_int(self) -> int {
        if self.missing { throw ValueError("a missing value has no integer") }
        if self.dtype.is_wide() { return self.integer }
        if self.dtype.is_dictionary() {
            throw TypeError(f"a {self.dtype.name()} value has no numeric payload")
        }
        return int(self.number)
    }

    #: The payload as a truth value: nonzero, or a non-empty string.
    pub fn to_bool(self) -> bool {
        if self.missing { throw ValueError("a missing value has no truth value") }
        if self.dtype.is_dictionary() { return (self.text ?? "").len() > 0 }
        if self.dtype.is_wide() { return self.integer != 0 }
        return self.number != 0.0
    }

    #: The cell as text. `NA` for a missing one.
    pub fn to_str(self) -> str {
        if self.missing { return "NA" }
        if self.dtype == Dtype.Bool { return if self.number != 0.0 { "true" } else { "false" } }
        return match self.dtype {
            Dtype.Str | Dtype.Category => self.text ?? "",
            Dtype.Int64 | Dtype.Datetime => f"{self.integer}",
            _ => f"{self.number}",
        }
    }

    #: Cells are equal when both are missing, or when the same dtype family
    #: carries the same payload.
    #:
    #: Two missing cells compare equal whatever their dtypes, which is what
    #: makes the dtype-less `NA` usable against a column of any type. Note this
    #: is `NA == NA`, unlike pandas' propagating `NA`; a cell comparison here
    #: is a test, not an element of an expression.
    pub fn __eq__(self, other: any) -> bool {
        if other is not Value { return false }
        if self.missing or other.missing { return self.missing and other.missing }
        if self.dtype.is_dictionary() != other.dtype.is_dictionary() { return false }
        if self.dtype.is_dictionary() { return self.text == other.text }
        if self.dtype.is_wide() and other.dtype.is_wide() {
            return self.integer == other.integer
        }
        return self.to_float() == other.to_float()
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: The missing cell, of no particular dtype.
#:
#: It equals the missing cell of every dtype, so `column.get(i) == NA` is the
#: null test whatever the column holds.
pub let NA = Value(Dtype.Float64, missing: true)

#: The aggregations reduce, group-by, rolling, resample and pivot all name.
#:
#: One enum rather than one per module, because a user writes
#: `df.groupby("k").agg(Agg.Mean)` and `df.rolling(3).agg(Agg.Mean)` and expects
#: the same word to mean the same thing.
pub enum Agg {
    #: Sum of the present values; 0 over an all-null group.
    Sum,
    #: Arithmetic mean of the present values.
    Mean,
    #: Smallest present value.
    Min,
    #: Largest present value.
    Max,
    #: Number of present values.
    Count,
    #: Number of rows, nulls included. This is what `Count` is not.
    Size,
    #: Product of the present values.
    Prod,
    #: Sample standard deviation, one degree of freedom.
    Std,
    #: Sample variance, one degree of freedom.
    Var,
    #: Middle present value, averaged over the two middles when even.
    Median,
    #: A named quantile; the caller carries the fraction alongside.
    Quantile,
    #: First present value in row order.
    First,
    #: Last present value in row order.
    Last,
    #: Number of distinct present values.
    Nunique,
    #: Whether any present value is truthy.
    Any,
    #: Whether every present value is truthy.
    All,

    #: The pandas spelling, which is also the suffix a named aggregation gets.
    pub fn name(self) -> str {
        return match self {
            Agg.Sum => "sum",
            Agg.Mean => "mean",
            Agg.Min => "min",
            Agg.Max => "max",
            Agg.Count => "count",
            Agg.Size => "size",
            Agg.Prod => "prod",
            Agg.Std => "std",
            Agg.Var => "var",
            Agg.Median => "median",
            Agg.Quantile => "quantile",
            Agg.First => "first",
            Agg.Last => "last",
            Agg.Nunique => "nunique",
            Agg.Any => "any",
            Agg.All => "all",
        }
    }

    #: The aggregation `name` spells, so `df.agg("mean")` reaches the enum.
    pub static fn from_name(text: str) -> Agg {
        return match text {
            "sum" => Agg.Sum,
            "mean" | "avg" | "average" => Agg.Mean,
            "min" => Agg.Min,
            "max" => Agg.Max,
            "count" => Agg.Count,
            "size" => Agg.Size,
            "prod" | "product" => Agg.Prod,
            "std" => Agg.Std,
            "var" => Agg.Var,
            "median" => Agg.Median,
            "quantile" => Agg.Quantile,
            "first" => Agg.First,
            "last" => Agg.Last,
            "nunique" => Agg.Nunique,
            "any" => Agg.Any,
            "all" => Agg.All,
            _ => throw ValueError(f"unknown aggregation {text}"),
        }
    }

    #: Whether the result is a count and so is `Int64` whatever went in.
    pub fn counts_rows(self) -> bool {
        return match self {
            Agg.Count | Agg.Size | Agg.Nunique => true,
            _ => false,
        }
    }

    #: Whether the result is one of the inputs rather than a new number, and so
    #: keeps the input's dtype and works over text as well as numbers.
    pub fn picks_a_row(self) -> bool {
        return match self {
            Agg.Min | Agg.Max | Agg.First | Agg.Last => true,
            _ => false,
        }
    }
}
