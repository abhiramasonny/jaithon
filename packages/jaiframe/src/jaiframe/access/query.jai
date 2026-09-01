#: The host-side predicate path: a row view, and the calls that walk it.
from std.math import NAN, is_nan

from ..column import Column, DataFrame, Dtype, Value
from .select import filter_rows

#: One row of a frame, as a predicate sees it.
#:
#: The lanes are downloaded once, when the view is built, and the view is then
#: pointed at each row in turn rather than rebuilt — a frame of a million rows
#: allocates one of these, not a million. That still leaves a boxed `Value`
#: per cell the predicate touches, which is why `query` is the slow path.
pub class Row {
    #: Which row the view is pointed at. `query` moves it; a predicate reads it.
    pub var at: int
    var lookup: dict[str, int]
    var dtypes: list[Dtype]
    var numbers: list[list[float]]
    var integers: list[list[int]]
    var texts: list[list[str?]]
    var absent: list[list[bool]]

    #: An empty view. `over` is the constructor.
    pub fn init(self) {
        self.at = 0
        self.lookup = {}
        self.dtypes = []
        self.numbers = []
        self.integers = []
        self.texts = []
        self.absent = []
    }

    #: A view over `frame`, downloading each column once.
    #:
    #: One round trip per column, not per row. A column with no validity lane
    #: contributes no missing list at all, which is the difference between one
    #: host allocation per nullable column and one per column.
    pub static fn over(frame: DataFrame) -> Row {
        let row = Row()
        for (at, name) in frame.names.enumerate() { row.lookup[name] = at }
        row.dtypes = frame.dtypes()
        for column in frame.columns {
            row.absent.push(if column.has_validity() { column.null_mask() } else { [] })
            if column.dtype.is_dictionary() {
                row.texts.push(column.to_strs())
                row.numbers.push([])
                row.integers.push([])
                continue
            }
            row.texts.push([])
            if column.dtype.is_wide() {
                row.integers.push(column.to_ints())
                row.numbers.push([])
                continue
            }
            row.integers.push([])
            row.numbers.push(column.to_floats())
        }
        return row
    }

    #: How a named column is stored.
    pub fn dtype(self, name: str) -> Dtype { return self.dtypes[self.slot(name)] }

    #: Whether the cell in a named column is missing.
    pub fn is_na(self, name: str) -> bool {
        let at = self.slot(name)
        let flags = self.absent[at]
        if flags.len() > 0 and flags[self.at] { return true }
        if self.dtypes[at].is_dictionary() { return self.texts[at][self.at] is null }
        if self.dtypes[at].is_wide() { return false }
        return is_nan(self.numbers[at][self.at])
    }

    #: The cell in a named column as a float; NaN when it is missing.
    pub fn number(self, name: str) -> float {
        let at = self.slot(name)
        if self.is_na(name) { return NAN }
        if self.dtypes[at].is_dictionary() {
            throw TypeError(f"{name} is {self.dtypes[at].name()} and has no numeric payload")
        }
        if self.dtypes[at].is_wide() { return float(self.integers[at][self.at]) }
        return self.numbers[at][self.at]
    }

    #: The cell in a named column as an integer.
    pub fn integer(self, name: str) -> int {
        let at = self.slot(name)
        if self.dtypes[at].is_wide() { return self.integers[at][self.at] }
        return int(self.number(name))
    }

    #: The cell in a named column as text, `null` when it is missing.
    pub fn text(self, name: str) -> str? {
        let at = self.slot(name)
        if self.dtypes[at].is_dictionary() { return self.texts[at][self.at] }
        if self.is_na(name) { return null }
        return self.value(name).to_str()
    }

    #: The cell in a named column, boxed.
    pub fn value(self, name: str) -> Value {
        let at = self.slot(name)
        let dtype = self.dtypes[at]
        if self.is_na(name) { return Value.null_of(dtype) }
        if dtype.is_dictionary() { return Value.of_str(self.texts[at][self.at] ?? "", dtype) }
        if dtype.is_wide() { return Value(dtype, integer: self.integers[at][self.at]) }
        if dtype == Dtype.Bool { return Value.of_bool(self.numbers[at][self.at] != 0.0) }
        return Value.of_float(self.numbers[at][self.at])
    }

    fn slot(self, name: str) -> int {
        let at = self.lookup.get(name, -1)
        if at < 0 { throw KeyError(f"no column named {name}") }
        return at
    }
}

#: A `Bool` column holding what `predicate` said about each row.
#:
#: One download per column, then one host call per row. That per-row call is
#: the cost the vectorised path does not pay: a mask built from `arith.jai`
#: comparisons is a dispatch per comparison and nothing per row, so a
#: predicate that could have been written as one belongs there instead. This
#: is here for the ones that could not.
pub fn query_flags(frame: DataFrame, predicate: fn(Row) -> bool) -> Column {
    let row = Row.over(frame)
    var flags: list[bool] = []
    for at in 0..frame.len() {
        row.at = at
        flags.push(predicate(row))
    }
    return Column.from_bools(flags)
}

#: The rows a predicate accepts.
#:
#: The callback runs once per row on the host. `query_mask` is the same
#: selection driven by a `Bool` column and is the one to reach for whenever
#: the predicate can be written as a comparison.
pub fn query(frame: DataFrame, predicate: fn(Row) -> bool) -> DataFrame {
    let mask = query_flags(frame, predicate)
    defer { mask.free() }
    return filter_rows(frame, mask)
}

#: The rows a `Bool` column marks — the vectorised counterpart of `query`.
#:
#: Named so that the fast path is findable from the slow one; it is
#: `filter_rows`, which is what a mask selection has been all along.
pub fn query_mask(frame: DataFrame, mask: Column) -> DataFrame {
    return filter_rows(frame, mask)
}
