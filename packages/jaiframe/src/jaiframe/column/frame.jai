#: The labelled containers a column goes into: `Index`, `Series`, `DataFrame`.
from std.gpu import Buffer
from .lanes import WIDE_LIMIT, cached_kernel, u32
from .store import Column
from .types import Dtype, Value

#: The row labels of a `Series` or a `DataFrame` — all three pandas index kinds
#: in one class.
#:
#: An empty `levels` is a range index, whose labels are `start + i * step` and
#: cost no storage. One level is a plain index. Several levels are a
#: multi-index, where a row's label is the tuple across them.
#:
#: pandas spells those RangeIndex, Index and MultiIndex, and then every
#: operation that touches an index has to ask which one it was handed. One
#: class with a list of levels makes `stack`, `unstack` and a multi-key
#: group-by fall out of the same code that handles a single key, and it is what
#: lets `DataFrame` name its index field before the reshaping module exists.
pub class Index {
    #: One column per level, empty for a range index.
    pub var levels: list[Column]
    #: A name per level, `null` where a level is unnamed. `nlevels()` long.
    pub var names: list[str?]
    #: How many rows are labelled.
    pub var length: int
    #: First label of a range index; unread once `levels` is occupied.
    pub var start: int
    #: Distance between the labels of a range index.
    pub var step: int

    #: The empty range index. `range` and `of` are the constructors.
    pub fn init(self) {
        self.levels = []
        self.names = [null]
        self.length = 0
        self.start = 0
        self.step = 1
    }

    #: A range index — the default labels of a frame nobody has indexed.
    pub static fn range(length: int, start: int = 0, step: int = 1, name: str? = null) -> Index {
        if length < 0 { throw ValueError(f"an index cannot label {length} rows") }
        if step == 0 { throw ValueError("a range index of step 0 repeats one label") }
        let last = start + (if length > 0 { length - 1 } else { 0 }) * step
        if not fits_wide(start) or not fits_wide(last) {
            throw ValueError(
                f"a range index from {start} to {last} leaves the 48 bits an Int64 column stores"
            )
        }
        let index = Index()
        index.levels = []
        index.names = [name]
        index.length = length
        index.start = start
        index.step = step
        return index
    }

    #: An index over stored labels: one level for a plain index, several for a
    #: multi-index. The levels are adopted, not copied.
    pub static fn of(levels: list[Column], names: list[str?]? = null) -> Index {
        if levels.len() == 0 {
            throw ValueError("an index with no levels is a range index; build it with Index.range")
        }
        let length = levels[0].len()
        for level in levels {
            if level.len() != length {
                throw ValueError(f"index levels disagree on length: {level.len()} against {length}")
            }
        }
        var labels: list[str?] = []
        if names is null {
            for _level in 0..levels.len() { labels.push(null) }
        } else {
            for name in names ?? [] { labels.push(name) }
        }
        if labels.len() != levels.len() {
            throw ValueError(f"{labels.len()} names cannot name {levels.len()} levels")
        }
        let index = Index()
        index.levels = [level for level in levels]
        index.names = labels
        index.length = length
        index.start = 0
        index.step = 1
        return index
    }

    #: Whether the labels are implied rather than stored.
    pub fn is_range(self) -> bool { return self.levels.len() == 0 }

    #: How many levels a label spans. A range index has one, as pandas' does.
    pub fn nlevels(self) -> int {
        return if self.is_range() { 1 } else { self.levels.len() }
    }

    #: How many rows are labelled.
    pub fn len(self) -> int { return self.length }

    #: The levels as stored columns, freshly allocated and owned by the caller.
    #:
    #: A range index materialises into the `Int64` ramp it stands for. The
    #: point of not storing that ramp is that most frames never ask for it; the
    #: ones that do — a reshape, a group-by keyed on the index — ask here.
    pub fn materialize(self) -> list[Column] {
        if self.is_range() { return [range_column(self.length, self.start, self.step)] }
        return self.levels.map(|level| level.copy())
    }

    #: The rows `positions` names, as a new index.
    pub fn take(self, positions: Buffer, length: int, allow_missing: bool = false) -> Index {
        if self.is_range() {
            let ramp = range_column(self.length, self.start, self.step)
            let taken = ramp.take(positions, length, allow_missing)
            ramp.free()
            return Index.of([taken], [name for name in self.names])
        }
        return Index.of(
            self.levels.map(|level| level.take(positions, length, allow_missing)),
            [name for name in self.names]
        )
    }

    #: The label of one row, one `Value` per level.
    #:
    #: A list even for a single level, so code written for a multi-index reads
    #: a plain one without a case for it.
    pub fn label_at(self, position: int) -> list[Value] {
        let at = if position < 0 { self.length + position } else { position }
        if at < 0 or at >= self.length {
            throw IndexError(f"row {position} is outside an index of {self.length}")
        }
        if self.is_range() { return [Value.of_int(self.start + at * self.step)] }
        return self.levels.map(|level| level.get(at))
    }

    #: An independent index with the same labels.
    pub fn copy(self) -> Index {
        if self.is_range() {
            return Index.range(self.length, start: self.start, step: self.step, name: self.names[0])
        }
        return Index.of(self.levels.map(|level| level.copy()), [name for name in self.names])
    }

    #: Release the level columns. A range index holds none to release.
    pub fn free(self) -> void {
        for level in self.levels { level.free() }
    }
}

#: One column with its labels: values, an index, and a name.
#:
#: Structure only. Arithmetic, alignment, reductions and display are other
#: modules; what lives here is what a `DataFrame` field has to be able to name.
pub class Series {
    #: The values.
    pub var values: Column
    #: One label per row.
    pub var index: Index
    #: What the series is called, which becomes a column name in a frame.
    pub var name: str?

    #: A series over `values`. A null index gets the default range one.
    pub fn init(self, values: Column, index: Index? = null, name: str? = null) {
        let labels = index ?? Index.range(values.len())
        if labels.len() != values.len() {
            throw ValueError(f"an index of {labels.len()} cannot label {values.len()} rows")
        }
        self.values = values
        self.index = labels
        self.name = name
    }

    #: Number of rows.
    pub fn len(self) -> int { return self.values.len() }

    #: How the values are stored and what operations mean over them.
    pub fn dtype(self) -> Dtype { return self.values.dtype }

    #: An independent series with the same contents.
    pub fn copy(self) -> Series {
        return Series(self.values.copy(), index: self.index.copy(), name: self.name)
    }

    #: Release the values and the index.
    pub fn free(self) -> void {
        self.values.free()
        self.index.free()
    }
}

#: A table: named columns of equal length over one shared index.
#:
#: Only the structure is here — what a frame is made of, and how to reach a
#: column by name. Selecting, arithmetic, missing data, reductions, group-by,
#: joins, reshaping, timeseries and display are separate modules, and every one
#: of them builds its result through `_replace`.
pub class DataFrame {
    #: The columns, in display order.
    pub var columns: list[Column]
    #: A name per column, in the same order.
    pub var names: list[str]
    #: One label per row, shared by every column.
    pub var index: Index
    var lookup: dict[str, int]

    #: A frame over `columns`, named, with `index` labelling the rows.
    #:
    #: Nothing is copied: the frame adopts the columns and the index it is
    #: given, and only the two list containers are fresh.
    #:
    #: Duplicate names are refused. pandas allows them, and then `df["a"]` is a
    #: `Series` sometimes and a frame other times. Every module here addresses
    #: a column by name, so a duplicate would make `column(name)` wrong rather
    #: than expressive.
    pub fn init(self, names: list[str], columns: list[Column], index: Index? = null) {
        if names.len() != columns.len() {
            throw ValueError(f"{names.len()} names cannot name {columns.len()} columns")
        }
        var lookup: dict[str, int] = {}
        for (at, name) in names.enumerate() {
            if lookup.get(name, -1) >= 0 { throw ValueError(f"duplicate column name {name}") }
            lookup[name] = at
        }
        let labels = index ?? Index.range(if columns.len() > 0 { columns[0].len() } else { 0 })
        for column in columns {
            if column.len() != labels.len() {
                throw ValueError(
                    f"a column of {column.len()} rows does not fit an index of {labels.len()}"
                )
            }
        }
        self.names = [name for name in names]
        self.columns = [column for column in columns]
        self.index = labels
        self.lookup = lookup
    }

    #: Rows and columns.
    pub fn shape(self) -> (int, int) { return (self.index.len(), self.columns.len()) }

    #: Number of rows, which is the index's length even with no columns.
    pub fn len(self) -> int { return self.index.len() }

    #: Number of columns.
    pub fn width(self) -> int { return self.columns.len() }

    #: Whether a column of that name exists.
    pub fn has(self, name: str) -> bool { return self.lookup.get(name, -1) >= 0 }

    #: Where a named column sits, raising `KeyError` when there is none.
    #:
    #: Resolved through a table built at construction, so this is a lookup and
    #: not a scan. That table is why renaming a column means building a new
    #: frame through `_replace` rather than writing into `names`.
    pub fn column_index(self, name: str) -> int {
        let at = self.lookup.get(name, -1)
        if at < 0 { throw KeyError(f"no column named {name}") }
        return at
    }

    #: A named column, raising `KeyError` when there is none.
    pub fn column(self, name: str) -> Column { return self.columns[self.column_index(name)] }

    #: How each column is stored, in column order.
    pub fn dtypes(self) -> list[Dtype] { return self.columns.map(|column| column.dtype) }

    #: An independent frame with the same contents.
    pub fn copy(self) -> DataFrame {
        return DataFrame(
            [name for name in self.names],
            self.columns.map(|column| column.copy()),
            index: self.index.copy()
        )
    }

    #: Release every column and the index.
    pub fn free(self) -> void {
        for column in self.columns { column.free() }
        self.index.free()
    }

    #: The same frame with some of its parts swapped out.
    #:
    #: This is how every other module returns a result: build the columns, hand
    #: them over, keep whatever did not change. Nothing is copied, so the result
    #: shares whichever parts came from this frame. Freeing either frame frees
    #: those shared buffers for both — `free` is idempotent, so the danger is a
    #: read after one of them was released, not a double release.
    pub fn _replace(
        self,
        names: list[str]? = null,
        columns: list[Column]? = null,
        index: Index? = null
    ) -> DataFrame {
        return DataFrame(names ?? self.names, columns ?? self.columns, index: index ?? self.index)
    }
}

#: Whether an integer survives the two-lane encoding.
fn fits_wide(value: int) -> bool { return value > -WIDE_LIMIT and value < WIDE_LIMIT }

#: The `Int64` ramp a range index stands for.
fn range_column(length: int, start: int, step: int) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    if length == 0 { return column }
    cached_kernel("col_ramp").dispatch_async(
        [column.values, column.wide ?? column.values],
        length,
        [length, u32(start), u32(step)]
    )
    return column
}
