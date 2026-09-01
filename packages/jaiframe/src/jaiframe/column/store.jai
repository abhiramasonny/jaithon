#: The `Column` class: the lanes a dtype needs, and the accessors over them.
from std.gpu import Buffer
from std.math import NAN, is_nan
from .lanes import COUNT_GROUP, COUNT_GROUPS, MAX_ROWS
from .lanes import cached_kernel, copy_lane, flag, prefix_sum, require_device
from .lanes import u32, wide_pack, wide_unpack
from .types import Dtype, Value

#: One column of a frame: a dtype, a length, and the lanes that dtype needs.
#:
#: Lane ownership is one column, one set of buffers. Nothing here hands out a
#: view onto another column's allocation, because a frame is reshaped far more
#: often than an image is cropped and a shared lane would make every `free` a
#: question. The cost is that `copy` really copies; the benefit is that every
#: module can `defer { column.free() }` without asking who else is looking.
pub class Column {
    #: How the lanes are to be read.
    pub let dtype: Dtype
    #: Number of rows.
    pub let length: int
    #: The payload lane, or the dictionary code lane.
    pub var values: Buffer
    #: The high half of a 48-bit integer, present only for a wide dtype.
    pub var wide: Buffer?
    #: 1.0 where a row holds a value. `null` means no row is missing, which
    #: lets a kernel skip the mask entirely.
    pub var valid: Buffer?
    #: The dictionary for `Str` and `Category`, `null` for every other dtype.
    pub var levels: list[str]?
    var intern: dict[str, int]?
    var freed: bool

    #: Allocate the lanes `dtype` needs, with the contents left undefined.
    #:
    #: The caller writes every slot. This is the constructor a kernel-producing
    #: operation uses; the `from_*` ones are for host data.
    pub static fn allocate(dtype: Dtype, length: int, with_validity: bool = false) -> Column {
        if length < 0 { throw ValueError(f"a column cannot have {length} rows") }
        if length > MAX_ROWS {
            throw ValueError(
                f"{length} rows exceeds the {MAX_ROWS} a float32 position lane addresses"
            )
        }
        require_device("allocate a Column")
        let column = Column()
        column.dtype = dtype
        column.length = length
        column.values = Buffer(length)
        column.wide = if dtype.is_wide() { Buffer(length) } else { null }
        column.valid = if with_validity { Buffer(length) } else { null }
        column.levels = if dtype.is_dictionary() { [] } else { null }
        column.intern = if dtype.is_dictionary() { {} } else { null }
        column.freed = false
        return column
    }

    #: An empty shell. `allocate` and the `from_*` constructors fill it in.
    pub fn init(self) {
        self.dtype = Dtype.Float64
        self.length = 0
        self.values = Buffer(0)
        self.wide = null
        self.valid = null
        self.levels = null
        self.intern = null
        self.freed = false
    }

    #: A `Float64` column from host floats. NaN counts as missing unless
    #: `nulls` says otherwise, which is what a CSV reader wants.
    pub static fn from_floats(values: list[float], nulls: list[bool]? = null) -> Column {
        let mask = validity_of(values.len(), nulls, values.map(|v| is_nan(v)))
        let column = Column.allocate(Dtype.Float64, values.len(), with_validity: mask is not null)
        if values.len() > 0 { column.values.upload(values) }
        column.upload_validity(mask)
        return column
    }

    #: An `Int64` column, or a `Datetime` one when `dtype` says so.
    pub static fn from_ints(
        values: list[int],
        nulls: list[bool]? = null,
        dtype: Dtype = Dtype.Int64
    ) -> Column {
        if not dtype.is_wide() { throw ValueError(f"{dtype.name()} is not stored as two lanes") }
        let mask = validity_of(values.len(), nulls, null)
        let column = Column.allocate(dtype, values.len(), with_validity: mask is not null)
        var low: list[float] = []
        var high: list[float] = []
        for value in values {
            let split = wide_pack(value)
            low.push(split[0])
            high.push(split[1])
        }
        if values.len() > 0 {
            column.values.upload(low)
            (column.wide ?? column.values).upload(high)
        }
        column.upload_validity(mask)
        return column
    }

    #: A `Bool` column from host booleans.
    pub static fn from_bools(values: list[bool], nulls: list[bool]? = null) -> Column {
        let mask = validity_of(values.len(), nulls, null)
        let column = Column.allocate(Dtype.Bool, values.len(), with_validity: mask is not null)
        if values.len() > 0 {
            column.values.upload(values.map(|v| if v { 1.0 } else { 0.0 }))
        }
        column.upload_validity(mask)
        return column
    }

    #: A `Str` column, or a `Category` one, dictionary-encoding as it goes.
    #:
    #: A null entry is a missing row, not the empty string. Levels come out in
    #: first-appearance order, the way `factorize` numbers them; a module that
    #: wants them sorted reorders them itself.
    pub static fn from_strs(values: list[str?], dtype: Dtype = Dtype.Str) -> Column {
        if not dtype.is_dictionary() { throw ValueError(f"{dtype.name()} does not hold text") }
        var levels: list[str] = []
        var intern: dict[str, int] = {}
        var codes: list[float] = []
        var present: list[bool] = []
        var any_missing = false
        for entry in values {
            if entry is null {
                codes.push(0.0)
                present.push(false)
                any_missing = true
                continue
            }
            var code = intern.get(entry, -1)
            if code < 0 {
                code = levels.len()
                levels.push(entry)
                intern[entry] = code
            }
            codes.push(float(code))
            present.push(true)
        }
        let column = Column.allocate(dtype, values.len(), with_validity: any_missing)
        if values.len() > 0 { column.values.upload(codes) }
        column.upload_validity(if any_missing { present } else { null })
        column.levels = levels
        column.intern = intern
        return column
    }

    #: `length` copies of one cell.
    pub static fn full(dtype: Dtype, length: int, value: Value) -> Column {
        if value.is_na() { return Column.nulls(dtype, length) }
        let column = Column.allocate(dtype, length)
        if length == 0 { return column }
        if dtype.is_dictionary() {
            column.levels = [value.to_str()]
            column.intern = {value.to_str(): 0}
            column.values.fill(0.0)
            return column
        }
        if dtype.is_wide() {
            let split = wide_pack(value.to_int())
            column.values.fill(split[0])
            (column.wide ?? column.values).fill(split[1])
            return column
        }
        column.values.fill(value.to_float())
        return column
    }

    #: `length` missing rows.
    pub static fn nulls(dtype: Dtype, length: int) -> Column {
        let column = Column.allocate(dtype, length, with_validity: true)
        if length == 0 { return column }
        column.values.fill_zero()
        let high = column.wide
        if high is not null { high.fill_zero() }
        let mask = column.valid
        if mask is not null { mask.fill_zero() }
        return column
    }

    #: Number of rows, so a column reads like the other containers.
    pub fn len(self) -> int { return self.length }

    #: Whether the column holds no rows.
    pub fn is_empty(self) -> bool { return self.length == 0 }

    #: Number of entries in the dictionary, 0 when there is none.
    pub fn level_count(self) -> int { return (self.levels ?? []).len() }

    #: The string a dictionary code stands for.
    pub fn level_at(self, code: int) -> str {
        let table = self.levels
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        if code < 0 or code >= table.len() {
            throw IndexError(f"code {code} is outside the {table.len()} levels")
        }
        return table[code]
    }

    #: The code `text` already has, or `null` when the dictionary lacks it.
    #:
    #: This is what makes `column == "widget"` one integer compare: the scalar
    #: is resolved to a code once, on the host, and a missing one short-circuits
    #: the whole comparison to false without a dispatch.
    pub fn level_code(self, text: str) -> int? {
        let table = self.intern
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        let code = table.get(text, -1)
        return if code < 0 { null } else { code }
    }

    #: The code for `text`, appending a level when the dictionary lacks it.
    pub fn intern_level(self, text: str) -> int {
        var table = self.intern
        var names = self.levels
        if table is null or names is null {
            throw TypeError(f"{self.dtype.name()} has no dictionary")
        }
        let existing = table.get(text, -1)
        if existing >= 0 { return existing }
        let code = names.len()
        names.push(text)
        table[text] = code
        return code
    }

    #: Replace the dictionary wholesale, for a reader that decoded the codes
    #: and the levels separately.
    pub fn set_levels(self, levels: list[str]) -> void {
        if not self.dtype.is_dictionary() {
            throw TypeError(f"{self.dtype.name()} has no dictionary")
        }
        var table: dict[str, int] = {}
        for (code, name) in levels.enumerate() { table[name] = code }
        self.levels = levels
        self.intern = table
    }

    #: Whether the column carries a validity lane at all.
    pub fn has_validity(self) -> bool { return self.valid is not null }

    #: The validity lane, creating a lane of ones when there was none.
    #:
    #: Call it when a kernel is about to write nulls into a column that has not
    #: had any. Reading validity does not need it — a `null` lane already means
    #: every row is present.
    pub fn materialize_validity(self) -> Buffer {
        self.check_live()
        let existing = self.valid
        if existing is not null { return existing }
        let mask = Buffer(self.length)
        if self.length > 0 { mask.fill(1.0) }
        self.valid = mask
        return mask
    }

    #: The same values under a different validity lane, which the result adopts.
    #:
    #: The value lanes are copied rather than shared. One column owning one set
    #: of buffers is the rule that makes `free` decidable, and a mask update is
    #: rare enough next to the arithmetic that produced it to pay for a copy.
    pub fn with_validity(self, valid: Buffer?) -> Column {
        let out = self.copy_lanes(with_validity: false)
        out.valid = valid
        return out
    }

    #: An independent column with the same contents.
    pub fn copy(self) -> Column { return self.copy_lanes(with_validity: true) }

    #: Release the lanes. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.values.free()
        let high = self.wide
        if high is not null { high.free() }
        let mask = self.valid
        if mask is not null { mask.free() }
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    #: Every row as a host float, missing rows as NaN.
    #:
    #: One boxed float per row: this is the boundary where a column stops being
    #: a device allocation, and it costs what materialising it costs.
    pub fn to_floats(self) -> list[float] {
        self.check_live()
        if self.length == 0 { return [] }
        var out = if self.dtype.is_wide() {
            let lane = self.as_float_lane()
            let downloaded = lane.download()
            lane.free()
            downloaded
        } else {
            self.values.download()
        }
        let mask = self.valid
        if mask is null { return out }
        for (row, present) in mask.download().enumerate() {
            if present == 0.0 { out[row] = NAN }
        }
        return out
    }

    #: Every row as a host integer. A missing row reads as 0.
    pub fn to_ints(self) -> list[int] {
        self.check_live()
        if self.length == 0 { return [] }
        if not self.dtype.is_wide() {
            return self.to_floats().map(|v| if is_nan(v) { 0 } else { int(v) })
        }
        let low = self.values.download()
        let high = (self.wide ?? self.values).download()
        var out: list[int] = []
        for (row, slot) in low.enumerate() { out.push(wide_unpack(slot, high[row])) }
        return out
    }

    #: Every row as a host boolean. A missing row reads as false.
    pub fn to_bools(self) -> list[bool] {
        return self.to_floats().map(|v| not is_nan(v) and v != 0.0)
    }

    #: Every row as host text, missing rows as `null`.
    pub fn to_strs(self) -> list[str?] {
        self.check_live()
        let table = self.levels
        if table is null { throw TypeError(f"{self.dtype.name()} has no dictionary") }
        if self.length == 0 { return [] }
        let codes = self.values.download()
        let mask = if self.valid is null { [] } else { (self.valid ?? self.values).download() }
        var out: list[str?] = []
        for (row, code) in codes.enumerate() {
            if mask.len() > 0 and mask[row] == 0.0 {
                out.push(null)
                continue
            }
            let at = int(code)
            out.push(if at >= 0 and at < table.len() { table[at] } else { null })
        }
        return out
    }

    #: True where a row is missing.
    pub fn null_mask(self) -> list[bool] {
        self.check_live()
        let mask = self.valid
        if mask is null { return [false for _row in 0..self.length] }
        return mask.download().map(|v| v == 0.0)
    }

    #: How many rows are missing.
    #:
    #: Reduced on the device into a few hundred partials and finished on the
    #: host, so a column of any size costs one dispatch and one short download
    #: instead of a boxed bool per row.
    pub fn null_count(self) -> int {
        self.check_live()
        let mask = self.valid
        if mask is null or self.length == 0 { return 0 }
        let partials = Buffer(COUNT_GROUPS)
        cached_kernel("col_count_nulls").dispatch_async(
            [mask, partials],
            COUNT_GROUPS * COUNT_GROUP,
            [self.length],
            group_size: COUNT_GROUP
        )
        var total = 0.0
        for part in partials.download() { total += part }
        partials.free()
        return int(total)
    }

    #: One cell.
    #:
    #: A device round trip per call. Reading a whole column goes through
    #: `to_floats` and friends, which pay one round trip for all of it.
    pub fn get(self, position: int) -> Value {
        self.check_live()
        let at = self.resolve(position)
        let mask = self.valid
        if mask is not null and mask.download(at, 1)[0] == 0.0 {
            return Value.null_of(self.dtype)
        }
        let slot = self.values.download(at, 1)[0]
        if self.dtype.is_wide() {
            let high = (self.wide ?? self.values).download(at, 1)[0]
            return Value(self.dtype, integer: wide_unpack(slot, high))
        }
        if self.dtype.is_dictionary() { return Value.of_str(self.level_at(int(slot)), self.dtype) }
        if self.dtype == Dtype.Bool { return Value.of_bool(slot != 0.0) }
        return Value.of_float(slot)
    }

    #: Write one cell, interning the text when the column is a dictionary one.
    pub fn set(self, position: int, value: Value) -> void {
        self.check_live()
        let at = self.resolve(position)
        if value.is_na() {
            self.materialize_validity().upload([0.0], at)
            return
        }
        let mask = self.valid
        if mask is not null { mask.upload([1.0], at) }
        if self.dtype.is_dictionary() {
            self.values.upload([float(self.intern_level(value.to_str()))], at)
            return
        }
        if self.dtype.is_wide() {
            let split = wide_pack(value.to_int())
            self.values.upload([split[0]], at)
            (self.wide ?? self.values).upload([split[1]], at)
            return
        }
        self.values.upload([value.to_float()], at)
    }

    #: The rows `positions` names, in that order.
    #:
    #: This is the one reordering primitive: sort, filter by index, join, and
    #: every reindex are a positions lane plus this call. `allow_missing` turns
    #: a `MISSING_POSITION` entry into a null instead of an error, which is what
    #: an outer join needs.
    pub fn take(self, positions: Buffer, length: int, allow_missing: bool = false) -> Column {
        self.check_live()
        let out = Column.allocate(
            self.dtype,
            length,
            with_validity: allow_missing or self.valid is not null
        )
        out.adopt_dictionary(self)
        if length == 0 { return out }
        cached_kernel("col_gather").dispatch_async(
            [
                self.values,
                self.wide ?? self.values,
                self.valid ?? self.values,
                positions,
                out.values,
                out.wide ?? out.values,
                out.valid ?? out.values,
            ],
            length,
            [
                length,
                self.length,
                flag(self.dtype.is_wide()),
                flag(self.valid is not null),
                flag(out.valid is not null),
                flag(allow_missing),
            ]
        )
        return out
    }

    #: The rows where `mask` is true, in order.
    #:
    #: A null in the mask drops its row, which is what `dropna` and a filter
    #: over a column that has nulls both want.
    pub fn compress(self, mask: Column) -> Column {
        self.check_live()
        if mask.dtype != Dtype.Bool {
            throw TypeError(f"a filter mask must be bool, not {mask.dtype.name()}")
        }
        if mask.length != self.length {
            throw ValueError(f"a mask of {mask.length} cannot filter {self.length} rows")
        }
        if self.length == 0 { return self.copy() }
        let flags = Buffer(self.length)
        cached_kernel("col_mask_flags").dispatch_async(
            [mask.values, mask.valid ?? mask.values, flags],
            self.length,
            [self.length, flag(mask.valid is not null)]
        )
        let scanned = prefix_sum(flags, self.length)
        let offsets = scanned[0]
        let kept = scanned[1]
        let out = Column.allocate(self.dtype, kept, with_validity: self.valid is not null)
        out.adopt_dictionary(self)
        if kept > 0 {
            cached_kernel("col_scatter").dispatch_async(
                [
                    self.values,
                    self.wide ?? self.values,
                    self.valid ?? self.values,
                    flags,
                    offsets,
                    out.values,
                    out.wide ?? out.values,
                    out.valid ?? out.values,
                ],
                self.length,
                [
                    self.length,
                    flag(self.dtype.is_wide()),
                    flag(self.valid is not null),
                    flag(out.valid is not null),
                ]
            )
        }
        flags.free()
        offsets.free()
        return out
    }

    #: The same rows read as `target`.
    #:
    #: `Str` and `Category` differ only in intent, so that pair is a relabel.
    #: `Int64` and `Datetime` share the two-lane layout, so that pair is a lane
    #: copy. Parsing text into numbers is the reader's job and formatting
    #: numbers into text is the display module's, so neither crosses here.
    pub fn cast(self, target: Dtype) -> Column {
        self.check_live()
        if target == self.dtype { return self.copy() }
        if self.dtype.is_dictionary() or target.is_dictionary() {
            if self.dtype.is_dictionary() and target.is_dictionary() {
                let relabelled = self.copy_lanes(with_validity: true, dtype: target)
                return relabelled
            }
            throw TypeError(
                f"casting between {self.dtype.name()} and {target.name()} needs a parser or a formatter"
            )
        }
        let out = Column.allocate(target, self.length, with_validity: self.valid is not null)
        let mask = self.valid
        if mask is not null and self.length > 0 {
            cached_kernel("col_copy_lane").dispatch_async(
                [mask, out.valid ?? out.values],
                self.length,
                [self.length, 0, 0]
            )
        }
        if self.length > 0 {
            cached_kernel("col_cast").dispatch_async(
                [self.values, self.wide ?? self.values, out.values, out.wide ?? out.values],
                self.length,
                [
                    self.length,
                    flag(self.dtype.is_wide()),
                    flag(target.is_wide()),
                    flag(target == Dtype.Bool),
                ]
            )
        }
        return out
    }

    #: One float32 lane holding every row's numeric value, freshly allocated.
    #:
    #: Wide columns are flattened here, and that is lossy past 2^24 — which is
    #: the honest precision of a mean or a standard deviation over int64
    #: anyway. An exact reduction reads both lanes instead. Dictionary columns
    #: give their codes.
    pub fn as_float_lane(self) -> Buffer {
        self.check_live()
        let lane = Buffer(self.length)
        if self.length == 0 { return lane }
        if self.dtype.is_wide() {
            cached_kernel("col_wide_to_float").dispatch_async(
                [self.values, self.wide ?? self.values, lane],
                self.length,
                [self.length]
            )
            return lane
        }
        cached_kernel("col_copy_lane").dispatch_async(
            [self.values, lane],
            self.length,
            [self.length, 0, 0]
        )
        return lane
    }

    #: The float lane and whether the caller owns it.
    #:
    #: A non-wide column already is its float lane, so a reduction over one
    #: should read `values` rather than copy it. This is the call that lets a
    #: hot path skip the copy without knowing which case it is in: free the
    #: buffer only when the flag says it is yours.
    pub fn borrow_float_lane(self) -> (Buffer, bool) {
        if self.dtype.is_wide() { return (self.as_float_lane(), true) }
        self.check_live()
        return (self.values, false)
    }

    #: Mix this column's per-row hash into `out`, one uint per row.
    #:
    #: `out` is device-side uint bits and must start zeroed; each call folds a
    #: further key column in, order-sensitively, so a multi-key group-by is one
    #: call per key over the same lane. Never download `out` as floats.
    #:
    #: Dictionary columns hash their codes, so two columns only hash alike
    #: after `unify_levels` has put them on one table.
    pub fn hash_into(self, out: Buffer, seed: int = 0) -> void {
        self.check_live()
        if self.length == 0 { return }
        cached_kernel("col_hash").dispatch_async(
            [self.values, self.wide ?? self.values, self.valid ?? self.values, out],
            self.length,
            [self.length, flag(self.dtype.is_wide()), flag(self.valid is not null), u32(seed)]
        )
    }

    #: Narrow `out` to the pairs whose rows are equal.
    #:
    #: `left[i]` indexes this column and `right[i]` indexes `other`. `out` must
    #: start at 1.0 and is only ever cleared, so chaining one call per key
    #: column leaves the pairs that agree on all of them — which is how a hash
    #: join confirms a bucket collision is a real match.
    #:
    #: Nulls compare equal to each other by default, matching what pandas'
    #: `merge` does with NaN keys; `dropna` group-by passes `false`.
    pub fn equal_rows(
        self,
        left: Buffer,
        other: Column,
        right: Buffer,
        length: int,
        out: Buffer,
        null_equal: bool = true
    ) -> void {
        self.check_live()
        other.check_live()
        if self.dtype != other.dtype {
            throw TypeError(f"cannot compare {self.dtype.name()} rows against {other.dtype.name()}")
        }
        if length == 0 { return }
        cached_kernel("col_equal_rows").dispatch_async(
            [
                self.values,
                self.wide ?? self.values,
                self.valid ?? self.values,
                other.values,
                other.wide ?? other.values,
                other.valid ?? other.values,
                left,
                right,
                out,
            ],
            length,
            [
                length,
                flag(self.dtype.is_wide()),
                flag(self.valid is not null),
                flag(other.valid is not null),
                flag(null_equal),
            ]
        )
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this Column has already been released") }
    }

    fn resolve(self, position: int) -> int {
        let at = if position < 0 { self.length + position } else { position }
        if at < 0 or at >= self.length {
            throw IndexError(f"row {position} is outside a column of {self.length}")
        }
        return at
    }

    fn upload_validity(self, present: list[bool]?) -> void {
        if present is null { return }
        let mask = self.valid
        if mask is null or self.length == 0 { return }
        mask.upload(present.map(|p| if p { 1.0 } else { 0.0 }))
    }

    fn adopt_dictionary(self, source: Column) -> void {
        if not self.dtype.is_dictionary() { return }
        self.levels = [name for name in source.levels ?? []]
        var table: dict[str, int] = {}
        for (code, name) in (self.levels ?? []).enumerate() { table[name] = code }
        self.intern = table
    }

    fn copy_lanes(self, with_validity: bool, dtype: Dtype? = null) -> Column {
        self.check_live()
        let out = Column.allocate(
            dtype ?? self.dtype,
            self.length,
            with_validity: with_validity and self.valid is not null
        )
        out.adopt_dictionary(self)
        if self.length == 0 { return out }
        copy_lane(self.values, out.values, self.length)
        let high = self.wide
        let out_high = out.wide
        if high is not null and out_high is not null { copy_lane(high, out_high, self.length) }
        let mask = self.valid
        let out_mask = out.valid
        if mask is not null and out_mask is not null { copy_lane(mask, out_mask, self.length) }
        return out
    }
}

fn validity_of(length: int, nulls: list[bool]?, derived: list[bool]?) -> list[bool]? {
    if nulls is not null {
        if nulls.len() != length {
            throw ValueError(f"{nulls.len()} null flags cannot cover {length} rows")
        }
        return nulls.map(|missing| not missing)
    }
    if derived is null { return null }
    var any = false
    for missing in derived {
        if missing {
            any = true
            break
        }
    }
    if not any { return null }
    return derived.map(|missing| not missing)
}
