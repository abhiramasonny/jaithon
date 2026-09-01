#: The `GroupBy` handle and the `group_by` that builds one.
#:
#: One class, because everything it does hangs off the same three cached
#: things — the codes lane, the group sizes and starts, and the stable layout —
#: and the choice between the scatter and the segmented path in `use_scatter`
#: reads all three.
from std.gpu import Buffer
from ..column import Agg, Column, DataFrame, Dtype, Index, Series, Value
from ..column import NA, f32_bits, prefix_sum
from ..keys import NO_GROUP
from .kernels import FINISH_ALL, FINISH_ANY, FINISH_DIV, FINISH_LERP
from .kernels import FINISH_STD, FINISH_VAR
from .kernels import LAYOUT_CHUNKS, LAYOUT_SLOTS, MAX_SLICES, SEGMENT_GROUP
from .kernels import OP_COUNT, OP_DEV2, OP_FIRST, OP_LAST, OP_MAX, OP_MIN
from .kernels import OP_NONZERO, OP_PROD, OP_SUM
from .kernels import PICK_HEAD, PICK_NTH, PICK_TAIL
from .kernels import SELECT_BITS, SELECT_BUCKETS, THREAD_BUDGET
from .kernels import adopt_int_lane, adopt_lane, bounded, copy_lane
from .kernels import flag, free_lane, kernel_for, require_device, u32
from .keying import as_frame, category_product, column_of_values, factorize_keys
from .keying import index_levels, key_specs, label_of, labelled_keys, render_labels
from .results import as_agg, as_bool, stack_frames, supports

#: The handle `group_by` returns: a codes lane, the distinct keys, and the
#: frame the rows came from.
#:
#: Every aggregation is the same three steps — reduce the codes lane into one
#: value per group, wrap the result lane in a `Column`, hang the distinct keys
#: off it as an index. What differs between `sum` and `median` is only which
#: kernel does the middle step.
#:
#: The handle caches what it derives: the group sizes, their starts, and the
#: stable layout of the rows by code. Aggregating twice therefore costs one
#: layout, not two, and that is why the crossover in `use_scatter` asks
#: whether the layout already exists before it asks anything else.
pub class GroupBy {
    #: The rows being grouped, as a frame even when the source was a `Series`.
    pub var frame: DataFrame
    #: One code per row: which group it belongs to, `NO_GROUP` when dropped.
    pub var codes: Buffer
    #: The distinct key values, one column per key level, `ngroups` rows.
    pub var uniques: list[Column]
    #: A name per key level.
    pub var key_names: list[str?]
    #: Whether the keys become the result's index or its leading columns.
    pub var as_index: bool
    var group_total: int
    var key_positions: list[int]
    var source_name: str?
    var sizes_lane: Buffer?
    var starts_lane: Buffer?
    var order_lane: Buffer?
    var rank_lane: Buffer?
    var label_cache: list[str]?
    var freed: bool

    #: Build a handle over an existing factorisation.
    #:
    #: `group_by` is the constructor to reach for. This one is here because a
    #: module that already has a codes lane — a resample, a cut — should not
    #: have to hash its keys a second time to get an aggregator over them.
    pub fn init(
        self,
        frame: DataFrame,
        codes: Buffer,
        ngroups: int,
        uniques: list[Column],
        key_names: list[str?]? = null,
        key_positions: list[int]? = null,
        as_index: bool = true,
        source_name: str? = null
    ) {
        if uniques.len() == 0 {
            throw ValueError("a grouping needs at least one key level")
        }
        for level in uniques {
            if level.len() != ngroups {
                throw ValueError(f"a key level of {level.len()} cannot label {ngroups} groups")
            }
        }
        self.frame = frame
        self.codes = codes
        self.group_total = ngroups
        self.uniques = uniques
        self.key_names = key_names ?? uniques.map(|_level| null)
        self.key_positions = key_positions ?? []
        self.as_index = as_index
        self.source_name = source_name
        self.sizes_lane = null
        self.starts_lane = null
        self.order_lane = null
        self.rank_lane = null
        self.label_cache = null
        self.freed = false
    }

    #: How many groups the keys fell into.
    pub fn ngroups(self) -> int { return self.group_total }

    #: How many rows were grouped.
    pub fn nrows(self) -> int { return self.frame.len() }

    #: The distinct key values, one column per level.
    pub fn key_frame(self) -> DataFrame {
        var names: list[str] = []
        for (at, name) in self.key_names.enumerate() { names.push(name ?? f"key{at}") }
        return DataFrame(names, self.uniques.map(|level| level.copy()))
    }

    #: The row positions of every group, keyed by the group's label.
    #:
    #: A host materialisation: one boxed integer per row, which is what a
    #: `dict` of row lists costs however it is built. Reach for `get_group`
    #: when one group is wanted, and for `agg` when the rows themselves are
    #: not.
    pub fn groups(self) -> dict[str, list[int]] {
        self.check_live()
        let placed = self.layout().download()
        let bounds = self.starts().download()
        let names = self.labels()
        var out: dict[str, list[int]] = {}
        for group in 0..self.group_total {
            var rows: list[int] = []
            for at in int(bounds[group])..int(bounds[group + 1]) { rows.push(int(placed[at])) }
            out[names[group]] = rows
        }
        return out
    }

    #: How many rows each group holds, nulls included.
    pub fn size(self) -> Series {
        self.check_live()
        let lane = Buffer(self.group_total)
        copy_lane(self.sizes(), lane, self.group_total)
        return Series(
            adopt_int_lane(lane, self.group_total, null),
            index: self.key_index(),
            name: "size"
        )
    }

    #: How many rows of each column each group holds a value for.
    pub fn count(self) -> DataFrame { return self.aggregate(Agg.Count) }

    #: Which group each row landed in, `NO_GROUP` for a dropped one.
    pub fn ngroup(self) -> Series {
        self.check_live()
        let lane = Buffer(self.frame.len())
        copy_lane(self.codes, lane, self.frame.len())
        return Series(
            adopt_int_lane(lane, self.frame.len(), null),
            index: self.frame.index.copy(),
            name: "ngroup"
        )
    }

    #: How many rows of the same group came before each row.
    pub fn cumcount(self) -> Series {
        self.check_live()
        let lane = Buffer(self.frame.len())
        copy_lane(self.ranks(), lane, self.frame.len())
        return Series(
            adopt_int_lane(lane, self.frame.len(), null),
            index: self.frame.index.copy(),
            name: "cumcount"
        )
    }

    #: The first `n` rows of every group, in row order.
    pub fn head(self, n: int = 5) -> DataFrame { return self.rank_slice(PICK_HEAD, n) }

    #: The last `n` rows of every group, in row order.
    pub fn tail(self, n: int = 5) -> DataFrame { return self.rank_slice(PICK_TAIL, n) }

    #: Row `n` of every group, counting from the end when `n` is negative.
    pub fn nth(self, n: int) -> DataFrame { return self.rank_slice(PICK_NTH, n) }

    #: The rows of one group, named the way `groups` names it.
    pub fn get_group(self, key: any) -> DataFrame {
        self.check_live()
        let wanted = self.code_of(key)
        let chosen = Buffer(self.group_total)
        if self.group_total > 0 {
            chosen.fill_zero()
            chosen.upload([1.0], wanted)
        }
        let taken = self.take_flagged(chosen)
        chosen.free()
        return taken
    }

    #: Aggregate, in every shape pandas spells it.
    #:
    #: `spec` may be one aggregation, a list of them, or a dict of column name
    #: to one or a list. `named` is pandas' named aggregation: the language has
    #: no `**kwargs`, so `agg(total=("sales", "sum"))` is written
    #: `agg(named: {"total": ("sales", Agg.Sum)})`, and the dict's insertion
    #: order is the column order of the result.
    #:
    #: A single aggregation keeps the source column's name. A list of them
    #: makes two-level columns, which a frame here cannot hold, so they come
    #: out flattened as `sales_sum` — the same string pandas' own
    #: `columns.map("_".join)` produces.
    #:
    #: ```
    #: sales.agg(Agg.Mean)
    #: sales.agg([Agg.Min, Agg.Max])
    #: sales.agg({"units": Agg.Sum, "price": [Agg.Min, Agg.Max]})
    #: sales.agg(named: {"total": ("units", Agg.Sum)})
    #: ```
    pub fn agg(self, spec: any = null, named: dict[str, (str, Agg)]? = null) -> DataFrame {
        self.check_live()
        let requested = named
        if requested is not null { return self.agg_named(requested) }
        if spec is null { throw ValueError("agg needs an aggregation, a list, a dict or `named`") }
        if spec is dict { return self.agg_mapping(spec) }
        if spec is list { return self.agg_list(spec) }
        return self.aggregate(as_agg(spec))
    }

    #: Sum of the present values in each group.
    pub fn sum(self) -> DataFrame { return self.aggregate(Agg.Sum) }

    #: Arithmetic mean of the present values in each group.
    pub fn mean(self) -> DataFrame { return self.aggregate(Agg.Mean) }

    #: Smallest present value in each group.
    pub fn min(self) -> DataFrame { return self.aggregate(Agg.Min) }

    #: Largest present value in each group.
    pub fn max(self) -> DataFrame { return self.aggregate(Agg.Max) }

    #: Middle present value in each group.
    pub fn median(self) -> DataFrame { return self.aggregate(Agg.Median) }

    #: Sample standard deviation of each group.
    pub fn std(self, ddof: int = 1) -> DataFrame {
        return self.aggregate(Agg.Std, ddof: ddof)
    }

    #: Sample variance of each group.
    #:
    #: pandas spells this `var`, which is a keyword here.
    pub fn variance(self, ddof: int = 1) -> DataFrame {
        return self.aggregate(Agg.Var, ddof: ddof)
    }

    #: Product of the present values in each group.
    pub fn prod(self) -> DataFrame { return self.aggregate(Agg.Prod) }

    #: The `q` quantile of each group, interpolated linearly between the two
    #: order statistics it falls between, as pandas' default does.
    pub fn quantile(self, q: float = 0.5) -> DataFrame {
        if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is in [0, 1], got {q}") }
        return self.aggregate(Agg.Quantile, fraction: q)
    }

    #: How many distinct present values each group holds.
    pub fn nunique(self) -> DataFrame { return self.aggregate(Agg.Nunique) }

    #: Whether any present value in the group is truthy.
    pub fn any(self) -> DataFrame { return self.aggregate(Agg.Any) }

    #: Whether every present value in the group is truthy.
    pub fn all(self) -> DataFrame { return self.aggregate(Agg.All) }

    #: First present value of each group, in row order.
    pub fn first(self) -> DataFrame { return self.aggregate(Agg.First) }

    #: Last present value of each group, in row order.
    pub fn last(self) -> DataFrame { return self.aggregate(Agg.Last) }

    #: The index label where each group reaches its largest value.
    pub fn idxmax(self) -> DataFrame { return self.index_of_extreme(OP_MAX) }

    #: The index label where each group reaches its smallest value.
    pub fn idxmin(self) -> DataFrame { return self.index_of_extreme(OP_MIN) }

    #: Every row replaced by its group's result, keeping the frame's shape.
    #:
    #: One `take` on the codes lane does the broadcast: a row's code is
    #: already the position of its group in the result, so reading the group
    #: result back onto the rows is a gather, not a join. A dropped row has
    #: code `NO_GROUP` and comes out null.
    #:
    #: `how` is an aggregation, or a callback handed one group as a `Series`
    #: and returning the `Value` that group broadcasts — the escape hatch,
    #: which walks the groups on the host.
    pub fn transform(self, how: any) -> DataFrame {
        self.check_live()
        let positions = self.codes
        var names: list[str] = []
        var columns: list[Column] = []
        if how is Agg or how is str {
            let agg = as_agg(how)
            for at in self.value_columns(agg) {
                let reduced = self.aggregate_column(self.frame.columns[at], agg, 0.5, 1)
                names.push(self.frame.names[at])
                columns.push(reduced.take(positions, self.frame.len(), allow_missing: true))
                reduced.free()
            }
        } else {
            for at in self.every_value_column() {
                let reduced = self.apply_to_column(at, how)
                names.push(self.frame.names[at])
                columns.push(reduced.take(positions, self.frame.len(), allow_missing: true))
                reduced.free()
            }
        }
        return DataFrame(names, columns, index: self.frame.index.copy())
    }

    #: Every row of the groups `pred` keeps.
    #:
    #: `pred` is a `Bool` column of `ngroups` rows — the per-group answer, from
    #: wherever it came — or a callback handed one group as a frame. The
    #: column form is one kernel and one compaction whatever the group count;
    #: the callback form builds each group on the host, which is what makes it
    #: the slow path and the general one.
    pub fn filter(self, pred: any) -> DataFrame {
        self.check_live()
        if pred is Column {
            if pred.dtype != Dtype.Bool {
                throw TypeError(f"a group filter must be bool, not {pred.dtype.name()}")
            }
            if pred.len() != self.group_total {
                throw ValueError(f"{pred.len()} flags cannot answer for {self.group_total} groups")
            }
            return self.take_flagged(pred.values)
        }
        let bounds = self.starts().download()
        var verdicts: list[bool] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            verdicts.push(as_bool(pred(piece)))
            piece.free()
        }
        let flags = Column.from_bools(verdicts)
        let taken = self.take_flagged(flags.values)
        flags.free()
        return taken
    }

    #: Run `fn_` once per group on the host — the documented escape hatch.
    #:
    #: Every other method here is one dispatch over the codes lane. This one is
    #: a loop over the groups, and each turn of it materialises a frame, so it
    #: costs a `take` per group per column. It is here because a per-group
    #: computation that is not an aggregation has nowhere else to go; reach for
    #: `agg`, `transform` or `filter` first.
    #:
    #: A callback returning a `Value` gives a `Series` over the group keys; one
    #: returning a `DataFrame` gives those frames stacked.
    pub fn apply(self, fn_: any) -> any {
        self.check_live()
        let bounds = self.starts().download()
        var cells: list[Value] = []
        var pieces: list[DataFrame] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            let produced = fn_(piece)
            if produced is DataFrame {
                pieces.push(produced)
                piece.free()
                continue
            }
            if produced is Value { cells.push(produced) } else { cells.push(NA) }
            piece.free()
        }
        if pieces.len() > 0 { return stack_frames(pieces) }
        return Series(column_of_values(cells), index: self.key_index(), name: self.source_name)
    }

    #: Release the codes lane, the distinct keys and everything derived.
    #:
    #: The frame is not released: the handle borrowed it and the caller still
    #: owns it, exactly as `DataFrame._replace` leaves its inputs alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.codes.free()
        for level in self.uniques { level.free() }
        free_lane(self.sizes_lane)
        free_lane(self.starts_lane)
        free_lane(self.order_lane)
        free_lane(self.rank_lane)
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this GroupBy has already been released") }
    }

    fn labels(self) -> list[str] {
        let cached = self.label_cache
        if cached is not null { return cached }
        let rendered = render_labels(self.uniques, self.group_total)
        self.label_cache = rendered
        return rendered
    }

    fn key_index(self) -> Index {
        return Index.of(self.uniques.map(|level| level.copy()), [name for name in self.key_names])
    }

    #: How many rows fell into each group, counted once and kept.
    fn sizes(self) -> Buffer {
        let cached = self.sizes_lane
        if cached is not null { return cached }
        let lane = Buffer(self.group_total)
        if self.group_total > 0 { lane.fill_zero() }
        kernel_for("gb_group_counts").dispatch_async(
            [self.codes, self.codes, lane],
            self.frame.len(),
            [self.frame.len(), 0]
        )
        self.sizes_lane = lane
        return lane
    }

    #: Where each group's run begins once the rows are laid out by code.
    fn starts(self) -> Buffer {
        let cached = self.starts_lane
        if cached is not null { return cached }
        let scanned = prefix_sum(self.sizes(), self.group_total)
        self.starts_lane = scanned[0]
        return scanned[0]
    }

    #: The rows laid out by code, stable in row order within each group.
    #:
    #: Two passes over the rows with a scan between them. A chunk of rows is
    #: one thread's, so pass one counts each chunk's rows per group into a
    #: group-major matrix and pass two walks the chunk again handing rows out
    #: from the scan of it. Laying the matrix out group-major is what makes one
    #: scan do both jobs: the entry for `(group, chunk)` already has every
    #: earlier group's rows ahead of it, so it *is* the destination slot, and
    #: the layout comes out stable because a chunk is contiguous and its thread
    #: walks it in order.
    fn layout(self) -> Buffer {
        let cached = self.order_lane
        if cached is not null { return cached }
        self.build_layout()
        return self.order_lane ?? self.codes
    }

    #: How many rows of the same group came before each row.
    fn ranks(self) -> Buffer {
        let cached = self.rank_lane
        if cached is not null { return cached }
        self.build_layout()
        return self.rank_lane ?? self.codes
    }

    fn build_layout(self) -> void {
        let rows = self.frame.len()
        let groups = max(1, self.group_total)
        let chunks = bounded(LAYOUT_SLOTS // groups, 1, LAYOUT_CHUNKS)
        let span = if rows == 0 { 1 } else { (rows + chunks - 1) // chunks }
        let histogram = Buffer(chunks * groups)
        histogram.fill_zero()
        kernel_for("gb_chunk_counts").dispatch_async(
            [self.codes, histogram],
            chunks,
            [rows, chunks, span]
        )
        let scanned = prefix_sum(histogram, chunks * groups)
        let placed = scanned[1]
        let order = Buffer(max(1, placed))
        let rank = Buffer(max(1, rows))
        kernel_for("gb_chunk_place").dispatch_async(
            [self.codes, scanned[0], self.starts(), order, rank],
            chunks,
            [rows, chunks, span]
        )
        histogram.free()
        scanned[0].free()
        self.order_lane = order
        self.rank_lane = rank
    }

    #: Whether an aggregation should scatter with atomics or walk segments.
    #:
    #: Once something has laid the rows out the question is settled: the layout
    #: is paid for, and walking a run beats fighting over an address. Before
    #: that, the scatter wins -- at every shape measured, not merely at the
    #: short groups the rule here used to allow.
    #:
    #: The rule was `rows < groups * 64`, on the reading that a tall group
    #: contends on its one slot until that costs more than laying the rows out.
    #: Measured at two million rows, that crossover is not there. Building the
    #: layout costs about 13 ms whatever the shape, and a single `sum` pays all
    #: of it, which is exactly what `df.groupby(k).sum()` does:
    #:
    #:     groups        4     32    256   2048   5000  20000
    #:     segment   26.96  32.96  22.48  17.43  18.17  24.06 ms
    #:     scatter    7.98   6.48   5.23   3.24   3.72   4.19 ms
    #:
    #: Even four groups -- half a million rows deep into one address -- come
    #: back 3.4x ahead. Small frames agree: 200k rows over 50 groups went 3.03
    #: to 1.34 ms, and by a thousand rows the two are level and the whole thing
    #: is dispatch-bound anyway.
    #:
    #: The atomic is on a float and float addition is not associative, so this
    #: widens how much of `groupby` has an order the hardware picks. Eight
    #: repeats of a five-thousand-group sum over two million rows came back
    #: bit-identical, so it is stable in practice for a fixed dispatch; it is
    #: not a guarantee, and a caller who needs one should reach for an
    #: aggregation that lays the rows out.
    fn use_scatter(self, op: int) -> bool {
        if op > OP_DEV2 { return false }
        return self.order_lane is null
    }

    #: One value per group, by whichever path suits the aggregation.
    fn reduce_lane(self, column: Column, op: int, centres: Buffer?) -> Buffer {
        let out = Buffer(max(1, self.group_total))
        if self.group_total > 0 {
            out.fill(if op == OP_PROD { 1.0 } else { 0.0 })
        }
        let rows = self.frame.len()
        if rows == 0 or self.group_total == 0 { return out }
        if not self.use_scatter(op) {
            self.segment_into(column, op, centres, out)
            return out
        }
        let quads = not column.dtype.is_wide() and rows % 4 == 0
        let threads = if quads { rows // 4 } else { rows }
        kernel_for("gb_scatter_add").dispatch_async(
            [
                column.values,
                column.wide ?? column.values,
                column.valid ?? column.values,
                self.codes,
                centres ?? column.values,
                out,
            ],
            threads,
            [rows, op, flag(column.dtype.is_wide()), flag(column.valid is not null), flag(quads)]
        )
        return out
    }

    fn segment_into(self, column: Column, op: int, centres: Buffer?, out: Buffer) -> void {
        let groups = self.group_total
        let unit = (groups + SEGMENT_GROUP - 1) // SEGMENT_GROUP * SEGMENT_GROUP
        let slices = bounded(THREAD_BUDGET // max(1, groups), 1, MAX_SLICES)
        let wide = column.wide ?? column.values
        let has_wide = flag(column.dtype.is_wide())
        let partials = Buffer(slices * groups)
        let picks = Buffer(slices * groups)
        kernel_for("gb_segment_reduce").dispatch_async(
            [
                column.values,
                wide,
                column.valid ?? column.values,
                self.layout(),
                self.starts(),
                centres ?? column.values,
                partials,
                picks,
            ],
            slices * unit,
            [groups, unit, slices, op, has_wide, flag(column.valid is not null)],
            group_size: SEGMENT_GROUP
        )
        if op >= OP_MIN {
            kernel_for("gb_pick_fold").dispatch_async(
                [column.values, wide, picks, out],
                groups,
                [groups, slices, op, has_wide]
            )
        } else {
            kernel_for("gb_segment_fold").dispatch_async(
                [partials, out],
                groups,
                [groups, slices, op]
            )
        }
        partials.free()
        picks.free()
    }

    #: The row each group's extreme, first or last present value sits on.
    fn pick_lane(self, column: Column, op: int) -> Buffer {
        let out = Buffer(max(1, self.group_total))
        if self.group_total > 0 { out.fill(float(NO_GROUP)) }
        if self.frame.len() == 0 or self.group_total == 0 { return out }
        self.segment_into(column, op, null, out)
        return out
    }

    fn aggregate(self, agg: Agg, fraction: float = 0.5, ddof: int = 1) -> DataFrame {
        self.check_live()
        if agg == Agg.Size { return self.finish_frame(["size"], [self.sizes_column()]) }
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.value_columns(agg) {
            names.push(self.frame.names[at])
            columns.push(self.aggregate_column(self.frame.columns[at], agg, fraction, ddof))
        }
        return self.finish_frame(names, columns)
    }

    fn aggregate_column(self, column: Column, agg: Agg, fraction: float, ddof: int) -> Column {
        let groups = self.group_total
        return match agg {
            Agg.Size => self.sizes_column(),
            Agg.Count => adopt_int_lane(self.reduce_lane(column, OP_COUNT, null), groups, null),
            Agg.Sum => adopt_lane(self.reduce_lane(column, OP_SUM, null), groups, Dtype.Float64, null),
            Agg.Prod => adopt_lane(self.reduce_lane(column, OP_PROD, null), groups, Dtype.Float64, null),
            Agg.Mean => self.mean_column(column),
            Agg.Std => self.spread_column(column, ddof, true),
            Agg.Var => self.spread_column(column, ddof, false),
            Agg.Any => self.truth_column(column, false),
            Agg.All => self.truth_column(column, true),
            Agg.Median => self.quantile_column(column, 0.5),
            Agg.Quantile => self.quantile_column(column, fraction),
            Agg.Nunique => self.nunique_column(column),
            _ => self.picked_column(column, agg),
        }
    }

    fn sizes_column(self) -> Column {
        let lane = Buffer(max(1, self.group_total))
        copy_lane(self.sizes(), lane, self.group_total)
        return adopt_int_lane(lane, self.group_total, null)
    }

    fn picked_column(self, column: Column, agg: Agg) -> Column {
        let op = match agg {
            Agg.Min => OP_MIN,
            Agg.Max => OP_MAX,
            Agg.First => OP_FIRST,
            _ => OP_LAST,
        }
        let rows = self.pick_lane(column, op)
        let picked = column.take(rows, self.group_total, allow_missing: true)
        rows.free()
        return picked
    }

    #: A column with no missing lane has one present value in every row, so its
    #: count per group is the group's own size -- which the grouping already
    #: knows and caches, and which reads the codes rather than the values.
    #: Counting it again was a second full pass over the column: `mean` cost
    #: twice what `sum` did over the same two million rows, and a frame of
    #: several value columns paid for it once per column.
    fn mean_column(self, column: Column) -> Column {
        let groups = self.group_total
        let total = self.reduce_lane(column, OP_SUM, null)
        let counted = if column.has_validity() {
            self.reduce_lane(column, OP_COUNT, null)
        } else {
            null
        }
        let present = counted ?? self.sizes()
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [total, present, total, out, valid],
            groups,
            [groups, FINISH_DIV, 0]
        )
        total.free()
        #: `sizes()` belongs to the handle and outlives this call.
        if counted is not null { counted.free() }
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    #: Variance from a second pass over the rows, not from `E[x^2] - E[x]^2`.
    #:
    #: The textbook identity subtracts two nearly equal float32 sums and keeps
    #: whatever is left, which for a column whose values sit far from zero is
    #: mostly rounding. Centring first costs one more read of a lane the
    #: previous dispatch just warmed.
    fn spread_column(self, column: Column, ddof: int, want_std: bool) -> Column {
        let groups = self.group_total
        let total = self.reduce_lane(column, OP_SUM, null)
        let present = self.reduce_lane(column, OP_COUNT, null)
        let centres = Buffer(max(1, groups))
        let centre_valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [total, present, total, centres, centre_valid],
            groups,
            [groups, FINISH_DIV, 0]
        )
        let deviation = self.reduce_lane(column, OP_DEV2, centres)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async([deviation, present, deviation, out, valid], groups, [groups, if want_std {
            FINISH_STD
        } else {
            FINISH_VAR
        }, u32(ddof)])
        total.free()
        present.free()
        centres.free()
        centre_valid.free()
        deviation.free()
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    fn truth_column(self, column: Column, want_all: bool) -> Column {
        let groups = self.group_total
        let truthy = self.reduce_lane(column, OP_NONZERO, null)
        let present = self.reduce_lane(column, OP_COUNT, null)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async([truthy, present, truthy, out, valid], groups, [groups, if want_all {
            FINISH_ALL
        } else {
            FINISH_ANY
        }, 0])
        truthy.free()
        present.free()
        valid.free()
        return adopt_lane(out, groups, Dtype.Bool, null)
    }

    #: A quantile per group without sorting a single segment.
    #:
    #: Selecting the k-th smallest by radix is eight rounds of "how many of the
    #: values still in the running have each of sixteen leading digits", and
    #: every round is one pass over the rows keyed by code — the same segmented
    #: shape as every other reduction here. Sorting the segments instead would
    #: cost a sort per group, and a sort of one enormous group is the case that
    #: has no good answer.
    fn quantile_column(self, column: Column, fraction: float) -> Column {
        let groups = self.group_total
        let present = self.reduce_lane(column, OP_COUNT, null)
        let lower = Buffer(max(1, groups))
        let upper = Buffer(max(1, groups))
        let weight = Buffer(max(1, groups))
        kernel_for("gb_quantile_targets").dispatch_async(
            [present, lower, upper, weight],
            groups,
            [groups, f32_bits(fraction)]
        )
        let low = self.select_lane(column, lower)
        let high = self.select_lane(column, upper)
        let out = Buffer(max(1, groups))
        let valid = Buffer(max(1, groups))
        kernel_for("gb_finish").dispatch_async(
            [low, high, weight, out, valid],
            groups,
            [groups, FINISH_LERP, 0]
        )
        present.free()
        lower.free()
        upper.free()
        weight.free()
        low.free()
        high.free()
        return adopt_lane(out, groups, Dtype.Float64, valid)
    }

    fn select_lane(self, column: Column, targets: Buffer) -> Buffer {
        let groups = self.group_total
        let rows = self.frame.len()
        let prefix = Buffer(max(1, groups))
        let want = Buffer(max(1, groups))
        kernel_for("gb_select_init").dispatch_async([targets, prefix, want], groups, [groups])
        let counts = Buffer(max(1, groups * SELECT_BUCKETS))
        var shift = 32 - SELECT_BITS
        while shift >= 0 {
            counts.fill_zero()
            kernel_for("gb_select_count").dispatch_async(
                [
                    column.values,
                    column.wide ?? column.values,
                    column.valid ?? column.values,
                    self.codes,
                    prefix,
                    counts,
                ],
                rows,
                [rows, shift, flag(column.dtype.is_wide()), flag(column.valid is not null)]
            )
            kernel_for("gb_select_step").dispatch_async([counts, prefix, want], groups, [groups])
            shift -= SELECT_BITS
        }
        let out = Buffer(max(1, groups))
        kernel_for("gb_select_value").dispatch_async([prefix, out], groups, [groups])
        prefix.free()
        want.free()
        counts.free()
        return out
    }

    #: Distinct present values per group, from a second factorisation.
    #:
    #: Factorising `(group code, value)` numbers the distinct pairs, and the
    #: count of pairs belonging to a group is exactly its `nunique`. That reuses
    #: the hash table `keys.jai` already has rather than growing a second one
    #: here, and it costs one pass over the rows plus one over the pairs.
    fn nunique_column(self, column: Column) -> Column {
        let groups = self.group_total
        let counts = Buffer(max(1, groups))
        if groups > 0 { counts.fill_zero() }
        if self.frame.len() == 0 or groups == 0 {
            return adopt_int_lane(counts, groups, null)
        }
        let keyed = self.code_column()
        let paired = factorize_keys([keyed, column], false, false)
        let owner = paired[1][0]
        let payload = paired[1][1]
        let lane = owner.as_float_lane()
        kernel_for("gb_group_counts").dispatch_async(
            [lane, payload.valid ?? lane, counts],
            paired[2],
            [paired[2], flag(payload.valid is not null)]
        )
        lane.free()
        paired[0].free()
        for level in paired[1] { level.free() }
        keyed.free()
        return adopt_int_lane(counts, groups, null)
    }

    fn code_column(self) -> Column {
        let lane = Buffer(max(1, self.frame.len()))
        copy_lane(self.codes, lane, self.frame.len())
        return adopt_int_lane(lane, self.frame.len(), null)
    }

    fn index_of_extreme(self, op: int) -> DataFrame {
        self.check_live()
        if self.frame.index.nlevels() != 1 {
            throw ValueError(
                "idxmax over a multi-index would return a tuple, which a column cannot hold"
            )
        }
        let agg = if op == OP_MAX { Agg.Max } else { Agg.Min }
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.value_columns(agg) {
            let rows = self.pick_lane(self.frame.columns[at], op)
            let labelled = self.frame.index.take(rows, self.group_total, allow_missing: true)
            names.push(self.frame.names[at])
            columns.push(labelled.levels[0])
            rows.free()
        }
        return self.finish_frame(names, columns)
    }

    fn agg_list(self, spec: list) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for at in self.every_value_column() {
            for entry in spec {
                let agg = as_agg(entry)
                if not supports(self.frame.columns[at].dtype, agg) { continue }
                names.push(f"{self.frame.names[at]}_{agg.name()}")
                columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
            }
        }
        return self.finish_frame(names, columns)
    }

    fn agg_mapping(self, spec: dict) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for key in spec {
            let at = self.frame.column_index(key)
            let entry = spec[key]
            if entry is list {
                for one in entry {
                    let agg = as_agg(one)
                    self.require_support(key, at, agg)
                    names.push(f"{key}_{agg.name()}")
                    columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
                }
                continue
            }
            let agg = as_agg(entry)
            self.require_support(key, at, agg)
            names.push(key)
            columns.push(self.aggregate_column(self.frame.columns[at], agg, 0.5, 1))
        }
        return self.finish_frame(names, columns)
    }

    fn agg_named(self, named: dict[str, (str, Agg)]) -> DataFrame {
        var names: list[str] = []
        var columns: list[Column] = []
        for label in named {
            let pair = named[label]
            let at = self.frame.column_index(pair[0])
            self.require_support(pair[0], at, pair[1])
            names.push(label)
            columns.push(self.aggregate_column(self.frame.columns[at], pair[1], 0.5, 1))
        }
        return self.finish_frame(names, columns)
    }

    fn require_support(self, name: str, at: int, agg: Agg) -> void {
        let dtype = self.frame.columns[at].dtype
        if supports(dtype, agg) { return }
        throw TypeError(f"{agg.name()} is not defined over the {dtype.name()} column {name}")
    }

    #: The columns an aggregation runs over: everything that is not a key and
    #: that the aggregation is defined for.
    fn value_columns(self, agg: Agg) -> list[int] {
        var out: list[int] = []
        for at in self.every_value_column() {
            if supports(self.frame.columns[at].dtype, agg) { out.push(at) }
        }
        return out
    }

    fn every_value_column(self) -> list[int] {
        var out: list[int] = []
        for at in 0..self.frame.width() {
            if self.key_positions.contains(at) { continue }
            out.push(at)
        }
        return out
    }

    fn finish_frame(self, names: list[str], columns: list[Column]) -> DataFrame {
        if self.as_index { return DataFrame(names, columns, index: self.key_index()) }
        var all_names: list[str] = []
        var all_columns: list[Column] = []
        for (at, name) in self.key_names.enumerate() {
            all_names.push(name ?? f"key{at}")
            all_columns.push(self.uniques[at].copy())
        }
        for (at, name) in names.enumerate() {
            all_names.push(name)
            all_columns.push(columns[at])
        }
        return DataFrame(all_names, all_columns, index: Index.range(self.group_total))
    }

    fn rank_slice(self, mode: int, wanted: int) -> DataFrame {
        self.check_live()
        let rows = self.frame.len()
        let flags = Buffer(max(1, rows))
        kernel_for("gb_rank_mask").dispatch_async(
            [self.codes, self.ranks(), self.sizes(), flags],
            rows,
            [rows, mode, u32(wanted)]
        )
        let taken = self.take_flags(flags, rows)
        flags.free()
        return taken
    }

    fn take_flagged(self, per_group: Buffer) -> DataFrame {
        let rows = self.frame.len()
        let flags = Buffer(max(1, rows))
        kernel_for("gb_group_flags").dispatch_async([self.codes, per_group, flags], rows, [rows])
        let taken = self.take_flags(flags, rows)
        flags.free()
        return taken
    }

    fn take_flags(self, flags: Buffer, count: int) -> DataFrame {
        let scanned = prefix_sum(flags, count)
        let kept = scanned[1]
        let rows = Buffer(max(1, kept))
        kernel_for("gb_compact").dispatch_async([flags, scanned[0], rows], count, [count])
        scanned[0].free()
        let taken = self.take_rows(rows, kept)
        rows.free()
        return taken
    }

    fn take_rows(self, rows: Buffer, count: int) -> DataFrame {
        return DataFrame(
            [name for name in self.frame.names],
            self.frame.columns.map(|column| column.take(rows, count)),
            index: self.frame.index.take(rows, count)
        )
    }

    #: One group's rows as a frame — the per-group materialisation `apply` and
    #: the callback form of `filter` are built on, and the reason both are slow.
    fn group_frame(self, group: int, bounds: list[float]) -> DataFrame {
        let begin = int(bounds[group])
        let stop = int(bounds[group + 1])
        let rows = Buffer(max(1, stop - begin))
        copy_lane(self.layout(), rows, stop - begin, begin, 0)
        let taken = self.take_rows(rows, stop - begin)
        rows.free()
        return taken
    }

    fn apply_to_column(self, at: int, callback: any) -> Column {
        let bounds = self.starts().download()
        var cells: list[Value] = []
        for group in 0..self.group_total {
            let piece = self.group_frame(group, bounds)
            let produced = callback(
                Series(
                    piece.columns[at],
                    index: piece.index,
                    name: self.frame.names[at]
                )
            )
            cells.push(if produced is Value { produced } else { NA })
            piece.free()
        }
        return column_of_values(cells)
    }

    fn code_of(self, key: any) -> int {
        let wanted = label_of(key)
        for (group, name) in self.labels().enumerate() {
            if name == wanted { return group }
        }
        throw KeyError(f"no group labelled {wanted}")
    }
}

#: Group a frame or a series by columns, by index levels, or by a callback.
#:
#: `by` takes a column name, a list of names, a `Column` of the same height, or
#: a callback handed each row's index label. `level` takes an index level by
#: position or by name, or a list of them. Exactly one of the two is given.
#:
#: `sort` and `dropna` mean what they do in pandas and go straight to
#: `factorize`: whether the groups come out in key order, and whether a missing
#: key is a group of its own rather than a dropped row.
#:
#: `observed` is the other pandas flag, and it defaults to `true` here as it
#: does in pandas 2.1 and later: only the key tuples rows actually carry become
#: groups. `observed: false` asks for a group per combination of the keys'
#: level tables, which is a question only a `Category` can answer, so it needs
#: every key to be one.
#:
#: `as_index` says where the keys go: the result's index when true, its leading
#: columns when false.
#:
#: ```
#: group_by(frame, "city")
#: group_by(frame, ["city", "quarter"], sort: false)
#: group_by(frame, level: 0)
#: ```
pub fn group_by(
    source: any,
    by: any = null,
    level: any = null,
    as_index: bool = true,
    sort: bool = true,
    dropna: bool = true,
    observed: bool = true
) -> GroupBy {
    require_device("group a frame")
    let unpacked = as_frame(source)
    let frame = unpacked[0]
    if by is null and level is null {
        throw ValueError("group_by needs `by` or `level`")
    }
    if by is not null and level is not null {
        throw ValueError("group_by takes `by` or `level`, not both")
    }
    var keys: list[Column] = []
    var names: list[str?] = []
    var positions: list[int] = []
    var built: list[Column] = []
    if level is not null {
        let wanted = index_levels(frame.index, level)
        for (at, one) in frame.index.materialize().enumerate() {
            if not wanted.contains(at) {
                one.free()
                continue
            }
            keys.push(one)
            built.push(one)
            names.push(frame.index.names[at])
        }
    } else {
        for one in key_specs(by) {
            if one is str {
                let at = frame.column_index(one)
                keys.push(frame.columns[at])
                names.push(one)
                positions.push(at)
                continue
            }
            if one is Column {
                if one.len() != frame.len() {
                    throw ValueError(f"a key of {one.len()} rows cannot group {frame.len()}")
                }
                keys.push(one)
                names.push(null)
                continue
            }
            let derived = labelled_keys(frame.index, one)
            keys.push(derived)
            built.push(derived)
            names.push(null)
        }
    }
    if keys.len() == 0 { throw ValueError("group_by needs at least one key") }
    let grouped = if observed {
        factorize_keys(keys, sort, dropna)
    } else {
        category_product(keys, frame.len(), dropna)
    }
    for one in built { one.free() }
    return GroupBy(
        frame,
        grouped[0],
        grouped[2],
        grouped[1],
        key_names: names,
        key_positions: positions,
        as_index: as_index,
        source_name: unpacked[1]
    )
}
