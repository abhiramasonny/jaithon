#: `group_by` and everything a `GroupBy` handle does: aggregate, transform,
#: filter, and the row-picking selectors.
#:
#: One idea carries the whole module. `keys.jai:factorize` numbers every row
#: with the group it belongs to, and that lane of codes is what turns a
#: group-wise reduction into one segmented dispatch: a kernel reads a row, asks
#: the codes lane which output slot it belongs to, and writes there. Nothing in
#: here ever loops over the groups and reduces each one — that is the shape
#: pandas has to take because its groups are Python objects, and it is the
#: shape that would make a hundred thousand groups a hundred thousand
#: dispatches.
#:
#: There are two ways to spend the codes lane, and the handle picks between
#: them per call:
#:
#: - **Scatter** — one thread per row, atomically accumulating into
#:   `out[code]`. Nothing to set up, perfectly coalesced reads, and the only
#:   cost is that every row of a group contends for one slot.
#: - **Segment** — lay the rows out by code once, then give one thread each
#:   `(group, column)` pair and let it walk its own contiguous run. The layout
#:   costs two passes over the rows and a scan, and it is the only path that
#:   serves an aggregation the device has no atomic for: minimum, maximum,
#:   product, and the row-pickers.
#:
#: Which of the two runs is decided in `use_scatter`, and the answer is the
#: scatter unless a layout already exists. There was a rows-per-group crossover
#: here once, reasoned rather than swept; sweeping it found no crossover at
#: all — see the note on `use_scatter` for the numbers. The layout is cached on
#: the handle, so once something has asked for a minimum every later
#: aggregation takes the segmented path for free.
#:
#: What is not here: `factorize` and the hashing under it are `keys.jai`,
#: `resample` is `timeseries.jai`, `pivot`/`unstack` are `reshape.jai`.
#:
#: ```
#: let sales = group_by(frame, "city")
#: defer { sales.free() }
#: let totals = sales.sum()
#: let ranked = sales.transform(Agg.Mean)
#: ```
#:
#: The work is in the sibling directory: `groupby.shader` and `groupby.kernels`
#: hold the Metal source and the plumbing that dispatches it,
#: `groupby.keying` how a grouping is asked for and numbered,
#: `groupby.results` the argument and answer vocabulary, and `groupby.handle`
#: the `GroupBy` class itself.
from .groupby.handle import GroupBy, group_by

export { GroupBy, group_by }
