#: Whole-column reductions, scans, and the moving-window forms of the same.
#:
#: Everything here collapses a column or runs along it. A reduction turns a
#: column into one `Value`; a scan turns it into another column of the same
#: length; a window turns it into a column of per-window answers. Group-wise
#: versions of all three live in `groupby`, and the time-based window offsets
#: in `timeseries`; what those modules share with this one is the `Agg`
#: vocabulary and `reduce_column`, so `df.sum()`, `df.groupby(k).sum()` and
#: `df.rolling(7).sum()` mean the same thing by construction.
#:
#: Two shapes of kernel do most of the work.
#:
#: A whole-column reduction is a two-stage tree, copied from jaicv's
#: `core/stats.jai` for its reason: a fixed grid of 256 groups strides the
#: column accumulating in registers, folds once per group, and a second tiny
#: dispatch folds the 256 partials — so a column of any height costs two
#: dispatches and one download of sixteen floats, rather than one boxed float
#: per row. Downloading the partials and adding them on the host cost jaicv
#: more than the reduction did, and a frame has many columns.
#:
#: A scan is a work-efficient Blelloch sweep over blocks of 256 with the block
#: totals scanned the same way and combined back, so 2^24 rows is three levels
#: of dispatch. `cumsum`, `cumprod`, `cummax` and `cummin` are the same kernel
#: under four operators.
#:
#: ```
#: let daily = pd.Series(pd.Column.from_floats([3.0, 1.0, 4.0, 1.0, 5.0]))
#: defer { daily.free() }
#: mean(daily).to_float()                   # 2.8
#: reduce_column(daily.values, Agg.Median)  # 3.0
#: let smooth = rolling(daily, 3).mean()
#: ```
#:
#: The work is in the sibling directory: `reduce.shader` and `reduce.kernels`
#: hold the Metal source and the grid it runs on, `reduce.stats` the batched
#: summary pass, `reduce.aggregate` the named reductions, `reduce.moments` the
#: shape statistics and the correlations, `reduce.scan` the running totals and
#: lags, and `reduce.window` the moving windows.
from .reduce.stats import ColumnStats, column_stats
from .reduce.aggregate import all_, any_, count, first_valid_index, idxmax, idxmin
from .reduce.aggregate import last_valid_index, maximum, mean, median, minimum, product
from .reduce.aggregate import quantile, reduce_column, std, sum, variance
from .reduce.moments import corr, corrwith, cov, describe, kurtosis, sem, skew
from .reduce.scan import cumcount_na, cumcount_na_column, cummax, cummin, cumprod, cumsum
from .reduce.scan import diff, pct_change, scan_column, shift, shift_column
from .reduce.window import Window, ewm, expanding, rolling

export { ColumnStats, column_stats, reduce_column }
export { sum, mean, minimum, maximum, product, count }
export { std, variance, sem, median, quantile, skew, kurtosis }
export { any_, all_ }
export { idxmin, idxmax, first_valid_index, last_valid_index }
export { scan_column, cumcount_na_column }
export { cumsum, cumprod, cummax, cummin, cumcount_na }
export { shift_column, shift, diff, pct_change }
export { describe, corr, cov, corrwith }
export { Window, rolling, expanding, ewm }
