# jaiframe

Data frames with pandas' API, on the GPU.

```jai
import jaiframe as pd

let frame = pd.DataFrame(
    ["city", "high"],
    [
        pd.Column.from_strs(["Austin", "Dallas", "Austin"]),
        pd.Column.from_floats([38.0, 41.0, 36.5]),
    ]
)
defer { frame.free() }
frame.shape()                    # (3, 2)
frame.column("high").get(1)      # 41.0
frame.dtypes()                   # [str, float64]
```

Names, argument order and defaults follow pandas, in snake case: `group_by`
for `groupby` where the language prefers it, `Agg.Mean` where pandas takes
`"mean"` — and `Agg.from_name("mean")` for when it does.

## What it is

A column is a `Column`: one contiguous float32 device buffer per lane, exactly
as jaicv's `Mat` is, so a whole column is one allocation and an operation over
it is one dispatch. A frame is a list of them under one `Index`, and nothing
copies a column that a reshape could have shared.

A float32 slot is exact only to 2^24, which is not enough for the two dtypes
pandas users rely on being exact. So a column is up to three lanes:

- `values` — the payload for `Float64` and `Bool`, and the dictionary code for
  `Str` and `Category`.
- `wide` — present only for `Int64` and `Datetime`, where a value is
  `wide * 2^24 + values`. Both halves are integral and under 2^24, so the pair
  is exact to 48 bits and Metal recombines it in `long`.
- `valid` — 1.0 where a row holds a value. A `null` lane means no row is
  missing, which is what lets a kernel skip the mask instead of reading a lane
  of ones.

`Datetime` is milliseconds since the Unix epoch — datetime64[ms]. 48 bits of
milliseconds is ±8900 years; 48 bits of nanoseconds, pandas' unit, would be
±3 days. `Float64` is a single float32 lane and is named for source
compatibility, the same compromise jaicv makes with `CV_64F`.

Text is dictionary-encoded: codes in `values`, the strings in `levels`. It is
the only representation that keeps a string column contiguous, and it is what
makes a string comparison an integer compare and a string key hashable by the
same kernel that hashes numbers. Two text columns have to be put on one table
with `unify_levels` before their codes mean the same thing.

`Index` is one class for all three pandas index kinds. No levels is a range
index, whose labels are implied and cost no storage; one level is a plain
index; several are a multi-index. That is what lets a multi-key group-by and a
single-key one run the same code.

`jaiframe` needs a Metal device. `is_available()` says whether there is one.

## Layout

`src/jaiframe/column.jai` is the foundation: the dtypes, the cell type, the
aggregation vocabulary, the four containers, and the primitives — `take`,
`compress`, `cast`, `prefix_sum`, `concat_columns` — that every other module
dispatches through. Nothing outside it reads a lane directly, because which
lanes exist depends on the dtype and on whether the column has ever held a
null.
