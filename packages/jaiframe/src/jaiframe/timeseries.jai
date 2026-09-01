#: Everything that only makes sense because a column holds timestamps.
#:
#: The storage is `column.jai`'s two-lane `Datetime`: milliseconds since the
#: Unix epoch, exact to 48 bits, recombined in `long` and never in `float`.
#: This module is what turns that integer into a calendar — fields, periods,
#: frequencies, bins — and back again.
#:
#: Three ideas carry the whole file.
#:
#: The first is that the calendar is arithmetic, not a lookup. A field
#: accessor is one kernel that divides the timestamp into a day number and a
#: millisecond of the day, runs Howard Hinnant's civil-from-days on the day
#: number, and reads the answer out. That kernel does its work in `int`: 48
#: bits of milliseconds is 3.2 million days, which fits an `int` with room to
#: spare, and a 32-bit division by a literal strength-reduces to a multiply
#: while the 64-bit one is a call. One `long` division per row buys every
#: field; nothing below it is 64-bit.
#:
#: The second is that a frequency is a period numbering. `bucket_of` maps each
#: timestamp to the index of the period it falls in, and that lane of codes is
#: what makes a resample a group-by: the same bins, the same aggregations, the
#: same result shape. Fixed frequencies — ms, s, min, H, D, W — divide by a
#: constant span; calendar ones — M, Q, A — divide a month index, because a
#: month is not a number of milliseconds and pretending otherwise is how a
#: monthly total quietly loses a day every February.
#:
#: The third is that a resample bucket and a time-offset window are the same
#: shape: a row range per output. Both are found by binary search over a
#: monotone lane and both are reduced by one kernel, so `resample("1D").mean()`
#: and `rolling("7D").mean()` cannot drift apart.
#:
#: Parsing and formatting are the exception. A string lives on the host and
#: there is no device path to one, so `to_datetime` and `dt_strftime` are host
#: loops — but they are host loops over the *distinct* strings, not over the
#: rows: a `Str` column is dictionary-encoded, so parsing one costs a parse per
#: level and a kernel over the codes.
#:
#: ```
#: let stamps = to_datetime(["2024-01-01", "2024-02-14", "2024-02-29"])
#: defer { stamps.free() }
#: dt_month(stamps).to_ints()          # [1, 2, 2]
#: dt_is_month_end(stamps).to_bools()  # [false, false, true]
#: parse_freq(5min).to_str()           # 5min
#: ```
from .timeseries.freq import Freq, FreqUnit, parse_freq, to_freq, to_datetime
from .timeseries.freq import MILLIS_PER_DAY, MILLIS_PER_HOUR, MILLIS_PER_MINUTE
from .timeseries.freq import MILLIS_PER_SECOND, MILLIS_PER_WEEK
from .timeseries.kernels import NO_BUCKET

export { MILLIS_PER_SECOND, MILLIS_PER_MINUTE, MILLIS_PER_HOUR }
export { MILLIS_PER_DAY, MILLIS_PER_WEEK, NO_BUCKET }
export { FreqUnit, Freq, parse_freq, to_freq, to_datetime }
