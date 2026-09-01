#: The calendar side of a column: the host parse that turns text into
#: timestamps, and the frequency aliases that name a period.
#:
#: What is worth checking about `to_datetime` is not that it reads a date —
#: `std.time` is tested for that — but the three things this module adds. The
#: first is the split: a millisecond count is a floored second plus a positive
#: remainder, so a moment before the epoch has to land below it rather than
#: being truncated towards zero. The second is that a repeated spelling is a
#: dictionary hit and not a second parse, which is only observable by the
#: answers agreeing. The third is that the reader can actually reach it: the
#: `parse_dates` route is this module's only caller, so a round trip through
#: `io.jai` is what proves the two agree about what a `Datetime` spells.
import std.gpu as gpu
from std.test import assert_eq, assert_false, assert_throws, assert_true
from jaiframe.column import Dtype
from jaiframe.io import read_csv_text, to_csv_text
from jaiframe.timeseries import FreqUnit, parse_freq, to_datetime, to_freq

fn test_to_datetime_reads_the_iso_spellings() -> void {
    if not gpu.is_available() { return }
    let stamps = to_datetime(
        [
            "2024-01-01",
            "2024-01-01T06:30:00Z",
            "2024-01-01 06:30:00",
            "2024-01-01T06:30:00+02:00",
        ]
    )
    defer { stamps.free() }
    assert_eq(stamps.dtype, Dtype.Datetime)
    assert_eq(stamps.to_ints(), [1704067200000, 1704090600000, 1704090600000, 1704083400000])
    assert_eq(stamps.null_count(), 0)
}

fn test_a_millisecond_is_a_floored_second_and_a_positive_remainder() -> void {
    if not gpu.is_available() { return }
    let stamps = to_datetime(
        [
            "1970-01-01T00:00:00Z",
            "1970-01-01T00:00:00.25Z",
            "1969-12-31T23:59:59.75Z",
            "1969-12-31T23:59:59Z",
        ]
    )
    defer { stamps.free() }
    assert_eq(stamps.to_ints(), [0, 250, -250, -1000])
}

fn test_a_missing_entry_stays_missing() -> void {
    if not gpu.is_available() { return }
    let stamps = to_datetime(["2024-01-01", null, "2024-01-02"])
    defer { stamps.free() }
    assert_eq(stamps.null_mask(), [false, true, false])
    assert_eq(stamps.null_count(), 1)
    assert_true(stamps.get(1).is_na(), "the null row is missing")
    assert_eq(stamps.get(2).to_int(), 1704153600000)
}

fn test_a_repeated_spelling_parses_once_and_answers_the_same() -> void {
    if not gpu.is_available() { return }
    let stamps = to_datetime(["2024-01-01", "2024-06-01", "2024-01-01", null, "2024-01-01"])
    defer { stamps.free() }
    let read = stamps.to_ints()
    assert_eq(read[0], read[2])
    assert_eq(read[0], read[4])
    assert_false(read[0] == read[1], "two different days are two different stamps")
}

#: A spelling that is not a timestamp is a `ParseError` and a date the calendar
#: does not have is a `ValueError`. Both are `ValueError`, which is what lets
#: `io.jai` promise one refusal type however a `parse_dates` column goes wrong.
fn test_text_that_is_not_a_timestamp_raises() -> void {
    if not gpu.is_available() { return }
    assert_throws(ParseError, || to_datetime(["2024-01-01", "sometime last tuesday"]))
    assert_throws(ParseError, || to_datetime(["2024-01-01T25"]))
    assert_throws(ValueError, || to_datetime(["2024-13-01T00:00:00Z"]))
    assert_throws(ValueError, || to_datetime(["2023-02-29"]))
}

fn test_no_rows_is_an_empty_datetime_column() -> void {
    if not gpu.is_available() { return }
    let stamps = to_datetime([])
    defer { stamps.free() }
    assert_eq(stamps.dtype, Dtype.Datetime)
    assert_eq(stamps.len(), 0)
    assert_true(stamps.is_empty(), "no rows is empty")
}

fn test_the_csv_reader_parses_dates_and_writes_them_back() -> void {
    if not gpu.is_available() { return }
    let frame = read_csv_text(
        "day,high\n2024-01-01,38.0\n,41.0\n2024-03-01T12:00:00Z,44.0\n",
        parse_dates: ["day"]
    )
    defer { frame.free() }
    assert_eq(frame.column("day").dtype, Dtype.Datetime)
    assert_eq(frame.column("day").to_ints(), [1704067200000, 0, 1709294400000])
    assert_eq(frame.column("day").null_mask(), [false, true, false])

    let rendered = to_csv_text(frame, index: false)
    let again = read_csv_text(rendered, parse_dates: ["day"])
    defer { again.free() }
    assert_eq(again.column("day").to_ints(), frame.column("day").to_ints())
    assert_eq(again.column("day").null_mask(), frame.column("day").null_mask())
}

fn test_a_field_that_is_not_a_date_refuses_the_column() -> void {
    if not gpu.is_available() { return }
    assert_throws(ValueError, || read_csv_text("day\nnotadate\n", parse_dates: ["day"]))
}

fn test_a_frequency_alias_names_a_period() -> void {
    let every = parse_freq("5min")
    assert_eq(every.unit, FreqUnit.Minute)
    assert_eq(every.count, 5)
    assert_eq(every.to_str(), "5min")
    assert_eq(parse_freq("ms").unit, FreqUnit.Milli)
    assert_eq(parse_freq("MS").unit, FreqUnit.Month)
    assert_eq(to_freq(every), every)
    assert_eq(to_freq("D").unit, FreqUnit.Day)
    assert_throws(ValueError, || parse_freq("fortnight"))
    assert_throws(TypeError, || to_freq(5))
}
