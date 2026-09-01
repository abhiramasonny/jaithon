#: `jaitoml.datetime` against Python's `datetime`, which is where every
#: expected string came from: `isoformat()` for `to_iso`, `str()` for
#: `to_str`, and the `datetime` constructor's own range checks for what
#: `init` refuses.
from std.test import assert_eq, assert_false, assert_throws, assert_true
from jaitoml.datetime import MAX_YEAR, MIN_YEAR, Date, DateTime, Time

fn test_date_renders_like_python() -> void {
    assert_eq(Date(1979, 5, 27).to_iso(), "1979-05-27")
    assert_eq(Date(1, 1, 1).to_iso(), "0001-01-01")
    assert_eq(Date(9999, 12, 31).to_str(), "9999-12-31")
    assert_eq(str(Date(2000, 2, 29)), "2000-02-29")
    assert_eq(repr(Date(2000, 2, 29)), "Date(2000, 2, 29)")
}

fn test_date_refuses_what_python_refuses() -> void {
    assert_throws(ValueError, fn() -> void { let _ = Date(0, 1, 1) }, "year 0")
    assert_throws(ValueError, fn() -> void { let _ = Date(10000, 1, 1) }, "year 10000")
    assert_throws(ValueError, fn() -> void { let _ = Date(2021, 13, 1) }, "month 13")
    assert_throws(ValueError, fn() -> void { let _ = Date(2021, 2, 29) }, "not a leap year")
    assert_throws(ValueError, fn() -> void { let _ = Date(2021, 4, 31) }, "April has 30")
    assert_eq(MIN_YEAR, 1)
    assert_eq(MAX_YEAR, 9999)
}

fn test_date_compares_chronologically() -> void {
    assert_true(Date(2020, 1, 1) == Date(2020, 1, 1))
    assert_false(Date(2020, 1, 1) == Date(2020, 1, 2))
    assert_true(Date(2019, 12, 31).compare(Date(2020, 1, 1)) < 0)
    assert_true(Date(2020, 2, 1).compare(Date(2020, 1, 31)) > 0)
    assert_eq(Date(2020, 1, 1).compare(Date(2020, 1, 1)), 0)
}

fn test_time_renders_like_python() -> void {
    assert_eq(Time(7, 32, 0).to_iso(), "07:32:00")
    assert_eq(Time(0, 0, 0, 1).to_iso(), "00:00:00.000001")
    assert_eq(Time(23, 59, 59, 999999).to_str(), "23:59:59.999999")
    assert_eq(Time(12).to_iso(), "12:00:00")
    assert_eq(str(Time(7, 32, 0, 500000)), "07:32:00.500000")
    assert_eq(repr(Time(7, 32, 0, 5)), "Time(7, 32, 0, 5)")
}

fn test_time_refuses_out_of_range_fields() -> void {
    assert_throws(ValueError, fn() -> void { let _ = Time(24, 0, 0) }, "hour 24")
    assert_throws(ValueError, fn() -> void { let _ = Time(0, 60, 0) }, "minute 60")
    assert_throws(ValueError, fn() -> void { let _ = Time(0, 0, 60) }, "second 60")
    assert_throws(ValueError, fn() -> void { let _ = Time(0, 0, 0, 1000000) }, "micros")
    assert_throws(ValueError, fn() -> void { let _ = Time(-1, 0, 0) }, "negative hour")
}

fn test_time_compares_within_the_day() -> void {
    assert_true(Time(1, 0, 0) == Time(1, 0, 0, 0))
    assert_false(Time(1, 0, 0) == Time(1, 0, 0, 1))
    assert_true(Time(1, 0, 0).compare(Time(1, 0, 1)) < 0)
    assert_true(Time(1, 0, 0, 2).compare(Time(1, 0, 0, 1)) > 0)
}

fn test_datetime_renders_like_python() -> void {
    let utc = DateTime(1979, 5, 27, 7, 32, 0, 0, 0)
    assert_eq(utc.to_iso(), "1979-05-27T07:32:00+00:00")
    assert_eq(utc.to_str(), "1979-05-27 07:32:00+00:00")
    assert_eq(str(utc), "1979-05-27 07:32:00+00:00")
    let west = DateTime(1979, 5, 27, 0, 32, 0, 999999, -420)
    assert_eq(west.to_iso(), "1979-05-27T00:32:00.999999-07:00")
    let east = DateTime(1979, 5, 27, 0, 32, 0, 0, 330)
    assert_eq(east.to_iso(), "1979-05-27T00:32:00+05:30")
    let local = DateTime(1979, 5, 27, 7, 32)
    assert_eq(local.to_iso(), "1979-05-27T07:32:00")
    assert_eq(local.to_str(), "1979-05-27 07:32:00")
    assert_eq(repr(local), "DateTime(1979, 5, 27, 7, 32, 0, 0)")
    assert_eq(repr(utc), "DateTime(1979, 5, 27, 7, 32, 0, 0, 0)")
}

fn test_datetime_knows_whether_it_has_an_offset() -> void {
    let aware = DateTime(2020, 1, 1, 0, 0, 0, 0, 60)
    let naive = DateTime(2020, 1, 1)
    assert_true(aware.is_aware())
    assert_false(naive.is_aware())
    assert_eq(aware.offset_minutes, 60)
    assert_eq(naive.offset_minutes, null)
    assert_eq(aware.date(), Date(2020, 1, 1))
    assert_eq(DateTime(2020, 1, 1, 8, 9, 10, 11).time(), Time(8, 9, 10, 11))
}

fn test_datetime_refuses_a_day_of_offset() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = DateTime(2020, 1, 1, 0, 0, 0, 0, 24 * 60)
    }, "+24:00")
    assert_throws(ValueError, fn() -> void {
        let _ = DateTime(2020, 1, 1, 0, 0, 0, 0, -24 * 60)
    }, "-24:00")
    assert_throws(ValueError, fn() -> void { let _ = DateTime(2020, 2, 30) }, "Feb 30")
    assert_throws(ValueError, fn() -> void { let _ = DateTime(2020, 1, 1, 25) }, "hour 25")
    let _ = DateTime(2020, 1, 1, 0, 0, 0, 0, 23 * 60 + 59)
}

#: Python compares offset date-times by instant, so 01:00+01:00 is 00:00Z.
fn test_offset_datetimes_compare_by_instant() -> void {
    let zulu = DateTime(2020, 1, 1, 0, 0, 0, 0, 0)
    let plus_one = DateTime(2020, 1, 1, 1, 0, 0, 0, 60)
    let minus_one = DateTime(2019, 12, 31, 23, 0, 0, 0, -60)
    assert_true(zulu == plus_one)
    assert_true(zulu == minus_one)
    assert_eq(zulu.compare(plus_one), 0)
    assert_true(zulu.compare(DateTime(2020, 1, 1, 0, 0, 1, 0, 0)) < 0)
    assert_true(DateTime(2020, 1, 1, 0, 0, 0, 1, 0).compare(zulu) > 0)
    assert_true(
        DateTime(2020, 3, 1, 0, 0, 0, 0, 0).compare(DateTime(2020, 2, 29, 0, 0, 0, 0, 0)) > 0
    )
}

fn test_local_and_offset_datetimes_never_equal() -> void {
    let aware = DateTime(2020, 1, 1, 0, 0, 0, 0, 0)
    let naive = DateTime(2020, 1, 1, 0, 0, 0, 0)
    assert_false(aware == naive)
    assert_false(naive == aware)
    assert_true(naive == DateTime(2020, 1, 1))
    assert_throws(TypeError, fn() -> void { let _ = aware.compare(naive) }, "mixed compare")
    assert_false(naive == Date(2020, 1, 1))
}

fn test_offset_datetime_converts_to_std_time() -> void {
    let moment = DateTime(2024, 2, 29, 12, 0, 0, 500000, 120)
    let converted = moment.to_std()
    assert_eq(converted.to_unix(), 1709200800)
    assert_eq(converted.nanosecond, 500000000)
    assert_eq(converted.offset_minutes, 120)
    assert_eq(DateTime(1970, 1, 1, 0, 0, 0, 0, 0).to_std().to_unix(), 0)
    assert_throws(ValueError, fn() -> void {
        let _ = DateTime(2024, 2, 29).to_std()
    }, "a local date-time has no instant")
}
