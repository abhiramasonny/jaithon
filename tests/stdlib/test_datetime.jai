#: `std.datetime` — the calendar is only worth anything if it agrees with the
#: one everyone else uses, so most of what is checked here is a value CPython
#: 3.13 prints for the same call: an ordinal, a `repr`, a `strftime` string.
#: The cases below are the named edges and the whole public surface; the
#: population is covered by the differential refuter, which generates random
#: ordinals, spans and moments across the entire MINYEAR..MAXYEAR range and
#: diffs both sides exactly:
#:
#:     uv run --python 3.13 python3 tests/stdlib/py/datetime_diff.py
import std.datetime as dt

from std.test import assert_eq, assert_false, assert_near, assert_throws, assert_true

fn test_year_bounds_are_cpythons() -> void {
    assert_eq(dt.MINYEAR, 1)
    assert_eq(dt.MAXYEAR, 9999)
    assert_eq(repr(dt.UTC), "datetime.timezone.utc")
}

fn test_timedelta_normalises_into_days_seconds_microseconds() -> void {
    let span = dt.timedelta(days: 1, seconds: 2, microseconds: 3)
    assert_eq(span.days, 1)
    assert_eq(span.seconds, 2)
    assert_eq(span.microseconds, 3)
    assert_eq(dt.timedelta(milliseconds: 1500).seconds, 1)
    assert_eq(dt.timedelta(milliseconds: 1500).microseconds, 500_000)
    assert_eq(dt.timedelta(minutes: 90).seconds, 5_400)
    assert_eq(dt.timedelta(hours: 25).days, 1)
    assert_eq(dt.timedelta(hours: 25).seconds, 3_600)
    assert_eq(dt.timedelta(weeks: 2).days, 14)
    assert_eq(dt.timedelta(days: 1, hours: -25), dt.timedelta(hours: -1))
    assert_eq(dt.timedelta(seconds: -1).days, -1)
    assert_eq(dt.timedelta(seconds: -1).seconds, 86_399)
    assert_eq(dt.timedelta().days, 0)
}

fn test_timedelta_rejects_a_span_past_the_bounds() -> void {
    assert_throws(OverflowError, || dt.timedelta(days: 1_000_000_000))
    assert_throws(OverflowError, || dt.timedelta(days: -1_000_000_000))
    assert_throws(OverflowError, || dt.timedelta.max + dt.timedelta(microseconds: 1))
    assert_throws(OverflowError, || dt.timedelta.max.negate())
    assert_eq(dt.timedelta.min.days, -999_999_999)
    assert_eq(dt.timedelta.max.days, 999_999_999)
    assert_eq(dt.timedelta.max.seconds, 86_399)
    assert_eq(dt.timedelta.max.microseconds, 999_999)
    assert_eq(dt.timedelta.resolution.microseconds, 1)
}

fn test_timedelta_total_seconds_matches_the_exact_division() -> void {
    assert_near(dt.timedelta(days: 1, microseconds: 1).total_seconds(), 86_400.000001)
    assert_eq(str(dt.timedelta(microseconds: -1).total_seconds()), "-1e-06")
    assert_eq(str(dt.timedelta(microseconds: 1).total_seconds()), "1e-06")
    assert_eq(dt.timedelta().total_seconds(), 0.0)
    assert_eq(str(dt.timedelta.max.total_seconds()), "86400000000000.0")
}

fn test_timedelta_arithmetic_and_ordering() -> void {
    let hour = dt.timedelta(hours: 1)
    assert_eq(hour + hour, dt.timedelta(hours: 2))
    assert_eq(hour - hour, dt.timedelta())
    assert_eq(hour * 3, dt.timedelta(hours: 3))
    assert_eq(-hour, dt.timedelta(hours: -1))
    assert_eq(hour.negate(), dt.timedelta(hours: -1))
    assert_eq(dt.timedelta(seconds: -1).absolute(), dt.timedelta(seconds: 1))
    assert_eq(hour.absolute(), hour)
    assert_true(dt.timedelta(seconds: 1) < dt.timedelta(seconds: 2))
    assert_true(dt.timedelta(seconds: 1) <= dt.timedelta(seconds: 1))
    assert_true(dt.timedelta(seconds: 2) > dt.timedelta(seconds: 1))
    assert_eq(dt.timedelta(seconds: 1).compare_to(dt.timedelta(seconds: 2)), -1)
    assert_eq(dt.timedelta(seconds: 2).compare_to(dt.timedelta(seconds: 2)), 0)
    assert_eq(dt.timedelta(seconds: 3).compare_to(dt.timedelta(seconds: 2)), 1)
    assert_false(dt.timedelta() == "not a span")
    assert_eq(hash(dt.timedelta(hours: 1)), hash(dt.timedelta(minutes: 60)))
}

fn test_timedelta_str_and_repr() -> void {
    assert_eq(str(dt.timedelta()), "0:00:00")
    assert_eq(str(dt.timedelta(seconds: -1)), "-1 day, 23:59:59")
    assert_eq(str(dt.timedelta(days: -1)), "-1 day, 0:00:00")
    assert_eq(str(dt.timedelta(days: -2)), "-2 days, 0:00:00")
    assert_eq(str(dt.timedelta(hours: 25, microseconds: 5)), "1 day, 1:00:00.000005")
    assert_eq(repr(dt.timedelta()), "datetime.timedelta(0)")
    assert_eq(repr(dt.timedelta(weeks: 2)), "datetime.timedelta(days=14)")
    assert_eq(repr(dt.timedelta(microseconds: 1)), "datetime.timedelta(microseconds=1)")
    assert_eq(
        repr(dt.timedelta(days: 1, seconds: 2, microseconds: 3)),
        "datetime.timedelta(days=1, seconds=2, microseconds=3)"
    )
}

fn test_date_ordinals_are_cpythons() -> void {
    assert_eq(dt.date(1, 1, 1).toordinal(), 1)
    assert_eq(dt.date(9999, 12, 31).toordinal(), 3_652_059)
    assert_eq(dt.date(1970, 1, 1).toordinal(), 719_163)
    assert_eq(dt.date(2024, 2, 29).toordinal(), 738_945)
    assert_eq(dt.date.fromordinal(1), dt.date(1, 1, 1))
    assert_eq(dt.date.fromordinal(3_652_059), dt.date(9999, 12, 31))
    for ordinal in [1, 2, 365, 366, 367, 719_163, 3_652_058, 3_652_059] {
        assert_eq(dt.date.fromordinal(ordinal).toordinal(), ordinal)
    }
    assert_throws(ValueError, || dt.date.fromordinal(0))
    assert_throws(ValueError, || dt.date.fromordinal(3_652_060))
}

fn test_date_rejects_a_day_that_is_not_on_the_calendar() -> void {
    assert_throws(ValueError, || dt.date(2023, 2, 29))
    assert_throws(ValueError, || dt.date(1900, 2, 29))
    assert_throws(ValueError, || dt.date(2024, 13, 1))
    assert_throws(ValueError, || dt.date(2024, 0, 1))
    assert_throws(ValueError, || dt.date(2024, 1, 0))
    assert_throws(ValueError, || dt.date(2024, 4, 31))
    assert_throws(ValueError, || dt.date(0, 1, 1))
    assert_throws(ValueError, || dt.date(10000, 1, 1))
    assert_eq(dt.date(2024, 2, 29).day, 29)
    assert_eq(dt.date(2000, 2, 29).day, 29)
    assert_eq(dt.date(400, 2, 29).day, 29)
}

fn test_date_weekday_and_isocalendar() -> void {
    assert_eq(dt.date(2024, 2, 29).weekday(), 3)
    assert_eq(dt.date(2024, 2, 29).isoweekday(), 4)
    assert_eq(dt.date(2024, 2, 29).isocalendar(), (2024, 9, 4))
    assert_eq(dt.date(2020, 1, 1).isocalendar(), (2020, 1, 3))
    assert_eq(dt.date(2021, 1, 1).isocalendar(), (2020, 53, 5))
    assert_eq(dt.date(2023, 1, 1).isocalendar(), (2022, 52, 7))
    assert_eq(dt.date(1, 1, 1).weekday(), 0)
    assert_eq(dt.date(9999, 12, 31).weekday(), 4)
}

fn test_date_isoformat_round_trip_and_week_dates() -> void {
    assert_eq(dt.date(2024, 2, 29).isoformat(), "2024-02-29")
    assert_eq(dt.date(1, 1, 1).isoformat(), "0001-01-01")
    assert_eq(dt.date.fromisoformat("2024-02-29"), dt.date(2024, 2, 29))
    assert_eq(dt.date.fromisoformat("20240229"), dt.date(2024, 2, 29))
    assert_eq(dt.date.fromisoformat("2024-W09-4"), dt.date(2024, 2, 29))
    assert_eq(dt.date.fromisoformat("2024W094"), dt.date(2024, 2, 29))
    assert_eq(dt.date.fromisoformat("2024-W09"), dt.date(2024, 2, 26))
    assert_eq(dt.date.fromisoformat("2020-W53-7"), dt.date(2021, 1, 3))
    for text in ["", "2024", "2024-2-9", "2024-02-2", "2024-060", "2024-W53-7", "2024-W00-1"] {
        assert_throws(ValueError, || dt.date.fromisoformat(text))
    }
}

fn test_date_ctime_replace_and_conversions() -> void {
    assert_eq(dt.date(2024, 3, 4).ctime(), "Mon Mar  4 00:00:00 2024")
    assert_eq(dt.date(2024, 3, 14).ctime(), "Thu Mar 14 00:00:00 2024")
    assert_eq(dt.date(2024, 1, 2).replace(month: 3), dt.date(2024, 3, 2))
    assert_eq(dt.date(2024, 1, 2).replace(year: 2020, day: 5), dt.date(2020, 1, 5))
    assert_eq(dt.date(2024, 1, 2).replace(), dt.date(2024, 1, 2))
    assert_eq(str(dt.date(2024, 2, 29)), "2024-02-29")
    assert_eq(repr(dt.date(2024, 2, 29)), "datetime.date(2024, 2, 29)")
    assert_eq(dt.date.fromtimestamp(0.0), dt.date(1970, 1, 1))
    assert_eq(dt.date.fromtimestamp(-1.0), dt.date(1969, 12, 31))
    assert_eq(dt.date.min, dt.date(1, 1, 1))
    assert_eq(dt.date.max, dt.date(9999, 12, 31))
    assert_eq(dt.date.resolution, dt.timedelta(days: 1))
    assert_true(dt.date.today().year >= 2024)
}

fn test_date_arithmetic_and_ordering() -> void {
    assert_eq(dt.date(2024, 2, 28) + dt.timedelta(days: 2), dt.date(2024, 3, 1))
    assert_eq(dt.date(2024, 3, 1) - dt.timedelta(days: 1), dt.date(2024, 2, 29))
    assert_eq(dt.date(2024, 3, 1) - dt.date(2024, 2, 29), dt.timedelta(days: 1))
    assert_eq(dt.date(2024, 1, 1) + dt.timedelta(days: 1, hours: 23), dt.date(2024, 1, 2))
    assert_throws(OverflowError, || dt.date.max + dt.timedelta(days: 1))
    assert_throws(OverflowError, || dt.date.min - dt.timedelta(days: 1))
    assert_throws(TypeError, || dt.date(2024, 1, 1) - 1)
    assert_true(dt.date(2024, 1, 1) < dt.date(2024, 1, 2))
    assert_true(dt.date(2024, 1, 1) <= dt.date(2024, 1, 1))
    assert_true(dt.date(2024, 1, 2) > dt.date(2024, 1, 1))
    assert_eq(dt.date(2024, 1, 1).compare_to(dt.date(2024, 1, 2)), -1)
    assert_false(dt.date(2024, 1, 1) == "2024-01-01")
    assert_eq(hash(dt.date(2024, 1, 1)), hash(dt.date.fromordinal(dt.date(2024, 1, 1).toordinal())))
}

fn test_time_fields_and_rendering() -> void {
    let value = dt.time(1, 2, 3, 4)
    assert_eq(value.hour, 1)
    assert_eq(value.minute, 2)
    assert_eq(value.second, 3)
    assert_eq(value.microsecond, 4)
    assert_eq(value.isoformat(), "01:02:03.000004")
    assert_eq(dt.time(1, 2, 3).isoformat(), "01:02:03")
    assert_eq(dt.time(1, 2).isoformat(), "01:02:00")
    assert_eq(dt.time(1, 2, 3, 4).isoformat(timespec: "hours"), "01")
    assert_eq(dt.time(1, 2, 3, 4).isoformat(timespec: "minutes"), "01:02")
    assert_eq(dt.time(1, 2, 3, 4).isoformat(timespec: "seconds"), "01:02:03")
    assert_eq(dt.time(1, 2, 3, 4).isoformat(timespec: "milliseconds"), "01:02:03.000")
    assert_eq(dt.time(1, 2, 3, 4).isoformat(timespec: "microseconds"), "01:02:03.000004")
    assert_throws(ValueError, || dt.time(1, 2).isoformat(timespec: "fortnights"))
    assert_eq(repr(dt.time(0, 0)), "datetime.time(0, 0)")
    assert_eq(repr(dt.time(1, 2, 3)), "datetime.time(1, 2, 3)")
    assert_eq(repr(dt.time(1, 2, 0, 5)), "datetime.time(1, 2, 0, 5)")
    assert_eq(str(dt.time(1, 2, 3, 4)), "01:02:03.000004")
    assert_eq(dt.time.min, dt.time(0, 0))
    assert_eq(dt.time.max, dt.time(23, 59, 59, 999_999))
    assert_eq(dt.time.resolution, dt.timedelta(microseconds: 1))
}

fn test_time_zone_awareness_and_comparison() -> void {
    let aware = dt.time(12, 0, tzinfo: dt.timezone.utc)
    assert_eq(aware.isoformat(), "12:00:00+00:00")
    assert_eq(aware.utcoffset(), dt.timedelta())
    assert_eq(aware.tzname(), "UTC")
    assert_eq(aware.offset_micros(), 0)
    assert_true(aware.dst() == null)
    assert_true(dt.time(12, 0).utcoffset() == null)
    assert_true(dt.time(12, 0).tzname() == null)
    assert_true(dt.time(12, 0).offset_micros() == null)
    assert_true(dt.time(1, 0) < dt.time(2, 0))
    assert_true(dt.time(1, 0) <= dt.time(1, 0))
    assert_eq(dt.time(1, 0).compare_to(dt.time(2, 0)), -1)
    assert_false(dt.time(12, 0) == aware)
    assert_throws(TypeError, || dt.time(12, 0) < aware)
    assert_eq(hash(dt.time(1, 2, 3)), hash(dt.time(1, 2, 3)))
    assert_eq(dt.time(1, 2, 3, 4).replace(second: 9), dt.time(1, 2, 9, 4))
    assert_eq(aware.replace(tzinfo: null, keep_tzinfo: false), dt.time(12, 0))
    assert_throws(ValueError, || dt.time(24, 0))
    assert_throws(ValueError, || dt.time(0, 60))
    assert_throws(ValueError, || dt.time(0, 0, 60))
    assert_throws(ValueError, || dt.time(0, 0, 0, 1_000_000))
}

fn test_time_fromisoformat_and_strftime() -> void {
    assert_eq(dt.time.fromisoformat("12:00:00"), dt.time(12, 0))
    assert_eq(dt.time.fromisoformat("12:00"), dt.time(12, 0))
    assert_eq(dt.time.fromisoformat("12"), dt.time(12, 0))
    assert_eq(dt.time.fromisoformat("120000"), dt.time(12, 0))
    assert_eq(dt.time.fromisoformat("12:00:00.5").microsecond, 500_000)
    assert_eq(dt.time.fromisoformat("12,5").microsecond, 500_000)
    assert_eq(dt.time.fromisoformat("12:00:00.1234567").microsecond, 123_456)
    assert_eq(dt.time.fromisoformat("12:00:00Z").tzname(), "UTC")
    assert_eq(dt.time.fromisoformat("12:00:00+05:30").offset_micros(), 19_800_000_000)
    for text in ["", "1:00", "12:0", "24:00", "12:00:00z", "12:00:00."] {
        assert_throws(ValueError, || dt.time.fromisoformat(text))
    }
    assert_eq(dt.time(13, 5).strftime("%I:%M %p"), "01:05 PM")
    assert_eq(dt.time(13, 5).strftime("%Y-%m-%d"), "1900-01-01")
}

fn test_timezone_offsets_and_names() -> void {
    assert_eq(str(dt.timezone.utc), "UTC")
    assert_eq(repr(dt.timezone.utc), "datetime.timezone.utc")
    assert_eq(repr(dt.timezone(dt.timedelta(0))), "datetime.timezone.utc")
    assert_eq(str(dt.timezone(dt.timedelta(minutes: 330))), "UTC+05:30")
    assert_eq(str(dt.timezone(dt.timedelta(minutes: -330))), "UTC-05:30")
    assert_eq(dt.timezone(dt.timedelta(seconds: 30)).tzname(), "UTC+00:00:30")
    assert_eq(dt.timezone(dt.timedelta(0), "Z").tzname(), "Z")
    assert_eq(dt.timezone(dt.timedelta(hours: 1)).offset_micros(), 3_600_000_000)
    assert_eq(dt.timezone(dt.timedelta(hours: 1)).utcoffset(), dt.timedelta(hours: 1))
    assert_true(dt.timezone.utc.dst() == null)
    assert_eq(dt.timezone(dt.timedelta(hours: 1)), dt.timezone(dt.timedelta(minutes: 60)))
    assert_false(dt.timezone.utc == "UTC")
    assert_eq(hash(dt.timezone.utc), hash(dt.timezone(dt.timedelta(0))))
    assert_eq(dt.timezone.min.tzname(), "UTC-23:59")
    assert_eq(dt.timezone.max.tzname(), "UTC+23:59")
    assert_throws(ValueError, || dt.timezone(dt.timedelta(hours: 24)))
    assert_throws(ValueError, || dt.timezone(dt.timedelta(hours: -24)))
    assert_throws(ValueError, || dt.timezone(dt.timedelta(days: 400)))
}

fn test_datetime_construction_and_parts() -> void {
    let moment = dt.datetime(2024, 2, 29, 12, 30, 15, 250_000)
    assert_eq(moment.year, 2024)
    assert_eq(moment.month, 2)
    assert_eq(moment.day, 29)
    assert_eq(moment.hour, 12)
    assert_eq(moment.minute, 30)
    assert_eq(moment.second, 15)
    assert_eq(moment.microsecond, 250_000)
    assert_eq(moment.date(), dt.date(2024, 2, 29))
    assert_eq(moment.time(), dt.time(12, 30, 15, 250_000))
    assert_eq(moment.timetz(), dt.time(12, 30, 15, 250_000))
    assert_eq(moment.toordinal(), 738_945)
    assert_eq(moment.weekday(), 3)
    assert_eq(moment.isoweekday(), 4)
    assert_eq(moment.isocalendar(), (2024, 9, 4))
    assert_eq(moment.ctime(), "Thu Feb 29 12:30:15 2024")
    assert_eq(dt.datetime.fromordinal(1), dt.datetime(1, 1, 1))
    assert_eq(dt.datetime.min, dt.datetime(1, 1, 1))
    assert_eq(dt.datetime.max, dt.datetime(9999, 12, 31, 23, 59, 59, 999_999))
    assert_eq(dt.datetime.resolution, dt.timedelta(microseconds: 1))
    assert_throws(ValueError, || dt.datetime(2023, 2, 29))
    assert_throws(ValueError, || dt.datetime(2024, 1, 1, 24))
    assert_throws(ValueError, || dt.datetime.fromordinal(0))
}

fn test_datetime_combine_and_clocks() -> void {
    let day = dt.date(2024, 1, 1)
    assert_eq(dt.datetime.combine(day, dt.time(1, 2, 3, 4)), dt.datetime(2024, 1, 1, 1, 2, 3, 4))
    let aware_time = dt.time(1, 2, 3, 4, dt.timezone.utc)
    assert_eq(dt.datetime.combine(day, aware_time).tzname(), "UTC")
    assert_true(dt.datetime.combine(day, aware_time, null).tzinfo == null)
    assert_eq(dt.datetime.combine(day, dt.time(1, 2), dt.timezone.utc).tzname(), "UTC")
    assert_true(dt.datetime.now().year >= 2024)
    assert_true(dt.datetime.utcnow().year >= 2024)
    assert_true(dt.datetime.today().year >= 2024)
    assert_true(dt.datetime.now(dt.timezone.utc).tzinfo is not null)
}

fn test_datetime_isoformat_and_fromisoformat() -> void {
    let moment = dt.datetime(2024, 1, 2, 3, 4, 5, 6)
    assert_eq(moment.isoformat(), "2024-01-02T03:04:05.000006")
    assert_eq(moment.isoformat(separator: " "), "2024-01-02 03:04:05.000006")
    assert_eq(moment.isoformat(timespec: "milliseconds"), "2024-01-02T03:04:05.000")
    assert_eq(moment.isoformat(timespec: "hours"), "2024-01-02T03")
    assert_eq(dt.datetime(2024, 1, 2, 3, 4, 5).isoformat(), "2024-01-02T03:04:05")
    assert_eq(str(moment), "2024-01-02 03:04:05.000006")
    assert_eq(repr(moment), "datetime.datetime(2024, 1, 2, 3, 4, 5, 6)")
    assert_eq(repr(dt.datetime(2024, 1, 2)), "datetime.datetime(2024, 1, 2, 0, 0)")
    assert_eq(
        repr(dt.datetime(2024, 1, 2, 3, 4, 5, 6, dt.timezone.utc)),
        "datetime.datetime(2024, 1, 2, 3, 4, 5, 6, tzinfo=datetime.timezone.utc)"
    )
    assert_eq(dt.datetime.fromisoformat(moment.isoformat()), moment)
    assert_eq(dt.datetime.fromisoformat("2024-02-29"), dt.datetime(2024, 2, 29))
    assert_eq(dt.datetime.fromisoformat("2024-02-29T12:00"), dt.datetime(2024, 2, 29, 12))
    assert_eq(dt.datetime.fromisoformat("20240229T120000"), dt.datetime(2024, 2, 29, 12))
    assert_eq(dt.datetime.fromisoformat("2024-02-29T12:00:00Z").tzname(), "UTC")
    assert_eq(
        dt.datetime.fromisoformat("2024-02-29T12:00:00+05:30").offset_micros(),
        19_800_000_000
    )
    for text in ["", "2024-02-29T", "2024-02-29T24:00:00", "2024-02-29T12:00:00."] {
        assert_throws(ValueError, || dt.datetime.fromisoformat(text))
    }
}

fn test_datetime_strftime_covers_every_specifier() -> void {
    let moment = dt.datetime(2024, 7, 4, 13, 5, 6, 7, dt.timezone(dt.timedelta(minutes: -330)))
    assert_eq(
        moment.strftime("%Y %m %d %H %M %S %f %z %a %A %b %B %j %U %W %%"),
        "2024 07 04 13 05 06 000007 -0530 Thu Thursday Jul July 186 26 27 %"
    )
    assert_eq(moment.strftime("%y|%I|%p|%w|%u|%G|%V|%Z"), "24|01|PM|4|4|2024|27|UTC-05:30")
    assert_eq(dt.datetime(1, 1, 1).strftime("%Y|%j|%U|%W|%z|%Z"), "0001|001|00|01||")
    assert_eq(dt.datetime(2024, 3, 4).strftime("%U|%W"), "09|10")
    assert_eq(dt.datetime(2024, 1, 1, 0, 0).strftime("%I %p"), "12 AM")
    assert_eq(dt.datetime(2024, 1, 1, 12, 0).strftime("%I %p"), "12 PM")
    assert_throws(ValueError, || dt.datetime(2024, 1, 1).strftime("%Q"))
    assert_throws(ValueError, || dt.datetime(2024, 1, 1).strftime("100%"))
}

fn test_datetime_strptime_round_trips() -> void {
    assert_eq(dt.datetime.strptime("2024-02-29", "%Y-%m-%d"), dt.datetime(2024, 2, 29))
    assert_eq(
        dt.datetime.strptime("29/Feb/2024:13:05:06", "%d/%b/%Y:%H:%M:%S"),
        dt.datetime(2024, 2, 29, 13, 5, 6)
    )
    assert_eq(
        dt.datetime.strptime("Thursday, 29 February 2024", "%A, %d %B %Y"),
        dt.datetime(2024, 2, 29)
    )
    assert_eq(dt.datetime.strptime("24-02-29", "%y-%m-%d"), dt.datetime(2024, 2, 29))
    assert_eq(dt.datetime.strptime("69-01-01", "%y-%m-%d"), dt.datetime(1969, 1, 1))
    assert_eq(dt.datetime.strptime("060 2024", "%j %Y"), dt.datetime(2024, 2, 29))
    assert_eq(dt.datetime.strptime("2024 08 4", "%Y %U %w"), dt.datetime(2024, 2, 29))
    assert_eq(dt.datetime.strptime("2024 09 Thu", "%Y %W %a"), dt.datetime(2024, 2, 29))
    assert_eq(dt.datetime.strptime("01:05 PM", "%I:%M %p").hour, 13)
    assert_eq(dt.datetime.strptime("12:00 AM", "%I:%M %p").hour, 0)
    assert_eq(dt.datetime.strptime("13:05:06.7", "%H:%M:%S.%f").microsecond, 700_000)
    assert_eq(dt.datetime.strptime("nothing", "nothing"), dt.datetime(1900, 1, 1))
    assert_eq(dt.datetime.strptime("+0530", "%z").offset_micros(), 19_800_000_000)
    assert_eq(dt.datetime.strptime("Z", "%z").offset_micros(), 0)
    assert_eq(dt.datetime.strptime("-05:30:45", "%z").offset_micros(), -19_845_000_000)
    for pair in [
        ["2024-02-29", "%Y/%m/%d"],
        ["2024-02-29 extra", "%Y-%m-%d"],
        ["Febtober 2024", "%B %Y"],
        ["", "%Y"],
        ["2024", "%Q"],
        ["2024", "%Y%"],
    ] {
        assert_throws(ValueError, || dt.datetime.strptime(pair[0], pair[1]))
    }
}

fn test_datetime_timestamps_and_zone_conversion() -> void {
    let epoch = dt.datetime(1970, 1, 1, tzinfo: dt.timezone.utc)
    assert_eq(epoch.timestamp(), 0.0)
    assert_eq(dt.datetime(1970, 1, 2, tzinfo: dt.timezone.utc).timestamp(), 86_400.0)
    assert_eq(
        dt.datetime(2024, 1, 1, 12, 0, 0, 500_000, tzinfo: dt.timezone.utc).timestamp(),
        1_704_110_400.5
    )
    assert_eq(dt.datetime.fromtimestamp(0.0, dt.timezone.utc), epoch)
    assert_eq(dt.datetime.fromtimestamp(0.0), dt.datetime(1970, 1, 1))
    assert_eq(dt.datetime.utcfromtimestamp(-1.0), dt.datetime(1969, 12, 31, 23, 59, 59))
    assert_eq(dt.datetime.fromtimestamp(1.5).microsecond, 500_000)
    assert_eq(dt.datetime.fromtimestamp(-1.5).microsecond, 500_000)
    assert_throws(ValueError, || dt.datetime.fromtimestamp(1.0e18))
    let india = dt.timezone(dt.timedelta(hours: 5, minutes: 30))
    assert_eq(epoch.astimezone(india), dt.datetime(1970, 1, 1, 5, 30, tzinfo: india))
    assert_eq(dt.datetime(2024, 1, 1).astimezone(dt.timezone.utc).hour, 0)
    assert_eq(epoch.astimezone(india).timestamp(), 0.0)
    assert_throws(OverflowError, || dt.datetime.max.astimezone(india))
}

fn test_datetime_arithmetic_and_ordering() -> void {
    let moment = dt.datetime(2024, 2, 28, 23, 0)
    assert_eq(moment + dt.timedelta(hours: 2), dt.datetime(2024, 2, 29, 1, 0))
    assert_eq(moment - dt.timedelta(hours: 24), dt.datetime(2024, 2, 27, 23, 0))
    assert_eq(dt.datetime(2024, 1, 2) - dt.datetime(2024, 1, 1), dt.timedelta(days: 1))
    assert_eq(
        dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc) -
            dt.datetime(2023, 1, 1, tzinfo: dt.timezone(dt.timedelta(hours: 2))),
        dt.timedelta(days: 365, hours: 2)
    )
    assert_throws(OverflowError, || dt.datetime.max + dt.timedelta(microseconds: 1))
    assert_throws(OverflowError, || dt.datetime.min - dt.timedelta(microseconds: 1))
    assert_throws(TypeError, || dt.datetime(2024, 1, 1) - "yesterday")
    assert_true(dt.datetime(2024, 1, 1) < dt.datetime(2024, 1, 2))
    assert_true(dt.datetime(2024, 1, 1) <= dt.datetime(2024, 1, 1))
    assert_eq(dt.datetime(2024, 1, 1).compare_to(dt.datetime(2024, 1, 2)), -1)
    assert_true(
        dt.datetime(2024, 1, 1, 5, tzinfo: dt.timezone(dt.timedelta(hours: 5))) ==
            dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc)
    )
    assert_false(dt.datetime(2024, 1, 1) == dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc))
    assert_throws(
        TypeError,
        || dt.datetime(2024, 1, 1) < dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc)
    )
    assert_throws(
        TypeError,
        || dt.datetime(2024, 1, 1) - dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc)
    )
    assert_eq(
        hash(dt.datetime(2024, 1, 1, 5, tzinfo: dt.timezone(dt.timedelta(hours: 5)))),
        hash(dt.datetime(2024, 1, 1, tzinfo: dt.timezone.utc))
    )
}

fn test_datetime_replace_and_zone_accessors() -> void {
    let moment = dt.datetime(2024, 1, 2, 3, 4, 5, 6, dt.timezone.utc)
    assert_eq(moment.replace(hour: 9).hour, 9)
    assert_eq(moment.replace(year: 2000, month: 12, day: 31).date(), dt.date(2000, 12, 31))
    assert_eq(moment.replace().tzname(), "UTC")
    assert_true(moment.replace(tzinfo: null, keep_tzinfo: false).tzinfo == null)
    assert_eq(moment.utcoffset(), dt.timedelta())
    assert_eq(moment.offset_micros(), 0)
    assert_true(moment.dst() == null)
    assert_true(dt.datetime(2024, 1, 2).utcoffset() == null)
    assert_true(dt.datetime(2024, 1, 2).tzname() == null)
    assert_true(dt.datetime(2024, 1, 2).offset_micros() == null)
}

fn test_leap_years_and_year_boundaries() -> void {
    for year in [4, 400, 1600, 2000, 2020, 2024, 2400, 9996] {
        assert_eq(dt.date(year, 2, 29).month, 2)
        assert_eq(dt.date(year, 12, 31).toordinal() - dt.date(year, 1, 1).toordinal(), 365)
    }
    for year in [1, 100, 1700, 1800, 1900, 2023, 2025, 2100, 9999] {
        assert_throws(ValueError, || dt.date(year, 2, 29))
        assert_eq(dt.date(year, 12, 31).toordinal() - dt.date(year, 1, 1).toordinal(), 364)
    }
    assert_eq(dt.date(2024, 12, 31) + dt.timedelta(days: 1), dt.date(2025, 1, 1))
    assert_eq(dt.date(2025, 1, 1) - dt.timedelta(days: 1), dt.date(2024, 12, 31))
    assert_eq(dt.date(1900, 2, 28) + dt.timedelta(days: 1), dt.date(1900, 3, 1))
    assert_eq(dt.date(2000, 2, 28) + dt.timedelta(days: 1), dt.date(2000, 2, 29))
    assert_eq(dt.date(2024, 12, 31).strftime("%j"), "366")
    assert_eq(dt.date(2023, 12, 31).strftime("%j"), "365")
}
