#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# ///
"""Differential test: `std.datetime` against CPython 3.13's `datetime`.

Every case is a tab-separated instruction. Both sides read the same case file,
run the same sequence of calls, and print one line per case: the case number,
then the fields that case produces, tab separated. Each field is guarded on
its own, so a call that raises contributes `!ExceptionType` and the fields
around it are still compared -- a case that must raise is checked as strictly
as one that must not. The two outputs are then diffed line by line and any
difference is a failure.

What the cases cover:

* random ordinals across the whole 0001-01-01..9999-12-31 range, through
  `fromordinal`, `toordinal`, `weekday`, `isoweekday`, `isocalendar`,
  `ctime`, `replace`, `isoformat` and `fromisoformat`;
* the leap-year and year-boundary edges by name -- 29 February in the
  Gregorian exception years, 1 January and 31 December either side of them,
  and both ends of the calendar;
* random `timedelta`s over the full +/-999999999-day range, built from all
  seven constructor units, through `repr`, `str`, `total_seconds`, negation,
  magnitude, sum, difference, ordering and multiplication;
* random moments with and without a fixed offset, through `isoformat` at
  every `timespec` and both separators, `timestamp`, `fromtimestamp`,
  `astimezone`, `date`, `time`, `timetz`, `utcoffset` and `tzname`;
* every `strftime` directive the module claims, on dates, times and moments,
  naive and aware;
* `strptime` round trips: CPython formats the moment and both sides parse
  that text back, so the input is CPython's own rendering rather than this
  module's;
* malformed ISO strings, malformed `strptime` inputs and out-of-range field
  values, where the exception type is the answer.

`TZ` is pinned to UTC for the CPython side. `std.datetime` has no zone
database, so where CPython reads the local zone -- `fromtimestamp` with no
zone, `timestamp` on a naive moment -- it reads UTC instead, and pinning `TZ`
is what lets those paths be diffed at all. The deviation is real and
documented in the module header; the pin only stops it from hiding the
arithmetic underneath. The one thing it cannot rescue is a naive moment in
year 1 or year 9999, where CPython's `mktime` probes a day outside the
calendar and raises; `local_readable` and `local_stamp_readable` leave those
fields out. A timestamp far enough outside the calendar that CPython's
`gmtime` itself fails -- 1e18 seconds, say -- is not generated either: CPython
surfaces the platform's `OSError` there where this module raises `ValueError`,
which is the module header's last documented deviation.

    uv run --python 3.13 python3 tests/stdlib/py/datetime_diff.py
    uv run --python 3.13 python3 tests/stdlib/py/datetime_diff.py --cases 3000 --seed 7
"""

from __future__ import annotations

import argparse
import os

os.environ["TZ"] = "UTC"

import random
import subprocess
import sys
import tempfile
import time as _time
from datetime import MAXYEAR, MINYEAR, date, datetime, time, timedelta, timezone
from pathlib import Path

_time.tzset()

ROOT = Path(__file__).resolve().parents[3]
JAITHON = ROOT / "jaithon"

MAXORDINAL = date.max.toordinal()
EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
UTC = timezone.utc

FULL = "%Y|%y|%m|%d|%j|%H|%I|%M|%S|%f|%p|%a|%A|%b|%B|%w|%u|%U|%W|%G|%V|%z|%Z|%%"

TIMESPECS = ["auto", "hours", "minutes", "seconds", "milliseconds", "microseconds"]

ROUND_TRIP_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%d/%b/%Y:%H:%M:%S %z",
    "%A, %d %B %Y",
    "%a %b %d %H:%M:%S %Y",
    "%y-%m-%d %I:%M %p",
    "%j %Y",
    "%Y %U %w",
    "%Y %W %w",
    "%Y %W %a",
    "%Y %U %A",
    "%%%Y%%",
    "%H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
]

BAD_ISO = [
    "",
    "2024",
    "2024-2-9",
    "2024-02-2",
    "24-02-29",
    "2024-0229",
    "202402-29",
    "2024-02-29 ",
    "+2024-02-29",
    "2024-060",
    "2024060",
    "2024-W099",
    "2024-W9-4",
    "2024-W00-1",
    "2024-W53-7",
    "2024-W54-1",
    "2020-W53-7",
    "2023-02-29",
    "2024-13-01",
    "2024-00-01",
    "2024-01-00",
    "2024-01-32",
    "0000-01-01",
    "2024-02-29T",
    "2024-02-29T24:00:00",
    "2024-02-29T12:00:00.",
    "2024-02-29T12:00:00+2400",
    "2024-02-29T12:60",
    "2024-02-29T12:00:61",
    "2024-02-29x12:00:00",
    "2024-02-29T12:00:00-05:30:45.000123",
    "2024-02-29 12:00:00.123456",
    "2024-02-29T12:00:00Z",
    "2024-02-29T12:00:00z",
    "2024-02-29T12:00:00-00:00",
    "2024W094",
    "2024-W09",
    "20240229T120000",
    "0001-01-01",
    "9999-12-31",
]

BAD_ISO_TIME = [
    "",
    "1:00",
    "12:0",
    "12",
    "12:00",
    "120000",
    "12:00:00",
    "12:00:00.5",
    "12:00:00.1234567",
    "12:00:00Z",
    "12:00:00z",
    "24:00",
    "12:00:00+05:30",
    "12:00:00+0530",
    "12:00:00-053045",
    "12:00:00+05",
    "12:00:00+053",
    "12,5",
    "12:00:00.",
]

BAD_STRPTIME = [
    ("2024-02-29", "%Y/%m/%d"),
    ("2024-13-01", "%Y-%m-%d"),
    ("2024-02-30", "%Y-%m-%d"),
    ("25", "%H"),
    ("2024-02-29 extra", "%Y-%m-%d"),
    ("2024-02", "%Y-%m-%d"),
    ("Febtober 2024", "%B %Y"),
    ("", "%Y"),
    ("2024", "%Q"),
    ("12 AM", "%I %p"),
    ("12 PM", "%I %p"),
    ("01 am", "%I %p"),
    ("1999", "%y"),
    ("68", "%y"),
    ("69", "%y"),
    ("366 2024", "%j %Y"),
    ("366 2023", "%j %Y"),
    ("2024   02   29", "%Y %m %d"),
    ("2024 02 29", "%Y   %m   %d"),
    ("2024-02-29+0530", "%Y-%m-%d%z"),
    ("2024-02-29Z", "%Y-%m-%d%z"),
    ("2024-02-29+05:30:45.123", "%Y-%m-%d%z"),
    ("FEB 2024", "%b %Y"),
    ("february 2024", "%B %Y"),
]


def bool_field(value: bool) -> str:
    return "1" if value else "0"


def safe(producer) -> str:
    """Render one field, or the name of the exception producing it raised.

    Each field is guarded on its own so that a case whose last call overflows
    still compares the twenty fields before it, instead of collapsing to a
    single exception name that hides whatever else was wrong.
    """
    try:
        return producer()
    except Exception as error:  # noqa: BLE001 - the exception type is the answer
        return "!" + type(error).__name__


def local_readable(moment: datetime) -> bool:
    """Whether CPython can read this moment through the local time zone.

    `timestamp` and `astimezone` on a naive moment go through `mktime`, which
    probes the day either side and so raises at both ends of the calendar.
    This module reads UTC and has no such edge, so those fields are left out
    for a naive moment in the first or last year rather than reported as a
    difference the documented deviation already explains.
    """
    return moment.tzinfo is not None or MINYEAR < moment.year < MAXYEAR


def local_stamp_readable(seconds: float) -> bool:
    """Whether CPython can turn `seconds` into a naive local moment.

    Same edge as `local_readable`, reached from the other side: `localtime`
    on a timestamp within a day of either end of the calendar probes past it
    and raises. A day of margin at each end keeps the naive field comparable
    everywhere else.
    """
    return -62_135_510_400.0 <= seconds <= 253_402_214_399.0


def iso_triple(value) -> str:
    return "%d-%d-%d" % (value[0], value[1], value[2])


def zone_of(micros: int | None):
    if micros is None:
        return None
    if micros == 0:
        return UTC
    return timezone(timedelta(microseconds=micros))


def moment_of(ordinal: int, hour: int, minute: int, second: int, micro: int, tz: int | None):
    return datetime.combine(
        date.fromordinal(ordinal), time(hour, minute, second, micro), zone_of(tz)
    )


def optional_int(text: str) -> int | None:
    return None if text == "-" else int(text)


# ------------------------------------------------------------------ payloads


def payload(kind: str, args: list[str]) -> list[str]:
    if kind == "date":
        day = date.fromordinal(int(args[0]))
        return [
            safe(lambda: day.isoformat()),
            safe(lambda: str(day.toordinal())),
            safe(lambda: str(day.weekday())),
            safe(lambda: str(day.isoweekday())),
            safe(lambda: iso_triple(day.isocalendar())),
            safe(lambda: day.ctime()),
            safe(lambda: repr(day)),
            safe(lambda: str(day)),
            safe(lambda: day.strftime(FULL)),
            safe(lambda: repr(date.fromisoformat(day.isoformat()))),
            safe(lambda: repr(day.replace(day=1))),
            safe(lambda: repr(day.replace(month=1, day=1))),
        ]

    if kind == "delta":
        span = timedelta(
            days=int(args[0]),
            seconds=int(args[1]),
            microseconds=int(args[2]),
            milliseconds=int(args[3]),
            minutes=int(args[4]),
            hours=int(args[5]),
            weeks=int(args[6]),
        )
        return [
            safe(lambda: repr(span)),
            safe(lambda: str(span)),
            safe(lambda: repr(span.total_seconds())),
            safe(lambda: str(span.days)),
            safe(lambda: str(span.seconds)),
            safe(lambda: str(span.microseconds)),
            safe(lambda: repr(-span)),
            safe(lambda: repr(abs(span))),
        ]

    if kind == "deltamul":
        span = timedelta(days=int(args[0]), seconds=int(args[1]), microseconds=int(args[2]))
        factor = int(args[3])
        return [safe(lambda: repr(span * factor))]

    if kind == "deltapair":
        left = timedelta(days=int(args[0]), seconds=int(args[1]), microseconds=int(args[2]))
        right = timedelta(days=int(args[3]), seconds=int(args[4]), microseconds=int(args[5]))
        return [
            safe(lambda: repr(left + right)),
            safe(lambda: repr(left - right)),
            safe(lambda: bool_field(left < right)),
            safe(lambda: bool_field(left <= right)),
            safe(lambda: bool_field(left == right)),
        ]

    if kind == "moment":
        tz = optional_int(args[5])
        moment = moment_of(
            int(args[0]), int(args[1]), int(args[2]), int(args[3]), int(args[4]), tz
        )
        fields = [
            safe(lambda: repr(moment)),
            safe(lambda: str(moment)),
            safe(lambda: moment.ctime()),
            safe(lambda: str(moment.toordinal())),
            safe(lambda: str(moment.weekday())),
            safe(lambda: iso_triple(moment.isocalendar())),
            safe(lambda: moment.strftime(FULL)),
            safe(lambda: repr(moment.date())),
            safe(lambda: repr(moment.time())),
            safe(lambda: repr(moment.timetz())),
        ]
        for spec in TIMESPECS:
            fields.append(safe(lambda spec=spec: moment.isoformat(timespec=spec)))
            fields.append(safe(lambda spec=spec: moment.isoformat(sep=" ", timespec=spec)))
        fields.append(safe(lambda: repr(datetime.fromisoformat(moment.isoformat()))))
        fields.append(safe(lambda: repr(datetime.fromisoformat(moment.isoformat(sep=" ")))))
        if tz is None:
            fields.append("-")
            fields.append("-")
        else:
            fields.append(safe(lambda: repr(moment.utcoffset())))
            fields.append(safe(lambda: moment.tzname()))
        if local_readable(moment):
            fields.append(safe(lambda: repr(moment.timestamp())))
            fields.append(safe(lambda: repr(datetime.fromtimestamp(moment.timestamp(), UTC))))
            fields.append(
                safe(lambda: repr(moment.astimezone(timezone(timedelta(hours=5, minutes=30)))))
            )
            fields.append(safe(lambda: repr(moment.astimezone(UTC))))
        else:
            fields += ["-", "-", "-", "-"]
        return fields

    if kind == "timeobj":
        value = time(
            int(args[0]), int(args[1]), int(args[2]), int(args[3]), zone_of(optional_int(args[4]))
        )
        fields = [
            safe(lambda: repr(value)),
            safe(lambda: str(value)),
            safe(lambda: value.strftime(FULL)),
        ]
        for spec in TIMESPECS:
            fields.append(safe(lambda spec=spec: value.isoformat(timespec=spec)))
        fields.append(safe(lambda: repr(time.fromisoformat(value.isoformat()))))
        return fields

    if kind == "momentadd":
        moment = moment_of(
            int(args[0]),
            int(args[1]),
            int(args[2]),
            int(args[3]),
            int(args[4]),
            optional_int(args[5]),
        )
        span = timedelta(days=int(args[6]), seconds=int(args[7]), microseconds=int(args[8]))
        return [safe(lambda: repr(moment + span)), safe(lambda: repr(moment - span))]

    if kind == "momentdiff":
        left = moment_of(
            int(args[0]),
            int(args[1]),
            int(args[2]),
            int(args[3]),
            int(args[4]),
            optional_int(args[5]),
        )
        right = moment_of(
            int(args[6]),
            int(args[7]),
            int(args[8]),
            int(args[9]),
            int(args[10]),
            optional_int(args[11]),
        )
        return [
            safe(lambda: repr(left - right)),
            safe(lambda: bool_field(left < right)),
            safe(lambda: bool_field(left <= right)),
            safe(lambda: bool_field(left == right)),
        ]

    if kind == "datediff":
        left = date.fromordinal(int(args[0]))
        right = date.fromordinal(int(args[1]))
        span = timedelta(days=int(args[2]))
        return [
            safe(lambda: repr(left - right)),
            safe(lambda: repr(left + span)),
            safe(lambda: repr(left - span)),
            safe(lambda: bool_field(left < right)),
            safe(lambda: bool_field(left == right)),
        ]

    if kind == "strptime":
        return [safe(lambda: repr(datetime.strptime(args[0], args[1])))]

    if kind == "isodate":
        return [safe(lambda: repr(date.fromisoformat(args[0])))]

    if kind == "isomoment":
        return [safe(lambda: repr(datetime.fromisoformat(args[0])))]

    if kind == "isotime":
        return [safe(lambda: repr(time.fromisoformat(args[0])))]

    if kind == "stamp":
        seconds = float(args[0])
        return [
            safe(lambda: repr(datetime.fromtimestamp(seconds, UTC))),
            safe(lambda: repr(datetime.fromtimestamp(seconds)))
            if local_stamp_readable(seconds)
            else "-",
            safe(lambda: repr(datetime.fromtimestamp(seconds, UTC).timestamp())),
        ]

    if kind == "ctor":
        parts = [int(part) for part in args[:7]]
        return [
            safe(lambda: repr(datetime(*parts))),
            safe(lambda: repr(date(parts[0], parts[1], parts[2]))),
            safe(lambda: repr(time(parts[3], parts[4], parts[5], parts[6]))),
        ]

    if kind == "zone":
        micros = int(args[0])
        return [
            safe(lambda: repr(timezone(timedelta(microseconds=micros)))),
            safe(lambda: str(timezone(timedelta(microseconds=micros)))),
            safe(lambda: timezone(timedelta(microseconds=micros)).tzname(None)),
            safe(lambda: repr(timezone(timedelta(microseconds=micros)).utcoffset(None))),
        ]

    raise AssertionError(f"unknown case kind {kind!r}")


def python_lines(cases: list[list[str]]) -> list[str]:
    out = []
    for index, case in enumerate(cases):
        try:
            fields = payload(case[0], case[1:])
        except AssertionError:
            raise
        except Exception as error:  # noqa: BLE001 - the exception type is the answer
            fields = ["!" + type(error).__name__]
        out.append(str(index) + "\t" + "\t".join(fields))
    return out


# ---------------------------------------------------------------- generation


def generate(rng: random.Random, count: int) -> list[list[str]]:
    cases: list[list[str]] = []

    def ordinal() -> int:
        if rng.random() < 0.1:
            return rng.choice(
                [1, 2, 365, 366, 367, MAXORDINAL, MAXORDINAL - 1, EPOCH_ORDINAL]
            )
        return rng.randint(1, MAXORDINAL)

    def offset_micros():
        pick = rng.random()
        if pick < 0.4:
            return None
        if pick < 0.55:
            return 0
        if pick < 0.9:
            return rng.randint(-23 * 60 - 59, 23 * 60 + 59) * 60_000_000
        return rng.randint(-86_399_999_999, 86_399_999_999)

    def span_parts() -> tuple[int, int, int]:
        scale = rng.choice([1, 10, 1000, 100_000, 10_000_000, 999_999_999])
        return (
            rng.randint(-scale, scale),
            rng.randint(-86_400, 86_400),
            rng.randint(-2_000_000, 2_000_000),
        )

    def clock() -> tuple[int, int, int, int]:
        if rng.random() < 0.15:
            return rng.choice(
                [
                    (0, 0, 0, 0),
                    (23, 59, 59, 999_999),
                    (12, 0, 0, 0),
                    (0, 0, 0, 1),
                    (13, 5, 6, 7),
                    (23, 59, 59, 0),
                ]
            )
        return (
            rng.randint(0, 23),
            rng.randint(0, 59),
            rng.randint(0, 59),
            rng.choice([0, 0, 1, 999_999, 500_000, rng.randint(0, 999_999)]),
        )

    def moment_args() -> list[str]:
        hour, minute, second, micro = clock()
        tz = offset_micros()
        return [
            str(ordinal()),
            str(hour),
            str(minute),
            str(second),
            str(micro),
            "-" if tz is None else str(tz),
        ]

    edge_years = [
        MINYEAR, 4, 100, 400, 1582, 1600, 1700, 1800, 1900, 1970, 2000, 2020,
        2023, 2024, 2025, 2100, 2400, 9996, MAXYEAR,
    ]
    for year in edge_years:
        for month, day in [(1, 1), (2, 28), (3, 1), (12, 31)]:
            cases.append(["date", str(date(year, month, day).toordinal())])
        try:
            cases.append(["date", str(date(year, 2, 29).toordinal())])
        except ValueError:
            pass
        cases.append(["ctor", str(year), "2", "29", "12", "30", "45", "123456"])
        cases.append(["ctor", str(year), "12", "31", "23", "59", "59", "999999"])
        cases.append(["ctor", str(year), "1", "1", "0", "0", "0", "0"])
    for bad in [
        ("0", "1", "1", "0", "0", "0", "0"),
        ("10000", "1", "1", "0", "0", "0", "0"),
        ("2024", "2", "30", "0", "0", "0", "0"),
        ("2024", "1", "1", "24", "0", "0", "0"),
        ("2024", "1", "1", "0", "60", "0", "0"),
        ("2024", "1", "1", "0", "0", "60", "0"),
        ("2024", "1", "1", "0", "0", "0", "1000000"),
        ("2024", "0", "1", "0", "0", "0", "0"),
        ("2024", "1", "0", "0", "0", "0", "0"),
    ]:
        cases.append(["ctor"] + list(bad))

    for text in BAD_ISO:
        cases.append(["isodate", text])
        cases.append(["isomoment", text])
    for text in BAD_ISO_TIME:
        cases.append(["isotime", text])
    for text, pattern in BAD_STRPTIME:
        cases.append(["strptime", text, pattern])

    for span in [
        timedelta.min,
        timedelta.max,
        timedelta.resolution,
        -timedelta.resolution,
        timedelta(0),
        timedelta(days=-1),
        timedelta(seconds=-1),
        timedelta(hours=25),
        timedelta(days=1, microseconds=1),
        timedelta(days=-1, microseconds=1),
    ]:
        cases.append(
            [
                "delta",
                str(span.days),
                str(span.seconds),
                str(span.microseconds),
                "0",
                "0",
                "0",
                "0",
            ]
        )
    cases.append(["delta", "999999999", "86400", "0", "0", "0", "0", "0"])
    cases.append(["delta", "-999999999", "-1", "0", "0", "0", "0", "0"])
    cases.append(["deltamul", "999999999", "0", "0", "2"])
    cases.append(["deltamul", "0", "0", "1", "1000000"])
    for micros in ["86400000000", "-86400000000", "0", "86399999999", "-86399999999",
                   "19800000000", "-19845000000", "1", "-1", "86400000001"]:
        cases.append(["zone", micros])
    for seconds in ["253402300799.0", "253402300800.0", "-62135596800.0",
                    "-62135596801.0", "0.0", "-0.0", "1e-06", "-1e-06",
                    "0.4999995", "-0.4999995", "1.9999995"]:
        cases.append(["stamp", seconds])

    while len(cases) < count:
        pick = rng.random()
        if pick < 0.16:
            cases.append(["date", str(ordinal())])
        elif pick < 0.28:
            days, seconds, micros = span_parts()
            cases.append(
                [
                    "delta",
                    str(days),
                    str(seconds),
                    str(micros),
                    str(rng.randint(-5000, 5000)),
                    str(rng.randint(-5000, 5000)),
                    str(rng.randint(-5000, 5000)),
                    str(rng.randint(-500, 500)),
                ]
            )
        elif pick < 0.33:
            days, seconds, micros = span_parts()
            cases.append(
                ["deltamul", str(days), str(seconds), str(micros), str(rng.randint(-1000, 1000))]
            )
        elif pick < 0.40:
            parts = span_parts() + span_parts()
            cases.append(["deltapair"] + [str(part) for part in parts])
        elif pick < 0.62:
            cases.append(["moment"] + moment_args())
        elif pick < 0.68:
            hour, minute, second, micro = clock()
            tz = offset_micros()
            cases.append(
                [
                    "timeobj",
                    str(hour),
                    str(minute),
                    str(second),
                    str(micro),
                    "-" if tz is None else str(tz),
                ]
            )
        elif pick < 0.74:
            days, seconds, micros = span_parts()
            cases.append(["momentadd"] + moment_args() + [str(days), str(seconds), str(micros)])
        elif pick < 0.79:
            cases.append(["momentdiff"] + moment_args() + moment_args())
        elif pick < 0.84:
            cases.append(
                [
                    "datediff",
                    str(ordinal()),
                    str(ordinal()),
                    str(rng.randint(-4_000_000, 4_000_000)),
                ]
            )
        elif pick < 0.95:
            hour, minute, second, micro = clock()
            moment = moment_of(ordinal(), hour, minute, second, micro, offset_micros())
            pattern = rng.choice(ROUND_TRIP_FORMATS)
            cases.append(["strptime", moment.strftime(pattern), pattern])
        else:
            seconds = rng.choice(
                [
                    0.0,
                    -1.0,
                    1.5,
                    -1.5,
                    0.0000005,
                    -0.0000005,
                    1704110400.5,
                    rng.uniform(-62_135_000_000.0, 253_402_000_000.0),
                    float(rng.randint(-62_135_000_000, 253_402_000_000)),
                ]
            )
            cases.append(["stamp", repr(seconds)])
    return cases


# --------------------------------------------------------------- the driver


DRIVER = r'''
from std.datetime import date, datetime, time, timedelta, timezone

from std.io import read_file
from std.str import to_float

const FULL = "%Y|%y|%m|%d|%j|%H|%I|%M|%S|%f|%p|%a|%A|%b|%B|%w|%u|%U|%W|%G|%V|%z|%Z|%%"

let TIMESPECS = ["auto", "hours", "minutes", "seconds", "milliseconds", "microseconds"]

fn safe(producer: fn() -> str) -> str {
    try {
        return producer()
    } catch e: Error {
        return "!" + type_of(e)
    }
}

fn flag(value: bool) -> str { return if value { "1" } else { "0" } }

fn triple(value: tuple[int, int, int]) -> str {
    return f"{value[0]}-{value[1]}-{value[2]}"
}

fn zone_of(text: str) -> timezone? {
    if text == "-" { return null }
    return timezone(timedelta(microseconds: int(text)))
}

fn moment_of(ordinal: int, hour: int, minute: int, second: int, micro: int,
             tz: str) -> datetime {
    let zone: any = zone_of(tz)
    return datetime.combine(date.fromordinal(ordinal),
        time(hour, minute, second, micro), zone)
}

fn local_readable(moment: datetime) -> bool {
    return moment.tzinfo is not null or (moment.year > 1 and moment.year < 9999)
}

fn local_stamp_readable(seconds: float) -> bool {
    return seconds >= -62135510400.0 and seconds <= 253402214399.0
}

fn date_case(args: list[str]) -> list[str] {
    let day = date.fromordinal(int(args[0]))
    return [
        safe(|| day.isoformat()),
        safe(|| str(day.toordinal())),
        safe(|| str(day.weekday())),
        safe(|| str(day.isoweekday())),
        safe(|| triple(day.isocalendar())),
        safe(|| day.ctime()),
        safe(|| repr(day)),
        safe(|| str(day)),
        safe(|| day.strftime(FULL)),
        safe(|| repr(date.fromisoformat(day.isoformat()))),
        safe(|| repr(day.replace(day: 1))),
        safe(|| repr(day.replace(month: 1, day: 1))),
    ]
}

fn delta_case(args: list[str]) -> list[str] {
    let span = timedelta(
        days: int(args[0]),
        seconds: int(args[1]),
        microseconds: int(args[2]),
        milliseconds: int(args[3]),
        minutes: int(args[4]),
        hours: int(args[5]),
        weeks: int(args[6]),
    )
    return [
        safe(|| repr(span)),
        safe(|| str(span)),
        safe(|| str(span.total_seconds())),
        safe(|| str(span.days)),
        safe(|| str(span.seconds)),
        safe(|| str(span.microseconds)),
        safe(|| repr(-span)),
        safe(|| repr(span.absolute())),
    ]
}

fn moment_case(args: list[str]) -> list[str] {
    let moment = moment_of(int(args[0]), int(args[1]), int(args[2]), int(args[3]),
        int(args[4]), args[5])
    var fields = [
        safe(|| repr(moment)),
        safe(|| str(moment)),
        safe(|| moment.ctime()),
        safe(|| str(moment.toordinal())),
        safe(|| str(moment.weekday())),
        safe(|| triple(moment.isocalendar())),
        safe(|| moment.strftime(FULL)),
        safe(|| repr(moment.date())),
        safe(|| repr(moment.time())),
        safe(|| repr(moment.timetz())),
    ]
    for spec in TIMESPECS {
        fields.push(safe(|| moment.isoformat(timespec: spec)))
        fields.push(safe(|| moment.isoformat(separator: " ", timespec: spec)))
    }
    fields.push(safe(|| repr(datetime.fromisoformat(moment.isoformat()))))
    fields.push(safe(|| repr(datetime.fromisoformat(moment.isoformat(separator: " ")))))
    if moment.tzinfo is null {
        fields.push("-")
        fields.push("-")
    } else {
        fields.push(safe(|| repr(moment.utcoffset() ?? timedelta())))
        fields.push(safe(|| moment.tzname() ?? ""))
    }
    if local_readable(moment) {
        fields.push(safe(|| str(moment.timestamp())))
        fields.push(safe(|| repr(datetime.fromtimestamp(moment.timestamp(), timezone.utc))))
        fields.push(safe(|| repr(moment.astimezone(
            timezone(timedelta(hours: 5, minutes: 30))))))
        fields.push(safe(|| repr(moment.astimezone(timezone.utc))))
    } else {
        fields.push("-")
        fields.push("-")
        fields.push("-")
        fields.push("-")
    }
    return fields
}

fn time_case(args: list[str]) -> list[str] {
    let value = time(int(args[0]), int(args[1]), int(args[2]), int(args[3]),
        zone_of(args[4]))
    var fields = [safe(|| repr(value)), safe(|| str(value)), safe(|| value.strftime(FULL))]
    for spec in TIMESPECS { fields.push(safe(|| value.isoformat(timespec: spec))) }
    fields.push(safe(|| repr(time.fromisoformat(value.isoformat()))))
    return fields
}

fn run_case(kind: str, args: list[str]) -> list[str] {
    if kind == "date" { return date_case(args) }
    if kind == "delta" { return delta_case(args) }
    if kind == "moment" { return moment_case(args) }
    if kind == "timeobj" { return time_case(args) }
    if kind == "deltamul" {
        let span = timedelta(days: int(args[0]), seconds: int(args[1]),
            microseconds: int(args[2]))
        let factor = int(args[3])
        return [safe(|| repr(span * factor))]
    }
    if kind == "deltapair" {
        let left = timedelta(days: int(args[0]), seconds: int(args[1]),
            microseconds: int(args[2]))
        let right = timedelta(days: int(args[3]), seconds: int(args[4]),
            microseconds: int(args[5]))
        return [
            safe(|| repr(left + right)),
            safe(|| repr(left - right)),
            safe(|| flag(left < right)),
            safe(|| flag(left <= right)),
            safe(|| flag(left == right)),
        ]
    }
    if kind == "momentadd" {
        let moment = moment_of(int(args[0]), int(args[1]), int(args[2]), int(args[3]),
            int(args[4]), args[5])
        let span = timedelta(days: int(args[6]), seconds: int(args[7]),
            microseconds: int(args[8]))
        return [safe(|| repr(moment + span)), safe(|| repr(moment - span))]
    }
    if kind == "momentdiff" {
        let left = moment_of(int(args[0]), int(args[1]), int(args[2]), int(args[3]),
            int(args[4]), args[5])
        let right = moment_of(int(args[6]), int(args[7]), int(args[8]), int(args[9]),
            int(args[10]), args[11])
        return [
            safe(|| repr(left - right)),
            safe(|| flag(left < right)),
            safe(|| flag(left <= right)),
            safe(|| flag(left == right)),
        ]
    }
    if kind == "datediff" {
        let left = date.fromordinal(int(args[0]))
        let right = date.fromordinal(int(args[1]))
        let span = timedelta(days: int(args[2]))
        return [
            safe(|| repr(left - right)),
            safe(|| repr(left + span)),
            safe(|| repr(left - span)),
            safe(|| flag(left < right)),
            safe(|| flag(left == right)),
        ]
    }
    if kind == "strptime" { return [safe(|| repr(datetime.strptime(args[0], args[1])))] }
    if kind == "isodate" { return [safe(|| repr(date.fromisoformat(args[0])))] }
    if kind == "isomoment" { return [safe(|| repr(datetime.fromisoformat(args[0])))] }
    if kind == "isotime" { return [safe(|| repr(time.fromisoformat(args[0])))] }
    if kind == "stamp" {
        let seconds = to_float(args[0]) ?? 0.0
        return [
            safe(|| repr(datetime.fromtimestamp(seconds, timezone.utc))),
            if local_stamp_readable(seconds) {
                safe(|| repr(datetime.fromtimestamp(seconds)))
            } else {
                "-"
            },
            safe(|| str(datetime.fromtimestamp(seconds, timezone.utc).timestamp())),
        ]
    }
    if kind == "ctor" {
        let parts = [int(part) for part in args[0:7]]
        return [
            safe(|| repr(datetime(parts[0], parts[1], parts[2], parts[3], parts[4],
                parts[5], parts[6]))),
            safe(|| repr(date(parts[0], parts[1], parts[2]))),
            safe(|| repr(time(parts[3], parts[4], parts[5], parts[6]))),
        ]
    }
    if kind == "zone" {
        let micros = int(args[0])
        return [
            safe(|| repr(timezone(timedelta(microseconds: micros)))),
            safe(|| str(timezone(timedelta(microseconds: micros)))),
            safe(|| timezone(timedelta(microseconds: micros)).tzname()),
            safe(|| repr(timezone(timedelta(microseconds: micros)).utcoffset())),
        ]
    }
    throw ValueError(f"unknown case kind {kind}")
}

fn main() -> int {
    let lines = read_file("__CASES__").split("\n")
    var index = 0
    for line in lines {
        if line.len() == 0 { continue }
        let parts = line.split("\t")
        var fields: list[str] = []
        try {
            fields = run_case(parts[0], parts[1:parts.len()])
        } catch e: Error {
            fields = ["!" + type_of(e)]
        }
        print(str(index) + "\t" + "\t".join(fields))
        index += 1
    }
    return 0
}
'''


def jaithon_lines(cases: list[list[str]], work: Path) -> list[str]:
    case_file = work / "datetime_cases.tsv"
    case_file.write_text("".join("\t".join(case) + "\n" for case in cases), encoding="utf-8")
    driver = work / "datetime_diff_driver.jai"
    driver.write_text(DRIVER.replace("__CASES__", str(case_file)), encoding="utf-8")
    proc = subprocess.run(
        [str(JAITHON), "run", str(driver)],
        cwd=ROOT,
        capture_output=True,
        env={**os.environ, "JAITHON_NO_COLOR": "1"},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        sys.stderr.write(proc.stdout.decode("utf-8", "replace")[-4000:])
        raise SystemExit(f"jaithon exited {proc.returncode}")
    return [line for line in proc.stdout.decode("utf-8").split("\n") if line]


def report(index: int, case: list[str], want: str, got: str) -> None:
    sys.stderr.write("case %d: %s\n" % (index, "\t".join(case)))
    left_fields = want.split("\t")
    right_fields = got.split("\t")
    if len(left_fields) != len(right_fields):
        sys.stderr.write(f"  cpython line: {want!r}\n")
        sys.stderr.write(f"  jaithon line: {got!r}\n")
        return
    for column, (left, right) in enumerate(zip(left_fields, right_fields)):
        if left != right:
            sys.stderr.write(f"  field {column}\n")
            sys.stderr.write(f"    cpython: {left!r}\n")
            sys.stderr.write(f"    jaithon: {right!r}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="differential test for std.datetime")
    parser.add_argument("--cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--show", type=int, default=12, help="how many differences to print")
    options = parser.parse_args()

    if not JAITHON.exists():
        sys.stderr.write(f"error: {JAITHON} not built; run 'make' first\n")
        return 1

    cases = generate(random.Random(options.seed), options.cases)
    expected = python_lines(cases)

    with tempfile.TemporaryDirectory(prefix="jaithon-datetime-") as directory:
        actual = jaithon_lines(cases, Path(directory))

    if len(actual) != len(expected):
        sys.stderr.write(
            f"error: jaithon produced {len(actual)} lines, cpython {len(expected)}\n"
        )
        return 1

    failures = 0
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want == got:
            continue
        failures += 1
        if failures <= options.show:
            report(index, cases[index], want, got)

    print(f"{len(cases)} cases, {len(cases) - failures} agreed, {failures} differed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
