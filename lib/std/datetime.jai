#: Dates, times and the arithmetic between them — CPython's `datetime`.
#:
#: Five types, with CPython's names, fields and semantics: `timedelta` for a
#: span, `date` for a calendar day, `time` for a time of day, `datetime` for a
#: moment, and `timezone` for a fixed offset. The calendar is proleptic
#: Gregorian over `MINYEAR..MAXYEAR`, and the ordinal is CPython's, so
#: `date.fromordinal(1)` is 0001-01-01 and `date(9999, 12, 31).toordinal()` is
#: 3652059.
#:
#: ```
#: date(2024, 2, 29).isoformat()                      # "2024-02-29"
#: datetime.fromisoformat("2024-02-29T12:00:00Z")     # aware, UTC
#: datetime(2024, 3, 1) - datetime(2024, 2, 29)       # timedelta(days=1)
#: datetime(2024, 2, 29).strftime("%A %j")            # "Thursday 060"
#: datetime.strptime("29/Feb/2024", "%d/%b/%Y").day   # 29
#: ```
#:
#: Use `std.time` for elapsed time. This module is a calendar; `Instant` and
#: `Stopwatch` there read a monotonic clock, and the wall clock this module
#: reads can jump backwards. `date.today` and `datetime.now` are the only
#: clock readers here and both go through `std.time.DateTime.now`, as
#: `is_leap_year` and `days_in_month` are `std.time`'s.
#:
#: ## Where this differs from CPython 3.13
#:
#: **No local time zone.** There is no zone database to consult, so every
#: place CPython reads the local zone this reads UTC: `date.today`,
#: `datetime.now()` and `datetime.today()` give the UTC wall clock,
#: `datetime.fromtimestamp(t)` with no zone gives naive UTC, and
#: `datetime.timestamp()` reads a naive value as UTC. `astimezone` needs its
#: argument, since "the local zone" has no meaning here. A value that carries
#: a `timezone` behaves exactly as CPython's does.
#:
#: **Integer components only.** `timedelta(days: 1.5)` is a type error;
#: CPython would accept it. Write `timedelta(hours: 36)`.
#:
#: **No division on `timedelta`.** `+`, `-`, unary `-`, `absolute()` and
#: multiplication by an integer are here; `/`, `//`, `%` and `divmod` are not.
#: A span's microsecond total reaches 8.6e19, past what a 64-bit integer
#: holds, and CPython's exact-rational division of those totals would need
#: bignum arithmetic at every call. `total_seconds()` gives the float a ratio
#: usually wants.
#:
#: **`abs` is `absolute()`.** Jaithon's `abs` is a numeric builtin with no
#: hook a class can answer. `-span` and `span * 3` are operators as usual.
#:
#: **`replace` keeps `tzinfo` unless told otherwise.** CPython distinguishes
#: "argument omitted" from "argument is None" with a private sentinel;
#: `replace` here takes `keep_tzinfo: false` alongside `tzinfo` to install a
#: new one, which may be null.
#:
#: **No `tzinfo` base class, no `fold`, no `timetuple`, no `%c`/`%x`/`%X`.**
#: `timezone` is the only zone type, ambiguous local times cannot arise
#: without a local zone, `struct_time` has no counterpart here, and the three
#: locale-dependent specifiers would differ by machine. Every other
#: `strftime` specifier CPython documents as portable is supported, and
#: `strptime` reads `%Y %y %m %d %H %I %M %S %f %j %U %W %w %a %A %b %B %p %z
#: %Z %%`.
#:
#: **Reverse operands.** `timedelta + date` and `int * timedelta` have no
#: hook; write the class on the left.
#:
#: **A timestamp far outside the calendar raises `ValueError`.** CPython hands
#: such a value to the platform's `gmtime` and surfaces its `OSError`; the
#: range is checked here before the conversion, so the error names the year
#: rather than an errno.
#:
#: Facade: imports and re-exports the public surface of every module in
#: `datetime/`.
from .datetime.common import MAXYEAR, MINYEAR
from .datetime.dates import date
from .datetime.delta import timedelta
from .datetime.moment import datetime
from .datetime.times import time
from .datetime.tz import UTC, timezone

export { MAXYEAR, MINYEAR, UTC, date, datetime, time, timedelta, timezone }
