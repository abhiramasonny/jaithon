#: The three calendar values a TOML document can hold, shaped like Python's
#: `datetime.date`, `datetime.time` and `datetime.datetime`.
#:
#: `tomllib` returns those three classes, and a TOML writer has to tell them
#: apart from strings to print them unquoted. `std.time.DateTime` cannot stand
#: in: it is always a moment with an offset, so it has nowhere to put a local
#: date-time, a date without a time, or a time without a date. These three
#: carry exactly what the TOML grammar wrote.
#:
#: ```
#: Date(1979, 5, 27).to_iso()                       # "1979-05-27"
#: Time(7, 32, 0, 999999).to_iso()                  # "07:32:00.999999"
#: DateTime(1979, 5, 27, 7, 32, 0, 0, 0).to_iso()   # "1979-05-27T07:32:00+00:00"
#: DateTime(1979, 5, 27, 7, 32).to_iso()            # "1979-05-27T07:32:00"
#: ```
#:
#: Fractional seconds are microseconds, as in Python: `tomllib` keeps the first
#: six fractional digits and drops the rest, and so does the decoder here.
#: `to_iso` is Python's `isoformat()` and `to_str` is Python's `str()`; they
#: differ only on a `DateTime`, where `str` puts a space between date and time.
#: An offset is minutes east of UTC, and null on a `DateTime` means local: the
#: document named no offset, which is what Python's `tzinfo=None` records.
from std.core import Comparable, Printable
from std.time import DateTime as StdDateTime
from std.time import days_in_month

#: The earliest year, as in Python's `datetime.MINYEAR`.
pub const MIN_YEAR: int = 1

#: The latest year, as in Python's `datetime.MAXYEAR`.
pub const MAX_YEAR: int = 9999

#: A calendar date without a time, like `datetime.date`.
pub class Date: Comparable, Printable {
    #: The year, 1 to 9999.
    pub let year: int

    #: The month, 1 to 12.
    pub let month: int

    #: The day of the month, 1 to the month's length.
    pub let day: int

    #: Create a date, which must exist in the proleptic Gregorian calendar.
    #:
    #: Raises `ValueError` for a year outside 1 to 9999, a month outside 1 to
    #: 12, or a day the month does not have.
    pub fn init(self, year: int, month: int, day: int) {
        _check_date(year, month, day)
        self.year = year
        self.month = month
        self.day = day
    }

    #: Render as `YYYY-MM-DD`, Python's `isoformat()`.
    pub fn to_iso(self) -> str {
        return _pad(self.year, 4) + "-" + _pad(self.month, 2) + "-" + _pad(self.day, 2)
    }

    #: Render as `YYYY-MM-DD`, Python's `str()`, which is the same text.
    pub fn to_str(self) -> str { return self.to_iso() }

    #: Order two dates chronologically.
    pub fn compare(self, other: Date) -> int {
        return _compare_fields(
            [self.year, self.month, self.day],
            [other.year, other.month, other.day]
        )
    }

    fn __eq__(self, other: any) -> bool {
        return isinstance(other, Date) and self.compare(other) == 0
    }

    fn __str__(self) -> str { return self.to_iso() }

    fn __repr__(self) -> str { return f"Date({self.year}, {self.month}, {self.day})" }
}

#: A time of day without a date or an offset, like a naive `datetime.time`.
pub class Time: Comparable, Printable {
    #: The hour, 0 to 23.
    pub let hour: int

    #: The minute, 0 to 59.
    pub let minute: int

    #: The second, 0 to 59.
    pub let second: int

    #: The microsecond, 0 to 999999.
    pub let microsecond: int

    #: Create a time of day.
    #:
    #: Raises `ValueError` when a field is out of range.
    pub fn init(self, hour: int, minute: int = 0, second: int = 0, microsecond: int = 0) {
        _check_time(hour, minute, second, microsecond)
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
    }

    #: Render as `HH:MM:SS`, with `.ffffff` appended when the microsecond is
    #: not zero, which is Python's `isoformat()`.
    pub fn to_iso(self) -> str {
        return _time_text(self.hour, self.minute, self.second, self.microsecond)
    }

    #: Render as `HH:MM:SS[.ffffff]`, Python's `str()`, which is the same text.
    pub fn to_str(self) -> str { return self.to_iso() }

    #: Order two times of day.
    pub fn compare(self, other: Time) -> int {
        return _compare_fields(
            [self.hour, self.minute, self.second, self.microsecond],
            [other.hour, other.minute, other.second, other.microsecond]
        )
    }

    fn __eq__(self, other: any) -> bool {
        return isinstance(other, Time) and self.compare(other) == 0
    }

    fn __str__(self) -> str { return self.to_iso() }

    fn __repr__(self) -> str {
        return f"Time({self.hour}, {self.minute}, {self.second}, {self.microsecond})"
    }
}

#: A date and a time, with or without an offset, like `datetime.datetime`.
#:
#: `offset_minutes` is minutes east of UTC. Null means the value is local:
#: the document wrote no offset, and Python would hold `tzinfo=None`. Two
#: offset date-times are equal when they name the same instant; an offset
#: date-time never equals a local one, and comparing the two raises
#: `TypeError`, both as in Python.
pub class DateTime: Comparable, Printable {
    #: The year, 1 to 9999.
    pub let year: int

    #: The month, 1 to 12.
    pub let month: int

    #: The day of the month.
    pub let day: int

    #: The hour, 0 to 23.
    pub let hour: int

    #: The minute, 0 to 59.
    pub let minute: int

    #: The second, 0 to 59.
    pub let second: int

    #: The microsecond, 0 to 999999.
    pub let microsecond: int

    #: Minutes east of UTC, or null for a local date-time.
    pub let offset_minutes: int?

    #: Create a date-time. Omit `offset_minutes` for a local one.
    #:
    #: Raises `ValueError` when a field is out of range or the offset is a day
    #: or more in magnitude.
    pub fn init(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        offset_minutes: int? = null
    ) {
        _check_date(year, month, day)
        _check_time(hour, minute, second, microsecond)
        if offset_minutes is not null {
            let magnitude = offset_minutes < 0 ? -offset_minutes : offset_minutes
            if magnitude >= 24 * 60 {
                throw ValueError(f"offset must be less than a day, got {offset_minutes} minutes")
            }
        }
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.offset_minutes = offset_minutes
    }

    #: Whether the value carries an offset, Python's `tzinfo is not None`.
    pub fn is_aware(self) -> bool { return self.offset_minutes is not null }

    #: The date part alone.
    pub fn date(self) -> Date { return Date(self.year, self.month, self.day) }

    #: The time part alone, without the offset.
    pub fn time(self) -> Time {
        return Time(self.hour, self.minute, self.second, self.microsecond)
    }

    #: Render as `YYYY-MM-DDTHH:MM:SS[.ffffff][+HH:MM]`, Python's `isoformat()`.
    #:
    #: A UTC value renders its offset as `+00:00`, never `Z`, which is what
    #: Python prints and therefore what `tomli_w` writes.
    pub fn to_iso(self) -> str { return self._render("T") }

    #: Render like `to_iso` with a space between the date and the time, which
    #: is Python's `str()` and the form `tomli_w` writes.
    pub fn to_str(self) -> str { return self._render(" ") }

    #: Convert an offset date-time to `std.time.DateTime`.
    #:
    #: Raises `ValueError` on a local date-time, which names no instant.
    pub fn to_std(self) -> StdDateTime {
        let offset = self.offset_minutes
        if offset is null {
            throw ValueError("a local date-time names no instant; it has no offset")
        }
        return StdDateTime.of(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond * 1000,
            offset
        )
    }

    #: Order two date-times, offset ones by instant.
    #:
    #: Raises `TypeError` when one has an offset and the other does not.
    pub fn compare(self, other: DateTime) -> int {
        if self.is_aware() != other.is_aware() {
            throw TypeError("can't compare offset-naive and offset-aware datetimes")
        }
        return _compare_fields(self._instant_fields(), other._instant_fields())
    }

    fn _instant_fields(self) -> list[int] {
        let offset = self.offset_minutes ?? 0
        let total = _days_from_civil(self.year, self.month, self.day) * 1440 + self.hour * 60 + self.minute -
            offset
        return [total, self.second, self.microsecond]
    }

    fn _render(self, separator: str) -> str {
        var text = self.date().to_iso() +
            separator +
            _time_text(self.hour, self.minute, self.second, self.microsecond)
        let offset = self.offset_minutes
        if offset is null { return text }
        let sign = offset < 0 ? "-" : "+"
        let magnitude = offset < 0 ? -offset : offset
        return text + sign + _pad(magnitude // 60, 2) + ":" + _pad(magnitude % 60, 2)
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, DateTime) { return false }
        if self.is_aware() != other.is_aware() { return false }
        return self.compare(other) == 0
    }

    fn __str__(self) -> str { return self.to_str() }

    fn __repr__(self) -> str {
        let offset = self.offset_minutes
        let tail = offset is null ? "" : f", {offset}"
        return f"DateTime({self.year}, {self.month}, {self.day}, {self.hour}, " +
            f"{self.minute}, {self.second}, {self.microsecond}{tail})"
    }
}

fn _check_date(year: int, month: int, day: int) -> void {
    if year < MIN_YEAR or year > MAX_YEAR {
        throw ValueError(f"year {year} is out of range")
    }
    if month < 1 or month > 12 { throw ValueError("month must be in 1..12") }
    if day < 1 or day > days_in_month(year, month) {
        throw ValueError("day is out of range for month")
    }
}

fn _check_time(hour: int, minute: int, second: int, microsecond: int) -> void {
    if hour < 0 or hour > 23 { throw ValueError("hour must be in 0..23") }
    if minute < 0 or minute > 59 { throw ValueError("minute must be in 0..59") }
    if second < 0 or second > 59 { throw ValueError("second must be in 0..59") }
    if microsecond < 0 or microsecond > 999999 {
        throw ValueError("microsecond must be in 0..999999")
    }
}

fn _time_text(hour: int, minute: int, second: int, microsecond: int) -> str {
    let base = _pad(hour, 2) + ":" + _pad(minute, 2) + ":" + _pad(second, 2)
    if microsecond == 0 { return base }
    return base + "." + _pad(microsecond, 6)
}

fn _compare_fields(left: list[int], right: list[int]) -> int {
    for (index, value) in left.enumerate() {
        if value < right[index] { return -1 }
        if value > right[index] { return 1 }
    }
    return 0
}

fn _pad(value: int, width: int) -> str {
    var text = str(value)
    while text.len() < width { text = "0" + text }
    return text
}

fn _days_from_civil(year: int, month: int, day: int) -> int {
    let shifted = month <= 2 ? year - 1 : year
    let era = (shifted >= 0 ? shifted : shifted - 399) // 400
    let year_of_era = shifted - era * 400
    let shifted_month = month > 2 ? month - 3 : month + 9
    let day_of_year = (153 * shifted_month + 2) // 5 + day - 1
    let day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468
}
