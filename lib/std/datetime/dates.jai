#: `date`, a proleptic Gregorian calendar day with no time and no zone.
#:
#: Ordinals are CPython's: 0001-01-01 is 1 and 9999-12-31 is 3652059, and
#: `fromordinal(d.toordinal())` is `d` for every day in between.
from .common import MAXORDINAL, MAXYEAR, MINYEAR, MONTH_NAMES
from .common import check_date, digits_at, iso_week1_monday, ordinal_to_ymd, pad
from .common import weekday_of, ymd_to_ordinal
from .delta import timedelta
from .fmt import format_fields, iso_calendar
from std.time import DateTime

let _KEEP = -1

#: A calendar day between 0001-01-01 and 9999-12-31.
#:
#: ```
#: date(2024, 2, 29).toordinal()      # 738945
#: date(2024, 2, 29).isoformat()      # "2024-02-29"
#: date(2024, 2, 29).weekday()        # 3
#: str(date(2024, 3, 1) - date(2024, 2, 29))    # "1 day, 0:00:00"
#: ```
pub class date {
    #: The year, 1 through 9999.
    pub let year: int

    #: The month, 1 for January through 12 for December.
    pub let month: int

    #: The day of the month, from 1.
    pub let day: int

    #: The earliest day, 0001-01-01.
    pub static var min: date

    #: The latest day, 9999-12-31.
    pub static var max: date

    #: The smallest difference between two unequal days, one day.
    pub static var resolution: timedelta

    #: Create the day `year`-`month`-`day`, which must be a real date.
    #:
    #: ```
    #: date(2024, 2, 29).day    # 29
    #: ```
    pub fn init(self, year: int, month: int, day: int) {
        check_date(year, month, day)
        self.year = year
        self.month = month
        self.day = day
    }

    #: Return today's date, read from the wall clock as UTC.
    #:
    #: CPython reads the local zone here. There is no zone database to consult
    #: — `std.time.DateTime.now()` is UTC and so is this.
    pub static fn today() -> date {
        let now = DateTime.now()
        return date(now.year, now.month, now.day)
    }

    #: Return the day whose proleptic Gregorian ordinal is `ordinal`.
    #:
    #: ```
    #: date.fromordinal(1).isoformat()    # "0001-01-01"
    #: ```
    pub static fn fromordinal(ordinal: int) -> date {
        if ordinal < 1 or ordinal > MAXORDINAL {
            throw ValueError(f"ordinal must be in 1..{MAXORDINAL}, not {ordinal}")
        }
        let (year, month, day) = ordinal_to_ymd(ordinal)
        return date(year, month, day)
    }

    #: Return the UTC day `timestamp` seconds after the Unix epoch.
    #:
    #: CPython reads the local zone here; this is UTC, as `today` is.
    pub static fn fromtimestamp(timestamp: float) -> date {
        let days = _floor_div_float(timestamp, 86_400.0)
        return date.fromordinal(days + 719_163)
    }

    #: Parse an ISO 8601 calendar or week date.
    #:
    #: Accepts `YYYY-MM-DD`, `YYYYMMDD`, `YYYY-Www-D` and `YYYYWwwD`, and the
    #: week forms with the day left off. Ordinal dates (`YYYY-DDD`) are
    #: rejected, exactly as CPython rejects them.
    #:
    #: ```
    #: date.fromisoformat("2024-02-29").month     # 2
    #: date.fromisoformat("2024-W09-4").day       # 29
    #: ```
    pub static fn fromisoformat(text: str) -> date {
        let parsed = parse_iso_date(text)
        if parsed is null { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        let (year, month, day, consumed) = parsed ?? (0, 0, 0, 0)
        if consumed != text.len() { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        return date(year, month, day)
    }

    #: Return the day's proleptic Gregorian ordinal.
    pub fn toordinal(self) -> int { return ymd_to_ordinal(self.year, self.month, self.day) }

    #: Return the weekday, 0 for Monday through 6 for Sunday.
    pub fn weekday(self) -> int { return weekday_of(self.toordinal()) }

    #: Return the weekday, 1 for Monday through 7 for Sunday.
    pub fn isoweekday(self) -> int { return weekday_of(self.toordinal()) + 1 }

    #: Return `(iso_year, iso_week, iso_weekday)` for this day.
    #:
    #: ```
    #: date(2023, 1, 1).isocalendar()    # (2022, 52, 7)
    #: ```
    pub fn isocalendar(self) -> tuple[int, int, int] {
        return iso_calendar(self.year, self.month, self.day)
    }

    #: Render the day as `YYYY-MM-DD`.
    pub fn isoformat(self) -> str {
        return pad(self.year, 4) + "-" + pad(self.month, 2) + "-" + pad(self.day, 2)
    }

    #: Render the day the way C's `ctime` does, at midnight.
    #:
    #: ```
    #: date(2024, 3, 4).ctime()    # "Mon Mar  4 00:00:00 2024"
    #: ```
    pub fn ctime(self) -> str {
        return ctime_of(self.year, self.month, self.day, 0, 0, 0)
    }

    #: Return a copy with the named fields replaced.
    #:
    #: ```
    #: date(2024, 1, 2).replace(month: 3).month    # 3
    #: ```
    pub fn replace(self, year: int = _KEEP, month: int = _KEEP, day: int = _KEEP) -> date {
        return date(if year == _KEEP {
            self.year
        } else {
            year
        }, if month == _KEEP {
            self.month
        } else {
            month
        }, if day == _KEEP {
            self.day
        } else {
            day
        })
    }

    #: Render the day with strftime-style specifiers.
    #:
    #: The time fields read as midnight and `%z` and `%Z` are empty, which is
    #: what CPython does for a `date`.
    #:
    #: ```
    #: date(2024, 2, 29).strftime("%A %d %B %Y")    # "Thursday 29 February 2024"
    #: ```
    pub fn strftime(self, pattern: str) -> str {
        return format_fields(pattern, self.year, self.month, self.day, 0, 0, 0, 0, null, null)
    }

    #: Order two days, returning a negative int, zero, or a positive int.
    pub fn compare_to(self, other: date) -> int {
        let mine = self.toordinal()
        let theirs = other.toordinal()
        if mine == theirs { return 0 }
        return if mine < theirs { -1 } else { 1 }
    }

    fn __add__(self, span: timedelta) -> date {
        return date.fromordinal(_shifted(self.toordinal(), span.days))
    }

    fn __sub__(self, other: any) -> any {
        if isinstance(other, timedelta) {
            return date.fromordinal(_shifted(self.toordinal(), -other.days))
        }
        if isinstance(other, date) { return timedelta(self.toordinal() - other.toordinal()) }
        throw TypeError(f"cannot subtract {type_of(other)} from a date")
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, date) { return false }
        return self.year == other.year and self.month == other.month and self.day == other.day
    }

    fn __lt__(self, other: date) -> bool { return self.compare_to(other) < 0 }

    fn __le__(self, other: date) -> bool { return self.compare_to(other) <= 0 }

    fn __hash__(self) -> int { return hash(self.toordinal()) }

    fn __str__(self) -> str { return self.isoformat() }

    fn __repr__(self) -> str {
        return f"datetime.date({self.year}, {self.month}, {self.day})"
    }
}

#: Render calendar fields the way C's `ctime` does.
#:
#: The day of the month is space padded, not zero padded, which is the one
#: place in the module where a field is not `%02d`.
#:
#: ```
#: ctime_of(2024, 3, 4, 5, 6, 7)    # "Mon Mar  4 05:06:07 2024"
#: ```
pub fn ctime_of(year: int, month: int, day: int, hour: int, minute: int, second: int) -> str {
    let ordinal = ymd_to_ordinal(year, month, day)
    let weekday = _CTIME_DAYS[ordinal % 7]
    let padded_day = if day < 10 { " " + str(day) } else { str(day) }
    return f"{weekday} {MONTH_NAMES[month][0:3]} {padded_day} " +
        f"{pad(hour, 2)}:{pad(minute, 2)}:{pad(second, 2)} {pad(year, 4)}"
}

#: Read an ISO 8601 date off the front of `text`.
#:
#: Returns `(year, month, day, characters_consumed)`, or null when the text
#: does not open with a date. `datetime.fromisoformat` needs the prefix form,
#: since the time and the offset follow it in the same string.
#:
#: ```
#: parse_iso_date("2024-02-29T12:00")    # (2024, 2, 29, 10)
#: ```
pub fn parse_iso_date(text: str) -> tuple[int, int, int, int]? {
    let year = digits_at(text, 0, 4)
    if year is null { return null }
    let calendar_year = year ?? 0
    if calendar_year < MINYEAR or calendar_year > MAXYEAR { return null }
    var at = 4
    var separated = false
    if at < text.len() and text[at] == "-" {
        separated = true
        at += 1
    }
    if at < text.len() and (text[at] == "W" or text[at] == "w") {
        at += 1
        let week = digits_at(text, at, 2)
        if week is null { return null }
        at += 2
        var weekday = 1
        if separated and at < text.len() and text[at] == "-" { at += 1 }
        let named = digits_at(text, at, 1)
        if named is not null {
            weekday = named ?? 1
            at += 1
        }
        let week_number = week ?? 0
        if week_number < 1 or week_number > 53 or weekday < 1 or weekday > 7 { return null }
        let ordinal = iso_week1_monday(calendar_year) + (week_number - 1) * 7 + weekday - 1
        if ordinal < 1 or ordinal > MAXORDINAL { return null }
        let (found_year, found_month, found_day) = ordinal_to_ymd(ordinal)
        if iso_calendar(found_year, found_month, found_day)[0] != calendar_year { return null }
        return (found_year, found_month, found_day, at)
    }
    let month = digits_at(text, at, 2)
    if month is null { return null }
    at += 2
    if separated {
        if at >= text.len() or text[at] != "-" { return null }
        at += 1
    }
    let day = digits_at(text, at, 2)
    if day is null { return null }
    at += 2
    let calendar_month = month ?? 0
    let calendar_day = day ?? 0
    if calendar_month < 1 or calendar_month > 12 { return null }
    if calendar_day < 1 or calendar_day > 31 { return null }
    return (calendar_year, calendar_month, calendar_day, at)
}

let _CTIME_DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

fn _shifted(ordinal: int, days: int) -> int {
    let moved = ordinal + days
    if moved < 1 or moved > MAXORDINAL { throw OverflowError("date value out of range") }
    return moved
}

fn _floor_div_float(value: float, divisor: float) -> int {
    let quotient = value / divisor
    let truncated = int(quotient)
    if float(truncated) > quotient { return truncated - 1 }
    return truncated
}

date.min = date(MINYEAR, 1, 1)
date.max = date(MAXYEAR, 12, 31)
date.resolution = timedelta(1)
