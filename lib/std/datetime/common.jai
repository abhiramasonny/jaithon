#: Calendar arithmetic shared by every class in `std.datetime`.
#:
#: The ordinal is CPython's: 0001-01-01 is day 1 of the proleptic Gregorian
#: calendar and 9999-12-31 is day 3652059. Every routine below is a
#: transcription of the one in CPython's `Lib/datetime.py`, so the two agree
#: on every day in `MINYEAR..MAXYEAR` rather than only on the days a test
#: happened to try.
#:
#: `is_leap_year` and `days_in_month` are `std.time`'s, not copies of them.
from std.time import days_in_month, is_leap_year

#: The earliest year `date` and `datetime` accept.
pub const MINYEAR: int = 1

#: The latest year `date` and `datetime` accept.
pub const MAXYEAR: int = 9999

#: The ordinal of 9999-12-31, the last day this module can name.
pub const MAXORDINAL: int = 3652059

#: Microseconds in one second.
pub const MICROS_PER_SECOND: int = 1_000_000

#: Seconds in one day, leap seconds ignored as they are everywhere here.
pub const SECONDS_PER_DAY: int = 86_400

#: Seconds in one hour.
pub const SECONDS_PER_HOUR: int = 3_600

#: Seconds in one minute.
pub const SECONDS_PER_MINUTE: int = 60

#: The largest `days` a `timedelta` may hold, as CPython bounds it.
pub const MAX_DELTA_DAYS: int = 999_999_999

const _DI400Y = 146_097
const _DI100Y = 36_524
const _DI4Y = 1_461

#: Length of each month, indexed from 1; slot 0 is a placeholder.
pub let DAYS_IN_MONTH = [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#: Days of a common year before each month, indexed from 1.
pub let DAYS_BEFORE_MONTH = [-1, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

#: Month names as the C locale spells them, indexed from 1.
pub let MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

#: Weekday names with Monday first, matching `weekday()`.
pub let DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#: Return the days from 0001-01-01 to the last day of `year - 1`.
#:
#: ```
#: days_before_year(1)    # 0
#: days_before_year(2)    # 365
#: ```
pub fn days_before_year(year: int) -> int {
    let previous = year - 1
    return previous * 365 + previous // 4 - previous // 100 + previous // 400
}

#: Return the days in `year` before the first of `month`.
#:
#: ```
#: days_before_month(2024, 3)    # 60
#: days_before_month(2023, 3)    # 59
#: ```
pub fn days_before_month(year: int, month: int) -> int {
    var count = DAYS_BEFORE_MONTH[month]
    if month > 2 and is_leap_year(year) { count += 1 }
    return count
}

#: Convert a calendar date to its proleptic Gregorian ordinal.
#:
#: ```
#: ymd_to_ordinal(1, 1, 1)         # 1
#: ymd_to_ordinal(9999, 12, 31)    # 3652059
#: ```
pub fn ymd_to_ordinal(year: int, month: int, day: int) -> int {
    return days_before_year(year) + days_before_month(year, month) + day
}

#: Convert a proleptic Gregorian ordinal back to `(year, month, day)`.
#:
#: ```
#: ordinal_to_ymd(1)          # (1, 1, 1)
#: ordinal_to_ymd(739000)     # (2024, 2, 26)
#: ```
pub fn ordinal_to_ymd(ordinal: int) -> tuple[int, int, int] {
    var rest = ordinal - 1
    let n400 = rest // _DI400Y
    rest -= n400 * _DI400Y
    var year = n400 * 400 + 1
    let n100 = rest // _DI100Y
    rest -= n100 * _DI100Y
    let n4 = rest // _DI4Y
    rest -= n4 * _DI4Y
    let n1 = rest // 365
    rest -= n1 * 365
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4 { return (year - 1, 12, 31) }
    let leap = n1 == 3 and (n4 != 24 or n100 == 3)
    var month = rest + 50 >> 5
    var preceding = DAYS_BEFORE_MONTH[month] + (if month > 2 and leap { 1 } else { 0 })
    if preceding > rest {
        month -= 1
        preceding -= DAYS_IN_MONTH[month] + (if month == 2 and leap { 1 } else { 0 })
    }
    return (year, month, rest - preceding + 1)
}

#: Return the weekday of an ordinal, 0 for Monday through 6 for Sunday.
#:
#: ```
#: weekday_of(ymd_to_ordinal(2024, 2, 29))    # 3
#: ```
pub fn weekday_of(ordinal: int) -> int { return (ordinal + 6) % 7 }

#: Return the ordinal of the Monday that opens ISO week 1 of `year`.
pub fn iso_week1_monday(year: int) -> int {
    let firstday = ymd_to_ordinal(year, 1, 1)
    let firstweekday = (firstday + 6) % 7
    var monday = firstday - firstweekday
    if firstweekday > 3 { monday += 7 }
    return monday
}

#: Raise `ValueError` unless `year`, `month` and `day` name a real date.
pub fn check_date(year: int, month: int, day: int) -> void {
    if year < MINYEAR or year > MAXYEAR {
        throw ValueError(f"year {year} is out of range")
    }
    if month < 1 or month > 12 {
        throw ValueError(f"month must be in 1..12, not {month}")
    }
    let last = days_in_month(year, month)
    if day < 1 or day > last {
        throw ValueError(f"day must be in 1..{last}, not {day}")
    }
}

#: Raise `ValueError` unless the four fields name a real time of day.
pub fn check_time(hour: int, minute: int, second: int, microsecond: int) -> void {
    if hour < 0 or hour > 23 { throw ValueError(f"hour must be in 0..23, not {hour}") }
    if minute < 0 or minute > 59 { throw ValueError(f"minute must be in 0..59, not {minute}") }
    if second < 0 or second > 59 { throw ValueError(f"second must be in 0..59, not {second}") }
    if microsecond < 0 or microsecond > 999_999 {
        throw ValueError(f"microsecond must be in 0..999999, not {microsecond}")
    }
}

#: Render `value` right-aligned in `width` columns, zero filled, as `%0Nd` does.
#:
#: A negative value keeps its sign inside the width, so `pad(-1, 4)` is
#: `"-001"` and not `"0-01"`.
#:
#: ```
#: pad(7, 2)      # "07"
#: pad(2024, 4)   # "2024"
#: ```
pub fn pad(value: int, width: int) -> str {
    if value < 0 { return "-" + pad(-value, width - 1) }
    var text = str(value)
    while text.len() < width { text = "0" + text }
    return text
}

#: Whether `character` is one of `0` through `9`.
pub fn is_digit(character: str) -> bool {
    let code = ord(character)
    return code >= 48 and code <= 57
}

#: Read `width` digits of `text` starting at `start`, or null if they are not
#: all digits or the string is too short.
pub fn digits_at(text: str, start: int, width: int) -> int? {
    if start < 0 or start + width > text.len() { return null }
    var value = 0
    for index in start..start + width {
        let character = text[index]
        if not is_digit(character) { return null }
        value = value * 10 + (ord(character) - 48)
    }
    return value
}
