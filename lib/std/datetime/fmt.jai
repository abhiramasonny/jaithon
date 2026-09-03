#: `strftime`, over plain calendar fields rather than over an object.
#:
#: `date`, `time` and `datetime` all format the same way in CPython, which
#: fills the fields its receiver lacks with a fixed default: a `date` formats
#: as midnight, and a `time` formats as 1900-01-01. Passing the fields in
#: rather than the object is what lets one routine serve all three, and keeps
#: this file below `timedelta` in the import order.
#:
#: The C locale is the only locale. `%a`, `%A`, `%b`, `%B` and `%p` are
#: English and nothing reads a locale database, so a formatted string is the
#: same on every machine — which is the behaviour a log or a wire format
#: wants anyway.
from .common import DAY_NAMES, MONTH_NAMES, days_before_month, iso_week1_monday, pad
from .common import weekday_of, ymd_to_ordinal
from .tz import format_offset

#: Render calendar fields with strftime-style specifiers.
#:
#: | Spec | Meaning | Spec | Meaning |
#: |---|---|---|---|
#: | `%Y` | year, four digits | `%y` | year modulo 100, two digits |
#: | `%m` | month, 01 to 12 | `%B` | month name |
#: | `%b` | month name, three letters | `%d` | day, 01 to 31 |
#: | `%j` | day of year, 001 to 366 | `%H` | hour, 00 to 23 |
#: | `%I` | hour, 01 to 12 | `%p` | `AM` or `PM` |
#: | `%M` | minute, 00 to 59 | `%S` | second, 00 to 59 |
#: | `%f` | microsecond, six digits | `%z` | offset as `+HHMM`, empty if naive |
#: | `%Z` | zone name, empty if naive | `%a` | weekday name, three letters |
#: | `%A` | weekday name | `%w` | weekday, 0 is Sunday |
#: | `%u` | weekday, 1 is Monday | `%U` | week of year, Sunday first |
#: | `%W` | week of year, Monday first | `%G` | ISO year |
#: | `%V` | ISO week, 01 to 53 | `%%` | a literal `%` |
#:
#: `offset` is microseconds east of UTC, or null for a naive value; `zone` is
#: what `%Z` prints. An unsupported specifier raises `ValueError` rather than
#: reaching a platform `strftime` whose answer would differ by machine.
#:
#: ```
#: format_fields("%Y-%m-%d", 2024, 2, 29, 0, 0, 0, 0, null, null)   # "2024-02-29"
#: format_fields("%A %j", 2024, 2, 29, 0, 0, 0, 0, null, null)      # "Thursday 060"
#: ```
pub fn format_fields(
    pattern: str,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    microsecond: int,
    offset: int?,
    zone: str?
) -> str {
    var parts: list[str] = []
    var index = 0
    while index < pattern.len() {
        let character = pattern[index]
        if character != "%" {
            parts.push(character)
            index += 1
            continue
        }
        if index + 1 >= pattern.len() {
            throw ValueError(f"format pattern ends with a lone `%`: {repr(pattern)}")
        }
        let code = pattern[index + 1]
        index += 2
        parts.push(
            _specifier(
                code,
                pattern,
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                offset,
                zone
            )
        )
    }
    return "".join(parts)
}

#: Return the day of the year, 1 through 366.
#:
#: ```
#: day_of_year(2024, 3, 1)    # 61
#: ```
pub fn day_of_year(year: int, month: int, day: int) -> int {
    return days_before_month(year, month) + day
}

#: Return `(iso_year, iso_week, iso_weekday)` for a calendar date.
#:
#: ISO 8601 puts the week containing the first Thursday of a year in that
#: year, so the first days of January often belong to the previous ISO year.
#:
#: ```
#: iso_calendar(2024, 2, 29)   # (2024, 9, 4)
#: iso_calendar(2023, 1, 1)    # (2022, 52, 7)
#: ```
pub fn iso_calendar(year: int, month: int, day: int) -> tuple[int, int, int] {
    var iso_year = year
    var monday = iso_week1_monday(iso_year)
    let today = ymd_to_ordinal(year, month, day)
    var week = (today - monday) // 7
    if week < 0 {
        iso_year -= 1
        monday = iso_week1_monday(iso_year)
        week = (today - monday) // 7
    } elif week >= 52 and today >= iso_week1_monday(iso_year + 1) {
        iso_year += 1
        monday = iso_week1_monday(iso_year)
        week = 0
    }
    return (iso_year, week + 1, today - monday - week * 7 + 1)
}

#: Return the week of the year, counting from the first `%U`/`%W` week start.
#:
#: `sunday_first` picks `%U` over `%W`. Days before the year's first such
#: weekday are week 0, which is what C's `strftime` reports.
#:
#: ```
#: week_of_year(2024, 3, 4, true)     # 9
#: week_of_year(2024, 3, 4, false)    # 10
#: ```
pub fn week_of_year(year: int, month: int, day: int, sunday_first: bool) -> int {
    let since_new_year = day_of_year(year, month, day) - 1
    let monday_based = weekday_of(ymd_to_ordinal(year, month, day))
    let weekday = if sunday_first { (monday_based + 1) % 7 } else { monday_based }
    return (since_new_year + 7 - weekday) // 7
}

fn _specifier(
    code: str,
    pattern: str,
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    microsecond: int,
    offset: int?,
    zone: str?
) -> str {
    let ordinal = ymd_to_ordinal(year, month, day)
    return match code {
        "Y" => pad(year, 4),
        "y" => pad(year % 100, 2),
        "m" => pad(month, 2),
        "d" => pad(day, 2),
        "H" => pad(hour, 2),
        "M" => pad(minute, 2),
        "S" => pad(second, 2),
        "f" => pad(microsecond, 6),
        "j" => pad(day_of_year(year, month, day), 3),
        "I" => pad(if hour % 12 == 0 {
            12
        } else {
            hour % 12
        }, 2),
        "p" => if hour < 12 {
            "AM"
        } else {
            "PM"
        },
        "a" => DAY_NAMES[weekday_of(ordinal)][0:3],
        "A" => DAY_NAMES[weekday_of(ordinal)],
        "b" => MONTH_NAMES[month][0:3],
        "B" => MONTH_NAMES[month],
        "w" => str((weekday_of(ordinal) + 1) % 7),
        "u" => str(weekday_of(ordinal) + 1),
        "U" => pad(week_of_year(year, month, day, true), 2),
        "W" => pad(week_of_year(year, month, day, false), 2),
        "G" => pad(iso_calendar(year, month, day)[0], 4),
        "V" => pad(iso_calendar(year, month, day)[1], 2),
        "z" => if offset is null {
            ""
        } else {
            format_offset(offset ?? 0, "")
        },
        "Z" => zone ?? "",
        "%" => "%",
        _ => throw ValueError(f"unsupported format specifier `%{code}` in {repr(pattern)}"),
    }
}
