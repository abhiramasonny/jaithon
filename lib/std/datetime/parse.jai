#: `strptime`, over plain calendar fields rather than over an object.
#:
#: CPython builds a regular expression from the format and matches it; this
#: reads the format and the data together, left to right. The two agree on
#: every format whose directives are separated by literal text, which is the
#: shape a real format has. Where they can differ is a format that runs two
#: variable-width directives together — `"%d%m"` against `"1234"` — since a
#: regex can backtrack across the pair and this cannot. Each directive does
#: try its widest reading first and falls back to a narrower one when the
#: value is out of range, which is what reproduces CPython on `"%H"` against
#: `"25"`.
#:
#: Defaults are CPython's: a format that names no year parses as 1900, and one
#: that names no month or day parses as the first of January.
#:
#: A reader returns `(value, end)` and reports failure as an `end` of -1,
#: rather than a null tuple: the caller unwraps a value on every one of the
#: twenty directives below, and an optional tuple would put a `??` on each.
from .common import DAY_NAMES, MICROS_PER_SECOND, MONTH_NAMES, is_digit
from .common import ordinal_to_ymd, weekday_of, ymd_to_ordinal
from std.time import is_leap_year

const _FAILED = -1

#: The calendar fields one `strptime` call recovered.
#:
#: `offset` is microseconds east of UTC and is null when the format had no
#: `%z`; `zone` is what a `%Z` matched, and is null otherwise.
pub class Parsed {
    #: The year, 1900 when the format named none.
    pub var year: int

    #: The month, 1 when the format named none.
    pub var month: int

    #: The day of the month, 1 when the format named none.
    pub var day: int

    #: The hour of the day, 0 when the format named none.
    pub var hour: int

    #: The minute of the hour, 0 when the format named none.
    pub var minute: int

    #: The second of the minute, 0 when the format named none.
    pub var second: int

    #: The microsecond, 0 when the format named none.
    pub var microsecond: int

    #: Microseconds east of UTC from a `%z`, or null for a naive result.
    pub var offset: int?

    #: The zone name a `%Z` matched, or null.
    pub var zone: str?

    #: Start from CPython's defaults, 1900-01-01T00:00:00, naive.
    pub fn init(self) {
        self.year = 1900
        self.month = 1
        self.day = 1
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.microsecond = 0
        self.offset = null
        self.zone = null
    }
}

#: Parse `text` against `pattern` and return the fields it named.
#:
#: Supported specifiers are `%Y %y %m %d %H %I %M %S %f %j %U %W %w %a %A %b
#: %B %p %z %Z %%`. A run of whitespace in the pattern matches a run of
#: whitespace in the text. Everything else is a literal, compared without
#: regard to case, as CPython's `IGNORECASE` regex does.
#:
#: ```
#: strptime_fields("2024-02-29", "%Y-%m-%d").day        # 29
#: strptime_fields("29 Feb 2024", "%d %b %Y").month     # 2
#: strptime_fields("12:30 PM", "%I:%M %p").hour         # 12
#: ```
pub fn strptime_fields(text: str, pattern: str) -> Parsed {
    var year = _FAILED
    var month = 1
    var day = 1
    var hour24 = _FAILED
    var hour12 = _FAILED
    var minute = 0
    var second = 0
    var fraction = 0
    var ampm = ""
    var weekday = _FAILED
    var julian = _FAILED
    var week = _FAILED
    var week_sunday_first = true
    var offset: int? = null
    var zone: str? = null

    var cursor = 0
    var at = 0
    while cursor < pattern.len() {
        let character = pattern[cursor]
        if _is_space(character) {
            while cursor < pattern.len() and _is_space(pattern[cursor]) { cursor += 1 }
            let start = at
            while at < text.len() and _is_space(text[at]) { at += 1 }
            if at == start { throw _no_match(text, pattern) }
            continue
        }
        if character != "%" {
            if at >= text.len() or text[at].lower() != character.lower() {
                throw _no_match(text, pattern)
            }
            cursor += 1
            at += 1
            continue
        }
        if cursor + 1 >= pattern.len() {
            throw ValueError(f"stray % in format {repr(pattern)}")
        }
        let code = pattern[cursor + 1]
        cursor += 2
        if code == "%" {
            if at >= text.len() or text[at] != "%" { throw _no_match(text, pattern) }
            at += 1
            continue
        }
        var value = _FAILED
        var next = _FAILED
        if code == "Y" {
            let (read, end) = _fixed(text, at, 4)
            value = read
            next = end
        } elif code == "y" {
            let (read, end) = _fixed(text, at, 2)
            value = if read <= 68 { read + 2000 } else { read + 1900 }
            next = end
        } elif code == "m" or code == "I" {
            let (read, end) = _ranged(text, at, 2, 1, 12)
            value = read
            next = end
        } elif code == "d" {
            let (read, end) = _ranged(text, at, 2, 1, 31)
            value = read
            next = end
        } elif code == "H" {
            let (read, end) = _ranged(text, at, 2, 0, 23)
            value = read
            next = end
        } elif code == "M" {
            let (read, end) = _ranged(text, at, 2, 0, 59)
            value = read
            next = end
        } elif code == "S" {
            let (read, end) = _ranged(text, at, 2, 0, 61)
            value = read
            next = end
        } elif code == "j" {
            let (read, end) = _ranged(text, at, 3, 1, 366)
            value = read
            next = end
        } elif code == "U" or code == "W" {
            let (read, end) = _ranged(text, at, 2, 0, 53)
            value = read
            next = end
        } elif code == "w" {
            let (read, end) = _ranged(text, at, 1, 0, 6)
            value = read
            next = end
        } elif code == "f" {
            let (read, end) = _fraction(text, at)
            value = read
            next = end
        } elif code == "a" or code == "A" {
            let (read, end) = _name(text, at, DAY_NAMES, code == "a")
            value = read
            next = end
        } elif code == "b" or code == "B" {
            let (read, end) = _name(text, at, MONTH_NAMES, code == "b")
            value = read
            next = end
        } elif code == "p" {
            let (read, end) = _name(text, at, _MERIDIEM, false)
            value = read
            next = end
        } elif code == "z" {
            let (read, end) = _zone_offset(text, at)
            value = read
            next = end
        } elif code == "Z" {
            let (read, end) = _zone_name(text, at)
            value = read
            next = end
        } else {
            throw ValueError(f"'{code}' is a bad directive in format {repr(pattern)}")
        }
        if next == _FAILED { throw _no_match(text, pattern) }
        if code == "Y" or code == "y" {
            year = value
        } elif code == "m" or code == "b" or code == "B" {
            month = value
        } elif code == "d" {
            day = value
        } elif code == "H" {
            hour24 = value
        } elif code == "I" {
            hour12 = value
        } elif code == "M" {
            minute = value
        } elif code == "S" {
            second = value
        } elif code == "f" {
            fraction = value
        } elif code == "j" {
            julian = value
        } elif code == "U" or code == "W" {
            week = value
            week_sunday_first = code == "U"
        } elif code == "w" {
            weekday = (value + 6) % 7
        } elif code == "a" or code == "A" {
            weekday = value
        } elif code == "p" {
            ampm = if value == 0 { "am" } else { "pm" }
        } elif code == "z" {
            offset = value
        } elif code == "Z" {
            zone = text[at:next]
        }
        at = next
    }
    if at != text.len() {
        throw ValueError(f"unconverted data remains: {text[at:text.len()]}")
    }

    var final_year = if year == _FAILED { 1900 } else { year }
    if julian == _FAILED and weekday != _FAILED and week != _FAILED {
        julian = _julian_from_week(final_year, week, weekday, not week_sunday_first)
    }
    if julian != _FAILED and julian <= 0 {
        final_year -= 1
        julian += if is_leap_year(final_year) { 366 } else { 365 }
    }
    let result = Parsed()
    if julian == _FAILED {
        result.year = final_year
        result.month = month
        result.day = day
    } else {
        let (recovered_year, recovered_month, recovered_day) = ordinal_to_ymd(
            julian - 1 +
                ymd_to_ordinal(final_year, 1, 1)
        )
        result.year = recovered_year
        result.month = recovered_month
        result.day = recovered_day
    }
    result.hour = _resolve_hour(hour24, hour12, ampm)
    result.minute = minute
    result.second = second
    result.microsecond = fraction
    result.offset = offset
    result.zone = zone
    return result
}

let _MERIDIEM = ["AM", "PM"]

fn _resolve_hour(hour24: int, hour12: int, ampm: str) -> int {
    if hour24 != _FAILED { return hour24 }
    if hour12 == _FAILED { return 0 }
    if ampm == "pm" { return if hour12 == 12 { 12 } else { hour12 + 12 } }
    return if hour12 == 12 { 0 } else { hour12 }
}

fn _julian_from_week(year: int, week: int, weekday: int, monday_first: bool) -> int {
    var first_weekday = weekday_of(ymd_to_ordinal(year, 1, 1))
    var day_of_week = weekday
    if not monday_first {
        first_weekday = (first_weekday + 1) % 7
        day_of_week = (day_of_week + 1) % 7
    }
    if week == 0 { return 1 + day_of_week - first_weekday }
    return 1 + (7 - first_weekday) % 7 + 7 * (week - 1) + day_of_week
}

fn _no_match(text: str, pattern: str) -> Error {
    return ValueError(f"time data {repr(text)} does not match format {repr(pattern)}")
}

fn _is_space(character: str) -> bool {
    return character == " " or
        character == "\t" or
        character == "\n" or
        character == "\r" or
        character == "\x0B" or
        character == "\x0C"
}

fn _fixed(text: str, start: int, width: int) -> tuple[int, int] {
    if start + width > text.len() { return (_FAILED, _FAILED) }
    var value = 0
    for index in start..start + width {
        if not is_digit(text[index]) { return (_FAILED, _FAILED) }
        value = value * 10 + (ord(text[index]) - 48)
    }
    return (value, start + width)
}

fn _ranged(text: str, start: int, width: int, low: int, high: int) -> tuple[int, int] {
    var tried = width
    while tried >= 1 {
        let (value, end) = _fixed(text, start, tried)
        if end != _FAILED and value >= low and value <= high { return (value, end) }
        tried -= 1
    }
    return (_FAILED, _FAILED)
}

fn _fraction(text: str, start: int) -> tuple[int, int] {
    var end = start
    while end < text.len() and end - start < 6 and is_digit(text[end]) { end += 1 }
    if end == start { return (_FAILED, _FAILED) }
    var value = 0
    for index in start..end { value = value * 10 + (ord(text[index]) - 48) }
    var missing = 6 - (end - start)
    while missing > 0 {
        value *= 10
        missing -= 1
    }
    return (value, end)
}

fn _name(text: str, start: int, candidates: list[str], abbreviated: bool) -> tuple[int, int] {
    var best = _FAILED
    var best_end = _FAILED
    for (index, candidate) in candidates.enumerate() {
        if candidate.len() == 0 { continue }
        let wanted = if abbreviated { candidate[0:3] } else { candidate }
        let end = start + wanted.len()
        if end > text.len() { continue }
        if text[start:end].lower() != wanted.lower() { continue }
        if end > best_end {
            best = index
            best_end = end
        }
    }
    return (best, best_end)
}

fn _zone_name(text: str, start: int) -> tuple[int, int] {
    var end = start
    while end < text.len() {
        let code = ord(text[end])
        let letter = code >= 65 and code <= 90 or code >= 97 and code <= 122
        if not (letter or text[end] == "+" or text[end] == "-" or is_digit(text[end])) { break }
        end += 1
    }
    if end == start { return (_FAILED, _FAILED) }
    return (0, end)
}

fn _zone_offset(text: str, start: int) -> tuple[int, int] {
    if start >= text.len() { return (_FAILED, _FAILED) }
    if text[start] == "Z" { return (0, start + 1) }
    let sign_character = text[start]
    if sign_character != "+" and sign_character != "-" { return (_FAILED, _FAILED) }
    var at = start + 1
    let (hours, after_hours) = _fixed(text, at, 2)
    if after_hours == _FAILED { return (_FAILED, _FAILED) }
    at = after_hours
    if at < text.len() and text[at] == ":" { at += 1 }
    let (minutes, after_minutes) = _fixed(text, at, 2)
    if after_minutes == _FAILED { return (_FAILED, _FAILED) }
    at = after_minutes
    var seconds = 0
    var micros = 0
    var scan = at
    if scan < text.len() and text[scan] == ":" { scan += 1 }
    let (read_seconds, after_seconds) = _fixed(text, scan, 2)
    if after_seconds != _FAILED {
        seconds = read_seconds
        at = after_seconds
        if at < text.len() and text[at] == "." {
            let (read_micros, after_micros) = _fraction(text, at + 1)
            if after_micros == _FAILED { return (_FAILED, _FAILED) }
            micros = read_micros
            at = after_micros
        }
    }
    let total = (hours * 3_600 + minutes * 60 + seconds) * MICROS_PER_SECOND + micros
    return (if sign_character == "-" { -total } else { total }, at)
}
