#: `time`, a time of day with no date, and the ISO 8601 time-part reader that
#: `datetime` shares with it.
#:
#: A `time` may carry a `timezone`, as CPython's may. Comparing a naive one
#: with an aware one raises `TypeError` for an ordering and answers false for
#: equality — CPython's rule, and the reason is that there is no moment the
#: naive one is known to name.
from .common import MICROS_PER_SECOND, check_time, digits_at, is_digit, pad
from .delta import timedelta
from .fmt import format_fields
from .tz import format_offset, timezone, zone_of_micros

let _KEEP = -1

#: A time of day, and the zone it is expressed in when it has one.
#:
#: ```
#: time(1, 2, 3, 4).isoformat()                    # "01:02:03.000004"
#: time(12, 0).isoformat(timespec: "minutes")      # "12:00"
#: repr(time(0, 0))                                # "datetime.time(0, 0)"
#: ```
pub class time {
    #: The hour, 0 through 23.
    pub let hour: int

    #: The minute, 0 through 59.
    pub let minute: int

    #: The second, 0 through 59.
    pub let second: int

    #: The microsecond, 0 through 999999.
    pub let microsecond: int

    #: The zone this time of day is expressed in, or null when it is naive.
    pub let tzinfo: timezone?

    #: The earliest time of day, midnight.
    pub static var min: time

    #: The latest time of day, 23:59:59.999999.
    pub static var max: time

    #: The smallest difference between two unequal times, one microsecond.
    pub static var resolution: timedelta

    #: Create a time of day, optionally in a fixed zone.
    #:
    #: ```
    #: time().isoformat()          # "00:00:00"
    #: time(23, 59, 59).second     # 59
    #: ```
    pub fn init(
        self,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        tzinfo: timezone? = null
    ) {
        check_time(hour, minute, second, microsecond)
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.tzinfo = tzinfo
    }

    #: Parse an ISO 8601 time of day, with an optional offset.
    #:
    #: ```
    #: time.fromisoformat("12:00:00.5").microsecond    # 500000
    #: time.fromisoformat("120000").hour               # 12
    #: ```
    pub static fn fromisoformat(text: str) -> time {
        let parsed = parse_iso_time(text)
        if parsed is null { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        let fields = parsed ?? IsoTime(0, 0, 0, 0, null)
        return time(fields.hour, fields.minute, fields.second, fields.microsecond, fields.zone)
    }

    #: Render the time of day as ISO 8601, to the precision `timespec` names.
    #:
    #: `timespec` is one of `"auto"`, `"hours"`, `"minutes"`, `"seconds"`,
    #: `"milliseconds"` or `"microseconds"`; `"auto"` drops the fraction when
    #: the microsecond is zero.
    #:
    #: ```
    #: time(1, 2, 3).isoformat()                         # "01:02:03"
    #: time(1, 2, 3, 4).isoformat(timespec: "seconds")   # "01:02:03"
    #: ```
    pub fn isoformat(self, timespec: str = "auto") -> str {
        let text = format_time_of_day(
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            timespec
        )
        let offset = self.offset_micros()
        if offset is null { return text }
        return text + format_offset(offset ?? 0, ":")
    }

    #: Return the offset east of UTC, or null when this time is naive.
    pub fn utcoffset(self) -> timedelta? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).utcoffset()
    }

    #: Return the zone's name, or null when this time is naive.
    pub fn tzname(self) -> str? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).tzname()
    }

    #: Return null: a fixed offset has no daylight-saving component.
    pub fn dst(self) -> timedelta? { return null }

    #: Return the offset in microseconds east of UTC, or null when naive.
    #:
    #: Not a CPython method; the conversions want one number, not a
    #: `timedelta` to unpack.
    pub fn offset_micros(self) -> int? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).offset_micros()
    }

    #: Return a copy with the named fields replaced.
    #:
    #: Passing `tzinfo: null` makes the copy naive, so unlike the integer
    #: fields there is no marker that means "leave it alone": pass
    #: `keep_tzinfo: false` together with a `tzinfo` to change it, and leave
    #: both out to keep what is there.
    #:
    #: ```
    #: time(1, 2).replace(second: 3).second    # 3
    #: ```
    pub fn replace(
        self,
        hour: int = _KEEP,
        minute: int = _KEEP,
        second: int = _KEEP,
        microsecond: int = _KEEP,
        tzinfo: timezone? = null,
        keep_tzinfo: bool = true
    ) -> time {
        return time(if hour == _KEEP {
            self.hour
        } else {
            hour
        }, if minute == _KEEP {
            self.minute
        } else {
            minute
        }, if second == _KEEP {
            self.second
        } else {
            second
        }, if microsecond == _KEEP {
            self.microsecond
        } else {
            microsecond
        }, if keep_tzinfo {
            self.tzinfo
        } else {
            tzinfo
        })
    }

    #: Render the time of day with strftime-style specifiers.
    #:
    #: The date fields read as 1900-01-01, which is what CPython uses for a
    #: bare `time`.
    #:
    #: ```
    #: time(13, 5).strftime("%I:%M %p")    # "01:05 PM"
    #: ```
    pub fn strftime(self, pattern: str) -> str {
        return format_fields(
            pattern,
            1900,
            1,
            1,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.offset_micros(),
            self.tzname()
        )
    }

    #: Order two times of day, returning a negative int, zero, or a positive int.
    #:
    #: Raises `TypeError` when one is naive and the other is aware.
    pub fn compare_to(self, other: time) -> int {
        let mine = self.offset_micros()
        let theirs = other.offset_micros()
        if (mine is null) != (theirs is null) {
            throw TypeError("can't compare offset-naive and offset-aware times")
        }
        let left = _micros_of_day(self) - (mine ?? 0)
        let right = _micros_of_day(other) - (theirs ?? 0)
        if left == right { return 0 }
        return if left < right { -1 } else { 1 }
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, time) { return false }
        let mine = self.offset_micros()
        let theirs = other.offset_micros()
        if (mine is null) != (theirs is null) { return false }
        return _micros_of_day(self) - (mine ?? 0) == _micros_of_day(other) - (theirs ?? 0)
    }

    fn __lt__(self, other: time) -> bool { return self.compare_to(other) < 0 }

    fn __le__(self, other: time) -> bool { return self.compare_to(other) <= 0 }

    fn __hash__(self) -> int { return hash(_micros_of_day(self) - (self.offset_micros() ?? 0)) }

    fn __str__(self) -> str { return self.isoformat() }

    fn __repr__(self) -> str {
        var text = f"datetime.time({self.hour}, {self.minute}"
        if self.microsecond != 0 {
            text = text + f", {self.second}, {self.microsecond}"
        } elif self.second != 0 {
            text = text + f", {self.second}"
        }
        if self.tzinfo is not null {
            text = text + f", tzinfo={repr(self.tzinfo ?? timezone.utc)}"
        }
        return text + ")"
    }
}

#: The four time fields and the zone an ISO 8601 time part named.
pub class IsoTime {
    #: The hour as written, before any range check.
    pub let hour: int

    #: The minute as written.
    pub let minute: int

    #: The second as written.
    pub let second: int

    #: The microsecond, already right-padded to six digits.
    pub let microsecond: int

    #: The zone the offset named, or null when there was none.
    pub let zone: timezone?

    #: Record the fields a reader recovered.
    pub fn init(self, hour: int, minute: int, second: int, microsecond: int, zone: timezone?) {
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.zone = zone
    }
}

#: Read a whole ISO 8601 time part, offset included, or null if malformed.
#:
#: The accepted shapes are CPython's: `HH`, `HH:MM`, `HH:MM:SS`, the same
#: three without colons, an optional fraction after `.` or `,` of any length
#: (truncated to six digits), and an optional `Z` or `+HH[:MM[:SS[.ffffff]]]`.
#:
#: ```
#: parse_iso_time("12:00:00.5") is not null    # true
#: parse_iso_time("12:0") is null              # true
#: ```
pub fn parse_iso_time(text: str) -> IsoTime? {
    if text.len() < 2 { return null }
    var marker = text.find("-") + 1
    if marker == 0 { marker = text.find("+") + 1 }
    if marker == 0 { marker = text.find("Z") + 1 }
    let body = if marker > 0 { text[0:marker - 1] } else { text }
    let fields = _hh_mm_ss_ff(body)
    if fields is null { return null }
    let (hour, minute, second, micro) = fields ?? (0, 0, 0, 0)
    var zone: timezone? = null
    if marker == text.len() and text[text.len() - 1] == "Z" {
        zone = timezone.utc
    } elif marker > 0 {
        let tail = text[marker:text.len()]
        if tail.len() == 0 or tail.len() == 1 or tail.len() == 3 { return null }
        let offset_fields = _hh_mm_ss_ff(tail)
        if offset_fields is null { return null }
        let (offset_hour, offset_minute, offset_second, offset_micro) = offset_fields ?? (0, 0, 0, 0)
        var micros = (offset_hour * 3_600 + offset_minute * 60 + offset_second) * MICROS_PER_SECOND +
            offset_micro
        if text[marker - 1] == "-" { micros = -micros }
        zone = zone_of_micros(micros)
    }
    return IsoTime(hour, minute, second, micro, zone)
}

#: Render the four time fields to the precision `timespec` names.
#:
#: ```
#: format_time_of_day(1, 2, 3, 0, "auto")            # "01:02:03"
#: format_time_of_day(1, 2, 3, 4, "milliseconds")    # "01:02:03.000"
#: ```
pub fn format_time_of_day(
    hour: int,
    minute: int,
    second: int,
    microsecond: int,
    timespec: str
) -> str {
    var wanted = timespec
    if wanted == "auto" { wanted = if microsecond == 0 { "seconds" } else { "microseconds" } }
    if wanted == "hours" { return pad(hour, 2) }
    if wanted == "minutes" { return pad(hour, 2) + ":" + pad(minute, 2) }
    let base = pad(hour, 2) + ":" + pad(minute, 2) + ":" + pad(second, 2)
    if wanted == "seconds" { return base }
    if wanted == "milliseconds" { return base + "." + pad(microsecond // 1_000, 3) }
    if wanted == "microseconds" { return base + "." + pad(microsecond, 6) }
    throw ValueError("Unknown timespec value")
}

fn _micros_of_day(value: time) -> int {
    return ((value.hour * 60 + value.minute) * 60 + value.second) * MICROS_PER_SECOND +
        value.microsecond
}

fn _hh_mm_ss_ff(text: str) -> tuple[int, int, int, int]? {
    var fields = [0, 0, 0, 0]
    var at = 0
    var has_separator = false
    var index = 0
    while index < 3 {
        if text.len() - at < 2 { return null }
        let value = digits_at(text, at, 2)
        if value is null { return null }
        fields[index] = value ?? 0
        at += 2
        let next = if at < text.len() { text[at] } else { "" }
        if index == 0 { has_separator = next == ":" }
        if next == "" or next == "." or next == "," or index >= 2 { break }
        if has_separator and next != ":" { return null }
        if has_separator { at += 1 }
        index += 1
    }
    if at < text.len() {
        if text[at] != "." and text[at] != "," { return null }
        at += 1
        let remaining = text.len() - at
        let width = if remaining >= 6 { 6 } else { remaining }
        if width == 0 { return null }
        let digits = digits_at(text, at, width)
        if digits is null { return null }
        var value = digits ?? 0
        var missing = 6 - width
        while missing > 0 {
            value *= 10
            missing -= 1
        }
        fields[3] = value
        at += width
        for scan in at..text.len() {
            if not is_digit(text[scan]) { return null }
        }
    }
    return (fields[0], fields[1], fields[2], fields[3])
}

time.min = time(0, 0)
time.max = time(23, 59, 59, 999_999)
time.resolution = timedelta(0, 0, 1)
