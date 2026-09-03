#: `datetime`, a calendar day and a time of day, optionally in a fixed zone.
#:
#: Every conversion here goes through one number: the microseconds since
#: 0001-01-01T00:00:00, offset removed for an aware value. That number peaks
#: near 3.2e17 for 9999-12-31, which a 64-bit integer holds with three digits
#: to spare, so `astimezone`, `timestamp` and subtraction are all the same
#: arithmetic and none of them can overflow inside the calendar's range.
from .common import MAXORDINAL, MAXYEAR, MICROS_PER_SECOND, MINYEAR
from .common import check_date, check_time, ordinal_to_ymd, pad, weekday_of, ymd_to_ordinal
from .dates import ctime_of, date, parse_iso_date
from .delta import timedelta
from .fmt import format_fields, iso_calendar
from .parse import strptime_fields
from .times import IsoTime, format_time_of_day, parse_iso_time, time
from .tz import Omitted, format_offset, timezone, zone_of_micros
from std.time import DateTime

const _MICROS_PER_DAY = 86_400_000_000
const _MICROS_PER_HOUR = 3_600_000_000
const _MICROS_PER_MINUTE = 60_000_000

let _KEEP = -1
let _OMITTED = Omitted()

#: A moment on the calendar, to microsecond resolution.
#:
#: ```
#: datetime(2024, 2, 29, 12, 30).isoformat()      # "2024-02-29T12:30:00"
#: datetime(1970, 1, 1, tzinfo: timezone.utc).timestamp()    # 0.0
#: str(datetime(2024, 1, 2) - datetime(2024, 1, 1))          # "1 day, 0:00:00"
#: ```
pub class datetime {
    #: The year, 1 through 9999.
    pub let year: int

    #: The month, 1 for January through 12 for December.
    pub let month: int

    #: The day of the month, from 1.
    pub let day: int

    #: The hour, 0 through 23.
    pub let hour: int

    #: The minute, 0 through 59.
    pub let minute: int

    #: The second, 0 through 59.
    pub let second: int

    #: The microsecond, 0 through 999999.
    pub let microsecond: int

    #: The zone the fields are expressed in, or null when this is naive.
    pub let tzinfo: timezone?

    #: The earliest moment, 0001-01-01T00:00:00.
    pub static var min: datetime

    #: The latest moment, 9999-12-31T23:59:59.999999.
    pub static var max: datetime

    #: The smallest difference between two unequal moments, one microsecond.
    pub static var resolution: timedelta

    #: Create a moment from calendar fields, which must name a real date.
    #:
    #: ```
    #: datetime(2024, 2, 29).hour    # 0
    #: ```
    pub fn init(
        self,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        microsecond: int = 0,
        tzinfo: timezone? = null
    ) {
        check_date(year, month, day)
        check_time(hour, minute, second, microsecond)
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.tzinfo = tzinfo
    }

    #: Read the wall clock, in `zone` when one is given.
    #:
    #: CPython reads the local zone for a naive result. There is no zone
    #: database here, so a naive result is UTC — the same reading `utcnow`
    #: gives.
    pub static fn now(zone: timezone? = null) -> datetime {
        let wall = DateTime.now()
        let moment = datetime(
            wall.year,
            wall.month,
            wall.day,
            wall.hour,
            wall.minute,
            wall.second,
            wall.nanosecond // 1_000
        )
        if zone is null { return moment }
        let target = zone ?? timezone.utc
        return _from_micros(_local_micros(moment) + target.offset_micros(), target, true)
    }

    #: Read the wall clock as a naive UTC moment.
    pub static fn utcnow() -> datetime { return datetime.now() }

    #: Read the wall clock, the same as `now` with no zone.
    pub static fn today() -> datetime { return datetime.now() }

    #: Join a `date` and a `time` into one moment.
    #:
    #: The time's zone comes along unless `zone` names another; passing
    #: `zone: null` makes the result naive.
    #:
    #: ```
    #: datetime.combine(date(2024, 1, 1), time(1, 2, 3)).minute    # 2
    #: ```
    pub static fn combine(day: date, moment: time, zone: any = _OMITTED) -> datetime {
        var chosen = moment.tzinfo
        if not isinstance(zone, Omitted) {
            chosen = if isinstance(zone, timezone) { zone } else { null }
        }
        return datetime(
            day.year,
            day.month,
            day.day,
            moment.hour,
            moment.minute,
            moment.second,
            moment.microsecond,
            chosen
        )
    }

    #: Return the moment `timestamp` seconds after the Unix epoch.
    #:
    #: With no zone the result is naive UTC, where CPython would give naive
    #: local time; with one it is that zone's rendering of the same instant.
    #:
    #: ```
    #: datetime.fromtimestamp(0.0).isoformat()    # "1970-01-01T00:00:00"
    #: ```
    pub static fn fromtimestamp(timestamp: float, zone: timezone? = null) -> datetime {
        let utc_micros = _EPOCH_MICROS + _timestamp_micros(timestamp)
        if zone is null { return _from_micros(utc_micros, null, false) }
        let target = zone ?? timezone.utc
        return _from_micros(utc_micros + target.offset_micros(), target, false)
    }

    #: Return the naive UTC moment `timestamp` seconds after the Unix epoch.
    pub static fn utcfromtimestamp(timestamp: float) -> datetime {
        return datetime.fromtimestamp(timestamp)
    }

    #: Return midnight of the day whose ordinal is `ordinal`.
    pub static fn fromordinal(ordinal: int) -> datetime {
        if ordinal < 1 or ordinal > MAXORDINAL {
            throw ValueError(f"ordinal must be in 1..{MAXORDINAL}, not {ordinal}")
        }
        let (year, month, day) = ordinal_to_ymd(ordinal)
        return datetime(year, month, day)
    }

    #: Parse an ISO 8601 date, optionally followed by a time and an offset.
    #:
    #: The character between the date and the time may be anything, as CPython
    #: allows since 3.11; `T` and a space are the usual ones.
    #:
    #: ```
    #: datetime.fromisoformat("2024-02-29T12:00:00+05:30").minute    # 0
    #: datetime.fromisoformat("2024-02-29").hour                     # 0
    #: ```
    pub static fn fromisoformat(text: str) -> datetime {
        let head = parse_iso_date(text)
        if head is null { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        let (year, month, day, consumed) = head ?? (0, 0, 0, 0)
        if consumed == text.len() { return datetime(year, month, day) }
        let tail = text[consumed + 1:text.len()]
        if tail.len() == 0 { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        let parsed = parse_iso_time(tail)
        if parsed is null { throw ValueError(f"Invalid isoformat string: {repr(text)}") }
        let fields = parsed ?? IsoTime(0, 0, 0, 0, null)
        return datetime(
            year,
            month,
            day,
            fields.hour,
            fields.minute,
            fields.second,
            fields.microsecond,
            fields.zone
        )
    }

    #: Parse `text` according to `pattern`, the inverse of `strftime`.
    #:
    #: ```
    #: datetime.strptime("2024-02-29", "%Y-%m-%d").day        # 29
    #: datetime.strptime("29/Feb/24 13:05", "%d/%b/%y %H:%M").hour    # 13
    #: ```
    pub static fn strptime(text: str, pattern: str) -> datetime {
        let fields = strptime_fields(text, pattern)
        var zone: timezone? = null
        if fields.offset is not null { zone = zone_of_micros(fields.offset ?? 0) }
        return datetime(
            fields.year,
            fields.month,
            fields.day,
            fields.hour,
            fields.minute,
            fields.second,
            fields.microsecond,
            zone
        )
    }

    #: Return the calendar day, dropping the time and the zone.
    pub fn date(self) -> date { return date(self.year, self.month, self.day) }

    #: Return the time of day, dropping the date and the zone.
    pub fn time(self) -> time {
        return time(self.hour, self.minute, self.second, self.microsecond)
    }

    #: Return the time of day, keeping the zone.
    pub fn timetz(self) -> time {
        return time(self.hour, self.minute, self.second, self.microsecond, self.tzinfo)
    }

    #: Return the day's proleptic Gregorian ordinal.
    pub fn toordinal(self) -> int { return ymd_to_ordinal(self.year, self.month, self.day) }

    #: Return the weekday, 0 for Monday through 6 for Sunday.
    pub fn weekday(self) -> int { return weekday_of(self.toordinal()) }

    #: Return the weekday, 1 for Monday through 7 for Sunday.
    pub fn isoweekday(self) -> int { return weekday_of(self.toordinal()) + 1 }

    #: Return `(iso_year, iso_week, iso_weekday)` for this moment's day.
    pub fn isocalendar(self) -> tuple[int, int, int] {
        return iso_calendar(self.year, self.month, self.day)
    }

    #: Render the moment as ISO 8601.
    #:
    #: `separator` sits between the date and the time; `timespec` is one of
    #: `"auto"`, `"hours"`, `"minutes"`, `"seconds"`, `"milliseconds"` or
    #: `"microseconds"`.
    #:
    #: ```
    #: datetime(2024, 1, 2, 3, 4, 5).isoformat()                    # "2024-01-02T03:04:05"
    #: datetime(2024, 1, 2, 3, 4, 5).isoformat(separator: " ")      # "2024-01-02 03:04:05"
    #: ```
    pub fn isoformat(self, separator: str = "T", timespec: str = "auto") -> str {
        var text = pad(self.year, 4) +
            "-" +
            pad(self.month, 2) +
            "-" +
            pad(self.day, 2) +
            separator +
            format_time_of_day(self.hour, self.minute, self.second, self.microsecond, timespec)
        let offset = self.offset_micros()
        if offset is not null { text = text + format_offset(offset ?? 0, ":") }
        return text
    }

    #: Render the moment the way C's `ctime` does.
    #:
    #: ```
    #: datetime(2024, 3, 4, 5, 6, 7).ctime()    # "Mon Mar  4 05:06:07 2024"
    #: ```
    pub fn ctime(self) -> str {
        return ctime_of(self.year, self.month, self.day, self.hour, self.minute, self.second)
    }

    #: Return the seconds between this moment and the Unix epoch.
    #:
    #: A naive moment counts as UTC. CPython reads the local zone for one,
    #: which is the module's one deliberate difference in a returned number
    #: rather than a spelling.
    #:
    #: The two roundings are CPython's two roundings, and they are not the
    #: same: an aware moment divides an exact microsecond total by a million,
    #: a naive one adds a whole second count to a rounded fraction, because
    #: CPython's naive path comes back through `mktime`. The results differ in
    #: the last place for a naive moment a microsecond either side of the
    #: epoch, so the split is kept rather than tidied away.
    #:
    #: ```
    #: datetime(1970, 1, 2, tzinfo: timezone.utc).timestamp()    # 86400.0
    #: ```
    pub fn timestamp(self) -> float {
        let micros = _utc_micros(self) - _EPOCH_MICROS
        let seconds = micros // MICROS_PER_SECOND
        let fraction = micros - seconds * MICROS_PER_SECOND
        if self.tzinfo is null {
            return float(seconds) + float(fraction) / 1_000_000.0
        }
        return timedelta(0, seconds, fraction).total_seconds()
    }

    #: Return the offset east of UTC, or null when this moment is naive.
    pub fn utcoffset(self) -> timedelta? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).utcoffset()
    }

    #: Return the zone's name, or null when this moment is naive.
    pub fn tzname(self) -> str? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).tzname()
    }

    #: Return null: a fixed offset has no daylight-saving component.
    pub fn dst(self) -> timedelta? { return null }

    #: Return the offset in microseconds east of UTC, or null when naive.
    pub fn offset_micros(self) -> int? {
        if self.tzinfo is null { return null }
        return (self.tzinfo ?? timezone.utc).offset_micros()
    }

    #: Return the same instant rendered in `zone`.
    #:
    #: A naive moment is read as UTC first. CPython's `astimezone` with no
    #: argument means "the local zone", which this module cannot name, so the
    #: zone is required.
    #:
    #: The conversion lands on UTC before it lands on `zone`, and that
    #: intermediate has to be a real moment: an hour before dawn on
    #: 0001-01-01 at a far eastern offset is UTC in year 0, and raises
    #: `OverflowError` even though the answer would fit. The arithmetic here
    #: would not have noticed; CPython's does, so this does.
    #:
    #: ```
    #: datetime(2024, 1, 1, tzinfo: timezone.utc)
    #:     .astimezone(timezone(timedelta(hours: 5))).hour    # 5
    #: ```
    pub fn astimezone(self, zone: timezone) -> datetime {
        let utc_micros = _utc_micros(self)
        let midpoint = _from_micros(utc_micros, null, true)
        return _from_micros(_local_micros(midpoint) + zone.offset_micros(), zone, true)
    }

    #: Return a copy with the named fields replaced.
    #:
    #: `tzinfo` works as it does on `time`: pass `keep_tzinfo: false` to
    #: install the `tzinfo` given, which may be null to make the copy naive.
    #:
    #: ```
    #: datetime(2024, 1, 2, 3).replace(hour: 9).hour    # 9
    #: ```
    pub fn replace(
        self,
        year: int = _KEEP,
        month: int = _KEEP,
        day: int = _KEEP,
        hour: int = _KEEP,
        minute: int = _KEEP,
        second: int = _KEEP,
        microsecond: int = _KEEP,
        tzinfo: timezone? = null,
        keep_tzinfo: bool = true
    ) -> datetime {
        return datetime(if year == _KEEP {
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
        }, if hour == _KEEP {
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

    #: Render the moment with strftime-style specifiers.
    #:
    #: ```
    #: datetime(2024, 2, 29, 13, 5).strftime("%Y-%m-%d %H:%M")    # "2024-02-29 13:05"
    #: ```
    pub fn strftime(self, pattern: str) -> str {
        return format_fields(
            pattern,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.offset_micros(),
            self.tzname()
        )
    }

    #: Order two moments, returning a negative int, zero, or a positive int.
    #:
    #: Raises `TypeError` when one is naive and the other is aware.
    pub fn compare_to(self, other: datetime) -> int {
        if (self.tzinfo is null) != (other.tzinfo is null) {
            throw TypeError("can't compare offset-naive and offset-aware datetimes")
        }
        let mine = _utc_micros(self)
        let theirs = _utc_micros(other)
        if mine == theirs { return 0 }
        return if mine < theirs { -1 } else { 1 }
    }

    fn __add__(self, span: timedelta) -> datetime {
        return _shifted(self, span)
    }

    fn __sub__(self, other: any) -> any {
        if isinstance(other, timedelta) { return _shifted(self, other.negate()) }
        if isinstance(other, datetime) {
            if (self.tzinfo is null) != (other.tzinfo is null) {
                throw TypeError("can't subtract offset-naive and offset-aware datetimes")
            }
            let micros = _utc_micros(self) - _utc_micros(other)
            let seconds = micros // MICROS_PER_SECOND
            return timedelta(0, seconds, micros - seconds * MICROS_PER_SECOND)
        }
        throw TypeError(f"cannot subtract {type_of(other)} from a datetime")
    }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, datetime) { return false }
        if (self.tzinfo is null) != (other.tzinfo is null) { return false }
        return _utc_micros(self) == _utc_micros(other)
    }

    fn __lt__(self, other: datetime) -> bool { return self.compare_to(other) < 0 }

    fn __le__(self, other: datetime) -> bool { return self.compare_to(other) <= 0 }

    fn __hash__(self) -> int { return hash(_utc_micros(self)) }

    fn __str__(self) -> str { return self.isoformat(" ") }

    fn __repr__(self) -> str {
        var text = f"datetime.datetime({self.year}, {self.month}, {self.day}, " +
            f"{self.hour}, {self.minute}"
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

const _EPOCH_MICROS = 62_135_596_800_000_000

fn _local_micros(moment: datetime) -> int {
    return (moment.toordinal() - 1) * _MICROS_PER_DAY +
        ((moment.hour * 60 + moment.minute) * 60 + moment.second) * MICROS_PER_SECOND +
        moment.microsecond
}

fn _utc_micros(moment: datetime) -> int {
    return _local_micros(moment) - (moment.offset_micros() ?? 0)
}

fn _from_micros(micros: int, zone: timezone?, overflows: bool) -> datetime {
    let days = micros // _MICROS_PER_DAY
    var rest = micros - days * _MICROS_PER_DAY
    let ordinal = days + 1
    if ordinal < 1 or ordinal > MAXORDINAL {
        if overflows { throw OverflowError("date value out of range") }
        throw ValueError(f"year {days // 366 + 1} is out of range")
    }
    let (year, month, day) = ordinal_to_ymd(ordinal)
    let hour = rest // _MICROS_PER_HOUR
    rest -= hour * _MICROS_PER_HOUR
    let minute = rest // _MICROS_PER_MINUTE
    rest -= minute * _MICROS_PER_MINUTE
    let second = rest // MICROS_PER_SECOND
    return datetime(year, month, day, hour, minute, second, rest - second * MICROS_PER_SECOND, zone)
}

fn _shifted(moment: datetime, span: timedelta) -> datetime {
    let base = timedelta(moment.toordinal(), 0, _local_micros(moment) % _MICROS_PER_DAY)
    let moved = base + span
    if moved.days < 1 or moved.days > MAXORDINAL { throw OverflowError("result out of range") }
    let (year, month, day) = ordinal_to_ymd(moved.days)
    let hour = moved.seconds // 3_600
    let minute = (moved.seconds - hour * 3_600) // 60
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        moved.seconds - hour * 3_600 - minute * 60,
        moved.microseconds,
        moment.tzinfo
    )
}

fn _timestamp_micros(timestamp: float) -> int {
    let (fraction, whole) = _split(timestamp)
    # Checked before the multiply, not after: a timestamp of 1e18 seconds
    # would overflow on the way to microseconds and raise about the integer
    # rather than about the year, which is what CPython names.
    if whole < -62_200_000_000.0 or whole > 253_500_000_000.0 {
        throw ValueError(f"timestamp {timestamp} is out of range")
    }
    var seconds = int(whole)
    var micros = _round_half_even(fraction * 1_000_000.0)
    if micros >= 1_000_000 {
        seconds += 1
        micros -= 1_000_000
    } elif micros < 0 {
        seconds -= 1
        micros += 1_000_000
    }
    return seconds * MICROS_PER_SECOND + micros
}

fn _split(value: float) -> tuple[float, float] { return __prim__.f64_modf(value) }

fn _round_half_even(value: float) -> int {
    let floor_value = __prim__.f64_floor(value)
    let remainder = value - floor_value
    let below = int(floor_value)
    if remainder > 0.5 { return below + 1 }
    if remainder < 0.5 { return below }
    return if below % 2 == 0 { below } else { below + 1 }
}

datetime.min = datetime(MINYEAR, 1, 1)
datetime.max = datetime(MAXYEAR, 12, 31, 23, 59, 59, 999_999)
datetime.resolution = timedelta(0, 0, 1)
