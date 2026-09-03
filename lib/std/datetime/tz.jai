#: `timezone`, a fixed offset from UTC.
#:
#: This is the whole of the module's time-zone support, exactly as
#: `datetime.timezone` is the whole of CPython's `datetime` without
#: `zoneinfo`: an offset and an optional name, with no rules and so no
#: daylight saving. `dst` is here so the shape matches, and it answers null
#: for every argument.
from .common import MICROS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE, pad
from .delta import timedelta

const _MICROS_PER_HOUR = 3_600_000_000
const _MICROS_PER_MINUTE = 60_000_000
const _MICROS_PER_DAY = 86_400_000_000

#: The stand-in for an argument the caller left out.
#:
#: `timezone(offset)` and `timezone(offset, null)` mean different things —
#: the first invents a name from the offset, the second is a request for no
#: name at all — so the default cannot itself be null.
pub class Omitted {
    #: Create the marker. There is never a reason to hold more than one.
    pub fn init(self) { }
}

let _OMITTED = Omitted()

#: A fixed offset east of UTC, optionally carrying a name.
#:
#: ```
#: str(timezone.utc)                                  # "UTC"
#: str(timezone(timedelta(hours: 5, minutes: 30)))    # "UTC+05:30"
#: timezone(timedelta(hours: -3)).utcoffset().days    # -1
#: ```
pub class timezone {
    let offset: timedelta
    let label: str?

    #: UTC itself, the offset every conversion in this module starts from.
    pub static var utc: timezone

    #: The westernmost offset, `-23:59`.
    pub static var min: timezone

    #: The easternmost offset, `+23:59`.
    pub static var max: timezone

    #: Create the zone `offset` east of UTC, named `name` if one is given.
    #:
    #: The offset must be strictly inside +/-24 hours. Sub-minute offsets are
    #: accepted, as CPython has accepted them since 3.7.
    #:
    #: ```
    #: timezone(timedelta(hours: 1)).tzname()    # "UTC+01:00"
    #: timezone(timedelta(0), "Z").tzname()      # "Z"
    #: ```
    pub fn init(self, offset: timedelta, name: any = _OMITTED) {
        if offset.days < -1 or offset.days > 0 or _delta_micros(offset) <= -_MICROS_PER_DAY {
            throw ValueError(
                "offset must be a timedelta strictly between -timedelta(hours=24) " +
                    f"and timedelta(hours=24), not {repr(offset)}."
            )
        }
        self.offset = offset
        var chosen: str? = null
        if isinstance(name, str) { chosen = str(name) }
        self.label = chosen
    }

    #: Return the offset east of UTC, ignoring the moment asked about.
    #:
    #: ```
    #: timezone.utc.utcoffset().days    # 0
    #: ```
    pub fn utcoffset(self, moment: any = null) -> timedelta { return self.offset }

    #: Return the zone's name: the one given to `init`, else `UTC+HH:MM`.
    #:
    #: ```
    #: timezone(timedelta(minutes: -330)).tzname()    # "UTC-05:30"
    #: ```
    pub fn tzname(self, moment: any = null) -> str {
        if self.label is null { return name_from_offset(_delta_micros(self.offset)) }
        return self.label ?? ""
    }

    #: Return null: a fixed offset has no daylight-saving component.
    pub fn dst(self, moment: any = null) -> timedelta? { return null }

    #: Return the offset in whole microseconds east of UTC.
    #:
    #: Not a CPython method. Every conversion in this module wants the offset
    #: as one number, and unpacking three `timedelta` fields at each of those
    #: sites would repeat the same line a dozen times.
    pub fn offset_micros(self) -> int { return _delta_micros(self.offset) }

    fn __eq__(self, other: any) -> bool {
        if not isinstance(other, timezone) { return false }
        return _delta_micros(self.offset) == other.offset_micros()
    }

    fn __hash__(self) -> int { return hash(_delta_micros(self.offset)) }

    fn __str__(self) -> str { return self.tzname() }

    fn __repr__(self) -> str {
        if self.label is null {
            if _delta_micros(self.offset) == 0 { return "datetime.timezone.utc" }
            return f"datetime.timezone({repr(self.offset)})"
        }
        let quoted = "'" + (self.label ?? "") + "'"
        return f"datetime.timezone({repr(self.offset)}, {quoted})"
    }
}

#: Render an offset the way `%z` and `isoformat` do, given its microseconds.
#:
#: `separator` is `":"` for ISO 8601 and `""` for `%z`. Seconds and
#: microseconds appear only when they are not zero, which is why an ordinary
#: offset stays four digits wide.
#:
#: ```
#: format_offset(19800000000, ":")     # "+05:30"
#: format_offset(-19845000000, "")     # "-053045"
#: ```
pub fn format_offset(micros: int, separator: str) -> str {
    let sign = if micros < 0 { "-" } else { "+" }
    var rest = if micros < 0 { -micros } else { micros }
    let hours = rest // _MICROS_PER_HOUR
    rest -= hours * _MICROS_PER_HOUR
    let minutes = rest // _MICROS_PER_MINUTE
    rest -= minutes * _MICROS_PER_MINUTE
    let seconds = rest // MICROS_PER_SECOND
    rest -= seconds * MICROS_PER_SECOND
    var text = sign + pad(hours, 2) + separator + pad(minutes, 2)
    if seconds != 0 or rest != 0 {
        text = text + separator + pad(seconds, 2)
        if rest != 0 { text = text + "." + pad(rest, 6) }
    }
    return text
}

#: Return the name CPython invents for an unnamed offset of `micros`.
#:
#: ```
#: name_from_offset(0)             # "UTC"
#: name_from_offset(3600000000)    # "UTC+01:00"
#: ```
pub fn name_from_offset(micros: int) -> str {
    if micros == 0 { return "UTC" }
    return "UTC" + format_offset(micros, ":")
}

#: Build the zone `micros` microseconds east of UTC, reusing `utc` at zero.
pub fn zone_of_micros(micros: int) -> timezone {
    if micros == 0 { return timezone.utc }
    let seconds = micros // MICROS_PER_SECOND
    return timezone(timedelta(0, seconds, micros - seconds * MICROS_PER_SECOND))
}

fn _delta_micros(offset: timedelta) -> int {
    return (offset.days * SECONDS_PER_DAY + offset.seconds) * MICROS_PER_SECOND +
        offset.microseconds
}

timezone.utc = timezone(timedelta(0))
timezone.min = timezone(timedelta(0, -SECONDS_PER_HOUR * 23 - SECONDS_PER_MINUTE * 59))
timezone.max = timezone(timedelta(0, SECONDS_PER_HOUR * 23 + SECONDS_PER_MINUTE * 59))

#: UTC, spelled as `datetime.UTC` is in CPython 3.11 and later.
pub let UTC: timezone = timezone.utc
