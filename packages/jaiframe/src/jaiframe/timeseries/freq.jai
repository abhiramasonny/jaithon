#: What a frequency is, and the pandas alias it is spelled with.
from std.time import DateTime
from ..column import Column, Dtype

#: Milliseconds in the fixed spans. `Datetime` is datetime64[ms], so these are
#: whole numbers and the arithmetic over them is exact.
pub const MILLIS_PER_SECOND = 1000
pub const MILLIS_PER_MINUTE = 60000
pub const MILLIS_PER_HOUR = 3600000
pub const MILLIS_PER_DAY = 86400000
pub const MILLIS_PER_WEEK = 604800000

let WEEKDAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

let MONTH_NAMES = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
]

#: What a frequency counts.
#:
#: The split that matters is `is_fixed`: the first six are a constant number of
#: milliseconds and bucket by division, the last three are calendar periods and
#: bucket by month index. A month is not 30 days, and a library that treats it
#: as one loses a day every February and two every leap year.
pub enum FreqUnit {
    #: Milliseconds — the resolution the store itself has.
    Milli,
    Second,
    Minute,
    Hour,
    Day,
    #: Seven days, phased to a weekday rather than to the epoch.
    Week,
    Month,
    Quarter,
    #: A year. pandas spells the alias `A` and, since 2.0, `Y`.
    Year,

    #: The pandas alias, which `parse_freq` reads back.
    pub fn name(self) -> str {
        return match self {
            FreqUnit.Milli => "ms",
            FreqUnit.Second => "s",
            FreqUnit.Minute => "min",
            FreqUnit.Hour => "H",
            FreqUnit.Day => "D",
            FreqUnit.Week => "W",
            FreqUnit.Month => "M",
            FreqUnit.Quarter => "Q",
            FreqUnit.Year => "A",
        }
    }

    #: Whether one period is always the same number of milliseconds.
    pub fn is_fixed(self) -> bool {
        return match self {
            FreqUnit.Month | FreqUnit.Quarter | FreqUnit.Year => false,
            _ => true,
        }
    }

    #: Milliseconds in one period, for the fixed units.
    pub fn millis(self) -> int {
        return match self {
            FreqUnit.Milli => 1,
            FreqUnit.Second => MILLIS_PER_SECOND,
            FreqUnit.Minute => MILLIS_PER_MINUTE,
            FreqUnit.Hour => MILLIS_PER_HOUR,
            FreqUnit.Day => MILLIS_PER_DAY,
            FreqUnit.Week => MILLIS_PER_WEEK,
            _ => throw ValueError(f"{self.name()} is a calendar period, not a fixed span"),
        }
    }

    #: Months in one period, for the calendar units.
    pub fn months(self) -> int {
        return match self {
            FreqUnit.Month => 1,
            FreqUnit.Quarter => 3,
            FreqUnit.Year => 12,
            _ => throw ValueError(f"{self.name()} is a fixed span, not a whole number of months"),
        }
    }
}

#: A frequency: how long a period is, and where its edges fall.
#:
#: `count` multiplies the unit, so `5min` and `2W` are one of these. `anchor`
#: is read differently by unit and by nothing else: for `Week` it is the
#: weekday a bin ends on, 0 for Monday, so `W` is `W-SUN` and bins run Monday
#: to Sunday; for `Month`, `Quarter` and `Year` it is the month a period ends
#: in, so `Q` is `Q-DEC` and `A-JUN` is a fiscal year closing in June.
#:
#: `at_end` records which edge the alias named — `M` is a month end and `MS` a
#: month start. It changes no bin, only which instant the bin is labelled with,
#: which is the one thing about resampling that surprises people.
pub class Freq {
    #: What a period counts.
    pub let unit: FreqUnit
    #: How many of them one period spans.
    pub let count: int
    #: The weekday a `Week` bin ends on, or the month a calendar period ends
    #: in. Unread for the other units.
    pub let anchor: int
    #: Whether the alias named the end of a period rather than its start.
    pub let at_end: bool

    #: A frequency of `count` units. The anchor defaults per unit, to Sunday
    #: for a week and December for a calendar period, as pandas' do.
    pub fn init(self, unit: FreqUnit, count: int = 1, anchor: int = -1, at_end: bool = true) {
        if count <= 0 {
            throw ValueError(f"a frequency spans at least one period, got {count}")
        }
        self.unit = unit
        self.count = count
        self.at_end = at_end
        if anchor >= 0 {
            self.anchor = anchor
            return
        }
        self.anchor = if unit == FreqUnit.Week { 6 } else { 12 }
    }

    #: Whether one period is always the same number of milliseconds.
    pub fn is_fixed(self) -> bool { return self.unit.is_fixed() }

    #: Milliseconds in one period, for a fixed frequency.
    pub fn millis(self) -> int { return self.count * self.unit.millis() }

    #: Months in one period, for a calendar frequency.
    pub fn total_months(self) -> int { return self.count * self.unit.months() }

    #: The month index a period boundary is offset by.
    #:
    #: Periods run `[p * total + shift, (p + 1) * total + shift)` over the
    #: 0-based month index `year * 12 + month - 1`. Solving for the shift that
    #: puts the last month of a period at `anchor` leaves the anchor's own
    #: residue, which is why `Q-DEC` shifts by nothing and `Q-JAN` by one.
    pub fn shift(self) -> int {
        let total = self.total_months()
        let residue = self.anchor - self.anchor // total * total
        return residue
    }

    #: The instant a fixed frequency's grid is phased from.
    #:
    #: Zero for everything but a week, whose bins are anchored to a weekday and
    #: not to the epoch — which is what makes `W` mean the same seven days in
    #: two frames that start on different days. 1970-01-01 was a Thursday, so
    #: the phase is the first day of 1970 whose weekday opens the bin.
    pub fn phase(self) -> int {
        if self.unit != FreqUnit.Week { return 0 }
        let opens = self.anchor + 1 - 7 * ((self.anchor + 1) // 7)
        let day = opens - 3 - 7 * ((opens - 3) // 7)
        return day * MILLIS_PER_DAY
    }

    #: The alias this frequency parses back from.
    pub fn to_str(self) -> str {
        var alias = self.unit.name()
        if not self.at_end and not self.unit.is_fixed() { alias = alias + "S" }
        var suffix = ""
        if self.unit == FreqUnit.Week and self.anchor != 6 {
            suffix = "-" + WEEKDAY_NAMES[self.anchor]
        }
        if not self.unit.is_fixed() and self.unit != FreqUnit.Month and self.anchor != 12 {
            suffix = "-" + MONTH_NAMES[self.anchor - 1]
        }
        let head = if self.count == 1 { "" } else { f"{self.count}" }
        return head + alias + suffix
    }
}

#: A pandas frequency alias — `5min`, `H`, `2W-MON`, `QS`, `A-JUN`.
#:
#: A leading integer multiplies, an alias names the unit, and a `-` suffix
#: anchors it. The alias table is case-sensitive where pandas' is: `ms` is
#: milliseconds and `MS` is a month start, and letting one stand for the other
#: is a two-order-of-magnitude bug in a resample nobody would look at twice.
pub fn parse_freq(text: str) -> Freq {
    let source = text.strip()
    if source.len() == 0 { throw ValueError("a frequency alias cannot be empty") }
    var at = 0
    while at < source.len() and source[at] >= "0" and source[at] <= "9" { at += 1 }
    let count = if at == 0 { 1 } else { source[0:at].to_int() ?? 0 }
    if count <= 0 {
        throw ValueError(f"a frequency spans at least one period, got {repr(text)}")
    }
    let rest = source[at:source.len()]
    if rest.len() == 0 { throw ValueError(f"{repr(text)} names no unit") }
    let dash = rest.find("-")
    let alias = if dash < 0 { rest } else { rest[0:dash] }
    let suffix = if dash < 0 { "" } else { rest[dash + 1:rest.len()].upper() }
    let named = unit_of_alias(alias)
    let unit = named[0]
    let at_end = named[1]
    if suffix.len() == 0 { return Freq(unit, count: count, at_end: at_end) }
    if unit == FreqUnit.Week {
        return Freq(unit, count: count, anchor: weekday_of_name(suffix, text), at_end: at_end)
    }
    if unit.is_fixed() {
        throw ValueError(f"{repr(text)} anchors a fixed frequency, which has nothing to anchor to")
    }
    return Freq(unit, count: count, anchor: month_of_name(suffix, text), at_end: at_end)
}

#: A frequency however it was spelled — an alias or an already-parsed `Freq`.
pub fn to_freq(spec: any) -> Freq {
    if isinstance(spec, Freq) { return spec }
    if isinstance(spec, str) { return parse_freq(spec) }
    throw TypeError("a frequency is an alias string or a Freq")
}

fn unit_of_alias(alias: str) -> (FreqUnit, bool) {
    return match alias {
        "ms" | "L" | "l" => (FreqUnit.Milli, true),
        "s" | "S" | "sec" | "secs" => (FreqUnit.Second, true),
        "min" | "T" | "t" | "mins" => (FreqUnit.Minute, true),
        "H" | "h" | "hr" | "hrs" => (FreqUnit.Hour, true),
        "D" | "d" => (FreqUnit.Day, true),
        "W" | "w" => (FreqUnit.Week, true),
        "M" | "ME" => (FreqUnit.Month, true),
        "MS" => (FreqUnit.Month, false),
        "Q" | "QE" => (FreqUnit.Quarter, true),
        "QS" => (FreqUnit.Quarter, false),
        "A" | "Y" | "AE" | "YE" => (FreqUnit.Year, true),
        "AS" | "YS" => (FreqUnit.Year, false),
        _ => match alias.lower() {
            "milli" | "millis" | "millisecond" | "milliseconds" => (FreqUnit.Milli, true),
            "second" | "seconds" => (FreqUnit.Second, true),
            "minute" | "minutes" => (FreqUnit.Minute, true),
            "hour" | "hours" => (FreqUnit.Hour, true),
            "day" | "days" => (FreqUnit.Day, true),
            "week" | "weeks" => (FreqUnit.Week, true),
            "month" | "months" => (FreqUnit.Month, true),
            "quarter" | "quarters" => (FreqUnit.Quarter, true),
            "year" | "years" | "annual" => (FreqUnit.Year, true),
            _ => throw ValueError(f"unknown frequency alias {repr(alias)}"),
        },
    }
}

fn weekday_of_name(name: str, whole: str) -> int {
    for (at, day) in WEEKDAY_NAMES.enumerate() {
        if name.startswith(day) { return at }
    }
    throw ValueError(f"{repr(whole)} anchors to no weekday this calendar has")
}

fn month_of_name(name: str, whole: str) -> int {
    for (at, month) in MONTH_NAMES.enumerate() {
        if name.startswith(month) { return at + 1 }
    }
    throw ValueError(f"{repr(whole)} anchors to no month this calendar has")
}

#: ```
#: let stamps = to_datetime(["2024-01-01", null, "2024-01-01T06:30:00Z"])
#: defer { stamps.free() }
#: stamps.to_ints()      # [1704067200000, 0, 1704090600000]
#: stamps.null_mask()    # [false, true, false]
#: ```
pub fn to_datetime(texts: list[str?]) -> Column {
    var stamps: list[int] = []
    var nulls: list[bool] = []
    var parsed: dict[str, int] = {}
    for text in texts {
        if text is null {
            stamps.push(0)
            nulls.push(true)
            continue
        }
        let spelling = text ?? ""
        if parsed.has(spelling) {
            stamps.push(parsed[spelling])
        } else {
            let stamp = millis_of(spelling)
            parsed[spelling] = stamp
            stamps.push(stamp)
        }
        nulls.push(false)
    }
    return Column.from_ints(stamps, nulls, Dtype.Datetime)
}

#: What `DateTime` counts its fraction in, against the millisecond this stores.
const NANOS_PER_MILLI = 1000000

#: The millisecond count one ISO 8601 timestamp names.
#:
#: `DateTime` has already normalised the fraction into `[0, 1e9)` against a
#: floored second, so this is exactly the inverse of the split `io.jai`'s
#: `iso_stamp` makes and a moment before the epoch survives the round trip.
fn millis_of(text: str) -> int {
    let moment = DateTime.from_iso(text)
    return moment.to_unix() * MILLIS_PER_SECOND + moment.nanosecond // NANOS_PER_MILLI
}
