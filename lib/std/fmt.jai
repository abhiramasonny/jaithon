#: Presentation: format specs, number and unit rendering, tables, progress
#: bars, colour, and pretty-printing.
#:
#: ## Format specs
#:
#: `format` and f-string interpolation share the built-in format engine. A
#: replacement field is
#:
#:     {[selector][:spec]}
#:
#: where `selector` is a positional index (`{0}`), a name (`{count}`, only
#: available to f-strings, which capture the binding), or empty for "the next
#: argument", and `spec` is
#:
#:     [[fill]align][sign][#][0][width][group][.precision][type]
#:
#: | Part | Values | Meaning |
#: |---|---|---|
#: | fill | any character | pad with this; requires an explicit align |
#: | align | `<` `>` `^` `=` | left, right, centre, pad between sign and digits |
#: | sign | `+` `-` space | always show, negative only (default), leading space |
#: | `#` | — | prefix `0b`, `0o` or `0x` on the integer bases |
#: | `0` | — | zero-pad the field, the same as `0=` |
#: | width | integer | minimum field width, in characters |
#: | group | `,` `_` | digit-group separator |
#: | precision | `.` integer | digits after the point, or max length for `s` |
#: | type | below | how to render the value |
#:
#: Integer types are `d` decimal, `b` binary, `o` octal, `x`/`X` hex and `c`
#: codepoint. Float types are `f` fixed, `e`/`E` scientific, `g`/`G` general and
#: `%` percentage. Any value accepts `s` for `str(value)` and `r` for
#: `repr(value)`. `{{` and `}}` are literal braces.
#:
#: Padding in a spec counts characters. When the text may contain wide or
#: combining characters, or ANSI colour, pad with `align_left`, `align_right`
#: and `center`, which count display cells instead.
#:
#: ```
#: format("{}: {:.2f}", "pi", 3.14159)   # "pi: 3.14"
#: format("[{:>5}]", 42)                 # "[   42]"
#: ```
from std.str import StringBuilder, quote, repeat

const _STDOUT = 1

let _BINARY_UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
let _DECIMAL_UNITS = ["B", "kB", "MB", "GB", "TB", "PB", "EB"]

# Scalars drawn in two terminal cells, and scalars drawn in none. Both are
# sorted, so `_in_ranges` can stop at the first range that starts too late.
let _WIDE_RANGES: list[(int, int)] = [
    (0x1100, 0x115F),
    (0x2E80, 0x303E),
    (0x3041, 0x33FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA000, 0xA4CF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE19),
    (0xFE30, 0xFE6F),
    (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6),
    (0x1F300, 0x1F64F),
    (0x1F900, 0x1F9FF),
    (0x20000, 0x3FFFD),
]

let _ZERO_RANGES: list[(int, int)] = [
    (0x0300, 0x036F),
    (0x0483, 0x0489),
    (0x0591, 0x05BD),
    (0x1AB0, 0x1AFF),
    (0x200B, 0x200F),
    (0x20D0, 0x20FF),
    (0xFE00, 0xFE0F),
    (0xFE20, 0xFE2F),
]

var _colors_enabled = true

#: How a table column places its text inside the column width.
pub enum Align {
    #: Text against the left edge, padding on the right. The default for a
    #: column, and what `TableStyle.Markdown` writes as a bare dashed marker.
    Left,
    #: Text against the right edge, padding on the left. What a numeric column
    #: wants, so that digits of the same place value line up.
    Right,
    #: Padding split between the two sides. An odd cell of padding goes to the
    #: right, so a one-cell-narrow text sits one column left of centre.
    Center,
}

#: The frame a table is drawn with.
pub enum TableStyle {
    #: A ruled frame of box-drawing characters, with the header divided from
    #: the body. The default of `render`.
    Box,
    #: Pipe-delimited rows and a marker row under the header. The column's
    #: `Align` is carried in that row's colons, so the alignment survives into
    #: whatever renders the Markdown rather than being baked into the padding.
    Markdown,
    #: Columns padded to width and separated by two spaces, with a dashed rule
    #: under the header and no vertical rules. The style to pipe into `cut`,
    #: `awk` or anything else that splits on whitespace.
    Plain,
}

#: A text table whose columns are sized from their widest display cell.
#:
#: Every column is declared before the first row, so a row with the wrong
#: number of cells is caught where it is added rather than at render time. Cell
#: text may contain ANSI colour: widths are measured with `display_width`, which
#: ignores escapes and counts wide characters as two cells.
pub class Table {
    var headers: list[str]
    var alignments: list[Align]
    var rows: list[list[str]]

    #: Create a table with no columns.
    pub fn init(self) {
        self.headers = []
        self.alignments = []
        self.rows = []
    }

    #: Append a column. Raises `ValueError` once any row has been added.
    pub fn add_column(self, header: str, alignment: Align = Align.Left) -> void {
        if self.rows.len() > 0 {
            throw ValueError("every column must be added before the first row")
        }
        self.headers.push(header)
        self.alignments.push(alignment)
    }

    #: Append a row of already-stringified cells, one per column.
    pub fn add_row(self, cells: list[str]) -> void {
        if self.headers.len() == 0 {
            throw ValueError("add_column must be called before add_row")
        }
        if cells.len() != self.headers.len() {
            throw ValueError(f"row has {cells.len()} cells, expected {self.headers.len()}")
        }
        self.rows.push(cells)
    }

    #: Number of rows added so far.
    pub fn row_count(self) -> int { return self.rows.len() }

    #: Render the table, with no trailing newline.
    #:
    #: An empty table renders as the empty string.
    pub fn render(self, style: TableStyle = TableStyle.Box) -> str {
        if self.headers.len() == 0 { return "" }
        let widths = self.column_widths()
        return match style {
            TableStyle.Box => self.render_box(widths),
            TableStyle.Markdown => self.render_markdown(widths),
            TableStyle.Plain => self.render_plain(widths),
        }
    }

    fn column_widths(self) -> list[int] {
        let widths = [display_width(header) for header in self.headers]
        for row in self.rows {
            for (index, cell) in row.enumerate() {
                widths[index] = max(widths[index], display_width(cell))
            }
        }
        return widths
    }

    fn pad(self, text: str, width: int, alignment: Align) -> str {
        return match alignment {
            Align.Left => align_left(text, width),
            Align.Right => align_right(text, width),
            Align.Center => center(text, width),
        }
    }

    fn render_box(self, widths: list[int]) -> str {
        let out = [self.rule(widths, "┌", "┬", "┐"), self.box_line(self.headers, widths)]
        out.push(self.rule(widths, "├", "┼", "┤"))
        for row in self.rows { out.push(self.box_line(row, widths)) }
        out.push(self.rule(widths, "└", "┴", "┘"))
        return "\n".join(out)
    }

    fn rule(self, widths: list[int], left: str, middle: str, right: str) -> str {
        let parts = [repeat("─", width + 2) for width in widths]
        return left + middle.join(parts) + right
    }

    fn box_line(self, cells: list[str], widths: list[int]) -> str {
        let parts: list[str] = []
        for (index, cell) in cells.enumerate() {
            parts.push(" " + self.pad(cell, widths[index], self.alignments[index]) + " ")
        }
        return "│" + "│".join(parts) + "│"
    }

    fn render_markdown(self, widths: list[int]) -> str {
        let out = [self.markdown_line(self.headers, widths)]
        let markers: list[str] = []
        for (index, width) in widths.enumerate() {
            let dashes = repeat("-", max(3, width))
            let marker = match self.alignments[index] {
                Align.Left => dashes,
                Align.Right => dashes[0:dashes.len() - 1] + ":",
                Align.Center => ":" + dashes[0:dashes.len() - 2] + ":",
            }
            markers.push(marker)
        }
        out.push("| " + " | ".join(markers) + " |")
        for row in self.rows { out.push(self.markdown_line(row, widths)) }
        return "\n".join(out)
    }

    fn markdown_line(self, cells: list[str], widths: list[int]) -> str {
        let parts: list[str] = []
        for (index, cell) in cells.enumerate() {
            # Widen the column by whatever the escaping added, so the rendered
            # table still lines up when a cell contains a pipe.
            let escaped = cell.replace("|", "\\|")
            let extra = display_width(escaped) - display_width(cell)
            parts.push(self.pad(escaped, widths[index] + extra, self.alignments[index]))
        }
        return "| " + " | ".join(parts) + " |"
    }

    fn render_plain(self, widths: list[int]) -> str {
        let out = [self.plain_line(self.headers, widths)]
        out.push("  ".join([repeat("-", width) for width in widths]))
        for row in self.rows { out.push(self.plain_line(row, widths)) }
        return "\n".join(out)
    }

    fn plain_line(self, cells: list[str], widths: list[int]) -> str {
        let parts: list[str] = []
        for (index, cell) in cells.enumerate() {
            parts.push(self.pad(cell, widths[index], self.alignments[index]))
        }
        return "  ".join(parts)
    }

    fn __str__(self) -> str { return self.render() }
}

#: A single-line progress bar for a terminal.
#:
#: `render` is pure, so it can be logged or tested; `draw` rewrites the current
#: line in place. Call `finish` once, which fills the bar and ends the line, so
#: later output does not land on top of it.
pub class ProgressBar {
    var total: int
    var current: int
    var width: int
    var label: str
    var filled: str
    var empty: str

    #: Create a bar for `total` units of work.
    pub fn init(self, total: int, width: int = 40, label: str = "") {
        if total <= 0 { throw ValueError(f"total must be positive, got {total}") }
        if width <= 0 { throw ValueError(f"width must be positive, got {width}") }
        self.total = total
        self.current = 0
        self.width = width
        self.label = label
        self.filled = "█"
        self.empty = "░"
    }

    #: Use `filled` and `empty` as the bar characters instead of the blocks.
    pub fn set_characters(self, filled: str, empty: str) -> void {
        self.filled = filled
        self.empty = empty
    }

    #: Move the bar to `value`, clamped to 0 and `total`.
    pub fn update(self, value: int) -> void {
        self.current = max(0, min(value, self.total))
    }

    #: Move the bar forward by `step` units.
    pub fn advance(self, step: int = 1) -> void { self.update(self.current + step) }

    #: Progress so far, from 0.0 to 1.0.
    pub fn fraction(self) -> float { return float(self.current) / float(self.total) }

    #: Whether every unit of work is accounted for.
    pub fn is_done(self) -> bool { return self.current >= self.total }

    #: Render the bar as one line, without a newline or carriage return.
    pub fn render(self) -> str {
        let done = int(self.fraction() * self.width)
        let bar = repeat(self.filled, done) + repeat(self.empty, self.width - done)
        let percent = align_right(fixed(self.fraction() * 100.0, 0) + "%", 4)
        let body = f"[{bar}] {percent} {self.current}/{self.total}"
        if self.label.len() == 0 { return body }
        return self.label + " " + body
    }

    #: Redraw the bar over the current terminal line.
    #:
    #: Writes to file descriptor 1 directly, because a carriage return has to
    #: reach the terminal without the newline `print` appends.
    pub fn draw(self) -> void {
        __prim__.io_write(_STDOUT, "\r" + self.render())
        __prim__.io_flush(_STDOUT)
    }

    #: Fill the bar, draw it a final time, and end the line.
    pub fn finish(self) -> void {
        self.update(self.total)
        self.draw()
        __prim__.io_write(_STDOUT, "\n")
        __prim__.io_flush(_STDOUT)
    }

    fn __str__(self) -> str { return self.render() }
}

#: Format `args` into `template` using the built-in format engine.
#:
#: Only positional fields are filled, since the arguments arrive without names;
#: use an f-string for `{name}` fields, or `std.str.expand` when the template
#: comes from outside the program.
#:
#: ```
#: format("{} of {}", 3, 10)     # "3 of 10"
#: format("{:08.3f}", 3.14159)   # "0003.142"
#: ```
pub fn format(template: str, ...args: any) -> str {
    return template.format(...args)
}

#: Render `value` with exactly `digits` digits after the decimal point.
#:
#: ```
#: fixed(3.14159, 2)   # "3.14"
#: fixed(2.0, 0)       # "2"
#: ```
pub fn fixed(value: float, digits: int = 2) -> str {
    if digits < 0 { throw ValueError(f"digits must be non-negative, got {digits}") }
    return format("{:." + str(digits) + "f}", value)
}

#: Group the integer digits of `value` in threes.
#:
#: ```
#: with_commas(1234567)        # "1,234,567"
#: with_commas(-1000, " ")     # "-1 000"
#: ```
pub fn with_commas(value: int, separator: str = ",") -> str {
    return group_digits(str(value), separator)
}

#: Group the integer digits of an already-formatted number in threes.
#:
#: Any sign, decimal part and exponent are preserved, so this composes with
#: `fixed` and `scientific`.
#:
#: ```
#: group_digits(fixed(1234567.891, 2))   # "1,234,567.89"
#: ```
pub fn group_digits(text: str, separator: str = ",") -> str {
    var start = 0
    if text.len() > 0 and (text[0] == "-" or text[0] == "+") { start = 1 }
    var end = start
    while end < text.len() and _is_ascii_digit(text[end]) { end += 1 }
    let body = text[start:end]
    if body.len() <= 3 { return text }
    let groups: list[str] = []
    var cut = body.len()
    while cut > 3 {
        groups.push(body[cut - 3:cut])
        cut -= 3
    }
    let ordered = [body[0:cut]]
    var index = groups.len() - 1
    while index >= 0 {
        ordered.push(groups[index])
        index -= 1
    }
    return text[0:start] + separator.join(ordered) + text[end:text.len()]
}

#: Render `value` as a percentage, where 1.0 is 100%.
#:
#: ```
#: as_percent(0.256)      # "25.6%"
#: as_percent(0.5, 0)     # "50%"
#: ```
pub fn as_percent(value: float, digits: int = 1) -> str {
    return fixed(value * 100.0, digits) + "%"
}

#: Render a byte count with a binary (KiB) or decimal (kB) unit.
#:
#: Counts below the base are shown as whole bytes. Rounding that would reach the
#: base carries into the next unit, so 1048575 bytes is "1.0 MiB", never
#: "1024.0 KiB".
#:
#: ```
#: as_bytes(1536)          # "1.5 KiB"
#: as_bytes(512)           # "512 B"
#: as_bytes(1500, false)   # "1.5 kB"
#: ```
pub fn as_bytes(count: int, binary: bool = true, digits: int = 1) -> str {
    let units = binary ? _BINARY_UNITS : _DECIMAL_UNITS
    let scale = binary ? 1024.0 : 1000.0
    let sign = count < 0 ? "-" : ""
    let magnitude = abs(count)
    if magnitude < int(scale) { return sign + str(magnitude) + " " + units[0] }
    var value = float(magnitude)
    var index = 0
    while value >= scale and index < units.len() - 1 {
        value = value / scale
        index += 1
    }
    if _round_to(value, digits) >= scale and index < units.len() - 1 {
        value = value / scale
        index += 1
    }
    return sign + fixed(value, digits) + " " + units[index]
}

#: Render a duration in seconds as its `parts` most significant units.
#:
#: Below a second the result is whole milliseconds, below ten seconds it is one
#: decimal, and above that it is days, hours, minutes and seconds with the
#: zero-valued units dropped.
#:
#: ```
#: as_duration(5025.0)    # "1h 23m"
#: as_duration(90.0)      # "1m 30s"
#: as_duration(0.25)      # "250ms"
#: ```
pub fn as_duration(seconds: float, parts: int = 2) -> str {
    if parts < 1 { throw ValueError(f"parts must be at least 1, got {parts}") }
    if seconds < 0.0 { return "-" + as_duration(-seconds, parts) }
    if seconds == 0.0 { return "0s" }
    if seconds < 1.0 { return fixed(seconds * 1000.0, 0) + "ms" }
    if seconds < 10.0 { return fixed(seconds, 1) + "s" }
    let units: list[(int, str)] = [(86400, "d"), (3600, "h"), (60, "m"), (1, "s")]
    let out: list[str] = []
    var remaining = int(seconds)
    for (size, suffix) in units {
        let amount = remaining // size
        remaining = remaining % size
        if amount > 0 { out.push(f"{amount}{suffix}") }
        if out.len() == parts { break }
    }
    return " ".join(out)
}

#: Render `value` as an English ordinal.
#:
#: ```
#: as_ordinal(1)    # "1st"
#: as_ordinal(12)   # "12th"
#: as_ordinal(22)   # "22nd"
#: ```
pub fn as_ordinal(value: int) -> str {
    let tens = abs(value) % 100
    if tens >= 11 and tens <= 13 { return str(value) + "th" }
    let suffix = match abs(value) % 10 {
        1 => "st",
        2 => "nd",
        3 => "rd",
        _ => "th",
    }
    return str(value) + suffix
}

#: Render `value` in scientific notation with `digits` decimals.
#:
#: ```
#: scientific(1500.0, 2)     # "1.50e+03"
#: scientific(-0.00042, 1)   # "-4.2e-04"
#: ```
pub fn scientific(value: float, digits: int = 3) -> str {
    if not _is_finite(value) { return str(value) }
    let (mantissa, exponent) = _decompose(value, digits)
    return fixed(mantissa, digits) + "e" + _exponent_suffix(exponent)
}

#: Render `value` in engineering notation: an exponent that is a multiple of 3.
#:
#: This is the form that lines up with SI prefixes, so 12000 reads as 12.0e+03
#: rather than 1.2e+04.
#:
#: ```
#: engineering(12000.0, 1)   # "12.0e+03"
#: ```
pub fn engineering(value: float, digits: int = 3) -> str {
    if not _is_finite(value) { return str(value) }
    let (mantissa, exponent) = _decompose(value, digits)
    var shift = exponent % 3
    if shift < 0 { shift += 3 }
    let scaled = mantissa * 10.0 ** shift
    return fixed(scaled, digits) + "e" + _exponent_suffix(exponent - shift)
}

#: Pad `text` on the right until it occupies `width` display cells.
#:
#: `fill` must be one cell wide. Text already at least that wide is returned
#: unchanged rather than truncated.
#:
#: ```
#: align_left("ab", 5) + "|"   # "ab   |"
#: ```
pub fn align_left(text: str, width: int, fill: str = " ") -> str {
    return text + repeat(_one_cell(fill), max(0, width - display_width(text)))
}

#: Pad `text` on the left until it occupies `width` display cells.
#:
#: ```
#: align_right("7", 3, "0")   # "007"
#: ```
pub fn align_right(text: str, width: int, fill: str = " ") -> str {
    return repeat(_one_cell(fill), max(0, width - display_width(text))) + text
}

#: Pad `text` on both sides until it occupies `width` display cells.
#:
#: An odd remainder goes on the right.
#:
#: ```
#: center("ab", 6, ".")   # "..ab.."
#: center("ab", 5, ".")   # ".ab.."
#: ```
pub fn center(text: str, width: int, fill: str = " ") -> str {
    let pad = max(0, width - display_width(text))
    let left = pad // 2
    let cell = _one_cell(fill)
    return repeat(cell, left) + text + repeat(cell, pad - left)
}

#: Count the terminal cells `text` occupies.
#:
#: ANSI escapes and control characters count as nothing, combining marks and
#: zero-width scalars count as nothing, East Asian wide scalars and emoji count
#: as two, and everything else counts as one. A tab counts as one cell, since
#: its real width depends on the column it starts in.
#:
#: ```
#: display_width("日本")   # 4
#: display_width("abc")   # 3
#: ```
pub fn display_width(text: str) -> int {
    # Stripping allocates, and the overwhelmingly common case has no escapes.
    let plain = "\x1B" in text ? strip_ansi(text) : text
    var total = 0
    for ch in plain {
        let code = ord(ch)
        if code < 32 or code == 127 { continue }
        if _in_ranges(code, _ZERO_RANGES) { continue }
        total += _in_ranges(code, _WIDE_RANGES) ? 2 : 1
    }
    return total
}

#: Remove every ANSI escape sequence from `text`.
#:
#: Handles CSI sequences such as colour codes and OSC sequences such as window
#: titles, so the result is what a terminal would actually show.
#:
#: ```
#: strip_ansi("\x1b[31mred\x1b[39m")   # "red"
#: ```
pub fn strip_ansi(text: str) -> str {
    let out = StringBuilder()
    let length = text.len()
    var index = 0
    while index < length {
        if text[index] != "\x1B" {
            out.push(text[index])
            index += 1
            continue
        }
        index += 1
        if index >= length { break }
        if text[index] == "[" {
            index += 1
            while index < length and not _is_csi_final(text[index]) { index += 1 }
            if index < length { index += 1 }
        } elif text[index] == "]" {
            index = _skip_osc(text, index + 1)
        } else {
            index += 1
        }
    }
    return out.build()
}

#: Pretty-print nested lists, dicts, sets and tuples over multiple lines.
#:
#: Containers already on the path are printed as `<cycle>` instead of recursing
#: forever, so a structure that refers to itself is still printable. Strings are
#: quoted and escaped; anything else falls back to `str(value)`.
#:
#: ```
#: pretty([1, 2])   # "[\n  1,\n  2\n]"
#: ```
pub fn pretty(value: any, indent: int = 2) -> str {
    if indent < 0 { throw ValueError(f"indent must be non-negative, got {indent}") }
    return _pretty(value, indent, 0, [])
}

#: Whether the colour helpers currently emit escape sequences.
pub fn colors_enabled() -> bool { return _colors_enabled }

#: Turn the colour helpers on or off for the whole program.
#:
#: Call this once at startup with the answer to "is stdout a terminal, and is
#: NO_COLOR unset". Every helper below then becomes the identity function, so no
#: call site needs a condition.
pub fn set_colors_enabled(enabled: bool) -> void {
    _colors_enabled = enabled
}

#: Colour `text` red.
#:
#: ```
#: red("x")   # "\x1b[31mx\x1b[39m"
#: ```
pub fn red(text: str) -> str { return _sgr("31", "39", text) }

#: Colour `text` green.
pub fn green(text: str) -> str { return _sgr("32", "39", text) }

#: Colour `text` yellow.
pub fn yellow(text: str) -> str { return _sgr("33", "39", text) }

#: Colour `text` blue.
pub fn blue(text: str) -> str { return _sgr("34", "39", text) }

#: Colour `text` magenta.
pub fn magenta(text: str) -> str { return _sgr("35", "39", text) }

#: Colour `text` cyan.
pub fn cyan(text: str) -> str { return _sgr("36", "39", text) }

#: Render `text` bold.
pub fn bold(text: str) -> str { return _sgr("1", "22", text) }

#: Render `text` dim.
pub fn dim(text: str) -> str { return _sgr("2", "22", text) }

#: Underline `text`.
pub fn underline(text: str) -> str { return _sgr("4", "24", text) }

# Each helper closes with the reset for its own attribute rather than `0`, so
# colouring part of an already-styled string leaves the outer style intact.
fn _sgr(code: str, reset: str, text: str) -> str {
    if not _colors_enabled { return text }
    return "\x1B[" + code + "m" + text + "\x1B[" + reset + "m"
}

fn _pretty(value: any, indent: int, depth: int, seen: list[int]) -> str {
    if value == null { return "null" }
    if isinstance(value, str) { return quote(value) }
    let is_list = isinstance(value, list)
    let is_tuple = isinstance(value, tuple)
    let is_dict = isinstance(value, dict)
    let is_set = isinstance(value, set)
    if not (is_list or is_tuple or is_dict or is_set) { return str(value) }
    if id(value) in seen { return "<cycle>" }
    let path = seen + [id(value)]
    let inner = repeat(" ", (depth + 1) * indent)
    let parts: list[str] = []
    if is_dict {
        for (key, item) in value.items() {
            let rendered_key = _pretty(key, indent, depth + 1, path)
            let rendered_item = _pretty(item, indent, depth + 1, path)
            parts.push(inner + rendered_key + ": " + rendered_item)
        }
    } else {
        for item in value { parts.push(inner + _pretty(item, indent, depth + 1, path)) }
    }
    if parts.len() == 0 {
        if is_dict { return "{}" }
        if is_set { return "set()" }
        return is_list ? "[]" : "()"
    }
    let open = is_list ? "[" : is_tuple ? "(" : "{"
    let close = is_list ? "]" : is_tuple ? ")" : "}"
    return open + "\n" + ",\n".join(parts) + "\n" + repeat(" ", depth * indent) + close
}

fn _in_ranges(code: int, ranges: list[(int, int)]) -> bool {
    for (low, high) in ranges {
        if code < low { return false }
        if code <= high { return true }
    }
    return false
}

fn _is_finite(value: float) -> bool {
    return value == value and abs(value) <= 1.7976931348623157e308
}

fn _round_to(value: float, digits: int) -> float {
    let scale = 10.0 ** digits
    return __prim__.f64_round(value * scale) / scale
}

# Split `value` into a mantissa in [1, 10) and a power of ten, then correct for
# a mantissa that rounding at `digits` would carry up to exactly 10.
fn _decompose(value: float, digits: int) -> (float, int) {
    if value == 0.0 { return (0.0, 0) }
    var mantissa = abs(value)
    var exponent = 0
    while mantissa >= 10.0 {
        mantissa = mantissa / 10.0
        exponent += 1
    }
    while mantissa < 1.0 {
        mantissa = mantissa * 10.0
        exponent -= 1
    }
    if _round_to(mantissa, digits) >= 10.0 {
        mantissa = mantissa / 10.0
        exponent += 1
    }
    let signed = value < 0.0 ? -mantissa : mantissa
    return (signed, exponent)
}

fn _exponent_suffix(exponent: int) -> str {
    let sign = exponent < 0 ? "-" : "+"
    let digits = str(abs(exponent))
    if digits.len() < 2 { return sign + "0" + digits }
    return sign + digits
}

fn _one_cell(fill: str) -> str {
    if display_width(fill) != 1 {
        throw ValueError(f"fill must be one display cell wide, got {quote(fill)}")
    }
    return fill
}

fn _is_ascii_digit(ch: str) -> bool {
    let code = ord(ch)
    return code >= 48 and code <= 57
}

fn _is_csi_final(ch: str) -> bool {
    let code = ord(ch)
    return code >= 0x40 and code <= 0x7E
}

# An OSC string runs to a BEL or to the two-character string terminator.
fn _skip_osc(text: str, start: int) -> int {
    var index = start
    while index < text.len() {
        if text[index] == "\x07" { return index + 1 }
        if text[index] == "\x1B" and index + 1 < text.len() and text[index + 1] == "\\" {
            return index + 2
        }
        index += 1
    }
    return index
}
