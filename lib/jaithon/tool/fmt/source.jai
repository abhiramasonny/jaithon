#: The input text, indexed by byte offset and by line.
from .text import _leading_space, _strip_end, _utf8_length

pub class _Source {
    pub let text: str
    pub let offsets: list[int]  # byte offset of each scalar, plus one past the end
    pub let starts: list[int]   # scalar index where each line begins

    pub fn init(self, text: str) {
        var byte = 0
        var index = 0
        pub let offsets: list[int] = []
        pub let starts: list[int] = [0]
        for character in text {
            offsets.push(byte)
            byte += _utf8_length(ord(character))
            index += 1
            if character == "\n" { starts.push(index) }
        }
        offsets.push(byte)
        self.text = text
        self.offsets = offsets
        self.starts = starts
    }

    pub fn line_count(self) -> int { return self.starts.len() }

    pub fn scalar_index(self, byte: int) -> int {
        var low = 0
        var high = self.offsets.len() - 1
        while low < high {
            let middle = (low + high + 1) // 2
            if self.offsets[middle] <= byte { low = middle } else { high = middle - 1 }
        }
        return low
    }

    pub fn line_of(self, byte: int) -> int {
        let target = self.scalar_index(byte)
        var low = 0
        var high = self.starts.len() - 1
        while low < high {
            let middle = (low + high + 1) // 2
            if self.starts[middle] <= target { low = middle } else { high = middle - 1 }
        }
        return low + 1
    }

    pub fn line_text(self, line: int) -> str {
        if line < 1 or line > self.starts.len() { return "" }
        let start = self.starts[line - 1]
        let end = if line < self.starts.len() { self.starts[line] - 1 } else { self.text.len() }
        return self.text[start:max(start, end)]
    }

    pub fn slice_span(self, span: any) -> str {
        if span is null { return "" }
        if span.end <= span.start { return "" }
        let start = self.scalar_index(span.start)
        let end = self.scalar_index(span.end)
        if end <= start { return "" }
        return self.text[start:end]
    }
}

pub fn _dedent_block(src: _Source, first: int, last: int) -> str {
    let lines: list[str] = []
    for line in first..=last { lines.push(src.line_text(line)) }
    if lines.len() == 0 { return "" }
    let margin = _leading_space(lines[0]).len()
    let shifted: list[str] = []
    for (index, line) in lines.enumerate() {
        if index == 0 {
            shifted.push(line.strip())
        } else {
            let cut = min(margin, _leading_space(line).len())
            shifted.push(_strip_end(line[cut:line.len()]))
        }
    }
    return "\n".join(shifted)
}
