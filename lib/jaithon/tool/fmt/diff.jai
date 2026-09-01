#: The unified diff `--diff` prints and the line `--check` names.
from .text import _lines_of

#: How many unchanged lines `unified_diff` prints around each change.
pub const DIFF_CONTEXT: int = 3

#: Return a unified diff from `before` to `after`, or the empty string.
#:
#: Standard format — `---`/`+++` headers, `@@` hunks, `DIFF_CONTEXT` lines of
#: context — so the output pipes into `patch`.
#:
#: ```
#: unified_diff("a\n", "a\n", "f.jai")                   # ""
#: unified_diff("a\n", "b\n", "f.jai").split("\n")[0]    # "--- a/f.jai"
#: ```
pub fn unified_diff(before: str, after: str, path: str) -> str {
    if before == after { return "" }
    let hunks = _hunks(_lines_of(before), _lines_of(after))
    if hunks.len() == 0 { return "" }
    let out: list[str] = [f"--- a/{path}", f"+++ b/{path}"]
    for hunk in hunks { out.extend(hunk) }
    return "\n".join(out) + "\n"
}

#: Return the 1-based number of the first line where `before` and `after`
#: differ, or 0 when they hold the same text.
#:
#: A shared line counts as the same only in full — trailing whitespace or a
#: changed indent counts as a difference too.
#:
#: ```
#: first_difference("a\nb\n", "a\nb\n")    # 0
#: first_difference("a\nb\n", "a\nc\n")    # 2
#: first_difference("a\n", "a\nb\n")       # 2
#: ```
pub fn first_difference(before: str, after: str) -> int {
    let old = _lines_of(before)
    let new = _lines_of(after)
    for index in 0..min(old.len(), new.len()) {
        if old[index] != new[index] { return index + 1 }
    }
    if old.len() == new.len() { return 0 }
    return min(old.len(), new.len()) + 1
}

#: Report where `path` stops being canonical, as one `file:line:` message and
#: the two spellings of the line beneath it.
#:
#: What `--check` prints — naming the line and both spellings means the log
#: alone is enough to see the change coming.
pub fn check_report(before: str, after: str, path: str) -> str {
    let line = first_difference(before, after)
    if line == 0 { return "" }
    let old = _lines_of(before)
    let new = _lines_of(after)
    let head = f"{path}:{line}: not formatted"
    let had = if line <= old.len() { f"\n  have: {old[line - 1]}" } else { "" }
    let want = if line <= new.len() { f"\n  want: {new[line - 1]}" } else { "" }
    return head + had + want
}

class _EditOp {
    pub let tag: str
    pub let text: str
    pub let old_line: int
    pub let new_line: int

    pub fn init(self, tag: str, text: str, old_line: int, new_line: int) {
        self.tag = tag
        self.text = text
        self.old_line = old_line
        self.new_line = new_line
    }
}

fn _hunks(old: list[str], new: list[str]) -> list[list[str]] {
    var head = 0
    while head < old.len() and head < new.len() and old[head] == new[head] { head += 1 }
    var tail = 0
    while tail < old.len() - head and
        tail < new.len() - head and
        old[old.len() - 1 - tail] == new[new.len() - 1 - tail] {
        tail += 1
    }

    let ops = _edit_ops(old[head:old.len() - tail], new[head:new.len() - tail])
    let keep = [false for _op in ops]
    for (index, op) in ops.enumerate() {
        if op.tag == " " { continue }
        for near in max(0, index - DIFF_CONTEXT)..min(ops.len(), index + DIFF_CONTEXT + 1) {
            keep[near] = true
        }
    }

    let hunks: list[list[str]] = []
    var index = 0
    while index < ops.len() {
        if not keep[index] {
            index += 1
            continue
        }
        var last = index
        while last < ops.len() and keep[last] { last += 1 }
        hunks.push(_render_hunk(ops, index, last, head))
        index = last
    }
    return hunks
}

fn _edit_ops(left: list[str], right: list[str]) -> list[_EditOp] {
    let n = left.len()
    let m = right.len()
    let ops: list[_EditOp] = []
    if n * m > 4_000_000 {
        for (index, text) in left.enumerate() { ops.push(_EditOp("-", text, index, 0)) }
        for (index, text) in right.enumerate() { ops.push(_EditOp("+", text, 0, index)) }
        return ops
    }

    let table = [[0 for _column in 0..=m] for _row in 0..=n]
    var row = n - 1
    while row >= 0 {
        var column = m - 1
        while column >= 0 {
            if left[row] == right[column] {
                table[row][column] = table[row + 1][column + 1] + 1
            } else {
                table[row][column] = max(table[row + 1][column], table[row][column + 1])
            }
            column -= 1
        }
        row -= 1
    }

    var i = 0
    var j = 0
    while i < n and j < m {
        if left[i] == right[j] {
            ops.push(_EditOp(" ", left[i], i, j))
            i += 1
            j += 1
        } elif table[i + 1][j] >= table[i][j + 1] {
            ops.push(_EditOp("-", left[i], i, j))
            i += 1
        } else {
            ops.push(_EditOp("+", right[j], i, j))
            j += 1
        }
    }
    while i < n {
        ops.push(_EditOp("-", left[i], i, j))
        i += 1
    }
    while j < m {
        ops.push(_EditOp("+", right[j], i, j))
        j += 1
    }
    return ops
}

fn _render_hunk(ops: list[_EditOp], first: int, last: int, offset: int) -> list[str] {
    var old_count = 0
    var new_count = 0
    let body: list[str] = []
    for index in first..last {
        let op = ops[index]
        if op.tag != "+" { old_count += 1 }
        if op.tag != "-" { new_count += 1 }
        body.push(op.tag + op.text)
    }
    let old_start = offset + ops[first].old_line + 1
    let new_start = offset + ops[first].new_line + 1
    let out: list[str] = [f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"]
    out.extend(body)
    return out
}
