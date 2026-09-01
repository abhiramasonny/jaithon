#: The message an `assert` bakes into its `OP_ASSERT_FAIL` constant: the
#: explicit string when the program wrote one, and otherwise the source text of
#: the condition, sliced out of the file and truncated to what the VM's buffer
#: holds.
#:
#: The truncation is why `_decoded` exists: the byte budget can land in the
#: middle of a multi-byte character, so the decode retreats a byte at a time
#: until it succeeds rather than assuming where a boundary is.
from jaithon.ast import Node, NodeKind

#: 256-byte buffer minus 32 reserved for prefix and terminator (`assertMessage`,
#: src/codegen/codegen_stmt.c).
const _ASSERT_TEXT_MAX: int = 224

pub fn _assert_message(node: Node, source: str?) -> str {
    let message = node.child("message")
    if message != null and message.kind == NodeKind.StrLit { return message.text("chars") }

    let cond = node.child("cond")
    if source != null and cond != null and cond.span.is_known() {
        let raw = __prim__.str_encode(source)
        let start = cond.span.start
        var stop = cond.span.end
        if stop > start and stop <= raw.len() {
            if stop - start > _ASSERT_TEXT_MAX { stop = start + _ASSERT_TEXT_MAX }
            var taken: list[int] = []
            var at = start
            while at < stop {
                taken.push(raw[at])
                at += 1
            }
            return f"assertion failed: {_decoded(taken)}"
        }
    }
    return "assertion failed"
}

fn _decoded(raw: list[int]) -> str {
    var length = raw.len()
    while length > 0 {
        try {
            return __prim__.str_decode(__prim__.bytes_new(raw[:length]))
        } catch _invalid: ValueError {
            length -= 1
        }
    }
    return ""
}
