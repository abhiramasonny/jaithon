#: Reading a file into a `ModuleDoc`: which `#:` block belongs to which
#: declaration, and which declarations are documented at all.
from jaithon.ast import DeclKind, Node, NodeKind, Visibility
from jaithon.tool.fmt import Comment, _utf8_length, expand_paths, freeze_literals
from jaithon.tool.fmt import parse_module, render_signature, scan_comments
from std.io import Path, print_err, read_file
from .model import DocComment, Entry, Example, ModuleDoc, _by_line

#: Return the dotted module name a path denotes.
#:
#: `mod.jai` names its directory (packages, per `spec/LANGUAGE.md` §8); a
#: path outside `lib/` falls back to its stem, which is what a test file
#: wants.
#:
#: ```
#: module_name_of("lib/std/str.jai")            # "std.str"
#: module_name_of("/w/lib/std/ds/mod.jai")      # "std.ds"
#: module_name_of("tests/lang/test_types.jai")  # "test_types"
#: ```
pub fn module_name_of(path: str) -> str {
    let parts = Path(path).text.split("/")
    var start = -1
    for (index, part) in parts.enumerate() {
        if part == "lib" { start = index + 1 }
    }
    if start < 0 or start >= parts.len() { return Path(path).stem() }
    let tail = parts[start:parts.len()]
    let last = tail.pop()
    if last != "mod.jai" { tail.push(Path(last).stem()) }
    if tail.len() == 0 { return Path(path).stem() }
    return ".".join(tail)
}

#: Read one file and return its documentation.
#:
#: Raises `ParseError` when the file does not parse: a tree is the only thing
#: that can say which comment belongs to which declaration.
pub fn collect_module(path: str) -> ModuleDoc {
    let source = read_file(path)
    let program = freeze_literals(parse_module(source, path), source)
    let blocks = _doc_blocks(scan_comments(source))
    let line_of = _LineIndex(source)

    let entries: list[Entry] = []
    let claimed: set[int] = set()
    var first = 0
    for statement in program.child_list("stmts") {
        let line = line_of.of(statement.span.start)
        if first == 0 { first = line }
        claimed.add(line)
        let entry = _entry_of(statement, blocks, line_of)
        if entry is not null { entries.push(entry) }
    }
    return ModuleDoc(module_name_of(path), path, _module_doc(blocks, claimed, first), entries)
}

#: Read every file in `paths` and return their documentation, by module name.
#:
#: A file that does not parse is reported on standard error and skipped, so one
#: broken file does not cost the whole site.
pub fn collect_all(paths: list[str]) -> list[ModuleDoc] {
    let modules: list[ModuleDoc] = []
    for path in expand_paths(paths) {
        try {
            modules.push(collect_module(path))
        } catch error: ParseError {
            print_err(f"doc: {error.message}")
        }
    }
    return modules
}

class _LineIndex {
    pub let starts: list[int]

    pub fn init(self, source: str) {
        var byte = 0
        let starts: list[int] = [0]
        for character in source {
            byte += _utf8_length(ord(character))
            if character == "\n" { starts.push(byte) }
        }
        self.starts = starts
    }

    pub fn of(self, byte: int) -> int {
        var low = 0
        var high = self.starts.len() - 1
        while low < high {
            let middle = (low + high + 1) // 2
            if self.starts[middle] <= byte { low = middle } else { high = middle - 1 }
        }
        return low + 1
    }
}

fn _doc_blocks(comments: list[Comment]) -> dict[int, DocComment] {
    let blocks: dict[int, DocComment] = {}
    var run: list[Comment] = []
    for comment in comments {
        let joins = run.len() > 0 and comment.line == run[run.len() - 1].end_line + 1
        if not comment.is_doc or not comment.own_line {
            if run.len() > 0 {
                blocks[run[run.len() - 1].end_line + 1] = _doc_of(run)
                run = []
            }
            continue
        }
        if run.len() > 0 and not joins {
            blocks[run[run.len() - 1].end_line + 1] = _doc_of(run)
            run = []
        }
        run.push(comment)
    }
    if run.len() > 0 { blocks[run[run.len() - 1].end_line + 1] = _doc_of(run) }
    return blocks
}

fn _doc_of(run: list[Comment]) -> DocComment {
    let body: list[str] = []
    let examples: list[Example] = []
    var fence: list[str] = []
    var fence_line = 0
    var open = false
    var runnable = false
    for comment in run {
        let text = comment.body()
        body.push(text)
        if text.starts_with("```") {
            if open {
                examples.push(Example(fence, fence_line, runnable))
                fence = []
                open = false
            } else {
                let info = text[3:text.len()].strip()
                runnable = info.len() == 0 or info == "jai"
                fence_line = comment.line + 1
                open = true
            }
            continue
        }
        if open { fence.push(text) }
    }
    if open and fence.len() > 0 { examples.push(Example(fence, fence_line, runnable)) }
    return DocComment("\n".join(body).strip(), examples)
}

fn _module_doc(blocks: dict[int, DocComment], claimed: set[int], first: int) -> DocComment? {
    var best: DocComment? = null
    var best_line = 0
    for (line, doc) in blocks.items() {
        if line in claimed { continue }
        if first > 0 and line > first { continue }
        if best is null or line < best_line {
            best = doc
            best_line = line
        }
    }
    return best
}

fn _entry_of(node: Node, blocks: dict[int, DocComment], line_of: _LineIndex) -> Entry? {
    let kind = _entry_kind(node)
    if kind is null { return null }
    if node.visibility() != Visibility.Public { return null }
    let line = line_of.of(node.span.start)
    let name = if node.has("name") { node.opt_text("name") } else { _pattern_name(node) }
    if name is null { return null }
    return Entry(
        name,
        kind,
        render_signature(node),
        blocks.get(line, null),
        line,
        _members_of(node, blocks, line_of)
    )
}

fn _entry_kind(node: Node) -> str? {
    match node.kind {
        NodeKind.FnDecl => { return "fn" }
        NodeKind.ClassDecl => { return "class" }
        NodeKind.TraitDecl => { return "trait" }
        NodeKind.EnumDecl => { return "enum" }
        NodeKind.TypeDecl => { return "type" }
        NodeKind.VarDecl => {
            return match node.get("varDeclKind") {
                DeclKind.Let => "let",
                DeclKind.Var => "var",
                DeclKind.Const => "const",
            }
        }
        _ => { return null }
    }
}

fn _pattern_name(node: Node) -> str? {
    let pattern = node.child("pattern")
    if pattern is null { return null }
    if pattern.kind != NodeKind.PatBind { return null }
    return pattern.text("name")
}

fn _members_of(node: Node, blocks: dict[int, DocComment], line_of: _LineIndex) -> list[Entry] {
    let members: list[Entry] = []
    if node.kind == NodeKind.ClassDecl {
        for field in node.records("fields") {
            if field["visibility"] != Visibility.Public { continue }
            let line = line_of.of(field["span"].start)
            let declared = field["type"]
            let annotation = if declared is null { "" } else { ": " + declared.to_source() }
            let keyword = if field["isLet"] { "let" } else { "var" }
            members.push(
                Entry(
                    field["name"],
                    "field",
                    "pub " + keyword + " " + field["name"] + annotation,
                    blocks.get(line, null),
                    line
                )
            )
        }
    }
    if node.kind == NodeKind.EnumDecl {
        for variant in node.records("variants") {
            let line = line_of.of(variant["span"].start)
            members.push(
                Entry(variant["name"], "variant", variant["name"], blocks.get(line, null), line)
            )
        }
    }
    for group in ["methods", "getters", "setters"] {
        if not node.has(group) { continue }
        for method in node.child_list(group) {
            if method.visibility() != Visibility.Public and node.kind != NodeKind.TraitDecl {
                continue
            }
            let line = line_of.of(method.span.start)
            members.push(
                Entry(
                    method.text("name"),
                    "method",
                    render_signature(method),
                    blocks.get(line, null),
                    line
                )
            )
        }
    }
    return _by_line(members)
}
