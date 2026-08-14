#: The documentation generator behind `jaithon doc`.
#:
#: Documentation is the `#:` comments and the declarations they sit above —
#: there is no separate manual to fall out of date. Only `pub` declarations
#: are documented (`spec/STYLE.md` §4); an undocumented one is still listed,
#: since hiding it would hide the hole. Fenced examples are executed by
#: `jaithon test --doc`, so pages can't drift from the code they document.
#:
#: ```
#: module_name_of("lib/std/algo/sort.jai")    # "std.algo.sort"
#: module_name_of("lib/std/ds/mod.jai")       # "std.ds"
#: ```
from jaithon.ast import Node, NodeKind, Visibility, DeclKind
from std.io import Path, print_err, read_file
from std.json import stringify
from std.str import escape_html, repeat
from .args import Arguments
from .fmt import Comment, expand_paths, freeze_literals, parse_module, render_signature
from .fmt import scan_comments, tail_arguments

#: The default directory `jaithon doc` writes into.
pub const DEFAULT_OUTPUT: str = "docs/api"

#: One fenced example lifted out of a doc comment.
#:
#: `line` is tracked so a failure reported by `jaithon test --doc` points at
#: the source, not a copy of it.
pub class Example {
    #: The example's source lines, in order.
    pub let lines: list[str]

    #: The 1-based line of the first example line in the file.
    pub let line: int

    #: Whether the fence claimed a language this tool can run.
    pub let runnable: bool

    #: Record an example found at `line`.
    pub fn init(self, lines: list[str], line: int, runnable: bool) {
        self.lines = lines
        self.line = line
        self.runnable = runnable
    }

    #: The example as one block of source.
    pub fn source(self) -> str { return "\n".join(self.lines) }
}

#: A parsed `#:` block: its summary, its body, and the examples inside it.
pub class DocComment {
    #: The first line, a complete sentence in the imperative mood.
    pub let summary: str

    #: The whole comment as Markdown, summary included.
    pub let body: str

    #: Every fenced block in the comment, in order.
    pub let examples: list[Example]

    #: Build a doc comment from its Markdown `body`.
    pub fn init(self, body: str, examples: list[Example]) {
        self.body = body
        self.examples = examples
        let lines = body.split("\n")
        self.summary = if lines.len() > 0 { lines[0] } else { "" }
    }

    #: The body with the summary line removed, for pages that print them apart.
    pub fn details(self) -> str {
        let lines = self.body.split("\n")
        if lines.len() <= 1 { return "" }
        return "\n".join(lines[1:lines.len()]).strip()
    }
}

#: One documented declaration, and the members below it when it is a type.
pub class Entry {
    #: The declared name.
    pub let name: str

    #: One of `fn`, `class`, `trait`, `enum`, `type`, `const`, `let` or `var`.
    pub let kind: str

    #: The declaration header, as `jaithon fmt` would print it.
    pub let signature: str

    #: The doc comment above it, or null when it has none.
    pub let doc: DocComment?

    #: The 1-based line the declaration starts on.
    pub let line: int

    #: Documented members, for a class, trait or enum.
    pub let members: list[Entry]

    #: Record a documented declaration.
    pub fn init(
        self,
        name: str,
        kind: str,
        signature: str,
        doc: DocComment?,
        line: int,
        members: list[Entry] = []
    ) {
        self.name = name
        self.kind = kind
        self.signature = signature
        self.doc = doc
        self.line = line
        self.members = members
    }

    #: The in-page anchor for this entry.
    #:
    #: ```
    #: Entry("read_file", "fn", "", null, 1).anchor()    # "read_file"
    #: ```
    pub fn anchor(self) -> str {
        let out: list[str] = []
        for character in self.name.lower() {
            if _is_ident_part(character) { out.push(character) } else { out.push("-") }
        }
        return "".join(out)
    }

    #: Whether this entry names a type other code can refer to.
    pub fn is_type(self) -> bool {
        return self.kind == "class" or
            self.kind == "trait" or
            self.kind == "enum" or
            self.kind == "type"
    }
}

#: One module's documentation: its own doc comment and its public API.
pub class ModuleDoc {
    #: The dotted module name, such as `std.collections`.
    pub let name: str

    #: The file the module was read from.
    pub let path: str

    #: The module's own leading `#:` block, or null.
    pub let doc: DocComment?

    #: Every documented declaration, in source order.
    pub let entries: list[Entry]

    #: Record a module's documentation.
    pub fn init(self, name: str, path: str, doc: DocComment?, entries: list[Entry]) {
        self.name = name
        self.path = path
        self.doc = doc
        self.entries = entries
    }

    #: The module's one-line summary, or the empty string.
    pub fn summary(self) -> str {
        let doc = self.doc
        if doc is null { return "" }
        return doc.summary
    }

    #: The page's file name, without a directory.
    pub fn page(self, extension: str) -> str { return self.name + "." + extension }
}

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

#: Return the link target of every documented type, keyed by name.
#:
#: A name declared in two modules resolves to the first, in module-name
#: order; both pages still document it.
pub fn build_links(modules: list[ModuleDoc], extension: str) -> dict[str, str] {
    let links: dict[str, str] = {}
    for mod in modules {
        for entry in mod.entries {
            if not entry.is_type() { continue }
            if entry.name in links { continue }
            links[entry.name] = mod.page(extension) + "#" + entry.anchor()
        }
    }
    return links
}

#: How many documented and how many undocumented `pub` declarations there
#: are, as `(documented, total)`.
#:
#: Members count too — a class whose methods carry no `#:` is as
#: undocumented as a bare function, and hiding them would flatter the figure.
pub fn coverage(modules: list[ModuleDoc]) -> tuple[int, int] {
    var documented = 0
    var total = 0
    for mod in modules {
        for entry in mod.entries {
            total += 1
            if entry.doc is not null { documented += 1 }
            for member in entry.members {
                total += 1
                if member.doc is not null { documented += 1 }
            }
        }
    }
    return (documented, total)
}

#: Render one module's page as Markdown.
#:
#: The signature goes in a fenced block, which Markdown will not put links
#: inside, so the types it names are linked underneath it instead.
pub fn render_markdown(mod: ModuleDoc, modules: list[ModuleDoc]) -> str {
    let links = build_links(modules, "md")
    let out: list[str] = [f"# `{mod.name}`", ""]
    out.push("[Index](index.md)")
    out.push("")
    let doc = mod.doc
    if doc is not null {
        out.push(doc.body)
        out.push("")
    }
    if mod.entries.len() > 0 {
        out.push("## Contents")
        out.push("")
        for entry in mod.entries {
            let summary = _entry_summary(entry)
            let tail = if summary.len() == 0 { "" } else { " — " + summary }
            out.push(f"- [`{entry.name}`](#{entry.anchor()}) *{entry.kind}*{tail}")
        }
        out.push("")
    }
    for entry in mod.entries {
        out.extend(_markdown_entry(entry, links, 2))
    }
    out.push(f"Source: `{mod.path}`")
    return "\n".join(out).strip() + "\n"
}

#: Render one module's page as a self-contained HTML document.
#:
#: HTML allows a link inside a code block, so the types named in a signature
#: are linked in place; every page also carries the module list so the site
#: can be walked without returning to the index.
pub fn render_html(mod: ModuleDoc, modules: list[ModuleDoc]) -> str {
    let links = build_links(modules, "html")
    let out: list[str] = [
        _html_head(mod.name),
        _html_sidebar(modules, mod.name),
        "<main>",
        f"<h1><code>{mod.name}</code></h1>",
    ]
    let doc = mod.doc
    if doc is not null { out.push(_html_block(doc.body)) }
    if mod.entries.len() > 0 {
        out.push("<h2>Contents</h2>")
        out.push("<ul class=\"toc\">")
        for entry in mod.entries {
            let summary = _inline_code(escape_html(_entry_summary(entry)))
            out.push(
                f"<li><a href=\"#{entry.anchor()}\"><code>{entry.name}</code></a> " +
                    f"<em>{entry.kind}</em> {summary}</li>"
            )
        }
        out.push("</ul>")
    }
    for entry in mod.entries { out.extend(_html_entry(entry, links, 2, "")) }
    out.push(f"<p class=\"source\">Source: <code>{escape_html(mod.path)}</code></p>")
    out.push("</main></body></html>")
    return "\n".join(out) + "\n"
}

#: Render the navigation index over every module.
#:
#: The HTML index inlines the search records instead of fetching
#: `search.json` — a browser reading the site off disk refuses that fetch.
pub fn render_index(modules: list[ModuleDoc], html: bool) -> str {
    let extension = if html { "html" } else { "md" }
    let ordered = _by_name(modules)
    let (documented, total) = coverage(ordered)
    let missing = total - documented
    let summary_line = f"{ordered.len()} modules, {total} public declarations, " +
        f"{missing} without a doc comment."
    if not html {
        let out: list[str] = ["# API reference", "", summary_line, ""]
        for mod in ordered {
            let summary = mod.summary()
            let tail = if summary.len() == 0 { "" } else { " — " + summary }
            out.push(f"- [`{mod.name}`]({mod.page(extension)}){tail}")
        }
        return "\n".join(out) + "\n"
    }
    let out: list[str] = [
        _html_head("API reference"),
        _html_sidebar(ordered, ""),
        "<main>",
        "<h1>API reference</h1>",
        f"<p class=\"coverage\">{escape_html(summary_line)}</p>",
        "<input id=\"q\" type=\"search\" placeholder=\"Search the API\" " +
            "autocomplete=\"off\" aria-label=\"Search the API\">",
        "<ul id=\"hits\"></ul>",
        "<ul class=\"toc\">",
    ]
    for mod in ordered {
        let summary = _inline_code(escape_html(mod.summary()))
        out.push(
            f"<li><a href=\"{mod.page(extension)}\"><code>{mod.name}</code></a> " +
                summary +
                "</li>"
        )
    }
    out.push("</ul></main>")
    let records = stringify(_brief_records(_records_of(ordered, extension)))
    out.push("<script>const RECORDS=" + _script_json(records) + ";" + _SEARCH_SCRIPT + "</script>")
    out.push("</body></html>")
    return "\n".join(out) + "\n"
}

#: Return the search index as JSON.
pub fn search_index(modules: list[ModuleDoc], extension: str = "html") -> str {
    return stringify(_records_of(modules, extension))
}

#: Run `jaithon doc`.
#:
#: ```text
#: jaithon doc [--out dir] [--html] [paths...]
#: ```
#:
#: Writes one page per module, an index, and `search.json`. Exits 1 when no
#: module could be read.
pub fn main(args: list[str]) -> int {
    let options = _Options.parse(tail_arguments(args, "doc"))
    if options.usage_error is not null {
        print_err(f"jaithon doc: {options.usage_error}")
        print_err("usage: jaithon doc [--out dir] [--html] [paths...]")
        return 2
    }

    let modules = _by_name(collect_all(options.paths))
    if modules.len() == 0 {
        print_err("jaithon doc: nothing to document")
        return 1
    }

    let directory = Path(options.output)
    directory.mkdir(true, true)
    let extension = if options.html { "html" } else { "md" }
    for mod in modules {
        let page = if options.html {
            render_html(mod, modules)
        } else {
            render_markdown(mod, modules)
        }
        directory.join(mod.page(extension)).write_text(page)
    }
    directory.join("index." + extension).write_text(render_index(modules, options.html))
    directory.join("search.json").write_text(search_index(modules, extension))

    let (documented, total) = coverage(modules)
    let missing = total - documented
    print(
        f"{modules.len()} modules, {total} public declarations, " +
            f"{missing} undocumented -> {options.output}"
    )
    return 0
}

# --- Collecting --------------------------------------------------------------
class _LineIndex {
    pub let starts: list[int]

    pub fn init(self, source: str) {
        var byte = 0
        pub let starts: list[int] = [0]
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

# --- Rendering ---------------------------------------------------------------
fn _entry_summary(entry: Entry) -> str {
    let doc = entry.doc
    if doc is null { return "" }
    return doc.summary
}

fn _variants_of(entry: Entry) -> list[Entry] {
    return [member for member in entry.members if member.kind == "variant"]
}

fn _other_members_of(entry: Entry) -> list[Entry] {
    return [member for member in entry.members if member.kind != "variant"]
}

fn _markdown_entry(entry: Entry, links: dict[str, str], level: int) -> list[str] {
    let heading = repeat("#", level)
    let out: list[str] = [f"{heading} `{entry.name}`", ""]
    out.push("```jai")
    out.push(entry.signature)
    out.push("```")
    out.push("")
    let doc = entry.doc
    if doc is null {
        out.push("*Undocumented.*")
        out.push("")
    } else {
        out.push(doc.body)
        out.push("")
    }
    let referenced = _referenced_types(entry.signature, links, entry.name)
    if referenced.len() > 0 {
        let targets = [f"[`{name}`]({links[name]})" for name in referenced]
        out.push("Types: " + ", ".join(targets))
        out.push("")
    }
    let variants = _variants_of(entry)
    if variants.len() > 0 {
        out.push(f"{heading}# Variants")
        out.push("")
        for variant in variants {
            let variant_doc = variant.doc
            if variant_doc is null {
                out.push(f"- `{variant.name}` — *undocumented*")
                continue
            }
            out.push(f"- `{variant.name}` — {variant_doc.summary}")
            let details = variant_doc.details()
            if details.len() > 0 {
                for line in details.split("\n") { out.push("  " + line) }
            }
        }
        out.push("")
    }
    for member in _other_members_of(entry) {
        out.extend(_markdown_entry(member, links, level + 1))
    }
    return out
}

fn _html_entry(entry: Entry, links: dict[str, str], level: int, prefix: str) -> list[str] {
    let tag = f"h{min(level, 6)}"
    let anchor = if prefix.len() == 0 { entry.anchor() } else { prefix + "-" + entry.anchor() }
    let out: list[str] = [
        f"<{tag} id=\"{anchor}\"><code>{entry.name}</code>" +
            f" <span class=\"kind\">{entry.kind}</span></{tag}>",
    ]
    out.push(
        "<pre class=\"sig\">" + _linked_signature(entry.signature, links, entry.name) + "</pre>"
    )
    let doc = entry.doc
    if doc is null {
        out.push("<p class=\"undocumented\">Undocumented.</p>")
    } else {
        out.push(_html_block(doc.body))
    }
    let variants = _variants_of(entry)
    if variants.len() > 0 {
        out.push("<dl class=\"variants\">")
        for variant in variants {
            out.push(f"<dt id=\"{anchor}-{variant.anchor()}\"><code>{variant.name}</code></dt>")
            let variant_doc = variant.doc
            if variant_doc is null {
                out.push("<dd class=\"undocumented\">Undocumented.</dd>")
            } else {
                out.push("<dd>" + _html_block(variant_doc.body) + "</dd>")
            }
        }
        out.push("</dl>")
    }
    for member in _other_members_of(entry) {
        out.extend(_html_entry(member, links, level + 1, anchor))
    }
    return out
}

fn _member_anchor(entry: Entry, member: Entry) -> str {
    return entry.anchor() + "-" + member.anchor()
}

fn _referenced_types(signature: str, links: dict[str, str], own: str) -> list[str] {
    let found: list[str] = []
    let seen: set[str] = set()
    for word in _identifiers(signature) {
        if word == own or word in seen { continue }
        if word in links {
            seen.add(word)
            found.push(word)
        }
    }
    return found
}

fn _linked_signature(signature: str, links: dict[str, str], own: str) -> str {
    let out: list[str] = []
    var index = 0
    while index < signature.len() {
        if not _is_ident_start(signature[index]) {
            out.push(escape_html(signature[index]))
            index += 1
            continue
        }
        var end = index
        while end < signature.len() and _is_ident_part(signature[end]) { end += 1 }
        let word = signature[index:end]
        if word in links and word != own {
            out.push(f"<a href=\"{links[word]}\">{escape_html(word)}</a>")
        } else {
            out.push(escape_html(word))
        }
        index = end
    }
    return "".join(out)
}

fn _identifiers(text: str) -> list[str] {
    let words: list[str] = []
    var index = 0
    while index < text.len() {
        if not _is_ident_start(text[index]) {
            index += 1
            continue
        }
        var end = index
        while end < text.len() and _is_ident_part(text[end]) { end += 1 }
        words.push(text[index:end])
        index = end
    }
    return words
}

fn _html_block(markdown: str) -> str {
    let out: list[str] = []
    var paragraph: list[str] = []
    var code: list[str] = []
    var fenced = false
    for line in markdown.split("\n") {
        if line.strip().starts_with("```") {
            if fenced {
                out.push("<pre class=\"example\">" + escape_html("\n".join(code)) + "</pre>")
                code = []
                fenced = false
            } else {
                if paragraph.len() > 0 {
                    out.push(_html_paragraph(paragraph))
                    paragraph = []
                }
                fenced = true
            }
            continue
        }
        if fenced {
            code.push(line)
        } elif line.strip().len() == 0 {
            if paragraph.len() > 0 {
                out.push(_html_paragraph(paragraph))
                paragraph = []
            }
        } else {
            paragraph.push(line.strip())
        }
    }
    if paragraph.len() > 0 { out.push(_html_paragraph(paragraph)) }
    if code.len() > 0 {
        out.push("<pre class=\"example\">" + escape_html("\n".join(code)) + "</pre>")
    }
    return "\n".join(out)
}

fn _html_paragraph(lines: list[str]) -> str {
    return "<p>" + _inline_code(escape_html(" ".join(lines))) + "</p>"
}

fn _inline_code(escaped: str) -> str {
    let parts = escaped.split("`")
    if parts.len() < 3 { return escaped }
    let out: list[str] = []
    for (index, part) in parts.enumerate() {
        if index % 2 == 0 {
            out.push(part)
        } elif index == parts.len() - 1 {
            out.push("`" + part)
        } else {
            out.push("<code>" + part + "</code>")
        }
    }
    return "".join(out)
}

fn _html_head(title: str) -> str {
    return "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">" +
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" +
        f"<title>{escape_html(title)}</title><style>{_STYLE}</style>" +
        "</head><body>"
}

fn _html_sidebar(modules: list[ModuleDoc], current: str) -> str {
    let out: list[str] = [
        "<nav class=\"sidebar\"><a class=\"home\" href=\"index.html\">" +
            "API reference</a><ul>",
    ]
    for mod in _by_name(modules) {
        let here = if mod.name == current { " class=\"here\"" } else { "" }
        let page = mod.page("html")
        out.push(f"<li{here}><a href=\"{page}\"><code>{mod.name}</code></a></li>")
    }
    out.push("</ul></nav>")
    return "".join(out)
}

fn _script_json(json: str) -> str {
    return json.replace("</", "<\\/")
}

let _SEARCH_SCRIPT: str = "const box=document.getElementById('q'),hits=document.getElementById('hits');" +
    "function show(){const needle=box.value.trim().toLowerCase();hits.textContent='';" +
    "if(needle.length<2){return}" +
    "const found=RECORDS.map(r=>[(r.mod+'.'+r.name).toLowerCase().indexOf(needle),r])" +
    ".filter(p=>p[0]>=0).sort((a,b)=>a[0]-b[0]||a[1].name.length-b[1].name.length)" +
    ".slice(0,60);" +
    "for(const [,r] of found){const li=document.createElement('li');" +
    "const a=document.createElement('a');" +
    "a.href=r.page+(r.anchor?'#'+r.anchor:'');" +
    "const code=document.createElement('code');code.textContent=r.mod+'.'+r.name;" +
    "a.appendChild(code);const kind=document.createElement('em');" +
    "kind.textContent=' '+r.kind;const text=document.createTextNode(' '+r.summary);" +
    "li.append(a,kind,text);hits.appendChild(li)}}" +
    "box.addEventListener('input',show);"

let _STYLE: str = "body{font:16px/1.6 system-ui,sans-serif;margin:0;background:#fbfbfd;" +
    "color:#1c1c1e;display:flex;align-items:flex-start}" +
    "main{flex:1;max-width:52rem;margin:0 auto;padding:2rem 1.25rem;min-width:0}" +
    "nav.sidebar{width:15rem;flex:none;position:sticky;top:0;max-height:100vh;" +
    "overflow:auto;padding:2rem 1rem;border-right:1px solid #e3e3e8;font-size:.9em}" +
    "nav.sidebar ul{list-style:none;padding:0;margin:.5rem 0 0}" +
    "nav.sidebar li.here>a{font-weight:700}nav.sidebar a{text-decoration:none}" +
    "nav.sidebar .home{font-weight:700;color:inherit}" +
    "code,pre{font-family:ui-monospace,Menlo,monospace}" +
    "pre{background:#f2f2f7;padding:.75rem 1rem;border-radius:6px;overflow-x:auto}" +
    "pre.sig{background:#eef2ff}h1,h2,h3,h4{line-height:1.25}" +
    ".kind{font:400 .7em system-ui;color:#6b6b70;text-transform:uppercase}" +
    ".undocumented{color:#a1000d}ul.toc{padding-left:1.1rem}" +
    ".coverage,.source{color:#6b6b70;font-size:.9em}" +
    "dl.variants dt{margin-top:.5rem}dl.variants dd{margin:0 0 0 1.5rem}" +
    "#q{width:100%;padding:.5rem .75rem;font:inherit;border:1px solid #c7c7cc;" +
    "border-radius:6px;background:inherit;color:inherit}" +
    "#hits{list-style:none;padding:0}#hits li{margin:.35rem 0}" +
    "a{color:#0b57d0}@media(max-width:48rem){body{display:block}" +
    "nav.sidebar{width:auto;position:static;max-height:none;border-right:0;" +
    "border-bottom:1px solid #e3e3e8}}" +
    "@media(prefers-color-scheme:dark){body{background:#101014;" +
    "color:#e8e8ed}pre{background:#1c1c22}pre.sig{background:#161a2b}a{color:#8ab4f8}" +
    "nav.sidebar{border-color:#2a2a32}#q{border-color:#3a3a44}}"

fn _records_of(modules: list[ModuleDoc], extension: str) -> list[any] {
    let records: list[any] = []
    for mod in _by_name(modules) {
        if mod.doc is not null {
            records.push(_record(mod, mod.name, "mod", "", mod.summary(), "", extension))
        }
        for entry in mod.entries {
            records.push(
                _record(
                    mod,
                    entry.name,
                    entry.kind,
                    entry.signature,
                    _entry_summary(entry),
                    entry.anchor(),
                    extension
                )
            )
            for member in entry.members {
                records.push(
                    _record(
                        mod,
                        entry.name + "." + member.name,
                        member.kind,
                        member.signature,
                        _entry_summary(member),
                        _member_anchor(entry, member),
                        extension
                    )
                )
            }
        }
    }
    return records
}

fn _brief_records(records: list[any]) -> list[any] {
    return [{"mod": record["mod"], "name": record["name"], "kind": record["kind"], "summary": record["summary"], "page": record["page"], "anchor": record["anchor"]} for record in records]
}

fn _record(
    mod: ModuleDoc,
    name: str,
    kind: str,
    signature: str,
    summary: str,
    anchor: str,
    extension: str
) -> dict[str, any] {
    return {
        "mod": mod.name,
        "name": name,
        "kind": kind,
        "signature": signature,
        "summary": summary,
        "page": mod.page(extension),
        "anchor": anchor,
        "path": mod.path,
    }
}

fn _by_name(modules: list[ModuleDoc]) -> list[ModuleDoc] {
    let names = [mod.name for mod in modules].sorted()
    let by_name: dict[str, ModuleDoc] = {mod.name: mod for mod in modules}
    return [by_name[name] for name in names]
}

fn _by_line(entries: list[Entry]) -> list[Entry] {
    let ordered = [entry for entry in entries]
    var index = 1
    while index < ordered.len() {
        let current = ordered[index]
        var position = index - 1
        while position >= 0 and ordered[position].line > current.line {
            ordered[position + 1] = ordered[position]
            position -= 1
        }
        ordered[position + 1] = current
        index += 1
    }
    return ordered
}

fn _is_ident_start(character: str) -> bool {
    return character == "_" or character.lower() != character.upper()
}

fn _is_ident_part(character: str) -> bool {
    return _is_ident_start(character) or character >= "0" and character <= "9"
}

fn _utf8_length(code_point: int) -> int {
    if code_point < 0x80 { return 1 }
    if code_point < 0x800 { return 2 }
    if code_point < 0x10000 { return 3 }
    return 4
}

class _Options {
    pub var output: str
    pub var html: bool
    pub var paths: list[str]
    pub var usage_error: str?

    pub fn init(self) {
        self.output = DEFAULT_OUTPUT
        self.html = false
        self.paths = []
        self.usage_error = null
    }

    pub static fn parse(args: list[str]) -> _Options {
        let options = _Options()
        let reader = Arguments(args)
        while true {
            let argument = reader.next()
            if argument is null { break }
            if not argument.is_flag() {
                options.paths.push(argument.text)
                continue
            }
            if argument.name == "--out" {
                let value = reader.value_of(argument)
                if value is null { break }
                options.output = value
            } elif argument.name == "--html" {
                if reader.is_bare(argument) { options.html = true }
            } elif argument.name == "--markdown" {
                if reader.is_bare(argument) { options.html = false }
            } else {
                reader.unknown(argument)
            }
        }
        options.usage_error = reader.error
        if options.paths.len() == 0 { options.paths = ["lib"] }
        return options
    }
}
