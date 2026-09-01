#: What a page is made of: a doc comment, a declaration, a module, and the few
#: questions the renderers ask of them.
from jaithon.tool.fmt import _is_ident_part

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

pub fn _entry_summary(entry: Entry) -> str {
    let doc = entry.doc
    if doc is null { return "" }
    return doc.summary
}

pub fn _variants_of(entry: Entry) -> list[Entry] {
    return [member for member in entry.members if member.kind == "variant"]
}

pub fn _other_members_of(entry: Entry) -> list[Entry] {
    return [member for member in entry.members if member.kind != "variant"]
}

pub fn _member_anchor(entry: Entry, member: Entry) -> str {
    return entry.anchor() + "-" + member.anchor()
}

pub fn _by_name(modules: list[ModuleDoc]) -> list[ModuleDoc] {
    let names = [mod.name for mod in modules].sorted()
    let by_name: dict[str, ModuleDoc] = {mod.name: mod for mod in modules}
    return [by_name[name] for name in names]
}

pub fn _by_line(entries: list[Entry]) -> list[Entry] {
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
