#: The Markdown page: a fenced signature, the doc comment, then the members.
from jaithon.tool.fmt import _is_ident_part, _is_ident_start
from std.str import repeat
from .model import Entry, ModuleDoc, _entry_summary, _other_members_of
from .model import _variants_of, build_links

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
