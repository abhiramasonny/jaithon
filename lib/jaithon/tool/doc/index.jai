#: The navigation index over every module, and the search records it carries.
from std.json import stringify
from std.str import escape_html
from .html import _SEARCH_SCRIPT, _html_head, _html_sidebar, _inline_code
from .html import _script_json
from .model import ModuleDoc, _by_name, _entry_summary, _member_anchor, coverage

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
