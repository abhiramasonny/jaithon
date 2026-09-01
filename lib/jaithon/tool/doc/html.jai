#: The HTML page, which unlike Markdown can link the types a signature names
#: where they are written.
from jaithon.tool.fmt import _is_ident_part, _is_ident_start
from std.str import escape_html
from .model import Entry, ModuleDoc, _by_name, _entry_summary
from .model import _other_members_of, _variants_of, build_links

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

pub fn _inline_code(escaped: str) -> str {
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

pub fn _html_head(title: str) -> str {
    return "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">" +
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">" +
        f"<title>{escape_html(title)}</title><style>{_STYLE}</style>" +
        "</head><body>"
}

pub fn _html_sidebar(modules: list[ModuleDoc], current: str) -> str {
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

pub fn _script_json(json: str) -> str {
    return json.replace("</", "<\\/")
}

pub let _SEARCH_SCRIPT: str = "const box=document.getElementById('q'),hits=document.getElementById('hits');" +
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
