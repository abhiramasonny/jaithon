#: `jaithon.tool.doc` — the pages `jaithon doc` writes.
#:
#: The tree is built by hand rather than read off disk: a renderer's job is to
#: turn a `ModuleDoc` into a page, and a test that first has to parse a file is
#: testing the parser. The one thing every case here is really checking is that
#: nothing public disappears — a member, a variant or a requirement that the
#: page leaves out is a lie about the API.
from std.test import assert_eq, assert_true, assert_false
from jaithon.tool.doc import DocComment, Entry, ModuleDoc, build_links, coverage
from jaithon.tool.doc import module_name_of, render_index, render_markdown, search_index

fn _documented(text: str) -> DocComment {
    return DocComment(text, [])
}

fn _shape() -> ModuleDoc {
    let colour = Entry(
        "Colour",
        "enum",
        "pub enum Colour",
        _documented("A colour."),
        3,
        [
            Entry("Red", "variant", "Red", _documented("The warm one."), 4),
            Entry("Blue", "variant", "Blue", null, 5),
        ]
    )
    let drawable = Entry(
        "Drawable",
        "trait",
        "pub trait Drawable",
        _documented("Anything that can be drawn."),
        10,
        [
            Entry(
                "draw",
                "method",
                "fn draw(self) -> void",
                _documented("Draw `self`."),
                11
            ),
        ]
    )
    let circle = Entry(
        "Circle",
        "class",
        "pub class Circle: Drawable",
        _documented("A round shape."),
        20,
        [
            Entry(
                "radius",
                "field",
                "pub let radius: float",
                _documented("Distance from the centre."),
                21
            ),
            Entry("draw", "method", "pub fn draw(self) -> void", null, 23),
        ]
    )
    let paint = Entry(
        "paint",
        "fn",
        "pub fn paint(shape: Circle, colour: Colour) -> void",
        _documented("Paint `shape`."),
        30
    )
    return ModuleDoc(
        "demo.shapes",
        "lib/demo/shapes.jai",
        _documented("Shapes and the colours they come in."),
        [colour, drawable, circle, paint]
    )
}

fn test_module_name_follows_the_path_under_lib() -> void {
    assert_eq(module_name_of("lib/std/str.jai"), "std.str")
    assert_eq(module_name_of("/w/lib/std/ds/mod.jai"), "std.ds")
    assert_eq(module_name_of("tests/lang/test_types.jai"), "test_types")
}

fn test_every_public_declaration_reaches_the_page() -> void {
    let page = render_markdown(_shape(), [_shape()])
    for name in ["Colour", "Drawable", "Circle", "paint", "radius", "draw", "Red", "Blue"] {
        assert_true(name in page)
    }
    assert_true("The warm one." in page)
    assert_true("Draw `self`." in page)
    assert_true("Distance from the centre." in page)
}

fn test_a_declaration_without_a_doc_comment_is_still_listed() -> void {
    let page = render_markdown(_shape(), [_shape()])
    assert_true("*Undocumented.*" in page)
    assert_true("`Blue` — *undocumented*" in page)
}

fn test_a_signature_links_to_the_types_it_names() -> void {
    let links = build_links([_shape()], "md")
    assert_eq(links["Circle"], "demo.shapes.md#circle")
    assert_eq(links["Colour"], "demo.shapes.md#colour")
    let page = render_markdown(_shape(), [_shape()])
    assert_true(
        "Types: [`Circle`](demo.shapes.md#circle), " + "[`Colour`](demo.shapes.md#colour)" in page
    )
    assert_false(
        "## `Circle`\n\n```jai\npub class Circle: Drawable\n```\n\n" + "A round shape.\n\nTypes: [`Circle`]" in
            page
    )
}

fn test_the_index_links_every_module_and_counts_the_holes() -> void {
    let index = render_index([_shape()], false)
    assert_true("[`demo.shapes`](demo.shapes.md)" in index)
    assert_true("Shapes and the colours they come in." in index)
    assert_true("1 modules, 9 public declarations, 2 without a doc comment." in index)
}

fn test_coverage_counts_members_as_declarations() -> void {
    let (documented, total) = coverage([_shape()])
    assert_eq(total, 9)
    assert_eq(documented, 7)
}

fn test_the_search_index_addresses_members_uniquely() -> void {
    let index = search_index([_shape()], "html")
    assert_true("\"name\":\"Circle.draw\"" in index)
    assert_true("\"anchor\":\"circle-draw\"" in index)
    assert_true("\"anchor\":\"drawable-draw\"" in index)
    assert_true("\"page\":\"demo.shapes.html\"" in index)
}
