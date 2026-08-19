#: `jaithon.tool.fmt` — the canonical formatter.
#:
#: Two properties matter here more than any particular choice of layout, and
#: they are what most of this file is about.
#:
#: The first is that no comment is ever lost. A comment is the one thing in a
#: source file the compiler does not read and the reader does, so a formatter
#: that drops one has destroyed the only copy. `test_every_comment_survives`
#: runs a file holding a comment in every position the grammar allows through
#: the formatter and checks all of them out the other side.
#:
#: The second is idempotence: formatting formatted text must change nothing.
#: Without it `fmt --check` is not a gate but a coin toss, since the tree can be
#: formatted and still fail the check. Every shape below that the printer treats
#: specially — a broken argument list, a broken operator chain, a collapsed
#: one-line block — is run through `format_source` twice.
#:
#: The rest are regressions, each one a case where the formatter used to emit
#: something that was not the program it was given.
from std.test import assert_eq, assert_true
from jaithon.tool.fmt import check_report, first_difference, format_source, is_formatted
from jaithon.tool.fmt import migrate2, scan_comments

const _EVERY_POSITION: str = """#: Module doc comment.
#:
#: Second paragraph of the module doc.

# Before an import.
from std.core import Iterable

# Before a constant.
pub const LIMIT: int = 10  # trailing the constant

#* A nesting block comment.
   #* with an inner block *#
   still inside the outer one. *#

#: Doc comment on a class.
pub class Widget {
    # Before a field.
    var count: int  # trailing the field

    #: Doc comment on a method.
    pub fn init(self) {
        # First in a block.
        self.count = 0  # trailing a statement
        # Last in a block.
    }

    pub fn empty(self) -> void {
        # The only thing inside this block is this comment.
    }

    pub fn classify(self, value: int) -> str {
        match value {
            # Before the first arm.
            0 => "zero",
            # Between two arms.
            1 => "one",
            _ => "many",  # trailing an arm
        }
        return "unreachable"
    }

    pub fn call_with_comment(self) -> int {
        return sum([
            1,
            # Inside an argument list.
            2,
        ])
    }

    pub fn branches(self, flag: bool) -> int {
        if flag {
            # Inside the then-branch.
            return 1
        } else {
            # Inside the else-branch.
            return 2
        }
    }
}

#: Doc comment on an enum.
pub enum Colour {
    # Before a variant.
    Red,
    Green,  # trailing a variant
    Blue,
}

# At the very end of the file.
"""

fn test_every_comment_survives_formatting() -> void {
    let formatted = format_source(_EVERY_POSITION, "every_position.jai")
    let before = scan_comments(_EVERY_POSITION)
    let after = scan_comments(formatted)
    assert_eq(after.len(), before.len(), "the formatter lost or invented a comment")
    for (index, comment) in before.enumerate() {
        assert_eq(after[index].text, comment.text, f"comment {index} changed")
        assert_eq(
            after[index].own_line,
            comment.own_line,
            f"comment {index} changed between own-line and trailing"
        )
        assert_eq(after[index].is_doc, comment.is_doc, f"comment {index} stopped being a doc")
    }
}

#: A comment the printer has nowhere to put means the file is left alone, so
#: this file is already canonical: every comment in it is somewhere the printer
#: knows about.
fn test_the_every_position_file_is_already_canonical() -> void {
    assert_true(is_formatted(_EVERY_POSITION, "every_position.jai"))
}

let _CORPUS: list[str] = [
    "let x = 1\n",
    "let x=1+2\n",
    "if a{f( 1,2 )}\n",
    "fn f( a:int )->int{return a}\n",
    "pub enum Ordering { Less, Equal, Greater }\n",
    "let pair = Foo[K, V]()\n",
    "let deque = Deque[tuple[int, int]]()\n",
    "let same = (a.y > p.y) != (b.y > p.y)\n",
    "let low = -(-9223372036854775808)\n",
    "let negated = - -x\n",
    "let one = (x,)\n",
    "let none = set()\n",
    "let mapping = {\"a\": 1, \"b\": 2}\n",
    "fn g() -> int { return 0 }\n",
    "let inline = if c { a } else { b }\n",
    "for item in items { total += item }\n",
    "while a and b { step() }\n",
    "match v {\n    0 => \"zero\",\n    _ => \"many\",\n}\n",
    "try {\n    risky()\n} catch error: ValueError {\n    recover()\n}\n",
    "let value = compute(first_argument_name, second_argument_name, third_argument_name, " +
        "fourth_argument)\n",
    "let ok = math.fabs(self.xxxxxx - other.xxxxxx) <= tol and " +
        "math.fabs(self.yyyyyy - other.yyyyyy) <= tol\n",
    "pub fn knapsack_unbounded(weights: list[int], values: list[int], capacity: int) " +
        "-> tuple[int, list[int]] {\n    return (0, [])\n}\n",
    "let along = (separation * separation + self.radius * self.radius - other.radius " +
        "* other.radius) / (2.0 * separation)\n",
]

fn test_formatting_is_idempotent() -> void {
    for (index, source) in _CORPUS.enumerate() {
        let once = format_source(source, f"corpus_{index}.jai")
        let twice = format_source(once, f"corpus_{index}.jai")
        assert_eq(twice, once, f"corpus entry {index} is not a fixed point:\n{once}")
    }
}

fn test_formatting_the_every_position_file_is_idempotent() -> void {
    let once = format_source(_EVERY_POSITION, "every_position.jai")
    assert_eq(format_source(once, "every_position.jai"), once)
}

#: `a[x, y]` is how a generic is instantiated. The parser reads the subscript as
#: a tuple, and printing that tuple back as one turned every `Dict[K, V]` in the
#: tree into `Dict[(K, V)]`.
fn test_generic_arguments_keep_their_spelling() -> void {
    assert_true(is_formatted("let pair = Foo[K, V]()\n", "d.jai"))
    assert_true(is_formatted("let deque = Deque[tuple[int, int]]()\n", "d.jai"))
    assert_true(is_formatted("let one = xs[(index,)]\n", "d.jai"))
}

#: Comparison is chained rather than left-associative (§4.2), so the brackets in
#: `(a > b) != (c > d)` are load-bearing: without them the three operands become
#: one chain, which is a different program.
fn test_chained_comparison_keeps_its_brackets() -> void {
    assert_true(is_formatted("let same = (a.y > p.y) != (b.y > p.y)\n", "d.jai"))
}

#: An enum variant carrying no payload is written bare.
fn test_enum_variant_without_payload_stays_bare() -> void {
    assert_true(is_formatted("pub enum Ordering { Less, Equal, Greater }\n", "d.jai"))
    assert_true(is_formatted("pub enum Shape { Circle(radius: float), Point }\n", "d.jai"))
}

#: `-9223372036854775808` is a single token, because its magnitude has no
#: positive spelling in `int`. Print it after an operator and the sign binds to
#: that operator instead, leaving digits that overflow — so it keeps brackets.
fn test_int_min_literal_keeps_its_brackets() -> void {
    assert_true(is_formatted("let low = -(-9223372036854775808)\n", "d.jai"))
    assert_true(is_formatted("let sum = 1 - (-9223372036854775808)\n", "d.jai"))
    assert_true(is_formatted("let negated = - -5\n", "d.jai"))
    assert_true(is_formatted("let plain = -5\n", "d.jai"))
}

#: A chain too long for one line breaks after the operator, not before it, and
#: the break must not disturb the brackets that fix the grouping.
fn test_long_operator_chain_breaks_after_the_operator() -> void {
    let source = "fn f() -> float {\n" +
        "    let along = (separation * separation + self.radius * self.radius" +
        " - other.radius * other.radius) / (2.0 * separation)\n" +
        "    return along\n}\n"
    let formatted = format_source(source, "d.jai")
    assert_true(formatted.contains(") /\n"), "the break belongs after the operator")
    assert_true(
        formatted.contains("(separation * separation"),
        "the grouping brackets must survive the break"
    )
}

#: The return type and the ` {` share the line with the parameters, so they
#: count towards the limit. Measuring the parameters alone was how a formatted
#: signature ended up 104 columns wide.
fn test_long_signature_counts_its_return_type() -> void {
    let source = "pub fn knapsack_unbounded(weights: list[int], values: list[int]," +
        " capacity: int) -> tuple[int, list[int]] {\n    return (0, [])\n}\n"
    let formatted = format_source(source, "d.jai")
    for line in formatted.split("\n") {
        assert_true(line.len() <= 100, f"line runs to {line.len()} columns: {line}")
    }
}

#: `to_source` opens a block for an `if` used as an expression, having no
#: column to lay it out against. The formatter has one, and a conditional that
#: fits stays on its line rather than becoming five.
fn test_short_conditional_expression_stays_on_one_line() -> void {
    assert_true(is_formatted("let normal = if front { outward } else { -outward }\n", "d.jai"))
    assert_true(is_formatted("push(if index < length { rank[index] } else { -1 })\n", "d.jai"))
    # A `match` has one arm to a line however short it is, so it stays open.
    let source = "let word = match v {\n    0 => \"zero\",\n    _ => \"many\",\n}\n"
    assert_true(is_formatted(source, "d.jai"))
}

fn test_first_difference_names_the_line() -> void {
    assert_eq(first_difference("a\nb\n", "a\nb\n"), 0)
    assert_eq(first_difference("a\nb\n", "a\nc\n"), 2)
    assert_eq(first_difference("a\n", "a\nb\n"), 2)
    assert_eq(first_difference("a\nb\n", "a\n"), 2)
    assert_eq(first_difference("x\n", "y\n"), 1)
}

#: `--check` used to print a bare path, which tells a CI log that something is
#: wrong and nothing about what.
fn test_check_report_shows_both_spellings_of_the_line() -> void {
    assert_eq(check_report("a\nb\n", "a\nb\n", "f.jai"), "")
    let report = check_report("let x=1\n", "let x = 1\n", "f.jai")
    assert_eq(report, "f.jai:1: not formatted\n  have: let x=1\n  want: let x = 1")
}

#: Every row of the migration table in §16, which `--migrate2` applies to text.
#:
#: `system "ls"` is the one row that does not read as the spec writes it: §16
#: says `os.system`, and the operation `std.os` actually offers is `os.shell`.
fn test_migrate2_rewrites_the_whole_table() -> void {
    assert_eq(migrate2("print x\n"), "print(x)\n")
    assert_eq(migrate2("print\n"), "print()\n")
    assert_eq(migrate2("input name\n"), "let name = input()\n")
    assert_eq(migrate2("system \"ls\"\n"), "os.shell(\"ls\")\n")
    assert_eq(migrate2("if c then do\n"), "if c {\n")
    assert_eq(migrate2("while c do\n"), "while c {\n")
    assert_eq(migrate2("func f(a)\n"), "fn f(a) {\n")
    assert_eq(migrate2("end\n"), "}\n")
    assert_eq(migrate2("var n = a eq b\n"), "var n = a == b\n")
    assert_eq(migrate2("var n = 2 ^ 3\n"), "var n = 2 ** 3\n")
    assert_eq(migrate2("var n = 5!\n"), "var n = math.factorial(5)\n")
    assert_eq(migrate2("var d = new Dog(\"Rex\")\n"), "var d = Dog(\"Rex\")\n")
    assert_eq(migrate2("var n = _alen(a)\n"), "var n = a.len()\n")
    assert_eq(migrate2("var n = arrayLen(a)\n"), "var n = a.len()\n")
    assert_eq(migrate2("import ds/array\n"), "import std.ds\n")
    assert_true(migrate2("namespace M\n").contains("TODO(migrate2)"))
}

#: `_get(a, i)` and `arrayGet(a, i)` become `a[i]`. Splitting those two
#: arguments used to raise `AttributeError` and abort the whole migration.
fn test_migrate2_rewrites_two_argument_array_access() -> void {
    assert_eq(migrate2("var n = arrayGet(items, 2)\n"), "var n = items[2]\n")
    assert_eq(migrate2("var n = _get(items, 2)\n"), "var n = items[2]\n")
    assert_eq(migrate2("var n = arrayGet(grid, f(a, b))\n"), "var n = grid[f(a, b)]\n")
}

#: No rewrite reaches inside a string literal.
fn test_migrate2_leaves_strings_and_comments_alone() -> void {
    assert_eq(migrate2("print \"a eq b\"\n"), "print(\"a eq b\")\n")
    assert_eq(migrate2("# a eq b, 2 ^ 3\n"), "# a eq b, 2 ^ 3\n")
}

#: A decorator is part of the program, not decoration.
#:
#: The formatter never wrote them, so every pass over a file silently deleted
#: every one: `@trace` is what puts a function in the traced tier and
#: `@gpu_kernel` is what makes one a kernel at all. A single `jaithon fmt` over
#: jaitensor turned its hot paths back into ordinary functions, and one test in
#: the whole tree noticed -- by asserting `is_traced`, which is not a thing most
#: code can assert about itself.
fn test_decorators_survive_formatting() -> void {
    let source = """@gpu_kernel
pub fn kernel_like(x: int) -> int {
    return x
}

pub class Holder {
    pub var n: int

    pub fn init(self) { self.n = 0 }

    @trace
    pub fn hot(self, x: int) -> int {
        return x + self.n
    }
}

@trace
fn also_traced(x: float) -> float {
    return x * 2.0
}
"""
    let once = format_source(source, "decorators.jai")
    assert_eq(once, source)
    assert_eq(format_source(once, "decorators.jai"), once)
    assert_true(is_formatted(source, "decorators.jai"))
}

#: Two decorators on one function, and one on a function whose body is an
#: expression rather than a block -- both shapes the printer reaches by a
#: different path.
fn test_several_decorators_and_a_short_body() -> void {
    let source = """@trace
@gpu_kernel
pub fn both(x: int) -> int { return x }
"""
    let once = format_source(source, "two.jai")
    assert_true(once.contains("@trace"))
    assert_true(once.contains("@gpu_kernel"))
    assert_eq(format_source(once, "two.jai"), once)
}

#: The lines inside a triple-quoted string are the program's data.
#:
#: The printer indented every output line to the current depth, and a string
#: literal spanning lines arrives as one rendered expression with real newlines
#: in it -- so its second and later lines were indented too, and the string
#: silently gained four characters a line. A program that formatted cleanly
#: printed something different afterwards.
fn test_a_multi_line_string_is_not_re_indented() -> void {
    let source = """fn main() -> int {
    let s = \"\"\"alpha
beta
\"\"\"
    print(s.len())
    return 0
}
"""
    let once = format_source(source, "literal.jai")
    assert_eq(once, source)
    assert_true(is_formatted(source, "literal.jai"))

    #: Nested one level deeper, where the wrong indent would be eight spaces.
    let nested = """fn main() -> int {
    if true {
        let s = \"\"\"alpha
beta
\"\"\"
        print(s.len())
    }
    return 0
}
"""
    assert_eq(format_source(nested, "nested.jai"), nested)
}
