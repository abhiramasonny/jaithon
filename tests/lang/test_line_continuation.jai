#: Which tokens let a line carry on to the next one.
#:
#: The lexer suppresses the newline after a token that cannot end an
#: expression, which is what makes a wrapped condition legal. `and`, `or` and
#: `not` were on that list from the start; `in` and `is` were not, and nothing
#: noticed because nobody writes the break by hand -- the FORMATTER does. `fmt`
#: breaks a line over its width after a binary operator, so a long
#: `x = a.text("name") in b[b.len() - 1]` came back as two lines with the first
#: ending in `in`, and the result did not parse. `make fmt-roundtrip` reported
#: it as "UNPARSEABLE AFTER lib/jaithon/compile/emit.jai".
#:
#: Every expression here is long enough that the formatter keeps the break
#: where it is: `make fmt-check` rejoins a short one, and a rejoined line
#: tests nothing. Run `make fmt-check` after editing this file -- if it wants
#: to join a case back, that case has stopped being a test.
from std.test import assert_true, assert_false, assert_eq

class Box {
    pub var tag: int

    pub fn init(self, tag: int) -> void {
        self.tag = tag
    }
}

let GROUPED_NAMES_BY_NESTING_LEVEL = [["alpha", "beta"], ["gamma", "delta", "epsilon"]]

fn test_a_line_carries_on_after_in() -> void {
    let found_the_wanted_name_here = "gamma" in
        GROUPED_NAMES_BY_NESTING_LEVEL[GROUPED_NAMES_BY_NESTING_LEVEL.len() - 1]
    let missing_the_wanted_name_here = "omega" in
        GROUPED_NAMES_BY_NESTING_LEVEL[GROUPED_NAMES_BY_NESTING_LEVEL.len() - 1]
    assert_true(found_the_wanted_name_here)
    assert_false(missing_the_wanted_name_here)
}

fn test_a_line_carries_on_after_not_in() -> void {
    let absent_from_the_list_of_names = "omega" not in
        GROUPED_NAMES_BY_NESTING_LEVEL[GROUPED_NAMES_BY_NESTING_LEVEL.len() - 1]
    assert_true(absent_from_the_list_of_names)
}

fn test_a_line_carries_on_after_is() -> void {
    let every_box_here_in_the_order_that_they_arrived: list[Box?] = [null, Box(3)]
    let the_very_first_one_really_is_a_null = every_box_here_in_the_order_that_they_arrived[0] is
        null
    let the_second_one_is_not_a_null_at_all = every_box_here_in_the_order_that_they_arrived[1] is
        null
    assert_true(the_very_first_one_really_is_a_null)
    assert_false(the_second_one_is_not_a_null_at_all)
}

fn test_a_line_carries_on_after_is_not() -> void {
    let every_box_here_in_the_order_that_they_arrived: list[Box?] = [null, Box(4)]
    let the_second_one_is_still_live_here = every_box_here_in_the_order_that_they_arrived[1] is not
        null
    assert_true(the_second_one_is_still_live_here)
}

#: The ones that already worked, kept beside the new ones so the rule reads as
#: one rule rather than as two lists.
fn test_a_line_carries_on_after_and_or_not() -> void {
    let both_of_these_conditions_hold = true and
        GROUPED_NAMES_BY_NESTING_LEVEL[GROUPED_NAMES_BY_NESTING_LEVEL.len() - 1].len() == 3
    let either_of_these_conditions_holds = false or
        GROUPED_NAMES_BY_NESTING_LEVEL[GROUPED_NAMES_BY_NESTING_LEVEL.len() - 1].len() == 3
    assert_true(both_of_these_conditions_hold)
    assert_true(either_of_these_conditions_holds)
}

#: A `for` head is the other place `in` appears, and the lexer rule reaches it
#: too -- but there is no test for it here on purpose. `fmt` formats a for-head
#: as one unit and joins it back whatever its width, so a wrapped one cannot
#: come out of the formatter and cannot be written in a file this repository
#: also requires to be formatter-clean. The rule it would exercise is the same
#: one the `let` cases above cover.
