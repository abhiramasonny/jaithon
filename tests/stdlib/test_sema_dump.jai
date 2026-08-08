#: The sema dump is the contract between the C checker and the one being
#: written to replace it, so the reader is tested against exact bytes, and
#: against a dump the C checker actually produced.
from std.test import assert_eq, assert_throws, assert_true
from jaithon.compile.check.dump import VERSION, Dump, Record
from jaithon.compile.check.dump import diff_dumps, parse_dump, render_dump

let _SAMPLE = "# jaithon sema dump v1\n" +
    "# file a.jai\n" +
    "14 19 AST_IDENT type float\n" +
    "33 38 AST_BINARY fold 5\n"

fn test_a_dump_round_trips() -> void {
    assert_eq(render_dump(parse_dump(_SAMPLE)), _SAMPLE)
}

fn test_the_header_names_the_file_and_the_records_follow() -> void {
    let dump = parse_dump(_SAMPLE)
    assert_eq(dump.source_path, "a.jai")
    assert_eq(dump.records.len(), 2)
    assert_eq(dump.records[0].span_start, 14)
    assert_eq(dump.records[0].span_end, 19)
    assert_eq(dump.records[0].kind, "AST_IDENT")
    assert_eq(dump.records[0].field, "type")
    assert_eq(dump.records[0].value, "float")
}

fn test_a_value_may_contain_spaces() -> void {
    # `dict[str, list[int]]` renders with a space in it, so the value is the
    # rest of the line and not the fifth whitespace-delimited field.
    let text = "# jaithon sema dump v1\n# file a.jai\n1 2 AST_IDENT type dict[str, list[int]]\n"
    assert_eq(parse_dump(text).records[0].value, "dict[str, list[int]]")
}

fn test_records_are_written_in_span_order_whatever_order_they_arrived_in() -> void {
    let jumbled = Dump(
        "a.jai",
        [
            Record(33, 38, "AST_BINARY", "fold", "5"),
            Record(14, 19, "AST_IDENT", "type", "float"),
        ]
    )
    assert_eq(render_dump(jumbled), _SAMPLE)
}

fn test_a_malformed_line_is_rejected_not_skipped() -> void {
    let error = assert_throws(
        ValueError,
        || parse_dump("# jaithon sema dump v1\n# file a.jai\nnonsense\n")
    )
    assert_true(error.message.contains("line 3"))
}

fn test_a_non_numeric_span_is_rejected() -> void {
    let error = assert_throws(
        ValueError,
        || parse_dump("# jaithon sema dump v1\n# file a.jai\nx y AST_IDENT type int\n")
    )
    assert_true(error.message.contains("is not a span"))
}

fn test_an_unknown_version_is_rejected() -> void {
    assert_throws(ValueError, || parse_dump("# jaithon sema dump v99\n# file a.jai\n"))
}

fn test_a_missing_header_is_rejected() -> void {
    assert_throws(ValueError, || parse_dump("14 19 AST_IDENT type float\n"))
}

fn test_identical_dumps_diff_to_nothing() -> void {
    assert_eq(diff_dumps(parse_dump(_SAMPLE), parse_dump(_SAMPLE)).len(), 0)
}

fn test_a_differing_value_is_reported_with_both_sides() -> void {
    let other = "# jaithon sema dump v1\n# file a.jai\n" +
        "14 19 AST_IDENT type int\n33 38 AST_BINARY fold 5\n"
    let found = diff_dumps(parse_dump(_SAMPLE), parse_dump(other))
    assert_eq(found.len(), 1)
    assert_eq(found[0].field, "type")
    assert_eq(found[0].ours, "float")
    assert_eq(found[0].theirs, "int")
}

fn test_a_missing_record_is_reported_as_null_on_that_side() -> void {
    let short = "# jaithon sema dump v1\n# file a.jai\n14 19 AST_IDENT type float\n"
    let found = diff_dumps(parse_dump(_SAMPLE), parse_dump(short))
    assert_eq(found.len(), 1)
    assert_eq(found[0].field, "fold")
    assert_eq(found[0].ours, "5")
    assert_true(found[0].theirs is null)
}

fn test_the_version_matches_the_c_writer() -> void {
    # `JAI_SEMADUMP_VERSION` in src/sema/semadump.h is the other half of this
    # number. They are one constant split across two languages.
    assert_eq(VERSION, 1)
}
