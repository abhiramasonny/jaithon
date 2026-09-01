#: The package as a caller reaches it: `import jaitoml`, then every name the
#: facade exports. The other test files import the modules directly, so this
#: is the only one that would notice a facade that names a module which is
#: not there.
import jaitoml

from std.test import assert_eq, assert_throws, assert_true
from jaitoml import Date, DateTime, TOMLDecodeError, Time

fn test_the_four_python_names_are_there() -> void {
    let value = jaitoml.loads("[server]\nport = 8080\nhosts = [\"a\", \"b\"]")
    assert_eq(value["server"]["port"], 8080)
    assert_eq(jaitoml.dumps({"server": {"port": 8080}}), "[server]\nport = 8080\n")
    assert_true(callable(jaitoml.load))
    assert_true(callable(jaitoml.dump))
}

fn test_the_error_and_the_calendar_types_are_there() -> void {
    let error = assert_throws(TOMLDecodeError, fn() -> void {
        let _ = jaitoml.loads("a = 1\na = 2\n")
    }, "a duplicate key")
    assert_eq(error.lineno, 2)
    assert_eq(error.colno, 6)
    let value = jaitoml.loads("d = 1979-05-27\nt = 07:32:00\ndt = 1979-05-27T07:32:00Z")
    assert_true(isinstance(value["d"], Date))
    assert_true(isinstance(value["t"], Time))
    assert_true(isinstance(value["dt"], DateTime))
    assert_eq(
        jaitoml.dumps(value),
        "d = 1979-05-27\nt = 07:32:00\ndt = 1979-05-27 07:32:00+00:00\n"
    )
}

fn test_the_constants_are_there() -> void {
    assert_eq(jaitoml.VERSION, "0.1.0")
    assert_eq(jaitoml.MIN_YEAR, 1)
    assert_eq(jaitoml.MAX_YEAR, 9999)
    assert_eq(jaitoml.MAX_LINE_LENGTH, 100)
}
