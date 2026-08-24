#: A first test, so the suite has something to find while the package is being
#: written.
import jainum
from std.test import assert_eq

fn test_the_package_loads() -> void {
    assert_eq(jainum.VERSION, "0.1.0")
}
