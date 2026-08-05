# expect: E0101
# A statement that opens with `{` is a block. The REPL reads such a line as a
# dict or set literal when it cannot be a block, but a file has no such second
# reading: here `{"a": 1}` is a block whose first statement is a string, and
# the `:` after it ends nothing.

{"a": 1}
