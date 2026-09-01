#: Files, paths, buffered streams, the standard streams, and delimited text.
#:
#: Four layers, from the shortest thing that works to the most controlled:
#:
#: - `read_file`, `write_file`, `read_lines` and friends do the whole job in one
#:   call and hold the whole file in memory.
#: - `open` returns a `File`, which is a thin, unbuffered wrapper over one
#:   operating-system handle. Pair it with `defer { file.close() }`.
#: - `BufferedReader` and `BufferedWriter` sit on a `File` and turn many small
#:   operations into few large ones. Reading a large file a line at a time goes
#:   through `BufferedReader`; anything else re-reads the file per line.
#: - `Path` is a path as a value: it joins, normalises, matches globs, walks
#:   directory trees, and asks the filesystem about itself.
#:
#: ```
#: write_file("/tmp/jaithon-io-doc.txt", "alpha\nbeta\n")
#: read_lines("/tmp/jaithon-io-doc.txt")        # ["alpha", "beta"]
#: Path("/tmp/jaithon-io-doc.txt").stem()       # "jaithon-io-doc"
#: ```
#:
#: Paths are separated by `/`, a file has no text mode to opt into, and text is
#: UTF-8 everywhere. Everything here reaches the operating system only through
#: the private primitive wrappers at the bottom of `std.io.stream`.
#:
#: Facade: imports and re-exports every sibling module in `io/`.
from .io.stream import SEEK_SET, SEEK_CURRENT, SEEK_END, DEFAULT_BUFFER_SIZE
from .io.stream import File, BufferedReader, BufferedWriter
from .io.stream import stdin, stdout, stderr
from .io.stream import open, read_file, read_lines, read_bytes
from .io.stream import write_file, write_bytes, append_file, write_lines
from .io.stream import split_lines, print_err, read_line, read_all_stdin, prompt
from .io.path import SEPARATOR, Path
from .io.csv import parse_csv, format_csv, read_csv, write_csv, read_tsv, write_tsv

export { SEEK_SET, SEEK_CURRENT, SEEK_END, DEFAULT_BUFFER_SIZE }
export { File, BufferedReader, BufferedWriter }
export { stdin, stdout, stderr }
export { open, read_file, read_lines, read_bytes }
export { write_file, write_bytes, append_file, write_lines }
export { split_lines, print_err, read_line, read_all_stdin, prompt }
export { SEPARATOR, Path }
export { parse_csv, format_csv, read_csv, write_csv, read_tsv, write_tsv }
