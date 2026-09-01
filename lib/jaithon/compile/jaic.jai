#: Read and write the `.jaic` container of `spec/BYTECODE.md` §4-§7, and
#: disassemble what comes out of it.
#:
#: This is the Jaithon half of `src/vm/bytecode/serialize.c` and must agree with it byte
#: for byte, so the layout below is normative rather than convenient:
#:
#: - Every integer is little-endian; a function record's pool is the chunk's
#:   pool plus whatever the record names by index that the chunk didn't
#:   already hold, appended after the chunk's own constants so a `u24` operand
#:   already in the code keeps pointing at the same value.
#: - A class spec's `superNameIndex` of 0 means "no superclass" (`OP_INHERIT`
#:   links the subclass).
#: - The line table is a *byte* length followed by an LTV1 stream — delta and
#:   LEB128 per entry, see `VERSION_LTV1`; it is debug data, ignored when
#:   comparing. A zero length means the spans were stripped into a `.jaid`
#:   beside the image.
#: - String constants are tag-10 indices into one module string table, not
#:   inline bytes; see `VERSION_STRTAB` and `StrTab`.
#: - Nothing is trusted: a short read, an out-of-range index or a bad CRC all
#:   mean "recompile" — `read_module` returns null, never a half-built mod.
#:
#: Facade: imports and re-exports every module of the `jaic/` directory. The
#: surface is what it was before the split, and the two importers — `mod.jai`
#: and `tests/checker/test_opt_dead.jai` — were not touched.
from jaithon.compile.jaic.format import COMPILER_VERSION, FLAG_DEBUG, FLAG_OPT_MASK
from jaithon.compile.jaic.format import FLAG_OPT_SHIFT, FLAG_RELEASE, MAGIC, MAX_DEPTH
from jaithon.compile.jaic.format import MIN_SIZE, NO_NAME, VERSION, VERSION_FNFLAGS16
from jaithon.compile.jaic.format import VERSION_LTV1, VERSION_MIN, VERSION_STRTAB
from jaithon.compile.jaic.format import build_id, flags_for
from jaithon.compile.jaic.bits import bits_to_float, crc32, float_bits, source_hash
from jaithon.compile.jaic.bits import utf8_bytes, utf8_text
from jaithon.compile.jaic.stream import Reader, Writer
from jaithon.compile.jaic.model import Import, Module, StrTab
from jaithon.compile.jaic.write import write_module
from jaithon.compile.jaic.read import read_module
from jaithon.compile.jaic.disassemble import disassemble, disassemble_instruction

export { COMPILER_VERSION, FLAG_DEBUG, FLAG_OPT_MASK, FLAG_OPT_SHIFT }
export { FLAG_RELEASE, MAGIC, MAX_DEPTH, MIN_SIZE, NO_NAME, VERSION }
export { VERSION_FNFLAGS16, VERSION_LTV1, VERSION_MIN, VERSION_STRTAB }
export { build_id, flags_for }
export { bits_to_float, crc32, float_bits, source_hash, utf8_bytes, utf8_text }
export { Reader, Writer }
export { Import, Module, StrTab }
export { write_module }
export { read_module }
export { disassemble, disassemble_instruction }
