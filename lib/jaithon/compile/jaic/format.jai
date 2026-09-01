#: The numbers the `.jaic` container is made of: its magic, the version
#: window the reader accepts, the header flags, and the fixed limits every
#: record is checked against.
#:
#: Most of these must equal a constant in `src/vm/bytecode/serialize.h` or
#: `src/common/common.h` -- one number split across two languages -- and each
#: says which where it stands. They are gathered here so that a format bump
#: is one edit to one file rather than a hunt through the reader and the
#: writer, and because a constant is the one thing every other module of
#: `jaic/` needs and none of them should have to import a codec to get.
#:
#: What is NOT here: anything that reads or writes a byte. This file has no
#: imports at all, which is what lets it sit at the bottom of the directory.

#: The four bytes every `.jaic` file starts with.
pub let MAGIC: list[int] = [0x4A, 0x41, 0x49, 0x43]

#: Container format version; a mismatch means recompile.
#:
#: Must equal `JAIC_VERSION` in `src/vm/bytecode/serialize.h` — one number split across
#: two languages.
pub const VERSION: int = 14

#: First container version whose function records carry u16 flags.
pub const VERSION_FNFLAGS16: int = 14

#: Oldest container this reader accepts. Must equal `JAIC_VERSION_MIN` in
#: `src/vm/bytecode/serialize.h`.
#:
#: A range, not an equality: across a format bump the seed's images were written
#: by the previous generation, and refusing them leaves the tree with no front
#: end and no way to build one. `make reseed` twice closes the gap — generation
#: one still runs the old compiler out of the old seed.
pub const VERSION_MIN: int = 13

#: First container version whose line table is LTV1 (delta + LEB128). Kept as
#: the record of when the encoding changed; `VERSION_MIN` is now past it, so no
#: accepted image predates LTV1 and the reader carries no other line format.
pub const VERSION_LTV1: int = 12

#: First container version carrying a module string table, whose strings every
#: pool then names by a tag-10 index. `VERSION_MIN` is now past it, so the
#: inline tag-4 form the reader used to accept is gone.
pub const VERSION_STRTAB: int = 13

#: `JAI_COMPILER_VERSION`, recorded so that a new compiler ignores old caches.
#:
#: Must equal `JAI_COMPILER_VERSION` in `src/common/common.h`.
pub const COMPILER_VERSION: int = 26

#: The `buildId` the running binary stamps into a `.jaic` and demands back.
#:
#: Keyed on a hash of the C sources (`build/jai_build_id.h`) rather than
#: written down here, since this compiler never reads C. An image missing it
#: is rejected by `src/vm/bytecode/serialize.c` with a header that otherwise validates
#: in full.
#:
#: ```
#: build_id() == build_id()      # true
#: build_id() >= 0               # true
#: ```
pub fn build_id() -> int { return __prim__.jaic_build_id() }

#: Header flag: debug information is present.
pub const FLAG_DEBUG: int = 1

#: Header flag: built with `--release`, so assertions were stripped.
pub const FLAG_RELEASE: int = 2

#: Where the optimisation level lives in the container flags, matching
#: `JAIC_FLAG_OPT_SHIFT`/`JAIC_FLAG_OPT_MASK` in `src/vm/bytecode/serialize.h`.
#:
#: Must match what `cacheFlagsFor` expects exactly: an image missing these
#: bits silently fails the cache check and falls back to the seed, so a
#: compiler edit can appear to never take effect.
pub const FLAG_OPT_SHIFT: int = 3
pub const FLAG_OPT_MASK: int = 3 << 3

#: `flags` for a build at `opt_level`, as the C loader expects them.
pub fn flags_for(release: bool, opt_level: int) -> int {
    var level = opt_level
    if level < 0 { level = 0 }
    if level > 3 { level = 3 }
    let base = if release { FLAG_RELEASE } else { FLAG_DEBUG }
    return base | level << FLAG_OPT_SHIFT & FLAG_OPT_MASK
}

#: The `nameIndex` that means "this record has no name".
pub const NO_NAME: int = 4294967295

#: Deepest nesting of function and tuple records the reader will follow.
pub const MAX_DEPTH: int = 64

#: Smallest file that could hold a header and a checksum.
pub const MIN_SIZE: int = 32

pub const _EXC_ROW: int = 16
pub const _CONST_LIMIT: int = 16777216
