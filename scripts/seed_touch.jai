# Compiling and running this populates __jaicache__ with every module the seed
# has to carry. `make reseed` runs it with JAITHON_NO_SEED=1, so each import
# below is compiled from source rather than served by the seed being replaced.
#
# `std.list` stands in for the stdlib: compiling it drags in the whole
# self-hosted front end, which is what the seed is mostly made of. The REPL's
# scanner is not on that path -- nothing compiles it unless a prompt asks a
# question -- so it is named here. Leaving it out is not a build failure; it is
# a cold `jaithon repl` compiling the scanner from source, which under
# `--gc-stress` takes minutes.
from std.list import partition
import jaithon.compile.repl

fn main() {
    print(1)
}
