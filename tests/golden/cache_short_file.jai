#: Driven by tests/vm/cache_corrupt.sh, which copies this file aside, warms its
#: cache, then truncates and corrupts the .jaic in every way the loader has to
#: survive. Each case must recompile from source and print this one line.
import std.io

print("recompiled and ran")
