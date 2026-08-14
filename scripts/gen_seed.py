#!/usr/bin/env python3
"""Generate boot/seed.c from the .jaic images the front end needs to start.

The self-hosted front end is written in Jaithon, so compiling it needs a
Jaithon compiler. The embedded seed breaks that circle. It holds the serialised
.jaic image of each module in the compiler's startup closure as a byte array in
the binary. The `sLoadingFrontEnd` window in src/runtime/modules/module.c limits
seed access to that startup phase.

A fresh clone can build offline with no second compiler. The loader in
src/vm/bytecode/serialize.c already reads .jaic, so loading one costs a memcpy
and a deserialise.

Usage:
    scripts/gen_seed.py <cache-root> <output.c> [subtree ...]

`cache-root` is the tree module names are relative to -- normally `lib`, so
that lib/std/__jaicache__/list.jaic is "std.list". The optional subtrees limit
which caches are collected; without them the whole root is walked.

The limit is not an optimisation. Running the compiler writes cache entries for
whatever it happens to import, so a walk of the whole root embeds a set that
varies from run to run: measured, 44 modules and then 47 across two reseeds of
an unchanged tree. A seed that does not converge is the one failure a bootstrap
cannot recover from on its own, so what gets embedded has to be exactly what
the populate step set out to build, not whatever was found afterwards.
"""

import os
import sys
import zlib


def module_name_for(jaic_path, root):
    """`<root>/std/__jaicache__/list.jaic` -> `std.list`.

    The cache mirrors the source tree with a __jaicache__ directory beside each
    module, so the module name is the path with that directory removed and the
    separators turned into dots. Deriving it from the path rather than reading
    it out of the image keeps this script independent of the container format.
    """
    rel = os.path.relpath(jaic_path, root)
    parts = rel.split(os.sep)
    if "__jaicache__" not in parts:
        return None
    parts.remove("__jaicache__")
    if not parts or not parts[-1].endswith(".jaic"):
        return None
    parts[-1] = parts[-1][: -len(".jaic")]

    if parts[-1] == "mod" and len(parts) > 1:
        parts.pop()

    return ".".join(parts)


def source_path_for(jaic_path, root):
    """`<root>/jaithon/compile/opt/__jaicache__/chunk.jaic` -> `jaithon/compile/opt/chunk.jai`.

    The seed is keyed on this rather than on the module name, because the name
    an importer gives a module is not always the dotted one: a relative import
    (`from .chunk import ...`) names it `chunk`, and a seed keyed on names is
    unreachable for every module reached that way.
    """
    rel = os.path.relpath(jaic_path, root)
    parts = rel.split(os.sep)
    parts.remove("__jaicache__")
    parts[-1] = parts[-1][: -len(".jaic")] + ".jai"
    return "/".join(parts)


def collect(root, subtrees):
    found = {}
    for subtree in subtrees:
        for dirpath, _dirnames, filenames in os.walk(subtree):
            if os.path.basename(dirpath) != "__jaicache__":
                continue
            for name in filenames:
                if not name.endswith(".jaic"):
                    continue
                path = os.path.join(dirpath, name)
                module = module_name_for(path, root)
                if module is None:
                    continue
                with open(path, "rb") as f:
                    found[source_path_for(path, root)] = f.read()
    return dict(sorted(found.items()))


def read_manifest(path):
    """The library-relative source paths the seed is allowed to carry.

    Blank lines and `#` comments are ignored so the list can explain itself.
    """
    wanted = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                wanted.append(line)
    return wanted


def main():
    argv = sys.argv[1:]
    manifest = None
    if "--manifest" in argv:
        i = argv.index("--manifest")
        if i + 1 >= len(argv):
            sys.stderr.write("gen_seed.py: --manifest needs a path\n")
            return 2
        manifest = argv[i + 1]
        del argv[i : i + 2]

    if len(argv) < 2:
        sys.stderr.write(
            "usage: gen_seed.py <cache-root> <output.c> [--manifest <file>] "
            "[subtree ...]\n"
        )
        return 2

    root, out_path = argv[0], argv[1]
    subtrees = argv[2:] or [root]
    images = collect(root, subtrees)

    if manifest is not None:
        wanted = read_manifest(manifest)
        missing = [m for m in wanted if m not in images]
        if missing:
            sys.stderr.write(
                "gen_seed.py: %d module(s) in %s were not found in the cache:\n"
                "  %s\n"
                "Run the compiler over the library first so __jaicache__ holds "
                "them.\n" % (len(missing), manifest, "\n  ".join(missing))
            )
            return 1
        dropped = len(images) - len(wanted)
        images = {m: images[m] for m in wanted}
        sys.stderr.write(
            "gen_seed.py: manifest keeps %d module(s), leaves out %d\n"
            % (len(wanted), dropped)
        )

    if not images:
        sys.stderr.write(
            "gen_seed.py: no .jaic images under %r.\n"
            "Run the compiler once so __jaicache__ is populated, then reseed.\n"
            "An empty seed would build and then fail at run time, so this is "
            "an error rather than an empty table.\n" % root
        )
        return 1

    total = sum(len(v) for v in images.values())

    bin_path = os.path.join(os.path.dirname(out_path) or ".", "seed.bin")
    packed = []
    offset = 0
    with open(bin_path, "wb") as bf:
        for module, blob in images.items():
            z = zlib.compress(blob, 9)
            bf.write(z)
            packed.append((module, offset, len(z), len(blob)))
            offset += len(z)

    with open(out_path, "w") as f:
        f.write(
            "/* Generated by scripts/gen_seed.py. Do not edit.\n"
            " *\n"
            " * The index over boot/seed.bin, which holds the .jaic images the\n"
            " * self-hosted front end needs before it can compile anything --\n"
            " * including itself. The images are deflated one per module and\n"
            " * inflated on first use, so a program pays for what it imports.\n"
            " *\n"
            " * %d modules, %d bytes of images, %d bytes packed.\n"
            " * Regenerate with `make reseed`.\n"
            " */\n\n"
            '#include "boot/seed.h"\n\n'
            "#include <stdlib.h>\n"
            "#include <string.h>\n"
            "#include <zlib.h>\n\n"
            "/* Defined by boot/seed_blob.S. */\n"
            "extern const unsigned char jaiSeedBlob[];\n\n"
            "typedef struct {\n"
            "    const char *module;\n"
            "    size_t      offset;   /* into jaiSeedBlob */\n"
            "    size_t      packed;   /* deflated size */\n"
            "    size_t      raw;      /* size once inflated */\n"
            "} SeedSource;\n\n" % (len(images), total, offset)
        )
        f.write("static const SeedSource kSources[] = {\n")
        for module, off, pk, rw in packed:
            f.write('    {"%s", %d, %d, %d},\n' % (module, off, pk, rw))
        f.write("};\n\n")
        f.write(
            "#define JAI_SEED_N (sizeof kSources / sizeof kSources[0])\n\n"
            "/* Inflated on demand and kept. Never freed: the images outlive\n"
            " * every caller and the process is the only thing that ends. */\n"
            "static JaiSeedEntry kEntries[JAI_SEED_N];\n\n"
            "static const JaiSeedEntry *unpack(size_t i) {\n"
            "    if (kEntries[i].image != NULL) return &kEntries[i];\n"
            "    size_t raw = kSources[i].raw;\n"
            "    unsigned char *out = malloc(raw ? raw : 1);\n"
            "    if (out == NULL) return NULL;\n"
            "    uLongf got = (uLongf)raw;\n"
            "    if (uncompress(out, &got, jaiSeedBlob + kSources[i].offset,\n"
            "                   (uLong)kSources[i].packed) != Z_OK ||\n"
            "        got != raw) {\n"
            "        /* A corrupt seed must fail to load rather than load\n"
            "         * something nearly right; the caller answers a NULL by\n"
            "         * compiling from source. */\n"
            "        free(out);\n"
            "        return NULL;\n"
            "    }\n"
            "    kEntries[i].module = kSources[i].module;\n"
            "    kEntries[i].image  = out;\n"
            "    kEntries[i].length = raw;\n"
            "    return &kEntries[i];\n"
            "}\n\n"
            "/* The caller has an absolute path and the table holds paths\n"
            " * relative to the library root, so a key matches when it is a\n"
            " * trailing path component run of the argument. Anchoring on the\n"
            " * separator is what stops `list.jai` from matching\n"
            " * `mylist.jai`. */\n"
            "const JaiSeedEntry *jaiSeedFind(const char *sourcePath) {\n"
            "    if (sourcePath == NULL) return NULL;\n"
            "    size_t pathLen = strlen(sourcePath);\n"
            "    for (size_t i = 0; i < JAI_SEED_N; i++) {\n"
            "        size_t keyLen = strlen(kSources[i].module);\n"
            "        if (keyLen > pathLen) continue;\n"
            "        const char *tail = sourcePath + (pathLen - keyLen);\n"
            "        if (strcmp(tail, kSources[i].module) != 0) continue;\n"
            "        if (tail != sourcePath && tail[-1] != '/') continue;\n"
            "        return unpack(i);\n"
            "    }\n"
            "    return NULL;\n"
            "}\n\n"
            "size_t jaiSeedCount(void) { return JAI_SEED_N; }\n"
        )

    sys.stderr.write(
        "gen_seed.py: %d modules, %d bytes -> %s\n" % (len(images), total, out_path)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
