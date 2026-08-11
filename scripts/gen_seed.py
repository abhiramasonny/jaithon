#!/usr/bin/env python3
"""Generate boot/seed.c from the .jaic images the front end needs to start.

The self-hosted front end is written in Jaithon, so compiling it needs a
Jaithon compiler. Today that circle is broken by the C front end, which
compiles the compiler's own closure inside the `sLoadingFrontEnd` window in
src/runtime/module.c. That window is the last thing keeping src/lang, src/sema
and src/codegen alive: everything outside it is already self-hosted.

The seed breaks the circle without a second compiler. It holds the serialised
.jaic image of every module in that closure, as a byte array compiled into the
binary, so a fresh clone builds offline with no extra files and no network.
src/vm/serialize.c already reads .jaic, so loading one costs a memcpy and a
deserialise.

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

import base64
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

    # `mod.jai` is a package's body, not a module inside it: the directory
    # `lib/jaithon/compile` is imported as `jaithon.compile`, and that is the
    # name the importer looks the seed up under. Keying it `jaithon.compile.mod`
    # left every package in the seed unreachable.
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


def c_bytes(blob, indent="    "):
    out = []
    for i in range(0, len(blob), 16):
        row = ", ".join("0x%02x" % b for b in blob[i : i + 16])
        out.append(indent + row + ",")
    return "\n".join(out)


def c_base85(blob, indent="    "):
    """The payload as adjacent C string literals.

    A `0x%02x, ` array costs six characters per byte, which turned 3.2 MB of
    images into a 20 MB source file. Deflate first and the payload is 1.4 MB;
    base64 spends four characters per three bytes rather than six per one, so
    what lands on disk is about a tenth of what it was.

    Escapes are what make a string literal awkward for binary, and base64 has
    none: every character it emits is printable ASCII outside the set C treats
    specially, so no byte here needs a backslash and the literal's length is
    exactly its byte count. Lines are wrapped because a single multi-megabyte
    literal is slow to parse, and adjacent literals concatenate at translation
    time into the same object.
    """
    text = base64.b64encode(zlib.compress(blob, 9)).decode("ascii")
    out = []
    for i in range(0, len(text), 96):
        out.append('%s"%s"' % (indent, text[i : i + 96]))
    return "\n".join(out)


def b64_table(indent="    "):
    """The base64 alphabet as a 256-entry lookup, -1 for everything else.

    A table rather than arithmetic on the character, so that a byte outside the
    alphabet is rejected by the same lookup that decodes a valid one and there
    is no second place for the two to disagree.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    values = [-1] * 256
    for i, ch in enumerate(alphabet):
        values[ord(ch)] = i
    out = []
    for i in range(0, 256, 16):
        row = ", ".join("%d" % v for v in values[i : i + 16])
        out.append(indent + row + ",")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: gen_seed.py <cache-root> <output.c> [subtree ...]\n"
        )
        return 2

    root, out_path = sys.argv[1], sys.argv[2]
    subtrees = sys.argv[3:] or [root]
    images = collect(root, subtrees)
    if not images:
        sys.stderr.write(
            "gen_seed.py: no .jaic images under %r.\n"
            "Run the compiler once so __jaicache__ is populated, then reseed.\n"
            "An empty seed would build and then fail at run time, so this is "
            "an error rather than an empty table.\n" % root
        )
        return 1

    total = sum(len(v) for v in images.values())

    with open(out_path, "w") as f:
        f.write(
            "/* Generated by scripts/gen_seed.py. Do not edit.\n"
            " *\n"
            " * The .jaic images the self-hosted front end needs before it can\n"
            " * compile anything, including itself. See scripts/gen_seed.py for\n"
            " * why this exists and `make reseed` for how to regenerate it.\n"
            " *\n"
            " * %d modules, %d bytes, deflated and base64'd -- see c_base85 in\n"
            " * the generator for why the payload is not a byte array.\n"
            " */\n\n"
            '#include "boot/seed.h"\n\n'
            "#include <stdlib.h>\n"
            "#include <string.h>\n"
            "#include <zlib.h>\n\n" % (len(images), total)
        )

        for i, (module, blob) in enumerate(images.items()):
            f.write("/* %s */\n" % module)
            f.write("static const char kImage%d[] =\n" % i)
            f.write(c_base85(blob))
            f.write(";\n\n")

        f.write(
            "typedef struct {\n"
            "    const char *module;\n"
            "    const char *packed;   /* deflated, then base64 */\n"
            "    size_t      rawLen;   /* bytes once inflated */\n"
            "} SeedSource;\n\n"
        )
        f.write("static const SeedSource kSources[] = {\n")
        for i, (module, blob) in enumerate(images.items()):
            f.write('    {"%s", kImage%d, %d},\n' % (module, i, len(blob)))
        f.write("};\n\n")
        f.write(
            "#define JAI_SEED_N (sizeof kSources / sizeof kSources[0])\n\n"
            "/* Unpacked on demand and kept. A module is sought once, so a\n"
            " * cache is only insurance against a caller that asks twice; what\n"
            " * it really buys is that a program pays for the modules it\n"
            " * imports and not for all %d of them. Never freed, because the\n"
            " * images it holds outlive every caller and the process is the\n"
            " * only thing that ends. */\n"
            "static JaiSeedEntry kEntries[JAI_SEED_N];\n\n"
            "static const signed char kB64[256] = {\n"
            "%s"
            "};\n\n"
            "/* base64 into `out`, which is exactly the size the decode must\n"
            " * produce. Returns 0 on any byte that is not base64 or any length\n"
            " * that does not match -- a corrupt seed must fail to load rather\n"
            " * than load something that is nearly right. */\n"
            "static int b64Decode(const char *in, size_t inLen,\n"
            "                     unsigned char *out, size_t outLen) {\n"
            "    if (inLen %% 4 != 0) return 0;\n"
            "    size_t pad = 0;\n"
            "    if (inLen >= 1 && in[inLen - 1] == '=') pad++;\n"
            "    if (inLen >= 2 && in[inLen - 2] == '=') pad++;\n"
            "    if (inLen / 4 * 3 - pad != outLen) return 0;\n"
            "    size_t o = 0;\n"
            "    for (size_t i = 0; i < inLen; i += 4) {\n"
            "        signed char a = kB64[(unsigned char)in[i]];\n"
            "        signed char b = kB64[(unsigned char)in[i + 1]];\n"
            "        signed char c = kB64[(unsigned char)in[i + 2]];\n"
            "        signed char d = kB64[(unsigned char)in[i + 3]];\n"
            "        if (a < 0 || b < 0) return 0;\n"
            "        unsigned long v = (unsigned long)a << 18 |\n"
            "                          (unsigned long)b << 12;\n"
            "        if (c >= 0) v |= (unsigned long)c << 6;\n"
            "        else if (in[i + 2] != '=') return 0;\n"
            "        if (d >= 0) v |= (unsigned long)d;\n"
            "        else if (in[i + 3] != '=') return 0;\n"
            "        if (o < outLen) out[o++] = (unsigned char)(v >> 16);\n"
            "        if (c >= 0 && o < outLen) out[o++] = (unsigned char)(v >> 8);\n"
            "        if (d >= 0 && o < outLen) out[o++] = (unsigned char)v;\n"
            "    }\n"
            "    return o == outLen;\n"
            "}\n\n"
            "/* Fills kEntries[i] the first time it is asked for. A failure\n"
            " * leaves the entry empty and returns NULL, which the caller\n"
            " * already treats as \"the seed does not carry this\" and answers\n"
            " * by compiling from source. */\n"
            "static const JaiSeedEntry *unpack(size_t i) {\n"
            "    if (kEntries[i].image != NULL) return &kEntries[i];\n"
            "    const char *packed = kSources[i].packed;\n"
            "    size_t packedLen = strlen(packed);\n"
            "    size_t rawLen = kSources[i].rawLen;\n"
            "    if (packedLen %% 4 != 0) return NULL;\n"
            "    size_t deflatedLen = packedLen / 4 * 3;\n"
            "    if (packedLen >= 1 && packed[packedLen - 1] == '=') deflatedLen--;\n"
            "    if (packedLen >= 2 && packed[packedLen - 2] == '=') deflatedLen--;\n"
            "    unsigned char *deflated = malloc(deflatedLen ? deflatedLen : 1);\n"
            "    unsigned char *raw = malloc(rawLen ? rawLen : 1);\n"
            "    if (deflated == NULL || raw == NULL) goto fail;\n"
            "    if (!b64Decode(packed, packedLen, deflated, deflatedLen)) goto fail;\n"
            "    uLongf got = (uLongf)rawLen;\n"
            "    if (uncompress(raw, &got, deflated, (uLong)deflatedLen) != Z_OK ||\n"
            "        got != rawLen) {\n"
            "        goto fail;\n"
            "    }\n"
            "    free(deflated);\n"
            "    kEntries[i].module = kSources[i].module;\n"
            "    kEntries[i].image  = raw;\n"
            "    kEntries[i].length = rawLen;\n"
            "    return &kEntries[i];\n"
            "fail:\n"
            "    free(deflated);\n"
            "    free(raw);\n"
            "    return NULL;\n"
            "}\n\n" % (len(images), b64_table())
        )

        f.write(
            "/* The caller has an absolute path and the table holds paths\n"
            " * relative to the library root, so a key matches when it is a\n"
            " * trailing path component run of the argument. Anchoring on the\n"
            " * separator is what stops `list.jai` from matching\n"
            " * `mylist.jai`. */\n"
            "const JaiSeedEntry *jaiSeedFind(const char *sourcePath) {\n"
            "    if (sourcePath == NULL) return NULL;\n"
            "    size_t pathLen = strlen(sourcePath);\n"
            "    for (size_t i = 0; i < JAI_SEED_N;\n"
            "         i++) {\n"
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
