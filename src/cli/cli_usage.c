/* cli_usage.c — the `--help` and `--version` text.
 *
 * Pure formatting: neither function touches JaiCliOptions, the diagnostic
 * bag, or any of the other CLI-internal state, which is what makes this the
 * easiest section of the old main.c to lift out whole. jaiCliPrintUsage and
 * jaiCliPrintVersion are declared in cli.h because main() calls them directly
 * for `-h`/`-v`/`help`/`version` before the rest of the CLI machinery (VM
 * init, module path, prelude) is even set up.
 */
#include "cli_internal.h"

#include "../native/native.h"
#include "../vm/serialize.h"

void jaiCliPrintUsage(FILE *out) {
    if (out == NULL) return;
    fputs(
        "jaithon " JAI_VERSION_STRING " — the Jaithon programming language\n"
        "\n"
        "usage:\n"
        "  jaithon                        start the REPL\n"
        "  jaithon FILE [args...]         run FILE (shorthand for `run`)\n"
        "  jaithon COMMAND [options] [paths...]\n"
        "\n"
        "commands:\n"
        "  run FILE [-- args...]      compile FILE, run it, then call its main()\n"
        "  repl                       interactive read-eval-print loop\n"
        "  check PATH...              compile and type-check without running\n"
        "  build FILE...              compile to a .jaic image\n"
        "  fmt [--check|--diff] PATH... format source          (jaithon.tool.fmt)\n"
        "  test [PATH...]             discover and run tests   (jaithon.tool.test)\n"
        "  doc [--out DIR] [PATH...]  generate documentation   (jaithon.tool.doc)\n"
        "  bench [PATH...]            run benchmarks           (jaithon.tool.bench)\n"
        "  disasm FILE...             print the compiled bytecode\n"
        "  ast FILE...                print the syntax tree\n"
        "  tokens FILE...             print the token stream\n"
        "  version                    print version information\n"
        "  help                       print this message\n"
        "\n"
        "options:\n"
        "  -h, --help                 print this message and exit\n"
        "  -v, --version              print version information and exit\n"
        "      --eval EXPR            evaluate EXPR, print its value, and exit\n"
        "  -O0 -O1 -O2 -O3            optimisation level (default -O2)\n"
        "      --release              strip asserts and debug information\n"
        "      --debug-trace          trace every instruction as it executes\n"
        "      --gc-stress            collect on every allocation\n"
        "      --no-cache             ignore and do not write __jaicache__\n"
        "      --no-prelude           do not load std.prelude\n"
        "      --strict               unannotated parameters are an error\n"
        "      --time                 print elapsed wall time\n"
        "      --stats                print VM, inline-cache and GC statistics\n"
        "      --threads=N            worker threads for std.thread\n"
        "      --no-gpu               disable GPU acceleration\n"
        "      --emit=ast|bc|tokens   compile and dump the given form\n"
        "      --json                 machine-readable output where supported\n"
        "      --color=auto|always|never\n"
        "      --out PATH             write the output to PATH\n"
        "\n"
        "Every option that takes a value accepts both `--name value` and\n"
        "`--name=value`, here and in fmt, test, doc and bench alike.\n"
        "  -I PATH                    add PATH to the module search path\n"
        "  --                         end options; the rest goes to the script\n",
        out);
}

void jaiCliPrintVersion(FILE *out) {
    if (out == NULL) return;
    fprintf(out, "jaithon %s\n", JAI_VERSION_STRING);
    fprintf(out, "bytecode %u, .jaic container %d\n",
            (unsigned)JAI_COMPILER_VERSION, JAIC_VERSION);
    if (jaiGpuAvailable()) {
        const char *device = jaiGpuDeviceName();
        fprintf(out, "gpu: %s\n", device != NULL ? device : "available");
    } else {
        fputs("gpu: unavailable\n", out);
    }
#ifdef JAI_HAVE_READLINE
    fputs("repl: readline\n", out);
#else
    fputs("repl: plain\n", out);
#endif
}
