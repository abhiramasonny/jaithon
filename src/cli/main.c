#include "../core/runtime.h"
#include "../core/parallel.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include "../vm/bytecode.h"
#include <getopt.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <sys/time.h>
#include <libgen.h>
#include <unistd.h>
#include <mach-o/dyld.h>

#define VERSION "2.2.2"

static bool readWholeFile(const char* path, char** outBuf, long* outSize) {
    FILE* f = fopen(path, "r");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* data = malloc(size + 1);
    if (!data) {
        fclose(f);
        return false;
    }
    size_t read = fread(data, 1, size, f);
    fclose(f);
    data[read] = '\0';
    if (outBuf) *outBuf = data; else free(data);
    if (outSize) *outSize = (long)read;
    return true;
}

static uint64_t hashFile(const char* path) {
    char* data = NULL;
    if (!readWholeFile(path, &data, NULL)) return 0;
    uint64_t h = hashSource(data);
    free(data);
    return h;
}

static void resolveSourcePath(const char* input, bool autoExt, const char* ext, char* out, size_t outSize) {
    strncpy(out, input, outSize - 1);
    out[outSize - 1] = '\0';
    if (autoExt && ext && !strstr(out, ext)) {
        strncat(out, ext, outSize - strlen(out) - 1);
    }
}

static void makeJaicPath(const char* sourcePath, char* out, size_t outSize) {
    strncpy(out, sourcePath, outSize - 1);
    out[outSize - 1] = '\0';
    char* dot = strrchr(out, '.');
    if (dot) {
        strcpy(dot, ".jaic");
    } else {
        strncat(out, ".jaic", outSize - strlen(out) - 1);
    }
}

static void loadStdLib(void) {
    const char* envLib = getenv("JAITHON_LIB");
    char stdPath[1024];
    const char* bases[] = {
        envLib && strlen(envLib) > 0 ? envLib : NULL,
        strlen(gExecDir) > 0 ? gExecDir : NULL,
        "/usr/local/share/jaithon",
        "/usr/local/lib/jaithon",
        "/Library/Jaithon",
        "/opt/homebrew/share/jaithon",
        NULL
    };
    FILE* f = NULL;
    for (int i = 0; bases[i]; i++) {
        snprintf(stdPath, sizeof(stdPath), "%s/lib/std.jai", bases[i]);
        f = fopen(stdPath, "r");
        if (f) break;
    }
    if (!f) {
        snprintf(stdPath, sizeof(stdPath), "lib/std.jai");
        f = fopen(stdPath, "r");
    }
    if (!f) return;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* code = malloc(size + 1);
    if (!code) {
        fclose(f);
        return;
    }
    
    fread(code, 1, size, f);
    code[size] = '\0';
    fclose(f);
    
    Lexer lex;
    lexerInit(&lex, code);
    parseProgram(&lex);
    if (eagerCompileEnabled()) {
        compileModuleFunctions(runtime.currentModule, eagerCompileStrict());
    }
    
    free(code);
}

static void runFile(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path);
        exit(1);
    }
    
    
    char absPath[1024];
    if (path[0] == '/') {
        strncpy(absPath, path, sizeof(absPath) - 1);
    } else {
        char cwd[512];
        if (getcwd(cwd, sizeof(cwd))) {
            snprintf(absPath, sizeof(absPath), "%s/%s", cwd, path);
        } else {
            strncpy(absPath, path, sizeof(absPath) - 1);
        }
    }
    strncpy(runtime.currentSourceFile, absPath, sizeof(runtime.currentSourceFile) - 1);
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* code = malloc(size + 1);
    if (!code) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(f);
        exit(1);
    }
    
    fread(code, 1, size, f);
    code[size] = '\0';
    fclose(f);
    
    Lexer lex;
    lexerInit(&lex, code);
    parseProgram(&lex);
    if (eagerCompileEnabled()) {
        compileModuleFunctions(runtime.currentModule, eagerCompileStrict());
    }
    
    free(code);
}

static void runShell(void) {
    runtime.shellMode = true;
    printf("JAITHON v%s - Interactive Shell\n", VERSION);
    printf("Type 'exit' to quit, 'help' for commands\n\n");
    
    for (;;) {
        char* line = readline("> ");
        if (!line) break;
        
        if (strlen(line) == 0) {
            free(line);
            continue;
        }
        
        add_history(line);
        
        if (strcmp(line, "exit") == 0 || strcmp(line, "quit") == 0) {
            free(line);
            break;
        }
        
        if (strcmp(line, "help") == 0) {
            printf("\nJAITHON Commands:\n");
            printf("  var x = value    - Define a variable\n");
            printf("  print expr       - Print an expression\n");
            printf("  func name(args)  - Define a function\n");
            printf("  if cond then do  - Conditional\n");
            printf("  while cond do    - Loop\n");
            printf("  import module    - Import a .jai file\n");
            printf("  exit             - Exit shell\n\n");
            printf("Built-in functions: sin, cos, tan, sqrt, abs, floor, ceil, round, time, rand, len, str, num\n");
            printf("Constants: PI, E\n\n");
            free(line);
            continue;
        }
        
        if (strcmp(line, "vars") == 0) {
            Module* m = runtime.currentModule;
            printf("\nVariables:\n");
            for (int i = 0; i < m->varCount; i++) {
                printf("  %s = ", m->variables[i].name);
                Value v = m->variables[i].value;
                switch (v.type) {
                    case VAL_NUMBER: printf("%g", v.as.number); break;
                    case VAL_STRING: printf("\"%s\"", v.as.string); break;
                    case VAL_BOOL: printf("%s", v.as.boolean ? "true" : "false"); break;
                    case VAL_FUNCTION: printf("<function>"); break;
                    case VAL_NATIVE_FUNC: printf("<native>"); break;
                    default: printf("null");
                }
                printf("\n");
            }
            printf("\n");
            free(line);
            continue;
        }
        
        if (strcmp(line, "funcs") == 0) {
            Module* m = runtime.currentModule;
            printf("\nFunctions:\n");
            for (int i = 0; i < m->funcCount; i++) {
                JaiFunction* f = m->functions[i];
                printf("  %s(", f->name);
                for (int j = 0; j < f->paramCount; j++) {
                    if (j > 0) printf(", ");
                    printf("%s", f->params[j]);
                }
                if (f->isVariadic) printf("...");
                printf(")\n");
            }
            printf("\n");
            free(line);
            continue;
        }
        
        Lexer lex;
        lexerInit(&lex, line);
        parseProgram(&lex);
        
        free(line);
    }
}

static void showVersion(void) {
    printf("JAITHON v%s\n", VERSION);
    printf("A simple programming language for learning\n");
}

static void showHelp(const char* prog) {
    printf("Usage: %s [options] [file]\n\n", prog);
    printf("Options:\n");
    printf("  -d, --debug      Enable debug mode\n");
    printf("  -s, --shell      Start interactive shell\n");
    printf("  -c, --compile    Compile a .jai file to .jaic and exit\n");
    printf("  -e, --execute    Execute a precompiled .jaic bundle\n");
    printf("  -v, --version    Show version\n");
    printf("  -h, --help       Show this help\n");
    printf("  --no-extension   Don't auto-append .jai extension\n");
    printf("  --serial         Disable parallel execution\n");
    printf("  --no-gpu         Disable GPU acceleration\n");
    printf("  --threads=N      Set max threads (default: auto)\n\n");
    printf("Parallelization:\n");
    printf("  Jaithon automatically parallelizes loops when safe.\n");
    printf("  Uses multi-threading, SIMD, and GPU (Metal) when available.\n\n");
    printf("If no file is given, starts interactive shell.\n");
}

int main(int argc, char* argv[]) {
    struct timeval start, stop;
    bool autoExt = true;
    bool forceShell = false;
    bool noStdLib = false;
    int maxThreads = 0;
    bool serialMode = false;
    bool noGpu = false;
    const char* compileTarget = NULL;
    const char* execTarget = NULL;
    
    static struct option longOpts[] = {
        {"debug", no_argument, 0, 'd'},
        {"shell", no_argument, 0, 's'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"no-extension", no_argument, 0, 'n'},
        {"no-stdlib", no_argument, 0, 'N'},
        {"serial", no_argument, 0, 'S'},
        {"no-gpu", no_argument, 0, 'G'},
        {"threads", required_argument, 0, 'T'},
        {"compile", required_argument, 0, 'c'},
        {"execute", required_argument, 0, 'e'},
        {0, 0, 0, 0}
    };
    
    char exePath[1024];
    uint32_t size = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &size) == 0) {
        char* dir = dirname(exePath);
        setExecDir(dir);
    }
    
    initRuntime();
    parallelInit();  
    registerGuiFunctions();
    registerBuiltinKeywords();
    initParser();
    cacheInit(gExecDir);
    
    setbuf(stdout, NULL);
    
    int opt;
    while ((opt = getopt_long(argc, argv, "dsvhNnSGT:c:e:", longOpts, NULL)) != -1) {
        switch (opt) {
            case 'd':
                runtime.debug = true;
                break;
            case 's':
                forceShell = true;
                break;
            case 'S':
                serialMode = true;
                break;
            case 'G':
                noGpu = true;
                break;
            case 'T':
                maxThreads = atoi(optarg);
                break;
            case 'N':
                noStdLib = true;
                break;
            case 'c':
                compileTarget = optarg;
                break;
            case 'e':
                execTarget = optarg;
                break;
            case 'v':
                showVersion();
                parallelShutdown();
                freeRuntime();
                return 0;
            case 'h':
                showHelp(argv[0]);
                parallelShutdown();
                freeRuntime();
                return 0;
            case 'n':
                autoExt = false;
                break;
            default:
                showHelp(argv[0]);
                parallelShutdown();
                freeRuntime();
                return 1;
        }
    }
    
    if (compileTarget && execTarget) {
        fprintf(stderr, "Cannot compile (-c) and execute (-e) in the same invocation.\n");
        parallelShutdown();
        freeRuntime();
        return 1;
    }
    
    
    if (serialMode) {
        parallelSetMode(PAR_MODE_SERIAL);
    }
    if (noGpu) {
        parallelEnableGPU(false);
    }
    if (maxThreads > 0) {
        parallelSetMaxThreads(maxThreads);
    }
    
    if (runtime.debug) {
        printf("==================== DEBUG MODE ====================\n");
        printf("Parallel mode: %s\n", serialMode ? "serial" : "auto");
        printf("GPU acceleration: %s\n", noGpu ? "disabled" : "enabled");
        printf("Max threads: %d\n", maxThreads > 0 ? maxThreads : parallelConfig.maxThreads);
    }
    
    if (!noStdLib) {
        loadStdLib();
    }
    
    gettimeofday(&start, NULL);
    
    int status = 0;
    
    if (compileTarget) {
        runtime.compileOnly = true;
        char srcPath[1024];
        resolveSourcePath(compileTarget, autoExt, ".jai", srcPath, sizeof(srcPath));
        
        int baseFuncCount = runtime.currentModule ? runtime.currentModule->funcCount : 0;
        runFile(srcPath);
        
        Module* mod = runtime.currentModule;
        int totalFuncs = mod ? mod->funcCount : 0;
        int newFuncs = totalFuncs - baseFuncCount;
        if (newFuncs < 0) newFuncs = 0;
        
        BundleEntry* entries = newFuncs > 0 ? malloc(sizeof(BundleEntry) * newFuncs) : NULL;
        int written = 0;
        bool ok = true;
        const char* firstName = NULL;
        for (int i = baseFuncCount; ok && i < totalFuncs; i++) {
            JaiFunction* f = mod->functions[i];
            if (!f || !f->body) continue;
            CompiledFunc* compiled = getCompiledFunc(f);
            if (!compiled) {
                fprintf(stderr, "Error: failed to compile '%s'\n", f->name);
                ok = false;
                break;
            }
            entries[written].func = f;
            entries[written].compiled = compiled;
            entries[written].bodyHash = functionBodyHash(f);
            if (!firstName) firstName = f->name;
            written++;
        }
        
        if (ok && written == 0) {
            fprintf(stderr, "No functions to compile in %s\n", srcPath);
            ok = false;
        }
        
        const char* entryName = "main";
        bool hasMain = false;
        for (int i = 0; i < written; i++) {
            if (strcmp(entries[i].func->name, "main") == 0) {
                hasMain = true;
                break;
            }
        }
        if (!hasMain && firstName) {
            entryName = firstName;
        }
        
        char outPath[1024];
        makeJaicPath(srcPath, outPath, sizeof(outPath));
        uint64_t srcHash = hashFile(srcPath);
        
        if (ok) {
            if (!saveJaicBundle(outPath, entries, written, entryName, srcHash)) {
                fprintf(stderr, "Failed to write bundle to %s\n", outPath);
                ok = false;
            } else {
                printf("Wrote %s with %d functions (entry: %s)\n", outPath, written, entryName);
            }
        }
        
        free(entries);
        
        gettimeofday(&stop, NULL);
        if (runtime.debug) {
            double elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0;
            printf("\n==================== Execution time: %.4fs ====================\n", elapsed);
        }
        printCompilationStats();
        parallelShutdown();
        cacheFree();
        freeRuntime();
        return ok ? 0 : 1;
    }
    
    if (execTarget) {
        char bundlePath[1024];
        resolveSourcePath(execTarget, autoExt, ".jaic", bundlePath, sizeof(bundlePath));
        strncpy(runtime.currentSourceFile, bundlePath, sizeof(runtime.currentSourceFile) - 1);
        
        char sourcePath[1024];
        strncpy(sourcePath, bundlePath, sizeof(sourcePath) - 1);
        char* dot = strrchr(sourcePath, '.');
        if (dot) {
            strcpy(dot, ".jai");
        }
        
        runtime.compileOnly = true;
        FILE* srcCheck = fopen(sourcePath, "r");
        if (srcCheck) {
            fclose(srcCheck);
            runFile(sourcePath);
        }
        runtime.compileOnly = false;
        
        char entryName[MAX_NAME_LEN] = "main";
        uint64_t srcHash = 0;
        if (!loadJaicBundle(bundlePath, runtime.currentModule, entryName, sizeof(entryName), &srcHash)) {
            fprintf(stderr, "Failed to load bundle: %s\n", bundlePath);
            status = 1;
        } else {
            JaiFunction* entryFunc = findFunction(entryName);
            if (!entryFunc) {
                fprintf(stderr, "Entry function '%s' not found in %s\n", entryName, bundlePath);
                status = 1;
            } else {
                callValue(makeFunction(entryFunc), NULL, 0);
            }
        }
    } else if (optind < argc && !forceShell) {
        char path[MAX_NAME_LEN];
        strncpy(path, argv[optind], sizeof(path) - 5);
        
        if (autoExt && !strstr(path, ".jai")) {
            strcat(path, ".jai");
        }
        
        runFile(path);
    } else {
        runShell();
    }
    
    gettimeofday(&stop, NULL);
    
    if (runtime.debug) {
        double elapsed = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) / 1000000.0;
        printf("\n==================== Execution time: %.4fs ====================\n", elapsed);
    }
    
    printCompilationStats();
    parallelShutdown();
    cacheFree();
    freeRuntime();
    return status;
}
