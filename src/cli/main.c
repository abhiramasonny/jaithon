#include "../core/runtime.h"
#include "../lang/lexer.h"
#include "../lang/parser.h"
#include <getopt.h>
#include <readline/readline.h>
#include <readline/history.h>
#include <sys/time.h>
#include <libgen.h>
#include <unistd.h>
#include <mach-o/dyld.h>

#define VERSION "2.0.0"

static char execDir[1024] = {0};

static void loadStdLib(void) {
    char stdPath[1024];
    snprintf(stdPath, sizeof(stdPath), "%s/lib/std.jai", execDir);
    
    FILE* f = fopen(stdPath, "r");
    if (!f) {
        snprintf(stdPath, sizeof(stdPath), "lib/std.jai");
        f = fopen(stdPath, "r");
    }
    if (!f) {
        return;
    }
    
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
    
    free(code);
}

static void runFile(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path);
        exit(1);
    }
    
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
    printf("  -v, --version    Show version\n");
    printf("  -h, --help       Show this help\n");
    printf("  --no-extension   Don't auto-append .jai extension\n\n");
    printf("If no file is given, starts interactive shell.\n");
}

int main(int argc, char* argv[]) {
    struct timeval start, stop;
    bool autoExt = true;
    bool forceShell = false;
    bool noStdLib = false;
    
    static struct option longOpts[] = {
        {"debug", no_argument, 0, 'd'},
        {"shell", no_argument, 0, 's'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"no-extension", no_argument, 0, 'n'},
        {"no-stdlib", no_argument, 0, 'N'},
        {0, 0, 0, 0}
    };
    
    char exePath[1024];
    uint32_t size = sizeof(exePath);
    if (_NSGetExecutablePath(exePath, &size) == 0) {
        char* dir = dirname(exePath);
        strncpy(execDir, dir, sizeof(execDir) - 1);
    }
    
    initRuntime();
    registerBuiltinKeywords();
    initParser();
    
    int opt;
    while ((opt = getopt_long(argc, argv, "dsvhN", longOpts, NULL)) != -1) {
        switch (opt) {
            case 'd':
                runtime.debug = true;
                break;
            case 's':
                forceShell = true;
                break;
            case 'N':
                noStdLib = true;
                break;
            case 'v':
                showVersion();
                freeRuntime();
                return 0;
            case 'h':
                showHelp(argv[0]);
                freeRuntime();
                return 0;
            case 'n':
                autoExt = false;
                break;
            default:
                showHelp(argv[0]);
                freeRuntime();
                return 1;
        }
    }
    
    if (runtime.debug) {
        printf("==================== DEBUG MODE ====================\n");
    }
    
    if (!noStdLib) {
        loadStdLib();
    }
    
    gettimeofday(&start, NULL);
    
    if (optind < argc && !forceShell) {
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
    
    freeRuntime();
    return 0;
}
