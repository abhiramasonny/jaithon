/* modsig.c — parse an imported module for its declaration signatures.
 *
 * The one place in the front end that reads a file other than the one being
 * compiled. It borrows the runtime's module search so that the checker looks a
 * module up exactly where the import instruction will, and it borrows nothing
 * else: no module object is created, no body runs, no cache is consulted or
 * written.
 */
#include "modsig.h"

#include "../common/diag.h"
#include "../lang/parser.h"
/* For the module search path only. The search rules are the runtime's (spec
 * §8.1) and having a second copy of them here would be a second answer to
 * "which file is `std.gui`". */
#include "../runtime/runtime.h"

/* One top-level name the module declares. */
typedef struct {
    const char *name;    /* interned in the signature's own AST arena */
    AstNode    *decl;
} SigDecl;

/* `from other import name as local` at the module's top level: the module does
 * not declare `local`, but importing it from here is legal and lands on
 * `other`'s declaration (spec §9's prelude is nothing but these). */
typedef struct {
    const char *local;
    const char *remote;
    const char *from;    /* dotted module name, as written */
} SigReexport;

struct ModuleSig {
    char       *path;        /* owned; absolute, and the cache key */
    char       *dotted;      /* owned; the name it was first loaded under */
    int         fileId;
    AstContext  ast;
    AstNode    *program;
    JAI_VEC(SigDecl)     decls;
    JAI_VEC(SigDecl)     types;       /* the class/trait/enum subset */
    JAI_VEC(SigReexport) reexports;
    JAI_VEC(const char *) wildcards;  /* `from x import *` */
    bool        visiting;             /* re-export cycle guard */
};

static JAI_VEC(ModuleSig *) sSigs;

/* A re-export chain longer than this is a mistake, not a design. */
#define SIG_MAX_REEXPORT_DEPTH 8

static bool nameEq(const char *a, const char *b) {
    if (a == b) return true;
    if (a == NULL || b == NULL) return false;
    return strcmp(a, b) == 0;
}

static void freeOwnedString(char *s) {
    if (s != NULL) (void)jaiRealloc(s, strlen(s) + 1, 0);
}

/* ------------------------------------------------------------------ */
/* Collecting the top-level declarations                                */
/* ------------------------------------------------------------------ */

static void addDecl(ModuleSig *sig, const char *name, AstNode *node, bool isType) {
    if (name == NULL || node == NULL) return;
    SigDecl entry = { name, node };
    JAI_VEC_PUSH(SigDecl, &sig->decls, entry);
    if (isType) JAI_VEC_PUSH(SigDecl, &sig->types, entry);
}

static void collect(ModuleSig *sig, AstNode **stmts, int count) {
    if (stmts == NULL) return;
    for (int i = 0; i < count; i++) {
        AstNode *n = stmts[i];
        if (n == NULL) continue;
        switch (n->kind) {
        case AST_FN_DECL:
            addDecl(sig, n->as.fn.name, n, false);
            break;
        case AST_CLASS_DECL:
            addDecl(sig, n->as.classDecl.name, n, true);
            break;
        case AST_TRAIT_DECL:
            addDecl(sig, n->as.traitDecl.name, n, true);
            break;
        case AST_ENUM_DECL:
            addDecl(sig, n->as.enumDecl.name, n, true);
            break;
        case AST_TYPE_DECL:
            addDecl(sig, n->as.typeDecl.name, n, false);
            break;
        case AST_FROM_IMPORT: {
            const char *from = n->as.fromImport.path;
            if (from == NULL) break;
            if (n->as.fromImport.isWildcard) {
                JAI_VEC_PUSH(const char *, &sig->wildcards, from);
                break;
            }
            for (int j = 0; j < n->as.fromImport.itemCount; j++) {
                AstImportItem *item = &n->as.fromImport.items[j];
                SigReexport re = { item->alias != NULL ? item->alias : item->name,
                                   item->name, from };
                JAI_VEC_PUSH(SigReexport, &sig->reexports, re);
            }
            break;
        }
        default:
            break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Loading                                                              */
/* ------------------------------------------------------------------ */

static ModuleSig *findCached(const char *path) {
    for (int i = 0; i < sSigs.count; i++)
        if (strcmp(sSigs.data[i]->path, path) == 0) return sSigs.data[i];
    return NULL;
}

ModuleSig *jaiModuleSigLoad(const char *dotted, int fromFileId) {
    if (dotted == NULL || dotted[0] == '\0') return NULL;

    char fromDir[JAI_MAX_PATH];
    fromDir[0] = '\0';
    JaiSourceFile *from = jaiSourceGet(fromFileId);
    if (from != NULL && from->path != NULL)
        jaiPathDirname(fromDir, sizeof fromDir, from->path);

    char resolved[JAI_MAX_PATH];
    if (!jaiResolveModulePathQuiet(dotted, fromDir, resolved, sizeof resolved))
        return NULL;

    ModuleSig *cached = findCached(resolved);
    if (cached != NULL) return cached;

    size_t length = 0;
    char *text = jaiReadFile(resolved, &length);
    if (text == NULL) return NULL;
    int fileId = jaiSourceAdd(resolved, text, length);   /* takes ownership */

    ModuleSig *sig = JAI_ALLOC(ModuleSig, 1);
    memset(sig, 0, sizeof *sig);
    sig->path = jaiStrdup(resolved);
    sig->dotted = jaiStrdup(dotted);
    sig->fileId = fileId;
    jaiAstContextInit(&sig->ast);

    /* Whatever this file has to say about itself belongs to its own
     * compilation. A syntax error two modules away must not surface as a
     * diagnostic of the file being checked, so the bag is swapped for a scratch
     * one and dropped — the same trick the import-cycle walk uses. */
    JaiDiagBag live = gDiags;
    jaiDiagInit(&gDiags);

    Lexer lex;
    AstNode *program = jaiParseSource(&sig->ast, &lex, text, length, fileId);
    jaiLexerFree(&lex);

    jaiDiagFree(&gDiags);
    gDiags = live;

    /* A file that does not parse has no signatures to offer, but it is still
     * cached: re-reading and re-parsing it once per import would be the whole
     * cost of the feature for no answer. */
    sig->program = program;
    if (program != NULL && program->kind == AST_PROGRAM)
        collect(sig, program->as.block.stmts, program->as.block.count);

    JAI_VEC_PUSH(ModuleSig *, &sSigs, sig);
    return sig;
}

/* ------------------------------------------------------------------ */
/* Lookup                                                               */
/* ------------------------------------------------------------------ */

static AstNode *findIn(ModuleSig *sig, const char *name, ModuleSig **outOwner,
                       int depth) {
    if (sig == NULL || name == NULL || depth <= 0 || sig->visiting) return NULL;

    for (int i = 0; i < sig->decls.count; i++) {
        if (!nameEq(sig->decls.data[i].name, name)) continue;
        if (outOwner != NULL) *outOwner = sig;
        return sig->decls.data[i].decl;
    }

    sig->visiting = true;
    AstNode *found = NULL;
    for (int i = 0; found == NULL && i < sig->reexports.count; i++) {
        const SigReexport *re = &sig->reexports.data[i];
        if (!nameEq(re->local, name)) continue;
        ModuleSig *next = jaiModuleSigLoad(re->from, sig->fileId);
        found = findIn(next, re->remote, outOwner, depth - 1);
    }
    for (int i = 0; found == NULL && i < sig->wildcards.count; i++) {
        ModuleSig *next = jaiModuleSigLoad(sig->wildcards.data[i], sig->fileId);
        found = findIn(next, name, outOwner, depth - 1);
    }
    sig->visiting = false;
    return found;
}

AstNode *jaiModuleSigFind(ModuleSig *sig, const char *name, ModuleSig **outOwner) {
    if (outOwner != NULL) *outOwner = sig;
    return findIn(sig, name, outOwner, SIG_MAX_REEXPORT_DEPTH);
}

int jaiModuleSigTypeCount(const ModuleSig *sig) {
    return sig == NULL ? 0 : sig->types.count;
}

AstNode *jaiModuleSigTypeAt(const ModuleSig *sig, int index) {
    if (sig == NULL || index < 0 || index >= sig->types.count) return NULL;
    return sig->types.data[index].decl;
}

const char *jaiModuleSigDotted(const ModuleSig *sig) {
    return sig == NULL ? NULL : sig->dotted;
}

ModuleSig *jaiModuleSigForFile(int fileId) {
    if (fileId < 0) return NULL;
    for (int i = 0; i < sSigs.count; i++)
        if (sSigs.data[i]->fileId == fileId) return sSigs.data[i];
    return NULL;
}

const char *jaiModuleSigQualify(ModuleSig *sig, const char *name) {
    if (sig == NULL || name == NULL) return name;
    /* A relative import is written from the importer's point of view (`.thing`),
     * which says nothing about identity; the file stem is the stable part. */
    const char *prefix = sig->dotted;
    while (prefix != NULL && *prefix == '.') prefix++;
    if (prefix == NULL || *prefix == '\0') prefix = sig->dotted;

    size_t prefixLen = strlen(prefix), nameLen = strlen(name);
    char *out = (char *)jaiArenaAlloc(&sig->ast.arena, prefixLen + nameLen + 2);
    memcpy(out, prefix, prefixLen);
    out[prefixLen] = '.';
    memcpy(out + prefixLen + 1, name, nameLen);
    out[prefixLen + nameLen + 1] = '\0';
    return out;
}

void jaiModuleSigFreeAll(void) {
    for (int i = 0; i < sSigs.count; i++) {
        ModuleSig *sig = sSigs.data[i];
        JAI_VEC_FREE(SigDecl, &sig->decls);
        JAI_VEC_FREE(SigDecl, &sig->types);
        JAI_VEC_FREE(SigReexport, &sig->reexports);
        JAI_VEC_FREE(const char *, &sig->wildcards);
        jaiAstContextFree(&sig->ast);
        freeOwnedString(sig->path);
        freeOwnedString(sig->dotted);
        JAI_FREE(ModuleSig, sig);
    }
    JAI_VEC_FREE(ModuleSig *, &sSigs);
}
