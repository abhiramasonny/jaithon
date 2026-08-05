/* codegen.c — the code generator's shared machinery and entry points.
 *
 * Normative references: spec/BYTECODE.md §2 (operand encoding) and §3 (opcode
 * table). This pass performs no name lookup: the resolver has already bound
 * every AST_IDENT to a Symbol and assigned every local slot and upvalue index,
 * so emission is a direct structural walk. The walk itself is in
 * codegen_expr.c, codegen_stmt.c, codegen_decl.c and codegen_pattern.c; what
 * lives here is what all four need — symbols, operators, literals, type
 * guards — plus jaiCompileProgram and jaiCompileFunction. codegen_internal.h
 * holds the emitter itself and says which file owns what.
 *
 * Two invariants make the rest readable.
 *
 *   1. Every expression emitter leaves exactly one value on the operand stack;
 *      every statement emitter leaves none. FnCtx.stackDepth tracks that
 *      statically, which is what lets break/continue emit the right POPN and
 *      what feeds ObjFunction.maxSlots.
 *
 *   2. Pattern matching never leaves partial results on the stack. The
 *      candidate value lives in a frame slot and every test is the triple
 *      `GET_LOCAL slot; MATCH_x -> miss; POP`, so at *every* miss edge the
 *      stack holds exactly one extra value. One POP at the miss label undoes
 *      it, whatever the pattern's shape.
 */
#include "codegen_internal.h"

/* Symbols                                                              */
/* ------------------------------------------------------------------ */

/* Symbol kinds that live in the module's global table. Everything the resolver
 * did not put in a frame slot or an upvalue is reached by interned name. */
static bool symbolIsGlobalLike(const Symbol *sym) {
    switch (sym->kind) {
    case SYM_LOCAL:
    case SYM_PARAM:
    case SYM_UPVALUE:
    case SYM_FIELD:
        return false;
    default:
        return true;
    }
}

void jaiCgGetGlobal(Emitter *e, const char *name, JaiSpan span) {
    uint32_t k = nameConst(e, name);
    uint16_t cache = newCache(e);
    emitOp(e, OP_GET_GLOBAL, span);
    emitU24(e, k, span);
    emitU16(e, cache, span);
}

void jaiCgLoadSymbol(Emitter *e, Symbol *sym, const char *name, JaiSpan span) {
    if (sym == NULL) {
        /* REPL and error-recovery paths: fall back to a late-bound global. */
        jaiCgGetGlobal(e, name, span);
        return;
    }
    switch (sym->kind) {
    case SYM_LOCAL:
    case SYM_PARAM:
        emitGetLocal(e, sym->slot, span);
        break;
    case SYM_UPVALUE:
        emitOp(e, OP_GET_UPVALUE, span);
        emitByte(e, (uint8_t)(sym->slot < 0 ? 0 : sym->slot), span);
        break;
    case SYM_FIELD: {
        /* An unqualified field name inside a method is `self.name`. */
        emitGetLocal(e, 0, span);
        uint32_t k = nameConst(e, sym->name != NULL ? sym->name : name);
        uint16_t cache = newCache(e);
        emitOp(e, OP_GET_FIELD, span);
        emitU24(e, k, span);
        emitU16(e, cache, span);
        break;
    }
    default:
        jaiCgGetGlobal(e, sym->name != NULL ? sym->name : name, span);
        break;
    }
}

/* Consumes the value on top of the stack. `declaring` selects DEF_GLOBAL over
 * SET_GLOBAL for the initial binding of a module-level name. */
void jaiCgStoreSymbol(Emitter *e, Symbol *sym, const char *name,
                      JaiSpan span, bool declaring) {
    if (sym == NULL || symbolIsGlobalLike(sym)) {
        const char *text = (sym != NULL && sym->name != NULL) ? sym->name : name;
        uint32_t k = nameConst(e, text);
        if (declaring) {
            emitOp(e, OP_DEF_GLOBAL, span);
            emitU24(e, k, span);
        } else {
            uint16_t cache = newCache(e);
            emitOp(e, OP_SET_GLOBAL, span);
            emitU24(e, k, span);
            emitU16(e, cache, span);
            emitPop(e, span);
        }
        return;
    }

    switch (sym->kind) {
    case SYM_LOCAL:
    case SYM_PARAM:
        emitBindLocal(e, sym->slot, span);
        break;
    case SYM_UPVALUE:
        emitOp(e, OP_SET_UPVALUE, span);
        emitByte(e, (uint8_t)(sym->slot < 0 ? 0 : sym->slot), span);
        emitPop(e, span);
        break;
    case SYM_FIELD: {
        /* The value is already on top; SET_FIELD wants (object, value). */
        emitGetLocal(e, 0, span);
        emitOp(e, OP_SWAP, span);
        uint32_t k = nameConst(e, sym->name != NULL ? sym->name : name);
        uint16_t cache = newCache(e);
        emitOp(e, OP_SET_FIELD, span);
        emitU24(e, k, span);
        emitU16(e, cache, span);
        break;
    }
    default:
        JAI_UNREACHABLE();
        break;
    }
}

/* Operators                                                            */
/* ------------------------------------------------------------------ */

/* OP_NOP means "not a single opcode": `and`, `or` and `@` are lowered by their
 * own emitters. */
OpCode jaiCgOpcodeForOp(OpKind op) {
    switch (op) {
    case OPK_ADD:       return OP_ADD;
    case OPK_SUB:       return OP_SUB;
    case OPK_MUL:       return OP_MUL;
    case OPK_DIV:       return OP_DIV;
    case OPK_FLOORDIV:  return OP_FLOORDIV;
    case OPK_MOD:       return OP_MOD;
    case OPK_POW:       return OP_POW;
    case OPK_ADD_WRAP:  return OP_ADD_WRAP;
    case OPK_SUB_WRAP:  return OP_SUB_WRAP;
    case OPK_MUL_WRAP:  return OP_MUL_WRAP;
    case OPK_BAND:      return OP_BAND;
    case OPK_BOR:       return OP_BOR;
    case OPK_BXOR:      return OP_BXOR;
    case OPK_SHL:       return OP_SHL;
    case OPK_SHR:       return OP_SHR;
    case OPK_EQ:        return OP_EQ;
    case OPK_NE:        return OP_NE;
    case OPK_LT:        return OP_LT;
    case OPK_LE:        return OP_LE;
    case OPK_GT:        return OP_GT;
    case OPK_GE:        return OP_GE;
    case OPK_IS:        return OP_IS;
    case OPK_IS_NOT:    return OP_IS_NOT;
    case OPK_IN:        return OP_IN;
    case OPK_NOT_IN:    return OP_NOT_IN;
    case OPK_NEG:       return OP_NEG;
    case OPK_POS:       return OP_POS;
    case OPK_NOT:       return OP_NOT;
    case OPK_BNOT:      return OP_BNOT;
    default:            return OP_NOP;
    }
}

void jaiCgInvokeName(Emitter *e, const char *name, int argc, JaiSpan span) {
    uint32_t k = nameConst(e, name);
    uint16_t cache = newCache(e);
    emitOp(e, OP_INVOKE, span);
    emitU24(e, k, span);
    emitByte(e, (uint8_t)argc, span);
    emitU16(e, cache, span);
    adjust(e, -argc);
}

/* `@` has no opcode: matrix multiplication is library-defined and dispatches
 * through the __matmul__ dunder like any other overload. */
void jaiCgBinaryOp(Emitter *e, OpKind op, JaiSpan span) {
    if (op == OPK_MATMUL) {
        jaiCgInvokeName(e, "__matmul__", 1, span);
        return;
    }
    OpCode code = jaiCgOpcodeForOp(op);
    if (code == OP_NOP) {
        CG_ERROR(e, span, E0902_INTERNAL_ERROR,
                 "no opcode for binary operator `%s`", jaiOpKindText(op));
        emitOp(e, OP_NOP, span);
        adjust(e, -1);
        return;
    }
    emitOp(e, code, span);
}

/* Literal constants                                                    */
/* ------------------------------------------------------------------ */

void jaiCgInt(Emitter *e, int64_t value, JaiSpan span) {
    if (value >= INT16_MIN && value <= INT16_MAX) {
        emitOp(e, OP_INT, span);
        emitI16(e, (int16_t)value, span);
        return;
    }
    emitOp(e, OP_CONST, span);
    emitU24(e, addConst(e, INT_VAL(value)), span);
}

void jaiCgConstValue(Emitter *e, Value v, JaiSpan span) {
    emitOp(e, OP_CONST, span);
    emitU24(e, addConst(e, v), span);
}

/* The compile-time value of a literal node, for MATCH_CONST / MATCH_RANGE
 * operands. Returns false for anything the constant folder left unfolded. */
bool jaiCgLiteralValue(AstNode *node, Value *out) {
    if (node == NULL) return false;
    switch (node->kind) {
    case AST_INT_LIT:   *out = INT_VAL(node->as.intLit); return true;
    case AST_FLOAT_LIT: *out = FLOAT_VAL(node->as.floatLit); return true;
    case AST_BOOL_LIT:  *out = BOOL_VAL(node->as.boolLit); return true;
    case AST_NULL_LIT:  *out = NULL_VAL; return true;
    case AST_STR_LIT:
        *out = OBJ_VAL(jaiStringIntern(node->as.strLit.chars,
                                       node->as.strLit.length));
        return true;
    case AST_UNARY:
        if (node->as.unary.op == OPK_NEG) {
            Value inner;
            if (!jaiCgLiteralValue(node->as.unary.operand, &inner)) return false;
            /* -INT64_MIN is not an int; negating it in C is undefined, so the
             * pattern is reported as non-constant instead of folded. */
            if (IS_INT(inner) && AS_INT(inner) == INT64_MIN) return false;
            if (IS_INT(inner))   { *out = INT_VAL(-AS_INT(inner)); return true; }
            if (IS_FLOAT(inner)) { *out = FLOAT_VAL(-AS_FLOAT(inner)); return true; }
        }
        return false;
    default:
        return false;
    }
}

/* Types                                                                */
/* ------------------------------------------------------------------ */

/* Type operands (MATCH_TYPE, TYPE_GUARD, IS_INSTANCE, exception-table entries)
 * are interned type *names*: the VM resolves them against the module's globals.
 * A serialised TYPE descriptor (pool tag 9) would need a Value representation
 * that object.h does not define. */
const char *jaiCgTypeNameOf(const AstType *type) {
    if (type == NULL) return "any";
    switch (type->kind) {
    case TYPE_NAME:
    case TYPE_GENERIC:
        return type->name != NULL ? type->name : "any";
    case TYPE_OPTIONAL:
        return jaiCgTypeNameOf(type->inner);
    default:
        if (type->resolved != NULL && type->resolved->name != NULL) {
            return type->resolved->name;
        }
        return "any";
    }
}

/* The name a *checked* type reduces to at run time. Type arguments are erased
 * (spec §6.1), so a guard can only test the shape a value actually carries:
 * `list[int]` is a list, `fn(int) -> int` is a fn, and a generic parameter
 * stands for nothing the VM can test, so it guards as `any`. Returns NULL for
 * a union, which needs one name per member and so cannot be a single string. */
static const char *runtimeTypeName(const JaiType *t) {
    for (int hops = 0; t != NULL && t->kind == TY_ALIAS && hops < 16; hops++) {
        t = t->ret;
    }
    if (t == NULL) return "any";
    switch (t->kind) {
    case TY_ANY:
    case TY_NEVER:
    case TY_GENERIC_PARAM:
    case TY_ALIAS:                 /* an unresolved alias cycle */
        return "any";
    case TY_NULL:   return "null";
    case TY_BOOL:   return "bool";
    case TY_INT:    return "int";
    case TY_FLOAT:  return "float";
    case TY_STR:    return "str";
    case TY_BYTES:  return "bytes";
    case TY_LIST:   return "list";
    case TY_DICT:   return "dict";
    case TY_SET:    return "set";
    case TY_TUPLE:  return "tuple";
    case TY_RANGE:  return "range";
    case TY_FN:     return "fn";
    case TY_MODULE: return "module";
    case TY_CLASS:
    case TY_TRAIT:
    case TY_ENUM: {
        /* Not `t->name`: an imported declaration's type is interned under its
         * qualified name so that two modules' `Node` stay two types, and the
         * VM resolves this operand against the module's globals, where the
         * binding is the bare name (or the alias it was imported under). */
        const char *bound = jaiTypeDeclBindingName(t);
        return bound != NULL ? bound : "any";
    }
    case TY_UNION:
        return NULL;
    }
    return "any";
}

/* A union wider than this guards as `any`: dropping members would turn the
 * check into a wrong rejection, and no hand-written union comes near it. */
#define MAX_UNION_GUARD 32

/* The constant OP_TYPE_GUARD reads. A union guards against each member in
 * turn, which the VM expects as a tuple of names — and every `T?` is such a
 * union, so without this an optional guard would reject `null`. */
uint32_t jaiCgTypeGuardConst(Emitter *e, const AstType *target) {
    const JaiType *t = target != NULL ? target->resolved : NULL;
    const char *name = t != NULL ? runtimeTypeName(t) : jaiCgTypeNameOf(target);
    if (name != NULL) return nameConst(e, name);
    if (t->argCount > MAX_UNION_GUARD) return nameConst(e, "any");

    Value members[MAX_UNION_GUARD];
    int count = 0;
    for (int i = 0; i < t->argCount; i++) {
        /* Unions are flattened, so a member is never itself a union. */
        const char *member = runtimeTypeName(t->args[i]);
        if (member == NULL) member = "any";
        members[count++] = OBJ_VAL(jaiStringIntern(member, strlen(member)));
        jaiGCPushRoot(members[count - 1]);
    }
    ObjTuple *tuple = jaiTupleNew(members, count);
    for (int i = 0; i < count; i++) jaiGCPopRoot();
    return addConst(e, OBJ_VAL(tuple));
}

/* Verification and optimisation                                        */
/* ------------------------------------------------------------------ */

void jaiCgFinishFunction(Emitter *e, ObjFunction *fn) {
    jaiOptimize(fn, e->opts);
#ifdef JAI_DEBUG
    char error[256];
    if (!jaiVerifyChunk(fn, error, sizeof error)) {
        CG_ERROR(e, fileSpan(e), E0902_INTERNAL_ERROR,
                 "generated bytecode for `%s` is malformed: %s",
                 fn->name != NULL ? fn->name->chars : "<anonymous>", error);
    }
#endif
}

/* Public entry points                                                  */
/* ------------------------------------------------------------------ */

CodegenOptions jaiCodegenDefaults(void) {
    CodegenOptions opts;
    opts.optLevel = 2;
    opts.debugInfo = true;
    opts.stripAsserts = false;
    opts.emitTailCalls = true;
    return opts;
}

static int sourceFileFor(const AstNode *node, const ObjModule *module) {
    if (node != NULL && node->span.file >= 0) return node->span.file;
    if (module != NULL) return module->sourceFileId;
    return -1;
}

ObjFunction *jaiCompileProgram(AstNode *program, ObjModule *module,
                               const CodegenOptions *opts) {
    if (program == NULL) return NULL;
    CodegenOptions defaults = jaiCodegenDefaults();
    if (opts == NULL) opts = &defaults;

    Emitter e;
    memset(&e, 0, sizeof e);
    e.opts = opts;
    e.module = module;
    e.fileId = sourceFileFor(program, module);

    ObjFunction *fn = jaiFunctionNew();
    jaiGCPushRoot(OBJ_VAL(fn));

    fn->module = module;
    jaiChunkInit(&fn->chunk, e.fileId);
    const char *moduleName = (module != NULL && module->name != NULL)
                                 ? module->name->chars
                                 : "__main__";
    fn->name = jaiStringIntern(moduleName, strlen(moduleName));
    fn->qualifiedName = fn->name;

    /* AST_PROGRAM's opaque scope pointer starts with the module body's
     * FunctionScope (see the comment on `struct Scope` in resolve.c). */
    FunctionScope *scope = NULL;
    if (program->as.block.scope != NULL) {
        scope = *(FunctionScope **)program->as.block.scope;
    }

    FnCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.closeBase = -1;       /* memset would read as "close at slot 0" */
    ctx.deepCloseBase = -1;
    ctx.fn = fn;
    ctx.scope = scope;
    ctx.decl = program;
    ctx.nextTemp = scope != NULL ? scope->localCount : 0;
    if (ctx.nextTemp < 1) ctx.nextTemp = 1;   /* slot 0 is the frame's callee */
    ctx.maxSlot = ctx.nextTemp;
    JAI_VEC_INIT(&ctx.exceptions);
    e.fn = &ctx;

    jaiEmitStmt(&e, program);
    if (ctx.hasDefer) emitOp(&e, OP_RUN_DEFERS, program->span);
    emitOp(&e, OP_RETURN_NULL, program->span);

    int slots = ctx.maxSlot + ctx.maxStackDepth;
    fn->maxSlots = (uint16_t)(slots > UINT16_MAX ? UINT16_MAX : slots);
    fn->upvalueCount = (uint16_t)(scope != NULL ? scope->upvalueCount : 0);

    if (ctx.exceptions.count > 0) {
        fn->exceptionCount = (uint16_t)ctx.exceptions.count;
        fn->exceptions = JAI_ALLOC(ExceptionEntry, ctx.exceptions.count);
        memcpy(fn->exceptions, ctx.exceptions.data,
               sizeof(ExceptionEntry) * (size_t)ctx.exceptions.count);
    }
    JAI_VEC_FREE(ExceptionEntry, &ctx.exceptions);
    e.fn = NULL;

    jaiCgFinishFunction(&e, fn);

    jaiGCPopRoot();
    if (e.errors > 0) return NULL;
    return fn;
}

ObjFunction *jaiCompileFunction(AstNode *fnDecl, ObjModule *module,
                                const CodegenOptions *opts) {
    if (fnDecl == NULL) return NULL;
    if (fnDecl->kind != AST_FN_DECL && fnDecl->kind != AST_ANON_FN &&
        fnDecl->kind != AST_LAMBDA) {
        jaiDiagError(E0902_INTERNAL_ERROR, fnDecl->span,
                     "jaiCompileFunction expects a function declaration, got `%s`",
                     jaiAstKindName(fnDecl->kind));
        return NULL;
    }

    CodegenOptions defaults = jaiCodegenDefaults();
    if (opts == NULL) opts = &defaults;

    Emitter e;
    memset(&e, 0, sizeof e);
    e.opts = opts;
    e.module = module;
    e.fileId = sourceFileFor(fnDecl, module);

    uint32_t flags = 0;
    if (fnDecl->as.fn.isGenerator) flags |= FN_GENERATOR;
    if (fnDecl->as.fn.isAsync)     flags |= FN_ASYNC;

    ObjFunction *fn = jaiEmitFunctionNode(&e, fnDecl, flags, NULL, NULL, 0);
    jaiCgFinishFunction(&e, fn);
    jaiGCPopRoot();   /* jaiEmitFunctionNode left it rooted */

    if (e.errors > 0) return NULL;
    return fn;
}
