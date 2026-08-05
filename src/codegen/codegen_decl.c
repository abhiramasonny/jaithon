/* codegen_decl.c — declarations: functions, classes, traits, enums
 * (spec §6, §7).
 *
 * A declaration is the one place lowering is not a walk of the node it is
 * given: a class body is emitted as a sequence of runtime construction
 * instructions against the class object, and a function is compiled into its
 * own chunk in a nested FnCtx before the enclosing code ever sees it.
 */
#include "codegen_internal.h"

/* Classes, traits, enums                                               */
/* ------------------------------------------------------------------ */

/* Kinds of the class-spec constant (spec/BYTECODE.md §6, pool tag 7). */
#define SPEC_CLASS 0
#define SPEC_TRAIT 1
#define SPEC_ENUM  2

/* The spec constant is a 6-tuple:
 *     (INT kind, STR name, STR|NULL superName, BOOL isAbstract,
 *      TUPLE required, TUPLE variants)
 * `required` holds a trait's (name, INT arity) pairs flattened; `variants`
 * holds an enum's (name, TUPLE fieldNames) pairs flattened; NULL_VAL for either
 * means "empty". Everything else a class needs arrives through
 * FIELD_DEF / METHOD / IMPL_TRAIT.
 *
 * Each object is rooted the instant it exists, because the next intern or tuple
 * allocation in this function can collect. */
static uint32_t classSpecConstant(Emitter *e, int kind, const char *name,
                                  const char *superName, bool isAbstract,
                                  Value required, Value variants) {
    int rooted = 0;
    jaiGCPushRoot(required);
    rooted++;
    jaiGCPushRoot(variants);
    rooted++;

    Value items[6];
    items[0] = INT_VAL(kind);
    items[1] = OBJ_VAL(jaiStringIntern(name != NULL ? name : "",
                                       name != NULL ? strlen(name) : 0));
    jaiGCPushRoot(items[1]);
    rooted++;

    items[2] = NULL_VAL;
    if (superName != NULL) {
        items[2] = OBJ_VAL(jaiStringIntern(superName, strlen(superName)));
        jaiGCPushRoot(items[2]);
        rooted++;
    }
    items[3] = BOOL_VAL(isAbstract);

    items[4] = required;
    if (IS_NULL(items[4])) {
        items[4] = OBJ_VAL(jaiTupleNew(NULL, 0));
        jaiGCPushRoot(items[4]);
        rooted++;
    }
    items[5] = variants;
    if (IS_NULL(items[5])) {
        items[5] = OBJ_VAL(jaiTupleNew(NULL, 0));
        jaiGCPushRoot(items[5]);
        rooted++;
    }

    ObjTuple *tuple = jaiTupleNew(items, 6);
    uint32_t k = addConst(e, OBJ_VAL(tuple));
    jaiGCPopRoots(rooted);
    return k;
}

static uint32_t methodFlags(const AstNode *fn, bool isGetter, bool isSetter) {
    uint32_t flags = FN_METHOD;
    if (fn->as.fn.isStatic)    flags |= FN_STATIC;
    if (fn->as.fn.isGenerator) flags |= FN_GENERATOR;
    if (fn->as.fn.isAsync)     flags |= FN_ASYNC;
    if (isGetter)              flags |= FN_GETTER;
    if (isSetter)              flags |= FN_SETTER;
    if (fn->as.fn.name != NULL && strcmp(fn->as.fn.name, "init") == 0) {
        flags |= FN_INIT;
    }
    return flags;
}

/* The runtime spelling of a declared visibility. The two enums are separate
 * because the AST must not depend on the object model. */
static uint8_t runtimeVisibility(AstVisibility vis) {
    switch (vis) {
    case AST_VIS_PRIVATE:   return (uint8_t)VIS_PRIVATE;
    case AST_VIS_PROTECTED: return (uint8_t)VIS_PROTECTED;
    case AST_VIS_PUBLIC:    return (uint8_t)VIS_PUBLIC;
    }
    return (uint8_t)VIS_PRIVATE;
}

/* Attaches one method to the class sitting on top of the stack. */
static void emitMethod(Emitter *e, AstNode *fn, const char *ownerName,
                       bool isGetter, bool isSetter,
                       const AstField *initFields, int initFieldCount) {
    uint32_t flags = methodFlags(fn, isGetter, isSetter);
    uint32_t k = jaiEmitFunctionConstant(e, fn, flags, ownerName, initFields,
                                         initFieldCount);
    jaiEmitClosure(e, fn, k, fn->span);
    emitOp(e, OP_METHOD, fn->span);
    emitU24(e, nameConst(e, fn->as.fn.name), fn->span);
    /* FN_INIT is bit 8 and does not survive the u8 operand; the VM recovers it
     * from the method name. See the header-change note in the report. */
    emitByte(e, (uint8_t)(flags & 0xffu), fn->span);
    emitByte(e, runtimeVisibility(fn->as.fn.visibility), fn->span);
}

void jaiEmitClassDecl(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    const char *name = node->as.classDecl.name;
    const AstType *super = node->as.classDecl.superclass;
    const char *superName = super != NULL ? jaiCgTypeNameOf(super) : NULL;

    if (superName != NULL) jaiCgGetGlobal(e, superName, span);

    uint32_t spec = classSpecConstant(e, SPEC_CLASS, name, superName,
                                      node->as.classDecl.isAbstract,
                                      NULL_VAL, NULL_VAL);
    emitOp(e, OP_CLASS, span);
    emitU24(e, spec, span);

    if (superName != NULL) {
        emitOp(e, OP_INHERIT, span);   /* peek(1) = super, peek(0) = subclass */
        emitOp(e, OP_SWAP, span);
        emitPop(e, span);
    }

    for (int i = 0; i < node->as.classDecl.traitCount; i++) {
        emitOp(e, OP_IMPL_TRAIT, span);
        emitU24(e, nameConst(e, jaiCgTypeNameOf(node->as.classDecl.traits[i])), span);
    }

    for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
        const AstField *field = &node->as.classDecl.fields[i];
        /* bits 0-1 visibility, bit 2 static, bit 3 immutable */
        uint8_t flags = (uint8_t)(field->visibility & 0x3);
        if (field->isStatic) flags |= 1u << 2;
        if (field->isLet)    flags |= 1u << 3;
        emitOp(e, OP_FIELD_DEF, field->span);
        emitU24(e, nameConst(e, field->name), field->span);
        emitByte(e, flags, field->span);
    }

    /* A static field belongs to the class, so its initialiser runs once, here,
     * with the class still on the stack. Instance fields are the opposite case
     * and are handled by the constructor below. */
    for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
        const AstField *field = &node->as.classDecl.fields[i];
        if (!field->isStatic || field->defaultValue == NULL) continue;
        emitOp(e, OP_DUP, field->span);
        jaiEmitExpr(e, field->defaultValue);
        uint16_t cache = newCache(e);
        emitOp(e, OP_SET_FIELD, field->span);
        emitU24(e, nameConst(e, field->name), field->span);
        emitU16(e, cache, field->span);
    }

    /* Field defaults belong to the constructor: they are re-evaluated per
     * instance, which is why they cannot live in the class spec. */
    AstNode *init = NULL;
    for (int i = 0; i < node->as.classDecl.methodCount; i++) {
        AstNode *method = node->as.classDecl.methods[i];
        if (method != NULL && method->as.fn.name != NULL &&
            strcmp(method->as.fn.name, "init") == 0) {
            init = method;
            break;
        }
    }

    bool needsSynthesizedInit = init == NULL;
    if (needsSynthesizedInit) {
        needsSynthesizedInit = false;
        for (int i = 0; i < node->as.classDecl.fieldCount; i++) {
            if (node->as.classDecl.fields[i].defaultValue != NULL &&
                !node->as.classDecl.fields[i].isStatic) {
                needsSynthesizedInit = true;
                break;
            }
        }
    }

    for (int i = 0; i < node->as.classDecl.methodCount; i++) {
        AstNode *method = node->as.classDecl.methods[i];
        if (method == NULL) continue;
        bool isInit = method == init;
        emitMethod(e, method, name, false, false,
                   isInit ? node->as.classDecl.fields : NULL,
                   isInit ? node->as.classDecl.fieldCount : 0);
    }
    for (int i = 0; i < node->as.classDecl.getterCount; i++) {
        if (node->as.classDecl.getters[i] == NULL) continue;
        emitMethod(e, node->as.classDecl.getters[i], name, true, false, NULL, 0);
    }
    for (int i = 0; i < node->as.classDecl.setterCount; i++) {
        if (node->as.classDecl.setters[i] == NULL) continue;
        emitMethod(e, node->as.classDecl.setters[i], name, false, true, NULL, 0);
    }

    if (needsSynthesizedInit) {
        /* No declared init but fields have defaults: emit a bare one whose only
         * body is the default prologue. */
        AstNode synthetic;
        memset(&synthetic, 0, sizeof synthetic);
        synthetic.kind = AST_FN_DECL;
        synthetic.span = span;
        synthetic.as.fn.name = "init";
        synthetic.as.fn.visibility = AST_VIS_PUBLIC;

        uint32_t flags = FN_METHOD | FN_INIT;
        uint32_t k = jaiEmitFunctionConstant(e, &synthetic, flags, name,
                                             node->as.classDecl.fields,
                                             node->as.classDecl.fieldCount);
        jaiEmitClosure(e, &synthetic, k, span);
        emitOp(e, OP_METHOD, span);
        emitU24(e, nameConst(e, "init"), span);
        emitByte(e, (uint8_t)(flags & 0xffu), span);
        /* The visibility byte is not optional: chunk.c declares OP_METHOD five
         * operand bytes wide, so omitting it here left the VM reading the NEXT
         * opcode as this method's visibility and then decoding the rest of the
         * stream one byte out of phase. A class with no explicit `init` did not
         * define its own name. A synthesised initialiser is always public, the
         * same as the implicit one a user would write. */
        emitByte(e, runtimeVisibility(AST_VIS_PUBLIC), span);
    }

    jaiCgStoreSymbol(e, node->as.classDecl.symbol, name, span, true);
}

void jaiEmitTraitDecl(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    const char *name = node->as.traitDecl.name;

    /* Required methods carry no body, so they cannot be METHODs: they travel in
     * the spec constant as (name, arity) pairs. */
    int requiredCount = 0;
    for (int i = 0; i < node->as.traitDecl.methodCount; i++) {
        AstNode *m = node->as.traitDecl.methods[i];
        if (m != NULL && m->as.fn.body == NULL) requiredCount++;
    }

    Value *pairs = NULL;
    int rooted = 0;
    if (requiredCount > 0) {
        pairs = JAI_ALLOC(Value, requiredCount * 2);
        int n = 0;
        for (int i = 0; i < node->as.traitDecl.methodCount; i++) {
            AstNode *m = node->as.traitDecl.methods[i];
            if (m == NULL || m->as.fn.body != NULL) continue;
            const char *mname = m->as.fn.name != NULL ? m->as.fn.name : "";
            pairs[n] = OBJ_VAL(jaiStringIntern(mname, strlen(mname)));
            jaiGCPushRoot(pairs[n]);
            rooted++;
            n++;
            pairs[n++] = INT_VAL(m->as.fn.paramCount);
        }
    }
    /* The temp-root stack is LIFO: drop the element roots (the tuple now owns
     * the names) and take one for the tuple before anything else allocates. */
    Value required = OBJ_VAL(jaiTupleNew(pairs, requiredCount * 2));
    jaiGCPopRoots(rooted);
    jaiGCPushRoot(required);
    if (pairs != NULL) JAI_FREE_ARRAY(Value, pairs, requiredCount * 2);

    uint32_t spec = classSpecConstant(e, SPEC_TRAIT, name, NULL, false, required,
                                      NULL_VAL);
    jaiGCPopRoot();

    emitOp(e, OP_CLASS, span);
    emitU24(e, spec, span);

    for (int i = 0; i < node->as.traitDecl.superCount; i++) {
        emitOp(e, OP_IMPL_TRAIT, span);
        emitU24(e, nameConst(e, jaiCgTypeNameOf(node->as.traitDecl.supers[i])), span);
    }

    for (int i = 0; i < node->as.traitDecl.methodCount; i++) {
        AstNode *m = node->as.traitDecl.methods[i];
        if (m == NULL || m->as.fn.body == NULL) continue;   /* default impls only */
        emitMethod(e, m, name, false, false, NULL, 0);
    }

    jaiCgStoreSymbol(e, node->as.traitDecl.symbol, name, span, true);
}

void jaiEmitEnumDecl(Emitter *e, AstNode *node) {
    JaiSpan span = node->span;
    const char *name = node->as.enumDecl.name;
    int count = node->as.enumDecl.variantCount;

    Value *entries = NULL;
    int rooted = 0;
    if (count > 0) entries = JAI_ALLOC(Value, count * 2);

    for (int i = 0; i < count; i++) {
        const AstVariant *variant = &node->as.enumDecl.variants[i];
        const char *vname = variant->name != NULL ? variant->name : "";
        entries[i * 2] = OBJ_VAL(jaiStringIntern(vname, strlen(vname)));
        jaiGCPushRoot(entries[i * 2]);
        rooted++;

        int fieldCount = variant->paramCount;
        Value *fields = fieldCount > 0 ? JAI_ALLOC(Value, fieldCount) : NULL;
        int fieldRoots = 0;
        for (int p = 0; p < fieldCount; p++) {
            const char *pname = variant->params[p].name != NULL
                                    ? variant->params[p].name
                                    : "";
            fields[p] = OBJ_VAL(jaiStringIntern(pname, strlen(pname)));
            jaiGCPushRoot(fields[p]);
            fieldRoots++;
        }
        entries[i * 2 + 1] = OBJ_VAL(jaiTupleNew(fields, fieldCount));
        jaiGCPopRoots(fieldRoots);
        jaiGCPushRoot(entries[i * 2 + 1]);
        rooted++;
        if (fields != NULL) JAI_FREE_ARRAY(Value, fields, fieldCount);
    }

    Value variants = OBJ_VAL(jaiTupleNew(entries, count * 2));
    jaiGCPopRoots(rooted);
    jaiGCPushRoot(variants);
    if (entries != NULL) JAI_FREE_ARRAY(Value, entries, count * 2);

    uint32_t spec = classSpecConstant(e, SPEC_ENUM, name, NULL, false, NULL_VAL,
                                      variants);
    jaiGCPopRoot();

    emitOp(e, OP_CLASS, span);
    emitU24(e, spec, span);

    for (int i = 0; i < node->as.enumDecl.methodCount; i++) {
        AstNode *m = node->as.enumDecl.methods[i];
        if (m == NULL) continue;
        emitMethod(e, m, name, false, false, NULL, 0);
    }

    jaiCgStoreSymbol(e, node->as.enumDecl.symbol, name, span, true);
}

/* Functions                                                            */
/* ------------------------------------------------------------------ */

/* True when params[0] is the explicit `self` of an instance method: it occupies
 * frame slot 0 rather than being a caller-supplied argument. */
static bool hasExplicitSelf(const AstNode *node, uint32_t flags) {
    if ((flags & FN_METHOD) == 0) return false;
    if ((flags & FN_STATIC) != 0) return false;
    if (node->as.fn.paramCount == 0) return false;
    const char *first = node->as.fn.params[0].name;
    return first != NULL && strcmp(first, "self") == 0;
}

static void setFunctionNames(Emitter *e, ObjFunction *fn, const char *name,
                             const char *ownerName) {
    const char *plain = name != NULL ? name : "<anonymous>";
    fn->name = jaiStringIntern(plain, strlen(plain));

    char buffer[512];
    const char *moduleName = (e->module != NULL && e->module->name != NULL)
                                 ? e->module->name->chars
                                 : NULL;
    int written;
    if (moduleName != NULL && ownerName != NULL) {
        written = snprintf(buffer, sizeof buffer, "%s.%s.%s", moduleName,
                           ownerName, plain);
    } else if (ownerName != NULL) {
        written = snprintf(buffer, sizeof buffer, "%s.%s", ownerName, plain);
    } else if (moduleName != NULL) {
        written = snprintf(buffer, sizeof buffer, "%s.%s", moduleName, plain);
    } else {
        written = snprintf(buffer, sizeof buffer, "%s", plain);
    }
    if (written < 0) written = 0;
    size_t length = (size_t)written < sizeof buffer ? (size_t)written
                                                    : sizeof buffer - 1;
    fn->qualifiedName = jaiStringIntern(buffer, length);
}

/* Records arity, flags and parameter names, and returns the number of trailing
 * parameters that carry a default. */
static int setupParameters(Emitter *e, ObjFunction *fn, AstNode *node,
                           bool selfIsSlot0) {
    int count = node->as.fn.paramCount;
    int first = selfIsSlot0 ? 1 : 0;
    int visible = count - first;
    if (visible < 0) visible = 0;

    if (visible > 0) {
        /* Zero before publishing the count: the array is a GC root the moment
         * paramCount is nonzero, and interning below can collect. */
        ObjString **names = JAI_ALLOC(ObjString *, visible);
        memset(names, 0, sizeof(ObjString *) * (size_t)visible);
        fn->paramNames = names;
        fn->paramCount = (uint16_t)visible;
        for (int i = 0; i < visible; i++) {
            const char *pname = node->as.fn.params[first + i].name;
            if (pname == NULL) pname = "_";
            names[i] = jaiStringIntern(pname, strlen(pname));
        }
    }

    int required = 0, defaults = 0;
    for (int i = first; i < count; i++) {
        const AstParam *param = &node->as.fn.params[i];
        if (param->isVariadic) {
            fn->flags |= FN_VARIADIC;
            continue;
        }
        if (param->isKwRest) {
            fn->flags |= FN_KWREST;
            continue;
        }
        if (param->defaultValue != NULL) defaults++;
        else required++;
    }

    if (required + defaults > 255) {
        CG_ERROR(e, node->span, E0600_ARITY_MISMATCH,
                 "a function may declare at most 255 positional parameters");
        if (defaults > 255) defaults = 255;
        required = 255 - defaults;
    }
    /* `arity` is every declared positional parameter, defaults included: the
     * frame layout puts them at slots[1..arity] and the variadic tail at
     * slots[arity+1], and the VM recovers the required count as
     * arity - defaultCount. */
    fn->arity = (uint8_t)(required + defaults);
    fn->defaultCount = (uint8_t)defaults;
    return defaults;
}

/* Default-value thunks (spec §5, `defaultOffsets`).
 *
 * Layout: offset 0 is `JUMP body` when there is at least one default, then one
 * self-contained thunk per defaulted parameter in declaration order, each
 * leaving its value on the stack and ending in OP_RETURN.
 *
 * They cannot fall through into one another: a keyword call may leave a hole
 * anywhere in the parameter list (`f(a, c: 3)` defaults only `b`), so the VM
 * runs each thunk on its own, in its own frame, and stores the result into the
 * caller-owned slot itself. Evaluating on every call is what keeps a mutable
 * default from being shared (spec §6). */
static void emitDefaultThunks(Emitter *e, ObjFunction *fn, AstNode *node,
                              int first, int defaults) {
    if (defaults == 0) return;

    JumpSite toBody = emitJump(e, OP_JUMP, node->span);
    fn->defaultOffsets = JAI_ALLOC(uint32_t, defaults);

    int index = 0;
    for (int i = first; i < node->as.fn.paramCount && index < defaults; i++) {
        const AstParam *param = &node->as.fn.params[i];
        if (param->isVariadic || param->isKwRest || param->defaultValue == NULL) {
            continue;
        }
        fn->defaultOffsets[index++] = (uint32_t)here(e);
        int depth = depthOf(e);
        jaiEmitExpr(e, param->defaultValue);
        emitOp(e, OP_RETURN, param->span);
        /* The thunk is entered directly, never fallen into, so its stack
         * effect is not part of the body's. */
        setDepth(e, depth);
    }
    patchJump(e, toBody);
}

/* `self.field = <default>` for every instance field that declares one; runs
 * before the body of `init` so an explicit assignment can override it. */
static void emitFieldDefaults(Emitter *e, const AstField *fields, int count) {
    for (int i = 0; i < count; i++) {
        const AstField *field = &fields[i];
        if (field->defaultValue == NULL || field->isStatic) continue;
        emitGetLocal(e, 0, field->span);
        jaiEmitExpr(e, field->defaultValue);
        uint32_t k = nameConst(e, field->name);
        uint16_t cache = newCache(e);
        emitOp(e, OP_SET_FIELD, field->span);
        emitU24(e, k, field->span);
        emitU16(e, cache, field->span);
    }
}

/* Leaves the finished function rooted; the caller pops once it is anchored in
 * a constant pool. */
ObjFunction *jaiEmitFunctionNode(Emitter *e, AstNode *node,
                                 uint32_t extraFlags,
                                 const char *ownerName,
                                 const AstField *initFields,
                                 int initFieldCount) {
    JaiSpan span = node->span;

    ObjFunction *fn = jaiFunctionNew();
    jaiGCPushRoot(OBJ_VAL(fn));

    fn->module = e->module;
    fn->flags = extraFlags;
    if (node->as.fn.isGenerator) fn->flags |= FN_GENERATOR;
    if (node->as.fn.isAsync)     fn->flags |= FN_ASYNC;
    if (node->as.fn.isStatic)    fn->flags |= FN_STATIC;
    jaiChunkInit(&fn->chunk, e->fileId);
    setFunctionNames(e, fn, node->as.fn.name, ownerName);

    bool selfIsSlot0 = hasExplicitSelf(node, fn->flags);
    int defaults = setupParameters(e, fn, node, selfIsSlot0);

    FunctionScope *scope = (FunctionScope *)node->as.fn.resolveInfo;
    int localCount = scope != NULL ? scope->localCount : node->as.fn.localCount;
    if (localCount < node->as.fn.paramCount + 1) {
        localCount = node->as.fn.paramCount + 1;
    }

    FnCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.closeBase = -1;       /* memset would read as "close at slot 0" */
    ctx.deepCloseBase = -1;
    ctx.enclosing = e->fn;
    ctx.fn = fn;
    ctx.scope = scope;
    ctx.decl = node;
    ctx.nextTemp = localCount;
    ctx.maxSlot = localCount;
    ctx.isGenerator = (fn->flags & FN_GENERATOR) != 0;
    JAI_VEC_INIT(&ctx.exceptions);
    e->fn = &ctx;

    emitDefaultThunks(e, fn, node, selfIsSlot0 ? 1 : 0, defaults);
    if (initFields != NULL) emitFieldDefaults(e, initFields, initFieldCount);

    if (node->as.fn.body == NULL) {
        emitOp(e, OP_RETURN_NULL, span);
    } else if (node->as.fn.isExprBody) {
        jaiEmitExpr(e, node->as.fn.body);
        if (ctx.hasDefer) emitOp(e, OP_RUN_DEFERS, span);
        emitOp(e, OP_RETURN, span);
    } else {
        jaiEmitStmt(e, node->as.fn.body);
        if (ctx.hasDefer) emitOp(e, OP_RUN_DEFERS, span);
        emitOp(e, OP_RETURN_NULL, span);
    }

    int slots = ctx.maxSlot + ctx.maxStackDepth;
    if (slots > UINT16_MAX) {
        CG_ERROR(e, span, E0208_TOO_MANY_LOCALS,
                 "frame window of %d slots exceeds the %d-slot limit", slots,
                 UINT16_MAX);
        slots = UINT16_MAX;
    }
    fn->maxSlots = (uint16_t)slots;
    fn->upvalueCount = (uint16_t)(scope != NULL ? scope->upvalueCount
                                                : node->as.fn.upvalueCount);

    if (ctx.exceptions.count > 0) {
        fn->exceptionCount = (uint16_t)ctx.exceptions.count;
        fn->exceptions = JAI_ALLOC(ExceptionEntry, ctx.exceptions.count);
        memcpy(fn->exceptions, ctx.exceptions.data,
               sizeof(ExceptionEntry) * (size_t)ctx.exceptions.count);
    }
    JAI_VEC_FREE(ExceptionEntry, &ctx.exceptions);

    e->fn = ctx.enclosing;
    return fn;
}
