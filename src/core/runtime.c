#include "runtime.h"
#include <stdarg.h>

Runtime runtime;

Value makeNumber(double n) {
    Value v;
    v.type = VAL_NUMBER;
    v.as.number = n;
    return v;
}

Value makeString(const char* s) {
    Value v;
    v.type = VAL_STRING;
    v.as.string = strdup(s);
    return v;
}

Value makeBool(bool b) {
    Value v;
    v.type = VAL_BOOL;
    v.as.boolean = b;
    return v;
}

Value makeNull(void) {
    Value v;
    v.type = VAL_NULL;
    return v;
}

Value makeFunction(JaiFunction* f) {
    Value v;
    v.type = VAL_FUNCTION;
    v.as.function = f;
    return v;
}

Value makeNativeFunc(NativeFunc f) {
    Value v;
    v.type = VAL_NATIVE_FUNC;
    v.as.nativeFunc = f;
    return v;
}

//cons cell (pair)
Value makeCell(void) {
    Value v;
    v.type = VAL_CELL;
    v.as.cell = malloc(sizeof(JaiCell));
    v.as.cell->car = malloc(sizeof(Value));
    v.as.cell->cdr = malloc(sizeof(Value));
    *v.as.cell->car = makeNull(); //car and cdr initialized to null
    *v.as.cell->cdr = makeNull();
    return v;
}

Value makeArray(int initialCapacity) {
    Value v;
    v.type = VAL_ARRAY;
    v.as.array = malloc(sizeof(JaiArray));
    v.as.array->capacity = initialCapacity > 0 ? initialCapacity : INITIAL_CAPACITY;
    v.as.array->items = malloc(sizeof(Value) * v.as.array->capacity);
    v.as.array->length = 0;
    return v;
}

Value makeObject(JaiClass* class) {
    Value v;
    v.type = VAL_OBJECT;
    v.as.object = malloc(sizeof(JaiObject));
    v.as.object->class = class;
    v.as.object->fieldCapacity = class ? class->fieldCount : INITIAL_CAPACITY;
    v.as.object->fieldCount = 0;
    v.as.object->fields = malloc(sizeof(Value) * v.as.object->fieldCapacity);
    v.as.object->fieldNames = malloc(sizeof(char*) * v.as.object->fieldCapacity);
    
    if (class) { //default from class
        for (int i = 0; i < class->fieldCount; i++) {
            v.as.object->fieldNames[i] = strdup(class->fieldNames[i]);
            v.as.object->fields[i] = makeNull();
            v.as.object->fieldCount++;
        }
    }
    return v;
}

void initRuntime(void) {
    memset(&runtime, 0, sizeof(Runtime));
    
    runtime.moduleCapacity = INITIAL_CAPACITY;
    runtime.modules = malloc(sizeof(Module*) * runtime.moduleCapacity);
    runtime.moduleCount = 0;
    
    runtime.classCapacity = INITIAL_CAPACITY;
    runtime.classes = malloc(sizeof(JaiClass*) * runtime.classCapacity);
    runtime.classCount = 0;
    
    runtime.keywords.capacity = INITIAL_CAPACITY;
    runtime.keywords.entries = malloc(sizeof(KeywordEntry) * runtime.keywords.capacity);
    runtime.keywords.count = 0;
    runtime.keywords.nextTokenType = 100;
    
    runtime.eventBus.subCapacity = INITIAL_CAPACITY;
    runtime.eventBus.subscriptions = malloc(sizeof(Subscription) * runtime.eventBus.subCapacity);
    runtime.eventBus.subCount = 0;
    
    runtime.currentModule = NULL;
    runtime.debug = false;
    runtime.shellMode = false;
    runtime.lineNumber = 1;
    runtime.callStackSize = 0;
    
    Module* main = createModule("__main__", "");
    runtime.currentModule = main;
}

static void freeFunction(JaiFunction* f) {
    if (!f) return;
    if (f->body) free(f->body);
    for (int i = 0; i < f->paramCount; i++) {
        if (f->params[i]) free(f->params[i]);
    }
    if (f->params) free(f->params);
    free(f);
}

static void freeModule(Module* m) {
    if (!m) return;
    for (int j = 0; j < m->varCount; j++) {
        if (m->variables[j].value.type == VAL_STRING && m->variables[j].value.as.string) {
            free(m->variables[j].value.as.string);
        }
    }
    if (m->variables) free(m->variables);
    for (int j = 0; j < m->funcCount; j++) {
        freeFunction(m->functions[j]);
    }
    if (m->functions) free(m->functions);
    free(m);
}

void freeRuntime(void) {
    for (int i = 0; i < runtime.moduleCount; i++) {
        freeModule(runtime.modules[i]);
    }
    if (runtime.modules) free(runtime.modules);
    
    for (int i = 0; i < runtime.classCount; i++) {
        JaiClass* c = runtime.classes[i];
        if (c) {
            for (int j = 0; j < c->fieldCount; j++) {
                if (c->fieldNames[j]) free(c->fieldNames[j]);
            }
            if (c->fieldNames) free(c->fieldNames);
            for (int j = 0; j < c->methodCount; j++) {
                if (c->methodNames[j]) free(c->methodNames[j]);
            }
            if (c->methodNames) free(c->methodNames);
            if (c->methods) free(c->methods);
            free(c);
        }
    }
    if (runtime.classes) free(runtime.classes);
    
    if (runtime.keywords.entries) free(runtime.keywords.entries);
    
    for (int i = 0; i < runtime.eventBus.subCount; i++) {
        if (runtime.eventBus.subscriptions[i].handlers) {
            free(runtime.eventBus.subscriptions[i].handlers);
        }
    }
    if (runtime.eventBus.subscriptions) free(runtime.eventBus.subscriptions);
}

static Subscription* findSubscription(const char* eventName) {
    for (int i = 0; i < runtime.eventBus.subCount; i++) {
        if (strcmp(runtime.eventBus.subscriptions[i].eventName, eventName) == 0) {
            return &runtime.eventBus.subscriptions[i];
        }
    }
    return NULL;
}

void subscribe(const char* eventName, EventHandler handler) {
    Subscription* sub = findSubscription(eventName);
    if (!sub) {
        if (runtime.eventBus.subCount >= runtime.eventBus.subCapacity) {
            runtime.eventBus.subCapacity *= GROWTH_FACTOR;
            runtime.eventBus.subscriptions = realloc(runtime.eventBus.subscriptions, 
                sizeof(Subscription) * runtime.eventBus.subCapacity);
        }
        sub = &runtime.eventBus.subscriptions[runtime.eventBus.subCount++];
        strncpy(sub->eventName, eventName, MAX_NAME_LEN - 1);
        sub->handlerCount = 0;
        sub->handlerCapacity = INITIAL_CAPACITY;
        sub->handlers = malloc(sizeof(EventHandler) * sub->handlerCapacity);
    }
    if (sub->handlerCount >= sub->handlerCapacity) {
        sub->handlerCapacity *= GROWTH_FACTOR;
        sub->handlers = realloc(sub->handlers, sizeof(EventHandler) * sub->handlerCapacity);
    }
    sub->handlers[sub->handlerCount++] = handler;
}

void publish(Event* event) {
    Subscription* sub = findSubscription(event->name);
    if (sub) {
        for (int i = 0; i < sub->handlerCount && !event->handled; i++) {
            sub->handlers[i](event);
        }
    }
}

Event createEvent(EventType type, const char* name, void* data) {
    Event e;
    e.type = type;
    strncpy(e.name, name, MAX_NAME_LEN - 1);
    e.data = data;
    e.result = makeNull();
    e.handled = false;
    return e;
}

int registerKeyword(const char* keyword) {
    int existing = lookupKeyword(keyword);
    if (existing >= 0) return existing;
    
    if (runtime.keywords.count >= runtime.keywords.capacity) {
        runtime.keywords.capacity *= GROWTH_FACTOR;
        runtime.keywords.entries = realloc(runtime.keywords.entries, 
            sizeof(KeywordEntry) * runtime.keywords.capacity);
    }
    
    KeywordEntry* entry = &runtime.keywords.entries[runtime.keywords.count++];
    strncpy(entry->keyword, keyword, MAX_NAME_LEN - 1);
    entry->tokenType = runtime.keywords.nextTokenType++;
    return entry->tokenType;
}

int lookupKeyword(const char* word) {
    for (int i = 0; i < runtime.keywords.count; i++) {
        if (strcmp(runtime.keywords.entries[i].keyword, word) == 0) {
            return runtime.keywords.entries[i].tokenType;
        }
    }
    return -1;
}

Module* createModule(const char* name, const char* path) {
    if (runtime.moduleCount >= runtime.moduleCapacity) {
        runtime.moduleCapacity *= GROWTH_FACTOR;
        runtime.modules = realloc(runtime.modules, sizeof(Module*) * runtime.moduleCapacity);
    }
    
    Module* m = malloc(sizeof(Module));
    runtime.modules[runtime.moduleCount++] = m;
    strncpy(m->name, name, MAX_NAME_LEN - 1);
    strncpy(m->path, path, MAX_NAME_LEN - 1);
    m->varCapacity = INITIAL_CAPACITY;
    m->variables = malloc(sizeof(Variable) * m->varCapacity);
    m->varCount = 0;
    m->funcCapacity = INITIAL_CAPACITY;
    m->functions = malloc(sizeof(JaiFunction*) * m->funcCapacity);
    m->funcCount = 0;
    m->loaded = false;
    return m;
}

Module* findModule(const char* name) {
    for (int i = 0; i < runtime.moduleCount; i++) {
        if (strcmp(runtime.modules[i]->name, name) == 0) {
            return runtime.modules[i];
        }
    }
    return NULL;
}

void setVariable(const char* name, Value value) {
    Module* m = runtime.currentModule;
    if (!m) {
        runtimeError("No current module");
        return;
    }
    
    for (int i = 0; i < m->varCount; i++) {
        if (strcmp(m->variables[i].name, name) == 0) {
            m->variables[i].value = value;
            return;
        }
    }
    
    if (m->varCount >= m->varCapacity) {
        m->varCapacity *= GROWTH_FACTOR;
        m->variables = realloc(m->variables, sizeof(Variable) * m->varCapacity);
    }
    
    Variable* var = &m->variables[m->varCount++];
    strncpy(var->name, name, MAX_NAME_LEN - 1);
    var->value = value;
}

Value getVariable(const char* name) {
    Module* m = runtime.currentModule;
    if (!m) {
        runtimeError("No current module");
        return makeNull();
    }
    
    for (int i = 0; i < m->varCount; i++) {
        if (strcmp(m->variables[i].name, name) == 0) {
            return m->variables[i].value;
        }
    }
    
    if (runtime.moduleCount > 1) {
        Module* global = runtime.modules[0];
        for (int i = 0; i < global->varCount; i++) {
            if (strcmp(global->variables[i].name, name) == 0) {
                return global->variables[i].value;
            }
        }
    }
    
    runtimeError("Undefined variable: %s", name);
    return makeNull();
}

bool hasVariable(const char* name) {
    Module* m = runtime.currentModule;
    if (!m) return false;
    
    for (int i = 0; i < m->varCount; i++) {
        if (strcmp(m->variables[i].name, name) == 0) {
            return true;
        }
    }
    
    if (runtime.moduleCount > 1) {
        Module* global = runtime.modules[0];
        for (int i = 0; i < global->varCount; i++) {
            if (strcmp(global->variables[i].name, name) == 0) {
                return true;
            }
        }
    }
    
    return false;
}

JaiFunction* defineFunction(const char* name, char** params, int paramCount, bool isVariadic, const char* body) {
    Module* m = runtime.currentModule;
    if (!m) {
        runtimeError("No current module");
        return NULL;
    }
    
    if (m->funcCount >= m->funcCapacity) {
        m->funcCapacity *= GROWTH_FACTOR;
        m->functions = realloc(m->functions, sizeof(JaiFunction*) * m->funcCapacity);
    }
    
    JaiFunction* f = malloc(sizeof(JaiFunction));
    m->functions[m->funcCount++] = f;
    strncpy(f->name, name, MAX_NAME_LEN - 1);
    f->paramCount = paramCount;
    f->paramCapacity = paramCount > 0 ? paramCount : INITIAL_CAPACITY;
    f->params = malloc(sizeof(char*) * f->paramCapacity);
    f->isVariadic = isVariadic;
    for (int i = 0; i < paramCount; i++) {
        f->params[i] = strdup(params[i]);
    }
    f->body = strdup(body);
    f->module = m;
    
    setVariable(name, makeFunction(f));
    
    return f;
}

JaiFunction* findFunction(const char* name) {
    Module* m = runtime.currentModule;
    if (!m) return NULL;
    
    for (int i = 0; i < m->funcCount; i++) {
        if (strcmp(m->functions[i]->name, name) == 0) {
            return m->functions[i];
        }
    }
    return NULL;
}

void arrayPush(JaiArray* arr, Value val) {
    if (arr->length >= arr->capacity) {
        arr->capacity *= GROWTH_FACTOR;
        arr->items = realloc(arr->items, sizeof(Value) * arr->capacity);
    }
    arr->items[arr->length++] = val;
}

Value arrayGet(JaiArray* arr, int index) {
    if (index < 0 || index >= arr->length) {
        runtimeError("Array index out of bounds: %d", index);
        return makeNull();
    }
    return arr->items[index];
}

void arraySet(JaiArray* arr, int index, Value val) {
    if (index < 0 || index >= arr->length) {
        runtimeError("Array index out of bounds: %d", index);
        return;
    }
    if (arr->items[index].type == VAL_STRING && arr->items[index].as.string) {
        free(arr->items[index].as.string);
    }
    arr->items[index] = val;
}

Value arrayPop(JaiArray* arr) {
    if (arr->length == 0) {
        runtimeError("Cannot pop from empty array");
        return makeNull();
    }
    return arr->items[--arr->length];
}

int arrayLen(JaiArray* arr) {
    return arr->length;
}

JaiClass* defineClass(const char* name, JaiClass* parent) {
    if (runtime.classCount >= runtime.classCapacity) {
        runtime.classCapacity *= GROWTH_FACTOR;
        runtime.classes = realloc(runtime.classes, sizeof(JaiClass*) * runtime.classCapacity);
    }
    
    JaiClass* c = malloc(sizeof(JaiClass));
    runtime.classes[runtime.classCount++] = c;
    strncpy(c->name, name, MAX_NAME_LEN - 1);
    c->parent = parent;
    c->fieldCapacity = INITIAL_CAPACITY;
    c->fieldNames = malloc(sizeof(char*) * c->fieldCapacity);
    c->fieldCount = 0;
    c->methodCapacity = INITIAL_CAPACITY;
    c->methods = malloc(sizeof(JaiFunction*) * c->methodCapacity);
    c->methodNames = malloc(sizeof(char*) * c->methodCapacity);
    c->methodCount = 0;
    c->constructor = NULL;
    
    if (parent) {
        for (int i = 0; i < parent->fieldCount; i++) {
            classAddField(c, parent->fieldNames[i]);
        }
        for (int i = 0; i < parent->methodCount; i++) {
            classAddMethod(c, parent->methodNames[i], parent->methods[i]);
        }
    }
    
    return c;
}

void classAddField(JaiClass* class, const char* name) {
    if (class->fieldCount >= class->fieldCapacity) {
        class->fieldCapacity *= GROWTH_FACTOR;
        class->fieldNames = realloc(class->fieldNames, sizeof(char*) * class->fieldCapacity);
    }
    class->fieldNames[class->fieldCount++] = strdup(name);
}

void classAddMethod(JaiClass* class, const char* name, JaiFunction* method) {
    for (int i = 0; i < class->methodCount; i++) {
        if (strcmp(class->methodNames[i], name) == 0) {
            class->methods[i] = method;
            if (strcmp(name, "__init__") == 0 || strcmp(name, "init") == 0) {
                class->constructor = method;
            }
            return;
        }
    }
    
    if (class->methodCount >= class->methodCapacity) {
        class->methodCapacity *= GROWTH_FACTOR;
        class->methods = realloc(class->methods, sizeof(JaiFunction*) * class->methodCapacity);
        class->methodNames = realloc(class->methodNames, sizeof(char*) * class->methodCapacity);
    }
    class->methodNames[class->methodCount] = strdup(name);
    class->methods[class->methodCount++] = method;
    
    if (strcmp(name, "__init__") == 0 || strcmp(name, "init") == 0) {
        class->constructor = method;
    }
}

Value objectGetField(JaiObject* obj, const char* name) {
    for (int i = 0; i < obj->fieldCount; i++) {
        if (strcmp(obj->fieldNames[i], name) == 0) {
            return obj->fields[i];
        }
    }
    runtimeError("Object has no field: %s", name);
    return makeNull();
}

void objectSetField(JaiObject* obj, const char* name, Value value) {
    for (int i = 0; i < obj->fieldCount; i++) {
        if (strcmp(obj->fieldNames[i], name) == 0) {
            if (obj->fields[i].type == VAL_STRING && obj->fields[i].as.string) {
                free(obj->fields[i].as.string);
            }
            obj->fields[i] = value;
            return;
        }
    }
    
    if (obj->fieldCount >= obj->fieldCapacity) {
        obj->fieldCapacity *= GROWTH_FACTOR;
        obj->fields = realloc(obj->fields, sizeof(Value) * obj->fieldCapacity);
        obj->fieldNames = realloc(obj->fieldNames, sizeof(char*) * obj->fieldCapacity);
    }
    obj->fieldNames[obj->fieldCount] = strdup(name);
    obj->fields[obj->fieldCount++] = value;
}

JaiFunction* objectGetMethod(JaiObject* obj, const char* name) {
    JaiClass* c = obj->class;
    while (c) {
        for (int i = 0; i < c->methodCount; i++) {
            if (strcmp(c->methodNames[i], name) == 0) {
                return c->methods[i];
            }
        }
        c = c->parent;
    }
    return NULL;
}

JaiClass* findClass(const char* name) {
    for (int i = 0; i < runtime.classCount; i++) {
        if (strcmp(runtime.classes[i]->name, name) == 0) {
            return runtime.classes[i];
        }
    }
    return NULL;
}

void runtimeError(const char* format, ...) {
    const char* modName = runtime.currentModule ? runtime.currentModule->name : "<no-module>";
    const char* modPath = runtime.currentModule ? runtime.currentModule->path : "";
    fprintf(stderr, "Error in %s (%s:%d): ", modName, modPath, runtime.lineNumber);
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
    
    if (runtime.callStackSize > 0) {
        fprintf(stderr, "Call stack:\n");
        for (int i = runtime.callStackSize - 1; i >= 0; i--) {
            fprintf(stderr, "  at %s\n", runtime.callStack[i]);
        }
    }
    
    if (!runtime.shellMode) {
        exit(1);
    }
}
