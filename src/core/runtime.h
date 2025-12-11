#ifndef RUNTIME_H
#define RUNTIME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define INITIAL_CAPACITY 64
#define GROWTH_FACTOR 2
#define MAX_NAME_LEN 256
#define MAX_CODE_LEN 65536
#define MAX_CALL_STACK 256

typedef enum {
    VAL_NUMBER,
    VAL_STRING,
    VAL_BOOL,
    VAL_NULL,
    VAL_FUNCTION,
    VAL_NATIVE_FUNC,
    VAL_CELL,
    VAL_ARRAY,
    VAL_OBJECT,
    VAL_FILE,
    VAL_NAMESPACE
} ValueType;

typedef struct Value Value;
typedef struct JaiFunction JaiFunction;
typedef struct JaiCell JaiCell;
typedef struct JaiArray JaiArray;
typedef struct JaiObject JaiObject;
typedef struct JaiClass JaiClass;
typedef struct JaiNamespace JaiNamespace;
typedef struct Variable Variable;
typedef struct Module Module;
typedef Value (*NativeFunc)(Value* args, int argc);

struct JaiFunction {
    char name[MAX_NAME_LEN];
    char** params;
    int paramCount;
    int paramCapacity;
    bool isVariadic;
    char* body;
    Module* module;
    JaiNamespace* namespace;
};

struct JaiCell {
    Value* car;  //first
    Value* cdr;  //rest
};
//dynamic
struct JaiArray {
    Value* items;
    int length;
    int capacity;
};

struct JaiObject {
    JaiClass* class;
    Value* fields;
    char** fieldNames;
    int fieldCount;
    int fieldCapacity;
};

struct JaiClass {
    char name[MAX_NAME_LEN];
    JaiClass* parent;
    char** fieldNames;
    int fieldCount;
    int fieldCapacity;
    JaiFunction** methods;
    char** methodNames;
    int methodCount;
    int methodCapacity;
    JaiFunction* constructor;
};

struct JaiNamespace {
    char name[MAX_NAME_LEN];
    Variable* variables;
    int varCount;
    int varCapacity;
    JaiFunction** functions;
    int funcCount;
    int funcCapacity;
};

struct Value {
    ValueType type;
    union {
        double number;
        char* string;
        bool boolean;
        JaiFunction* function;
        NativeFunc nativeFunc;
        JaiCell* cell;
        JaiArray* array;
        JaiObject* object;
        FILE* file;
        JaiNamespace* namespace;
    } as;
};

struct Variable {
    char name[MAX_NAME_LEN];
    Value value;
};

struct Module {
    char name[MAX_NAME_LEN];
    char path[MAX_NAME_LEN];
    Variable* variables;
    int varCount;
    int varCapacity;
    JaiFunction** functions;
    int funcCount;
    int funcCapacity;
    bool loaded;
};

typedef enum {
    EVENT_TOKEN,
    EVENT_STATEMENT,
    EVENT_EXPRESSION,
    EVENT_FUNCTION_CALL,
    EVENT_MODULE_LOAD,
    EVENT_ERROR
} EventType;

typedef struct {
    EventType type;
    char name[MAX_NAME_LEN];
    void* data;
    Value result;
    bool handled;
} Event;

typedef void (*EventHandler)(Event* event);

typedef struct {
    char eventName[MAX_NAME_LEN];
    EventHandler* handlers;
    int handlerCount;
    int handlerCapacity;
} Subscription;

typedef struct {
    Subscription* subscriptions;
    int subCount;
    int subCapacity;
} EventBus;

typedef struct {
    char keyword[MAX_NAME_LEN];
    int tokenType;
} KeywordEntry;

typedef struct {
    KeywordEntry* entries;
    int count;
    int capacity;
    int nextTokenType;
} KeywordRegistry;

typedef struct {
    Module** modules;
    int moduleCount;
    int moduleCapacity;
    Module* currentModule;
    EventBus eventBus;
    KeywordRegistry keywords;
    JaiClass** classes;
    int classCount;
    int classCapacity;
    bool debug;
    bool shellMode;
    int lineNumber;
    char callStack[MAX_CALL_STACK][MAX_NAME_LEN];
    int callStackSize;
} Runtime;

extern Runtime runtime;

Value makeNumber(double n);
Value makeString(const char* s);
Value makeBool(bool b);
Value makeNull(void);
Value makeFunction(JaiFunction* f);
Value makeNativeFunc(NativeFunc f);
Value makeCell(void);
Value makeArray(int initialCapacity);
Value makeObject(JaiClass* class);
Value makeFile(FILE* f);
Value makeNamespace(const char* name);

void initRuntime(void);
void freeRuntime(void);

void subscribe(const char* eventName, EventHandler handler);
void publish(Event* event);
Event createEvent(EventType type, const char* name, void* data);

int registerKeyword(const char* keyword);
int lookupKeyword(const char* word);

Module* createModule(const char* name, const char* path);
Module* findModule(const char* name);
Module* loadModule(const char* path);

void setVariable(const char* name, Value value);
Value getVariable(const char* name);
bool hasVariable(const char* name);

JaiFunction* defineFunction(const char* name, char** params, int paramCount, bool isVariadic, const char* body);
JaiFunction* findFunction(const char* name);
Value callFunction(JaiFunction* func, Value* args, int argc);

void arrayPush(JaiArray* arr, Value val);
Value arrayGet(JaiArray* arr, int index);
void arraySet(JaiArray* arr, int index, Value val);
Value arrayPop(JaiArray* arr);
int arrayLen(JaiArray* arr);

JaiClass* defineClass(const char* name, JaiClass* parent);
void classAddField(JaiClass* class, const char* name);
void classAddMethod(JaiClass* class, const char* name, JaiFunction* method);
Value objectGetField(JaiObject* obj, const char* name);
void objectSetField(JaiObject* obj, const char* name, Value value);
JaiFunction* objectGetMethod(JaiObject* obj, const char* name);
JaiClass* findClass(const char* name);

void runtimeError(const char* format, ...);

#endif
