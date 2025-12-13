#ifndef JAI_BYTECODE_H
#define JAI_BYTECODE_H

#include "vm.h"
#include <stdint.h>
#include <stdbool.h>
#include <time.h>


#define JAIC_MAGIC "JAIC"
#define JAIC_VERSION 3
#define JAIC_BUNDLE_MAGIC "JAIB"
#define JAIC_BUNDLE_VERSION 4

typedef enum {
    CONST_NULL = 0,
    CONST_BOOL = 1,
    CONST_NUMBER = 2,
    CONST_STRING = 3,
} ConstType;

typedef struct {
    char name[MAX_NAME_LEN];
    CompiledFunc* func;
    uint64_t sourceHash;
    time_t sourceTime;
    bool valid;
} CacheEntry;

typedef struct {
    CacheEntry* entries;
    int count;
    int capacity;
    char cacheDir[1024];
} BytecodeCache;

typedef struct {
    JaiFunction* func;
    CompiledFunc* compiled;
    uint64_t bodyHash;
} BundleEntry;

extern BytecodeCache gBytecodeCache;

void cacheInit(const char* baseDir);
void cacheFree(void);
uint64_t hashSource(const char* source);

void getCachePath(const char* sourceFile, char* outPath, size_t outSize);
CompiledFunc* cacheLoad(const char* funcName, const char* sourceFile, const char* funcBody);

bool cacheSave(const char* funcName, const char* sourceFile, CompiledFunc* func, const char* funcBody);

uint8_t* serializeFunc(CompiledFunc* func, size_t* outSize);
CompiledFunc* deserializeFunc(uint8_t* data, size_t size);

bool cacheIsValid(const char* sourceFile, uint64_t cachedHash);
bool ensureCacheDir(const char* baseDir);

bool saveJaicBundle(const char* bundlePath, BundleEntry* entries, int count, const char* entryName, uint64_t sourceHash);
bool loadJaicBundle(const char* bundlePath, Module* module, char* entryOut, size_t entryOutSize, uint64_t* sourceHashOut);

#endif
