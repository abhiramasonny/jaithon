#ifndef JAI_BYTECODE_H
#define JAI_BYTECODE_H

#include "vm.h"
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

// ============================================================================
// .jaic Cache File Format
// Header:
//   Magic: "JAIC" (4 bytes)
//   Version: uint16_t
//   Source hash: uint64_t (for invalidation)
//   Timestamp: int64_t (source file mtime)
//   Num functions: uint32_t
//
// For each function:
//   Name length: uint16_t
//   Name: char[name_length]
//   Arity: uint8_t
//   Is variadic: uint8_t
//   Num params: uint8_t
//   For each param:
//     Param name length: uint16_t
//     Param name: char[length]
//   Code length: uint32_t
//   Code: uint8_t[code_length]
//   Num constants: uint32_t
//   For each constant:
//     Type: uint8_t
//     Value data (varies by type)
// ============================================================================

#define JAIC_MAGIC "JAIC"
#define JAIC_VERSION 1

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

#endif
