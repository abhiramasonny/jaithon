#include "bytecode.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <libgen.h>
#include <unistd.h>
#include "../lang/parser.h"

BytecodeCache gBytecodeCache = {0};

void cacheInit(const char* baseDir) {
    gBytecodeCache.entries = NULL;
    gBytecodeCache.count = 0;
    gBytecodeCache.capacity = 0;
    
    if (baseDir) {
        snprintf(gBytecodeCache.cacheDir, sizeof(gBytecodeCache.cacheDir), 
                 "%s/__jaicache__", baseDir);
    } else {
        strcpy(gBytecodeCache.cacheDir, "./__jaicache__");
    }
    
    ensureCacheDir(gBytecodeCache.cacheDir);
}

void cacheFree(void) {
    for (int i = 0; i < gBytecodeCache.count; i++) {
        if (gBytecodeCache.entries[i].func) {
            compiledFuncFree(gBytecodeCache.entries[i].func);
        }
    }
    if (gBytecodeCache.entries) {
        free(gBytecodeCache.entries);
    }
    gBytecodeCache.entries = NULL;
    gBytecodeCache.count = 0;
    gBytecodeCache.capacity = 0;
}

uint64_t hashSource(const char* source) {
    uint64_t hash = 14695981039346656037ULL;
    while (*source) {
        hash ^= (uint8_t)*source++;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool ensureCacheDir(const char* dir) {
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0755) != 0) {
            return false;
        }
    }
    return true;
}

void getCachePath(const char* sourceFile, char* outPath, size_t outSize) {
    char srcCopy[1024];
    strncpy(srcCopy, sourceFile, sizeof(srcCopy) - 1);
    char* dir = dirname(srcCopy);
    
    char srcCopy2[1024];
    strncpy(srcCopy2, sourceFile, sizeof(srcCopy2) - 1);
    char* base = basename(srcCopy2);
    
    char* dot = strrchr(base, '.');
    if (dot) *dot = '\0';
    
    char cacheDir[1024];
    snprintf(cacheDir, sizeof(cacheDir), "%s/__jaicache__", dir);
    ensureCacheDir(cacheDir);
    
    snprintf(outPath, outSize, "%s/%s.jaic", cacheDir, base);
}

static void writeU8(uint8_t** ptr, uint8_t val) {
    **ptr = val;
    (*ptr)++;
}

static void writeU16(uint8_t** ptr, uint16_t val) {
    **ptr = (val >> 8) & 0xFF;
    (*ptr)++;
    **ptr = val & 0xFF;
    (*ptr)++;
}

static void writeU32(uint8_t** ptr, uint32_t val) {
    **ptr = (val >> 24) & 0xFF; (*ptr)++;
    **ptr = (val >> 16) & 0xFF; (*ptr)++;
    **ptr = (val >> 8) & 0xFF; (*ptr)++;
    **ptr = val & 0xFF; (*ptr)++;
}

static void writeU64(uint8_t** ptr, uint64_t val) {
    for (int i = 7; i >= 0; i--) {
        **ptr = (val >> (i * 8)) & 0xFF;
        (*ptr)++;
    }
}

static void writeDouble(uint8_t** ptr, double val) {
    memcpy(*ptr, &val, sizeof(double));
    *ptr += sizeof(double);
}

static void writeString(uint8_t** ptr, const char* str) {
    uint16_t len = strlen(str);
    writeU16(ptr, len);
    memcpy(*ptr, str, len);
    *ptr += len;
}


static uint8_t readU8(uint8_t** ptr) {
    return *(*ptr)++;
}

static uint16_t readU16(uint8_t** ptr) {
    uint16_t val = (**ptr << 8);
    (*ptr)++;
    val |= **ptr;
    (*ptr)++;
    return val;
}

static uint32_t readU32(uint8_t** ptr) {
    uint32_t val = 0;
    for (int i = 0; i < 4; i++) {
        val = (val << 8) | **ptr;
        (*ptr)++;
    }
    return val;
}

static uint64_t readU64(uint8_t** ptr) {
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val = (val << 8) | **ptr;
        (*ptr)++;
    }
    return val;
}

static double readDouble(uint8_t** ptr) {
    double val;
    memcpy(&val, *ptr, sizeof(double));
    *ptr += sizeof(double);
    return val;
}

static char* readString(uint8_t** ptr) {
    uint16_t len = readU16(ptr);
    char* str = malloc(len + 1);
    memcpy(str, *ptr, len);
    str[len] = '\0';
    *ptr += len;
    return str;
}

uint8_t* serializeFunc(CompiledFunc* func, size_t* outSize) {
    size_t size = 0;
    size += strlen(func->name) + 2;
    size += 1;
    size += 1;
    size += 1;
    for (int i = 0; i < func->arity; i++) {
        size += 2 + strlen(func->paramNames[i]);
    }
    size += 4;
    size += func->chunk.count;
    size += 4;

    for (int i = 0; i < func->chunk.constCount; i++) {
        Value v = func->chunk.constants[i];
        size += 1;
        switch (v.type) {
            case VAL_NULL:
            case VAL_BOOL:
                size += 1;
                break;
            case VAL_NUMBER:
                size += sizeof(double);
                break;
            case VAL_STRING:
                size += 2 + strlen(v.as.string);
                break;
            default:
                break;
        }
    }
    
    uint8_t* data = malloc(size + 64);
    uint8_t* ptr = data;
    
    writeString(&ptr, func->name);
    writeU8(&ptr, func->arity);
    writeU8(&ptr, func->isVariadic ? 1 : 0);
    writeU8(&ptr, func->arity);
    
    for (int i = 0; i < func->arity; i++) {
        writeString(&ptr, func->paramNames[i]);
    }
    
    writeU32(&ptr, func->chunk.count);
    memcpy(ptr, func->chunk.code, func->chunk.count);
    ptr += func->chunk.count;
    
    writeU32(&ptr, func->chunk.constCount);
    for (int i = 0; i < func->chunk.constCount; i++) {
        Value v = func->chunk.constants[i];
        switch (v.type) {
            case VAL_NULL:
                writeU8(&ptr, CONST_NULL);
                break;
            case VAL_BOOL:
                writeU8(&ptr, CONST_BOOL);
                writeU8(&ptr, v.as.boolean ? 1 : 0);
                break;
            case VAL_NUMBER:
                writeU8(&ptr, CONST_NUMBER);
                writeDouble(&ptr, v.as.number);
                break;
            case VAL_STRING:
                writeU8(&ptr, CONST_STRING);
                writeString(&ptr, v.as.string);
                break;
            default:
                writeU8(&ptr, CONST_NULL);
                break;
        }
    }
    
    *outSize = ptr - data;
    return data;
}

CompiledFunc* deserializeFunc(uint8_t* data, size_t size) {
    (void)size;
    uint8_t* ptr = data;
    
    char* name = readString(&ptr);
    uint8_t arity = readU8(&ptr);
    bool isVariadic = readU8(&ptr) != 0;
    uint8_t numParams = readU8(&ptr);
    
    CompiledFunc* func = compiledFuncNew(name, arity);
    func->isVariadic = isVariadic;
    free(name);
    
    if (numParams > 0) {
        func->paramNames = malloc(sizeof(char*) * numParams);
        for (int i = 0; i < numParams; i++) {
            func->paramNames[i] = readString(&ptr);
        }
    }

    uint32_t codeLen = readU32(&ptr);
    func->chunk.code = malloc(codeLen);
    func->chunk.count = codeLen;
    func->chunk.capacity = codeLen;
    memcpy(func->chunk.code, ptr, codeLen);
    ptr += codeLen;
    
    uint32_t numConsts = readU32(&ptr);
    func->chunk.constants = malloc(sizeof(Value) * numConsts);
    func->chunk.constCount = numConsts;
    func->chunk.constCapacity = numConsts;
    
    for (uint32_t i = 0; i < numConsts; i++) {
        uint8_t type = readU8(&ptr);
        switch (type) {
            case CONST_NULL:
                func->chunk.constants[i] = makeNull();
                break;
            case CONST_BOOL:
                func->chunk.constants[i] = makeBool(readU8(&ptr) != 0);
                break;
            case CONST_NUMBER:
                func->chunk.constants[i] = makeNumber(readDouble(&ptr));
                break;
            case CONST_STRING: {
                char* str = readString(&ptr);
                func->chunk.constants[i] = makeString(str);
                free(str);
                break;
            }
            default:
                func->chunk.constants[i] = makeNull();
                break;
        }
    }
    
    return func;
}

CompiledFunc* cacheLoad(const char* funcName, const char* sourceFile, const char* funcBody) {
    char cachePath[1024];
    getCachePath(sourceFile, cachePath, sizeof(cachePath));
    
    char funcCachePath[1100];
    snprintf(funcCachePath, sizeof(funcCachePath), "%s.%s", cachePath, funcName);
    
    FILE* f = fopen(funcCachePath, "rb");
    if (!f) return NULL;
    
    char magic[5] = {0};
    fread(magic, 1, 4, f);
    if (strcmp(magic, JAIC_MAGIC) != 0) {
        fclose(f);
        return NULL;
    }
    
    uint16_t version;
    fread(&version, sizeof(version), 1, f);
    if (version != JAIC_VERSION) {
        fclose(f);
        return NULL;
    }
    
    uint64_t storedHash;
    fread(&storedHash, sizeof(storedHash), 1, f);
    
    uint64_t currentHash = hashSource(funcBody);
    if (storedHash != currentHash) {
        fclose(f);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long totalSize = ftell(f);
    long dataSize = totalSize - 4 - 2 - 8;
    fseek(f, 4 + 2 + 8, SEEK_SET);
    
    uint8_t* data = malloc(dataSize);
    fread(data, 1, dataSize, f);
    fclose(f);
    
    CompiledFunc* func = deserializeFunc(data, dataSize);
    free(data);
    
    return func;
}

bool cacheSave(const char* funcName, const char* sourceFile, CompiledFunc* func, const char* funcBody) {
    char cachePath[1024];
    getCachePath(sourceFile, cachePath, sizeof(cachePath));
    
    char funcCachePath[1100];
    snprintf(funcCachePath, sizeof(funcCachePath), "%s.%s", cachePath, funcName);
    
    FILE* f = fopen(funcCachePath, "wb");
    if (!f) return false;
    
    fwrite(JAIC_MAGIC, 1, 4, f);
    uint16_t version = JAIC_VERSION;
    fwrite(&version, sizeof(version), 1, f);
    
    uint64_t hash = hashSource(funcBody);
    fwrite(&hash, sizeof(hash), 1, f);
    
    size_t size;
    uint8_t* data = serializeFunc(func, &size);
    fwrite(data, 1, size, f);
    free(data);
    
    fclose(f);
    return true;
}

bool cacheIsValid(const char* sourceFile, uint64_t cachedHash) {
    FILE* f = fopen(sourceFile, "r");
    if (!f) return false;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* content = malloc(size + 1);
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    uint64_t currentHash = hashSource(content);
    free(content);
    
    return currentHash == cachedHash;
}

static bool writeStringFile(FILE* f, const char* str) {
    uint16_t len = (uint16_t)(str ? strlen(str) : 0);
    if (fwrite(&len, sizeof(len), 1, f) != 1) return false;
    if (len == 0) return true;
    return fwrite(str, 1, len, f) == len;
}

static char* readStringFile(FILE* f) {
    uint16_t len = 0;
    if (fread(&len, sizeof(len), 1, f) != 1) return NULL;
    char* buf = malloc(len + 1);
    if (len > 0) {
        if (fread(buf, 1, len, f) != len) {
            free(buf);
            return NULL;
        }
    }
    buf[len] = '\0';
    return buf;
}

bool saveJaicBundle(const char* bundlePath, BundleEntry* entries, int count, const char* entryName, uint64_t sourceHash) {
    if (!bundlePath || !entries || count <= 0) return false;
    
    FILE* f = fopen(bundlePath, "wb");
    if (!f) return false;
    
    bool ok = true;
    
    if (fwrite(JAIC_BUNDLE_MAGIC, 1, 4, f) != 4) ok = false;
    
    uint16_t version = JAIC_BUNDLE_VERSION;
    if (ok && fwrite(&version, sizeof(version), 1, f) != 1) ok = false;
    
    if (ok && fwrite(&sourceHash, sizeof(sourceHash), 1, f) != 1) ok = false;
    
    const char* entry = (entryName && strlen(entryName) > 0) ? entryName : "main";
    if (ok && !writeStringFile(f, entry)) ok = false;
    
    uint32_t funcCount = (uint32_t)count;
    if (ok && fwrite(&funcCount, sizeof(funcCount), 1, f) != 1) ok = false;
    
    for (int i = 0; ok && i < count; i++) {
        BundleEntry* be = &entries[i];
        if (!be->func) {
            ok = false;
            break;
        }
        
        if (!writeStringFile(f, be->func->name)) { ok = false; break; }
        
        uint8_t arity = (uint8_t)be->func->paramCount;
        uint8_t isVar = be->func->isVariadic ? 1 : 0;
        if (fwrite(&arity, sizeof(arity), 1, f) != 1) { ok = false; break; }
        if (fwrite(&isVar, sizeof(isVar), 1, f) != 1) { ok = false; break; }
        for (int p = 0; p < be->func->paramCount; p++) {
            if (!writeStringFile(f, be->func->params[p])) { ok = false; break; }
        }
        if (!ok) break;
        
        if (fwrite(&be->bodyHash, sizeof(be->bodyHash), 1, f) != 1) { ok = false; break; }
        size_t bcSize = 0;
        uint8_t* data = NULL;
        if (be->compiled) {
            data = serializeFunc(be->compiled, &bcSize);
            if (!data) { ok = false; break; }
        }
        
        uint32_t bcSize32 = (uint32_t)bcSize;
        if (fwrite(&bcSize32, sizeof(bcSize32), 1, f) != 1) {
            ok = false;
            break;
        }
        if (bcSize > 0) {
            if (fwrite(data, 1, bcSize, f) != bcSize) {
                free(data);
                ok = false;
                break;
            }
        }
        free(data);
    }
    
    fclose(f);
    if (!ok) {
        remove(bundlePath);
    }
    return ok;
}

bool loadJaicBundle(const char* bundlePath, Module* module, char* entryOut, size_t entryOutSize, uint64_t* sourceHashOut) {
    if (!bundlePath) return false;
    if (!module) module = runtime.currentModule;
    
    FILE* f = fopen(bundlePath, "rb");
    if (!f) return false;
    
    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strcmp(magic, JAIC_BUNDLE_MAGIC) != 0) {
        fclose(f);
        return false;
    }
    
    uint16_t version = 0;
    if (fread(&version, sizeof(version), 1, f) != 1 ||
        (version != JAIC_BUNDLE_VERSION && version != 2 && version != 3)) {
        fclose(f);
        return false;
    }
    bool hasBodyString = version < 3;
    
    uint64_t srcHash = 0;
    if (fread(&srcHash, sizeof(srcHash), 1, f) != 1) {
        fclose(f);
        return false;
    }
    if (sourceHashOut) {
        *sourceHashOut = srcHash;
    }
    
    char* entryName = readStringFile(f);
    if (!entryName) {
        fclose(f);
        return false;
    }
    if (entryOut && entryOutSize > 0) {
        strncpy(entryOut, entryName, entryOutSize - 1);
        entryOut[entryOutSize - 1] = '\0';
    }
    free(entryName);
    
    uint32_t funcCount = 0;
    if (fread(&funcCount, sizeof(funcCount), 1, f) != 1) {
        fclose(f);
        return false;
    }
    
    Module* old = runtime.currentModule;
    runtime.currentModule = module;
    
    bool ok = true;
    for (uint32_t i = 0; i < funcCount; i++) {
        char* name = readStringFile(f);
        uint8_t arity = 0;
        uint8_t isVar = 0;
        uint64_t bodyHash = 0;
        char** paramNames = NULL;
        char* body = NULL;
        
        if (!name) { ok = false; break; }
        if (fread(&arity, sizeof(arity), 1, f) != 1) { ok = false; free(name); break; }
        if (fread(&isVar, sizeof(isVar), 1, f) != 1) { ok = false; free(name); break; }
        if (arity > 0) {
            paramNames = malloc(sizeof(char*) * arity);
            for (int p = 0; p < arity; p++) {
                paramNames[p] = readStringFile(f);
                if (!paramNames[p]) {
                    ok = false;
                    for (int j = 0; j < p; j++) free(paramNames[j]);
                    free(paramNames);
                    free(name);
                    break;
                }
            }
            if (!ok) break;
        }
        if (fread(&bodyHash, sizeof(bodyHash), 1, f) != 1) {
            ok = false;
            free(name);
            if (paramNames) {
                for (int p = 0; p < arity; p++) free(paramNames[p]);
                free(paramNames);
            }
            break;
        }
        
        if (hasBodyString) {
            body = readStringFile(f);
            if (!body) {
                ok = false;
                free(name);
                if (paramNames) {
                    for (int p = 0; p < arity; p++) free(paramNames[p]);
                    free(paramNames);
                }
                break;
            }
        } else {
            body = strdup("");
        }
        
        uint32_t bcSize = 0;
        if (fread(&bcSize, sizeof(bcSize), 1, f) != 1) {
            ok = false;
            free(name);
            free(body);
            if (paramNames) {
                for (int p = 0; p < arity; p++) free(paramNames[p]);
                free(paramNames);
            }
            break;
        }
        CompiledFunc* compiled = NULL;
        if (bcSize > 0) {
            uint8_t* data = malloc(bcSize);
            if (!data) {
                ok = false;
                free(name);
                free(body);
                if (paramNames) {
                    for (int p = 0; p < arity; p++) free(paramNames[p]);
                    free(paramNames);
                }
                break;
            }
            if (fread(data, 1, bcSize, f) != bcSize) {
                ok = false;
                free(name);
                free(body);
                if (paramNames) {
                    for (int p = 0; p < arity; p++) free(paramNames[p]);
                    free(paramNames);
                }
                free(data);
                break;
            }
            
            compiled = deserializeFunc(data, bcSize);
            free(data);
            if (!compiled) {
                ok = false;
                free(name);
                free(body);
                if (paramNames) {
                    for (int p = 0; p < arity; p++) free(paramNames[p]);
                    free(paramNames);
                }
                break;
            }
            compiled->arity = arity;
            compiled->isVariadic = isVar != 0;
        }
        
        char** params = NULL;
        if (arity > 0) {
            params = malloc(sizeof(char*) * arity);
            for (int p = 0; p < arity; p++) {
                const char* pname = (paramNames && paramNames[p]) ? paramNames[p] : NULL;
                if (!pname) {
                    char tmp[16];
                    snprintf(tmp, sizeof(tmp), "arg%d", p);
                    params[p] = strdup(tmp);
                } else {
                    params[p] = strdup(pname);
                }
            }
        }
        
        JaiFunction* jf = defineFunction(name, params, arity, isVar != 0, body);
        if (jf) {
            jf->bodyHash = bodyHash;
            jf->hasBodyHash = true;
            if (!hasBodyString && jf->body) {
                free(jf->body);
                jf->body = strdup("");
            }
        }
        for (int p = 0; p < arity; p++) {
            free(params[p]);
        }
        free(params);
        if (paramNames) {
            for (int p = 0; p < arity; p++) {
                free(paramNames[p]);
            }
            free(paramNames);
        }
        free(name);
        free(body);
        if (!jf) {
            if (compiled) compiledFuncFree(compiled);
            ok = false;
            break;
        }
        
        uint64_t regHash = jf ? jf->bodyHash : bodyHash;
        if (compiled) {
            if (!registerCompiledFunction(jf, compiled, regHash)) {
                compiledFuncFree(compiled);
                ok = false;
                break;
            }
        }
    }
    
    runtime.currentModule = old;
    fclose(f);
    return ok;
}
