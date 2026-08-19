/* builtins_compress.c — __prim__.deflate/adler32/crc32: a native DEFLATE (RFC 1951 fixed Huffman) and checksums, since zlib is not guaranteed to be linked and PNG output needs it. */

#include "runtime/builtins/builtins.h"
#include "runtime/runtime.h"

#include <zlib.h>

typedef struct {
    uint8_t *bytes;
    size_t   count;
    size_t   capacity;
    uint32_t bits;
    int      bitCount;
} BitWriter;

static void writerReserve(BitWriter *writer, size_t extra) {
    if (writer->count + extra <= writer->capacity) return;
    size_t capacity = writer->capacity < 512 ? 512 : writer->capacity;
    while (capacity < writer->count + extra) capacity *= 2;
    writer->bytes = JAI_GROW_ARRAY(uint8_t, writer->bytes, writer->capacity, capacity);
    writer->capacity = capacity;
}

static void writerPush(BitWriter *writer, uint8_t byte) {
    writerReserve(writer, 1);
    writer->bytes[writer->count++] = byte;
}

static void writerBits(BitWriter *writer, uint32_t value, int count) {
    writer->bits |= (value & ((1u << count) - 1u)) << writer->bitCount;
    writer->bitCount += count;
    while (writer->bitCount >= 8) {
        writerPush(writer, (uint8_t)(writer->bits & 0xFFu));
        writer->bits >>= 8;
        writer->bitCount -= 8;
    }
}

static void writerAlign(BitWriter *writer) {
    if (writer->bitCount == 0) return;
    writerPush(writer, (uint8_t)(writer->bits & 0xFFu));
    writer->bits = 0;
    writer->bitCount = 0;
}

static void writerCode(BitWriter *writer, uint32_t code, int bits) {
    uint32_t reversed = 0;
    for (int i = 0; i < bits; i++) reversed = (reversed << 1) | ((code >> i) & 1u);
    writerBits(writer, reversed, bits);
}

static void writeSymbol(BitWriter *writer, int symbol) {
    if (symbol <= 143)      writerCode(writer, 0x30u + (uint32_t)symbol, 8);
    else if (symbol <= 255) writerCode(writer, 0x190u + (uint32_t)(symbol - 144), 9);
    else if (symbol <= 279) writerCode(writer, (uint32_t)(symbol - 256), 7);
    else                    writerCode(writer, 0xC0u + (uint32_t)(symbol - 280), 8);
}

static const int kLengthBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51,
    59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
};
static const int kLengthExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
    3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
};
static const int kDistanceBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
};
static const int kDistanceExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
};

static void writeMatch(BitWriter *writer, int length, int64_t distance) {
    int code = 0;
    while (code < 28 && length >= kLengthBase[code + 1]) code++;
    writeSymbol(writer, 257 + code);
    if (kLengthExtra[code] > 0)
        writerBits(writer, (uint32_t)(length - kLengthBase[code]), kLengthExtra[code]);

    int distanceCode = 0;
    while (distanceCode < 29 && distance >= kDistanceBase[distanceCode + 1])
        distanceCode++;
    writerCode(writer, (uint32_t)distanceCode, 5);
    if (kDistanceExtra[distanceCode] > 0) {
        writerBits(writer, (uint32_t)(distance - kDistanceBase[distanceCode]),
                   kDistanceExtra[distanceCode]);
    }
}

#define WINDOW_SIZE 32768
#define WINDOW_MASK (WINDOW_SIZE - 1)
#define HASH_BITS   15
#define HASH_SIZE   (1 << HASH_BITS)
#define MIN_MATCH   3
#define MAX_MATCH   258

typedef struct {
    int32_t head[HASH_SIZE];
    int32_t prev[WINDOW_SIZE];
} MatchTable;

static uint32_t hashAt(const uint8_t *data) {
    return (uint32_t)(((uint32_t)data[0] << 10) ^ ((uint32_t)data[1] << 5) ^
                      (uint32_t)data[2]) & (HASH_SIZE - 1u);
}

static int32_t tableInsert(MatchTable *table, const uint8_t *data, int64_t length,
                           int64_t position) {
    if (position + MIN_MATCH > length) return -1;
    uint32_t slot = hashAt(data + position);
    int32_t previous = table->head[slot];
    table->prev[position & WINDOW_MASK] = previous;
    table->head[slot] = (int32_t)position;
    return previous;
}

static int findMatch(const MatchTable *table, const uint8_t *data, int64_t length,
                     int64_t position, int64_t candidate, int chainLimit,
                     int niceLength, int64_t *outDistance) {
    int64_t available = length - position;
    if (available < MIN_MATCH) return 0;
    int64_t maxLength = available < MAX_MATCH ? available : MAX_MATCH;
    int64_t oldest = position > WINDOW_SIZE ? position - WINDOW_SIZE : 0;

    const uint8_t *here = data + position;
    int best = 0;
    int64_t bestDistance = 0;

    for (int chain = 0; chain < chainLimit && candidate >= oldest; chain++) {
        const uint8_t *there = data + candidate;
        if (there[best] == here[best] && there[0] == here[0] && there[1] == here[1]) {
            int64_t matched = 0;
            while (matched < maxLength && there[matched] == here[matched]) matched++;
            if (matched > best) {
                best = (int)matched;
                bestDistance = position - candidate;
                if (best >= niceLength || best >= maxLength) break;
            }
        }
        candidate = table->prev[candidate & WINDOW_MASK];
    }

    if (best < MIN_MATCH) return 0;
    *outDistance = bestDistance;
    return best;
}

static size_t storedSize(size_t length) {
    size_t blocks = length == 0 ? 1 : (length + 65534) / 65535;
    return blocks * 5 + length;
}

static void writeStoredBlocks(BitWriter *writer, const uint8_t *data, size_t length) {
    size_t offset = 0;
    do {
        size_t chunk = length - offset;
        if (chunk > 65535) chunk = 65535;
        writerBits(writer, (offset + chunk == length) ? 1u : 0u, 1);
        writerBits(writer, 0u, 2);
        writerAlign(writer);
        writerPush(writer, (uint8_t)(chunk & 0xFFu));
        writerPush(writer, (uint8_t)((chunk >> 8) & 0xFFu));
        writerPush(writer, (uint8_t)(~chunk & 0xFFu));
        writerPush(writer, (uint8_t)((~chunk >> 8) & 0xFFu));
        writerReserve(writer, chunk);
        if (chunk > 0) {
            memcpy(writer->bytes + writer->count, data + offset, chunk);
            writer->count += chunk;
        }
        offset += chunk;
    } while (offset < length);
}

static const int  kChainLimit[10] = {0, 4, 8, 16, 32, 64, 128, 256, 1024, 4096};
static const int  kNiceLength[10] = {0, 16, 24, 32, 48, 64, 128, 192, 258, 258};
static const bool kUseLazy[10] = {
    false, false, false, false, true, true, true, true, true, true,
};

static void writeFixedBlock(BitWriter *writer, const uint8_t *data, int64_t length,
                            int level) {
    writerBits(writer, 1u, 1);
    writerBits(writer, 1u, 2);

    MatchTable *table = JAI_ALLOC(MatchTable, 1);
    memset(table->head, 0xFF, sizeof table->head);
    memset(table->prev, 0xFF, sizeof table->prev);

    int chainLimit = kChainLimit[level];
    int niceLength = kNiceLength[level];
    bool useLazy = kUseLazy[level];

    int64_t position = 0;
    int     heldLength = 0;
    int64_t heldDistance = 0;
    bool    holding = false;

    while (position < length) {
        int64_t chainStart = tableInsert(table, data, length, position);
        int64_t distance = 0;
        int found = findMatch(table, data, length, position, chainStart, chainLimit,
                              niceLength, &distance);

        if (holding) {
            if (found > heldLength) {
                writeSymbol(writer, data[position - 1]);
                heldLength = found;
                heldDistance = distance;
                position++;
                continue;
            }
            writeMatch(writer, heldLength, heldDistance);
            int64_t end = position - 1 + heldLength;
            for (int64_t i = position + 1; i < end; i++)
                (void)tableInsert(table, data, length, i);
            position = end;
            holding = false;
            continue;
        }

        if (found >= MIN_MATCH) {
            if (useLazy && position + 1 < length && found < niceLength) {
                heldLength = found;
                heldDistance = distance;
                holding = true;
                position++;
                continue;
            }
            writeMatch(writer, found, distance);
            for (int64_t i = position + 1; i < position + found; i++)
                (void)tableInsert(table, data, length, i);
            position += found;
            continue;
        }

        writeSymbol(writer, data[position]);
        position++;
    }

    if (holding) writeMatch(writer, heldLength, heldDistance);

    writeSymbol(writer, 256);
    JAI_FREE(MatchTable, table);
}

#define ADLER_BASE  65521u
#define ADLER_BLOCK 5552u

static uint32_t adler32Of(const uint8_t *data, size_t length) {
    uint32_t low = 1, high = 0;
    while (length > 0) {
        size_t block = length < ADLER_BLOCK ? length : ADLER_BLOCK;
        for (size_t i = 0; i < block; i++) {
            low += data[i];
            high += low;
        }
        low %= ADLER_BASE;
        high %= ADLER_BASE;
        data += block;
        length -= block;
    }
    return (high << 16) | low;
}

static bool compressArgBytes(Value v, int index, const char *fnName, ObjBytes **out) {
    if (!IS_BYTES(v)) return jaiBuiltinArgTypeError(index, fnName, "bytes", v);
    *out = AS_BYTES(v);
    return true;
}

static bool nDeflate(int argc, Value *args, Value *out) {
    (void)argc;
    ObjBytes *data;
    int64_t level;
    if (!compressArgBytes(args[0], 1, "deflate", &data)) return false;
    if (!jaiArgInt(args[1], 2, "deflate", &level)) return false;
    if (level < 0 || level > 9) {
        return jaiThrow(vm.cValueError,
                        "deflate(): level must be between 0 and 9, got %lld",
                        (long long)level);
    }

    uint32_t flevel = level == 0 ? 0u : (level < 6 ? 1u : (level == 6 ? 2u : 3u));
    uint32_t header = (0x78u << 8) | (flevel << 6);
    header += 31u - (header % 31u);

    BitWriter writer = {NULL, 0, 0, 0, 0};
    writerPush(&writer, (uint8_t)((header >> 8) & 0xFFu));
    writerPush(&writer, (uint8_t)(header & 0xFFu));

    size_t blockStart = writer.count;
    if (level == 0) {
        writeStoredBlocks(&writer, data->data, data->length);
    } else {
        writeFixedBlock(&writer, data->data, (int64_t)data->length, (int)level);
        writerAlign(&writer);
        if (writer.count - blockStart > storedSize(data->length)) {
            writer.count = blockStart;
            writeStoredBlocks(&writer, data->data, data->length);
        }
    }

    writerAlign(&writer);
    uint32_t checksum = adler32Of(data->data, data->length);
    writerPush(&writer, (uint8_t)((checksum >> 24) & 0xFFu));
    writerPush(&writer, (uint8_t)((checksum >> 16) & 0xFFu));
    writerPush(&writer, (uint8_t)((checksum >> 8) & 0xFFu));
    writerPush(&writer, (uint8_t)(checksum & 0xFFu));

    ObjBytes *result = jaiBytesNew(writer.bytes, writer.count);
    JAI_FREE_ARRAY(uint8_t, writer.bytes, writer.capacity);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

static bool nAdler32(int argc, Value *args, Value *out) {
    (void)argc;
    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "adler32", &data)) return false;
    *out = INT_VAL((int64_t)adler32Of(data->data, data->length));
    return true;
}


static uint32_t sCrcTable[256];
static bool sCrcReady = false;

static void crcInit(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        sCrcTable[n] = c;
    }
    sCrcReady = true;
}

static bool nCrc32(int argc, Value *args, Value *out) {
    (void)argc;
    if (!sCrcReady) crcInit();

    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "crc32", &data)) return false;
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < data->length; i++) {
        crc = sCrcTable[(crc ^ data->data[i]) & 0xFFu] ^ (crc >> 8);
    }
    *out = INT_VAL((int64_t)(crc ^ 0xFFFFFFFFu));
    return true;
}

/* The decoder is zlib's. Writing one here would mean a Huffman decoder, a
 * sliding window and a stored-block path in C for no gain: -lz is already on
 * the link line for the seed's images, so PNG input costs nothing but this
 * wrapper. */
static bool nInflate(int argc, Value *args, Value *out) {
    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "inflate", &data)) return false;

    int64_t hint = 0;
    if (argc > 1 && !IS_NULL(args[1])) {
        if (!jaiArgInt(args[1], 2, "inflate", &hint)) return false;
        if (hint < 0) return jaiThrow(vm.cValueError,
                                      "inflate(): the size hint must not be negative");
    }

    size_t capacity = hint > 0 ? (size_t)hint : (data->length * 4u + 1024u);
    uint8_t *buffer = JAI_GROW_ARRAY(uint8_t, NULL, 0, capacity);
    if (buffer == NULL) return jaiThrow(vm.cRuntimeError, "inflate(): out of memory");

    z_stream stream;
    memset(&stream, 0, sizeof stream);
    stream.next_in = (Bytef *)data->data;
    stream.avail_in = (uInt)data->length;
    /* 47 = 15 window bits plus the flag that accepts either a zlib or a gzip
     * header, so a caller does not have to know which one it holds. */
    if (inflateInit2(&stream, 47) != Z_OK) {
        JAI_FREE_ARRAY(uint8_t, buffer, capacity);
        return jaiThrow(vm.cRuntimeError, "inflate(): the decoder would not start");
    }

    size_t produced = 0;
    for (;;) {
        stream.next_out = buffer + produced;
        stream.avail_out = (uInt)(capacity - produced);
        int status = inflate(&stream, Z_NO_FLUSH);
        produced = capacity - stream.avail_out;
        if (status == Z_STREAM_END) break;
        if (status == Z_BUF_ERROR || (status == Z_OK && stream.avail_out == 0)) {
            size_t grown = capacity * 2u;
            uint8_t *bigger = JAI_GROW_ARRAY(uint8_t, buffer, capacity, grown);
            if (bigger == NULL) {
                inflateEnd(&stream);
                JAI_FREE_ARRAY(uint8_t, buffer, capacity);
                return jaiThrow(vm.cRuntimeError, "inflate(): out of memory");
            }
            buffer = bigger;
            capacity = grown;
            continue;
        }
        if (status != Z_OK) {
            inflateEnd(&stream);
            JAI_FREE_ARRAY(uint8_t, buffer, capacity);
            return jaiThrow(vm.cValueError, "inflate(): the stream is not valid deflate data");
        }
        if (stream.avail_in == 0 && stream.avail_out > 0) break;
    }
    inflateEnd(&stream);

    ObjBytes *result = jaiBytesNew(buffer, produced);
    JAI_FREE_ARRAY(uint8_t, buffer, capacity);
    if (result == NULL) return false;
    *out = OBJ_VAL(result);
    return true;
}

void jaiRegisterCompressPrimitives(void) {
    jaiDefineNative("__prim__.deflate",  nDeflate,  2, 2);
    jaiDefineNative("__prim__.inflate",  nInflate,  1, 2);
    jaiDefineNative("__prim__.crc32",   nCrc32,   1, 1);
    jaiDefineNative("__prim__.adler32",  nAdler32,  1, 1);
}
