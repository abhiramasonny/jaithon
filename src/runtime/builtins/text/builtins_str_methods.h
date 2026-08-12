/* builtins_str_methods.h — what the str method files share with each other.
 *
 * The `str` type is one method table (kStrMethods, built in builtins_str.c),
 * but its ~40 methods are implemented across five translation units split by
 * sub-concern: builtins_str.c itself keeps the table, the scalar/byte-offset
 * plumbing every method needs, and the __prim__ operator surface; the other
 * four hold case/classification (builtins_str_case.c), search and matching
 * (builtins_str_search.c), split/join/format (builtins_str_split.c), and
 * conversions to and from other representations (builtins_str_convert.c).
 *
 * This header is what lets that split compile: forward declarations for the
 * handful of helpers genuinely shared between two or more of those files, and
 * for every JaiNativeFn that kStrMethods or jaiRegisterStringPrimitives binds
 * from outside builtins_str.c. It is not a public interface — builtins_str.h
 * is the contract with builtins_format.c and builtins_bytes.c, and nothing
 * declared only here should ever need to cross that line. Only the five
 * str-method files include this header.
 */
#ifndef JAI_BUILTINS_STR_METHODS_H
#define JAI_BUILTINS_STR_METHODS_H

#include "runtime/builtins/text/builtins_str.h"

/* --- shared plumbing (defined in builtins_str.c) -------------------- */

/* A bound native receives the receiver as args[0]; every method below reads
 * its declared arguments from args[1] onwards. */
bool strReceiver(int argc, Value *args, const char *method, ObjString **out);

/* An absent or null argument yields NULL, read as "use the default set". */
bool optStr(int argc, Value *args, int slot, const char *method,
           const char *what, ObjString **out);

/* Resolves an optional [start, end) scalar window: negatives count from the
 * end, out-of-range bounds clamp, an inverted window comes back empty. */
void resolveWindow(ObjString *s, int64_t start, int64_t end,
                   size_t *outStart, size_t *outEnd);

/* Scalar index of a byte offset that sits on a scalar boundary. */
size_t scalarIndexOf(ObjString *s, size_t offset);

/* Last occurrence of `needle` in `hay`, or NULL — jaiStrFindBytes's mirror. */
const char *rfindBytes(const char *hay, size_t hayLen, const char *needle,
                       size_t needleLen);

/* --- shared plumbing (defined in builtins_str_case.c) --------------- */

/* True for any Unicode scalar that isSpaceCp treats as whitespace. Shared
 * with split's whitespace-mode split()/rsplit(), which draws the same
 * boundary strip() does. */
bool isSpaceCp(int32_t c);

/* --- shared plumbing (defined in builtins_str_split.c) --------------- */

/* Appends one substring to a list under construction as a fresh str. The
 * list is rooted by the caller. Shared with convert's chars(), which is
 * "split on every scalar boundary" in every way that matters here. */
bool pushSlice(ObjList *list, const char *chars, size_t length);

/* --- str methods defined in builtins_str_case.c ---------------------- */

bool strUpper(int argc, Value *args, Value *out);
bool strLower(int argc, Value *args, Value *out);
bool strTitle(int argc, Value *args, Value *out);
bool strCapitalize(int argc, Value *args, Value *out);
bool strStrip(int argc, Value *args, Value *out);
bool strLstrip(int argc, Value *args, Value *out);
bool strRstrip(int argc, Value *args, Value *out);
bool strIsDigit(int argc, Value *args, Value *out);
bool strIsAlpha(int argc, Value *args, Value *out);
bool strIsAlnum(int argc, Value *args, Value *out);
bool strIsSpace(int argc, Value *args, Value *out);
bool strIsUpper(int argc, Value *args, Value *out);
bool strIsLower(int argc, Value *args, Value *out);
bool strPadLeft(int argc, Value *args, Value *out);
bool strPadRight(int argc, Value *args, Value *out);
bool strCenter(int argc, Value *args, Value *out);
bool strRepeat(int argc, Value *args, Value *out);

/* --- str methods defined in builtins_str_search.c --------------------- */

bool strFind(int argc, Value *args, Value *out);
bool strRfind(int argc, Value *args, Value *out);
bool strIndex(int argc, Value *args, Value *out);
bool strCount(int argc, Value *args, Value *out);
bool strStartsWith(int argc, Value *args, Value *out);
bool strEndsWith(int argc, Value *args, Value *out);
bool strContains(int argc, Value *args, Value *out);

/* --- str methods defined in builtins_str_split.c ---------------------- */

bool strSplit(int argc, Value *args, Value *out);
bool strRsplit(int argc, Value *args, Value *out);
bool strSplitlines(int argc, Value *args, Value *out);
bool strJoin(int argc, Value *args, Value *out);
bool strReplace(int argc, Value *args, Value *out);
bool strFormat(int argc, Value *args, Value *out);

/* --- str methods defined in builtins_str_convert.c --------------------- */

bool strChars(int argc, Value *args, Value *out);
bool strBytes(int argc, Value *args, Value *out);
bool strCodePoints(int argc, Value *args, Value *out);
bool strToBytes(int argc, Value *args, Value *out);
bool strParseInt(int argc, Value *args, Value *out);
bool strParseFloat(int argc, Value *args, Value *out);
bool strToInt(int argc, Value *args, Value *out);
bool strToFloat(int argc, Value *args, Value *out);
bool strToStr(int argc, Value *args, Value *out);

/* --- __prim__ functions defined in builtins_str_convert.c --------------- */

/* jaiRegisterStringPrimitives (builtins_str.c) binds these four alongside the
 * ones it defines itself; they live in builtins_str_convert.c because each is
 * a codepoint or bytes conversion, the same concern as str's own chars(),
 * code_points() and to_bytes(). */
bool primStrFromCodepoint(int argc, Value *args, Value *out);
bool primStrToCodepoint(int argc, Value *args, Value *out);
bool primStrEncode(int argc, Value *args, Value *out);
bool primStrDecode(int argc, Value *args, Value *out);

#endif /* JAI_BUILTINS_STR_METHODS_H */
