/* The seed's payload, embedded straight from boot/seed.bin.
 *
 * The images used to live inside boot/seed.c as C data -- first a byte array,
 * then base64 string literals. Both are the same mistake in different clothes:
 * a source file that is not source. The byte array cost six characters per
 * byte, base64 cost four per three, and neither is something anyone would ever
 * read or edit.
 *
 * `.incbin` puts the file in the binary verbatim, so the data is a data file
 * and boot/seed.c goes back to being code: an index, a lookup and a decoder.
 * Nothing is generated at build time and nothing is read at run time -- the
 * bytes are linked in, which is what lets a fresh clone build offline and run
 * with no files beside the binary.
 *
 * This is a `.S` rather than a `.s` so the C preprocessor runs first, which is
 * how the two spellings below are chosen. Mach-O prefixes C symbols with an
 * underscore and ELF does not, and the two put read-only data in differently
 * named sections; everything else here is the same on both. */

#if defined(__APPLE__)
#  define JAI_SYM(name) _##name
    .section __TEXT,__const
#else
#  define JAI_SYM(name) name
    .section .rodata
#endif

    .globl JAI_SYM(jaiSeedBlob)
    .globl JAI_SYM(jaiSeedBlobEnd)
    .p2align 4

JAI_SYM(jaiSeedBlob):
    .incbin "boot/seed.bin"
JAI_SYM(jaiSeedBlobEnd):

/* A zero byte past the end so nothing can walk off the section, and so a build
 * whose seed.bin is empty still links rather than failing on an absent symbol. */
    .byte 0

#if defined(__linux__)
/* Mark the stack non-executable; without this GNU ld infers an executable
 * stack from any assembly object and the whole binary inherits it. */
    .section .note.GNU-stack,"",%progbits
#endif
