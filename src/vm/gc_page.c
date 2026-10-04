/* gc_page.c -- the page space: where small objects that own nothing but their
 * own block are allocated, and how the collector frees them without visiting
 * them. The design and its reasons are in gc.h above JaiPage. */

#include <stdlib.h>
#include <sys/mman.h>

#include "vm/gc.h"
#include "vm/object/object.h"
#include "vm/vm.h"

/* The reserved range. Address space only: a page is touched, and so costs
 * memory, when the allocator first carves it. 32 GiB of small objects is far
 * past anything this VM can mark; past it the allocator just declines and
 * objects go down the list path again. */
#define JAI_PAGE_RESERVE ((size_t)1 << 35)

/* The page header rounded up to a 128-byte line, so a 64-byte block never
 * straddles one (Apple cores have 128-byte lines). In grains. */
#define JAI_PAGE_HEADER_GRAINS                                                  \
    ((unsigned)(((sizeof(JaiPage) + 127u) & ~(size_t)127u) >> 4))

JaiPageCursor jaiPageCursor[JAI_SMALL_CLASSES + 1];
JaiPageCursor jaiPageCursorFin[JAI_SMALL_CLASSES + 1];
uint8_t       jaiPageKind[OBJ_TYPE_COUNT];
uintptr_t     jaiPageBase;
uintptr_t     jaiPageSpan;

static char    *gCarve;          /* next never-used page in the range */
static char    *gEnd;
static JaiPage *gPool;           /* empty pages, bitmaps zero, any class */
static size_t   gLiveBytes;      /* page bytes marked at the last collection */
static size_t   gInUseAtBegin;   /* page bytes in use when this one began */
static bool     gPoison;
static unsigned gPrefetch;       /* words ahead to prefetch for store; 0 = off */

/* Block starts of each class, per bitmap word. Built the first time a class
 * gets a page. */
static uint64_t gPattern[JAI_SMALL_CLASSES + 1][JAI_PAGE_WORDS];
static uint32_t gCapacity[JAI_SMALL_CLASSES + 1];   /* blocks per page; 0 = unbuilt */

static void buildPattern(unsigned cls) {
    unsigned n = 0;
    for (unsigned g = JAI_PAGE_HEADER_GRAINS; g + cls <= JAI_PAGE_GRAINS; g += cls) {
        gPattern[cls][g >> 6] |= (uint64_t)1 << (g & 63u);
        n++;
    }
    gCapacity[cls] = n;
}

void jaiPageSpaceInit(void) {
    static bool done;
    if (done) return;
    done = true;

    const char *e = getenv("JAITHON_GC_PAGES");
    if (e != NULL && e[0] == '0') return;
    /* The snapshot audit enumerates the heap by walking GCState.objects, which
     * page objects are not on. */
    if (getenv("JAITHON_SNAPSHOT_AUDIT") != NULL) return;
    gPoison = getenv("JAITHON_GC_POISON") != NULL;
    /* JAITHON_GC_PREFETCH=<words> sets how far ahead the allocator prefetches
     * for store; 0 turns it off. */
    {
        const char *pf = getenv("JAITHON_GC_PREFETCH");
        gPrefetch = pf != NULL ? (unsigned)atoi(pf) : 2u;
    }

    size_t want = JAI_PAGE_RESERVE + JAI_PAGE_BYTES;
    void *raw = mmap(NULL, want, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (raw == MAP_FAILED) return;   /* the list path serves everything */

    uintptr_t base = ((uintptr_t)raw + JAI_PAGE_BYTES - 1) &
                     ~(uintptr_t)(JAI_PAGE_BYTES - 1);
    gCarve = (char *)base;
    gEnd = (char *)base + JAI_PAGE_RESERVE;
    jaiPageBase = base;
    jaiPageSpan = JAI_PAGE_RESERVE;

    for (unsigned c = 0; c <= JAI_SMALL_CLASSES; c++) {
        jaiPageCursor[c].cls = (uint16_t)c;
        jaiPageCursorFin[c].cls = (uint16_t)c;
        jaiPageCursorFin[c].fin = 1;
    }

    /* Exactly the kinds jaiObjSoleBlock sizes: nothing to free but the block. */
    jaiPageKind[OBJ_STRING] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_STRBUF] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_BYTES] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_TUPLE] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_INSTANCE] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_ENUM_VAL] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_RANGE] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_UPVALUE] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_NATIVE] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_BOUND] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_ITER] = JAI_PAGE_PLAIN;
    jaiPageKind[OBJ_ENUM_CTOR] = JAI_PAGE_PLAIN;

    /* The common kinds that own one array or table, which jaiObjFinalize
     * frees. The rare ones stay on the list. */
    const char *f = getenv("JAITHON_GC_FIN_PAGES");
    if (!(f != NULL && f[0] == '0')) {
        jaiPageKind[OBJ_LIST] = JAI_PAGE_FIN;
        jaiPageKind[OBJ_DICT] = JAI_PAGE_FIN;
        jaiPageKind[OBJ_SET] = JAI_PAGE_FIN;
        jaiPageKind[OBJ_CLOSURE] = JAI_PAGE_FIN;
    }
}

/* Every cursor, plain then finalizing. */
#define FOR_EACH_CURSOR(pc)                                                     \
    for (unsigned fi_ = 0; fi_ < 2u; fi_++)                                     \
        for (unsigned c_ = 1; c_ <= JAI_SMALL_CLASSES; c_++)                    \
            for (JaiPageCursor *pc = fi_ ? &jaiPageCursorFin[c_]                \
                                         : &jaiPageCursor[c_];                  \
                 pc != NULL; pc = NULL)

static JaiPage *newPage(JaiPageCursor *pc) {
    const unsigned cls = pc->cls;
    JaiPage *pg = gPool;
    if (pg != NULL) {
        gPool = pg->next;
    } else {
        if (gCarve == NULL || (size_t)(gEnd - gCarve) < JAI_PAGE_BYTES) return NULL;
        pg = (JaiPage *)(void *)gCarve;   /* fresh mapping: already zero */
        gCarve += JAI_PAGE_BYTES;
    }
    if (gCapacity[cls] == 0) buildPattern(cls);
    pg->cls = cls;
    pg->liveBlocks = 0;
    pg->fin = pc->fin;
    pg->next = pc->pages;
    pc->pages = pg;
    return pg;
}

/* Gives the fast path the free blocks `m` of the current word, charging all of
 * them to jaiHeapBytes now, and returns the first. While jaiGCLimit is zero --
 * --gc-stress, a disabled collector -- it gives ONE block and stashes the rest,
 * so every allocation comes back through here and through the jaiGCWanted()
 * test its caller makes on the way; a whole word handed out would let sixteen
 * allocations pass a stress collection by. */
static void *handOut(JaiPageCursor *pc, uint64_t m) {
    const unsigned cls = pc->cls;
    uint64_t take = m;
    pc->stash = 0;
    if (JAI_UNLIKELY(jaiGCLimit == 0)) {
        take = m & (~m + 1u);
        pc->stash = m & (m - 1u);
    }
    /* In use from now: the ones never popped are taken back at the next
     * collection's start (jaiPageCollectBegin). */
    if (pc->fin) pc->page->inuse[pc->word] |= take;
    unsigned n = (unsigned)__builtin_popcountll(take);
    pc->handedOut += n;
    jaiHeapBytes += (size_t)n * cls * 16u;
    vm.allocCount += n;
    pc->freeMask = take & (take - 1u);
    return pc->wordBase + ((size_t)__builtin_ctzll(take) << 4);
}

void *jaiPageRefill(JaiPageCursor *pc) {
    if (jaiPageSpan == 0 || jaiGCInCollect) return NULL;

    const unsigned cls = pc->cls;
    if (pc->stash != 0) return handOut(pc, pc->stash);
    const uint64_t *pat = gPattern[cls];
    JaiPage *pg = pc->page;
    unsigned w = pc->word + 1u;

    for (;;) {
        if (pg != NULL) {
            for (; w < JAI_PAGE_WORDS; w++) {
                uint64_t m = pat[w] & ~pg->live[w];
                if (m != 0) {
                    pc->page = pg;
                    pc->word = w;
                    pc->wordBase = (char *)pg + ((size_t)w << 10);
                    /* Nothing reads a block before writing it, so nothing
                     * pulls the next ones into the cache ahead of the stores
                     * the way the old free-list link load did: ask for them
                     * gPrefetch words ahead. -3% cycles on alloc_churn at 2
                     * while every allocation was a call; since compiled code
                     * pops the mask inline it measures inside the noise.
                     * Tried and worse: DC ZVA of a wholly free word (+3%), a
                     * prefetch per allocation (+4%), and aiming at the next
                     * words that have free blocks rather than at w + 2 (+4%
                     * binary_trees, +7% alloc_churn). */
                    if (gPrefetch != 0 && w + gPrefetch < JAI_PAGE_WORDS) {
                        const char *ahead = pc->wordBase + ((size_t)gPrefetch << 10);
                        for (unsigned l = 0; l < 1024u; l += 128u)
                            __builtin_prefetch(ahead + l, 1, 3);
                    }
                    return handOut(pc, m);
                }
            }
        }
        /* This page is used up: the next one with room, else a new one. */
        pg = pc->nextPage;
        while (pg != NULL && pg->liveBlocks >= gCapacity[cls]) pg = pg->next;
        /* Past every full page it skipped, whether or not it found one with
         * room: leaving nextPage behind them made each new page walk the
         * whole run of full ones again, quadratic in a large live heap. */
        pc->nextPage = pg != NULL ? pg->next : NULL;
        if (pg == NULL) {
            pg = newPage(pc);
            if (pg == NULL) {
                pc->page = NULL;
                return NULL;
            }
            /* Linked at the head, behind the cursor: nextPage is unchanged. */
        }
        w = 0;
    }
}

uint64_t jaiPageUnpopped(void) {
    uint64_t n = 0;
    if (jaiPageSpan == 0) return 0;
    FOR_EACH_CURSOR(pc) n += (uint64_t)__builtin_popcountll(pc->freeMask);
    return n;
}

static void resetCursor(JaiPageCursor *pc) {
    pc->freeMask = 0;
    pc->epochHi = (uint64_t)jaiGCEpoch << 32;
    pc->stash = 0;
    pc->wordBase = NULL;
    pc->page = NULL;
    pc->nextPage = pc->pages;
    pc->handedOut = 0;
    pc->word = 0;
}

void jaiPageCollectBegin(void) {
    if (jaiPageSpan == 0) return;
    size_t inUse = gLiveBytes;
    FOR_EACH_CURSOR(pc) {
        const unsigned cls = pc->cls;
        uint64_t unused = (uint64_t)__builtin_popcountll(pc->freeMask);
        /* Handed out but never popped: no object there to finalize. */
        if (pc->fin && pc->page != NULL) pc->page->inuse[pc->word] &= ~pc->freeMask;
        uint64_t used = pc->handedOut - unused;
        inUse += (size_t)used * cls * 16u;
        /* Charged when the word was handed out, never allocated. */
        jaiHeapAccountFreed((size_t)unused * cls * 16u);
        vm.allocCount -= unused;
        /* No allocation from here until the sweep is done: refill declines
         * while jaiGCInCollect, and this makes every allocation reach it. */
        pc->freeMask = 0;
        pc->stash = 0;
        pc->handedOut = 0;
    }
    gInUseAtBegin = inUse;
}

static void poisonFree(JaiPage *pg, unsigned cls) {
    const uint64_t *pat = gPattern[cls];
    for (unsigned w = 0; w < JAI_PAGE_WORDS; w++) {
        uint64_t m = pat[w] & ~pg->live[w];
        while (m != 0) {
            unsigned g = (w << 6) + (unsigned)__builtin_ctzll(m);
            m &= m - 1;
            memset((char *)pg + ((size_t)g << 4), 0xDB, (size_t)cls << 4);
        }
    }
}

/* Frees what each dead object of a finalizing page owns -- in use, unmarked --
 * and makes the marked ones the in-use set. Before any poisoning, since it
 * reads the objects. */
static void finalizeDead(JaiPage *pg) {
    for (unsigned w = 0; w < JAI_PAGE_WORDS; w++) {
        uint64_t dead = pg->inuse[w] & ~pg->mark[w];
        pg->inuse[w] = pg->mark[w];
        while (dead != 0) {
            unsigned g = (w << 6) + (unsigned)__builtin_ctzll(dead);
            dead &= dead - 1;
            jaiObjFinalize((Obj *)(void *)((char *)pg + ((size_t)g << 4)));
        }
    }
}

size_t jaiPageCollectEnd(void) {
    if (jaiPageSpan == 0) return 0;
    size_t liveBytes = 0;
    /* Emptied pages, kept in list order -- most recently filled first, since a
     * class links each new page at its head -- and put in front of the pool as
     * one chain, so the next cycle reuses the memory it wrote last, which is
     * the memory still in cache. Pushing them one by one reversed that and had
     * every cycle start on the coldest page it owned. */
    JaiPage *emptied = NULL;
    JaiPage **emptiedTail = &emptied;
    FOR_EACH_CURSOR(pc) {
        const unsigned cls = pc->cls;
        JaiPage **link = &pc->pages;
        while (*link != NULL) {
            JaiPage *pg = *link;
            if (pg->fin) finalizeDead(pg);
            uint32_t n = 0;
            for (unsigned w = 0; w < JAI_PAGE_WORDS; w++) {
                uint64_t m = pg->mark[w];
                pg->live[w] = m;
                pg->mark[w] = 0;
                n += (uint32_t)__builtin_popcountll(m);
            }
            if (n == 0) {
                *link = pg->next;
                if (gPoison) poisonFree(pg, cls);
                pg->cls = 0;
                pg->liveBlocks = 0;
                pg->fin = 0;
                pg->next = NULL;
                *emptiedTail = pg;
                emptiedTail = &pg->next;
                continue;
            }
            pg->liveBlocks = n;
            liveBytes += (size_t)n * cls * 16u;
            if (gPoison) poisonFree(pg, cls);
            link = &pg->next;
        }
        resetCursor(pc);
    }
    if (emptied != NULL) {
        *emptiedTail = gPool;
        gPool = emptied;
    }
    gLiveBytes = liveBytes;
    return gInUseAtBegin > liveBytes ? gInUseAtBegin - liveBytes : 0;
}

size_t jaiPageSpaceReset(void) {
    if (jaiPageSpan == 0) return 0;
    jaiPageCollectBegin();
    size_t inUse = gInUseAtBegin;
    FOR_EACH_CURSOR(pc) {
        JaiPage *pg = pc->pages;
        while (pg != NULL) {
            JaiPage *next = pg->next;
            memset(pg->mark, 0, sizeof pg->mark);
            if (pg->fin) finalizeDead(pg);   /* nothing marked: all of them */
            memset(pg->live, 0, sizeof pg->live);
            memset(pg->inuse, 0, sizeof pg->inuse);
            pg->cls = 0;
            pg->fin = 0;
            pg->liveBlocks = 0;
            pg->next = gPool;
            gPool = pg;
            pg = next;
        }
        pc->pages = NULL;
        resetCursor(pc);
    }
    gLiveBytes = 0;
    gInUseAtBegin = 0;
    return inUse;
}
