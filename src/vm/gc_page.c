/* gc_page.c -- the page space: where small objects that own nothing but their
 * own block are allocated, and how the collector frees them without visiting
 * them. The design and its reasons are in gc.h above JaiPage. */

#include <stdlib.h>
#include <sys/mman.h>

#include "vm/gc.h"

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
bool          jaiPageKind[OBJ_TYPE_COUNT];
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

    /* Exactly the kinds jaiObjSoleBlock sizes: nothing to free but the block. */
    jaiPageKind[OBJ_STRING] = true;
    jaiPageKind[OBJ_STRBUF] = true;
    jaiPageKind[OBJ_BYTES] = true;
    jaiPageKind[OBJ_TUPLE] = true;
    jaiPageKind[OBJ_INSTANCE] = true;
    jaiPageKind[OBJ_ENUM_VAL] = true;
    jaiPageKind[OBJ_RANGE] = true;
    jaiPageKind[OBJ_UPVALUE] = true;
    jaiPageKind[OBJ_NATIVE] = true;
    jaiPageKind[OBJ_BOUND] = true;
    jaiPageKind[OBJ_ITER] = true;
    jaiPageKind[OBJ_ENUM_CTOR] = true;
}

static JaiPage *newPage(unsigned cls) {
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
    JaiPageCursor *pc = &jaiPageCursor[cls];
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
static void *handOut(JaiPageCursor *pc, unsigned cls, uint64_t m) {
    uint64_t take = m;
    pc->stash = 0;
    if (JAI_UNLIKELY(jaiGCLimit == 0)) {
        take = m & (~m + 1u);
        pc->stash = m & (m - 1u);
    }
    unsigned n = (unsigned)__builtin_popcountll(take);
    pc->handedOut += n;
    jaiHeapBytes += (size_t)n * cls * 16u;
    pc->freeMask = take & (take - 1u);
    return pc->wordBase + ((size_t)__builtin_ctzll(take) << 4);
}

void *jaiPageRefill(unsigned cls) {
    if (jaiPageSpan == 0 || jaiGCInCollect) return NULL;

    JaiPageCursor *pc = &jaiPageCursor[cls];
    if (pc->stash != 0) return handOut(pc, cls, pc->stash);
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
                     * gPrefetch words ahead. -3% cycles on alloc_churn at 2;
                     * DC ZVA of a wholly free word instead measured +3%. */
                    if (gPrefetch != 0 && w + gPrefetch < JAI_PAGE_WORDS) {
                        const char *ahead = pc->wordBase + ((size_t)gPrefetch << 10);
                        for (unsigned l = 0; l < 1024u; l += 128u)
                            __builtin_prefetch(ahead + l, 1, 3);
                    }
                    return handOut(pc, cls, m);
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
            pg = newPage(cls);
            if (pg == NULL) {
                pc->page = NULL;
                return NULL;
            }
            /* Linked at the head, behind the cursor: nextPage is unchanged. */
        }
        w = 0;
    }
}

static void resetCursor(JaiPageCursor *pc) {
    pc->freeMask = 0;
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
    for (unsigned cls = 1; cls <= JAI_SMALL_CLASSES; cls++) {
        JaiPageCursor *pc = &jaiPageCursor[cls];
        uint64_t unused = (uint64_t)__builtin_popcountll(pc->freeMask);
        uint64_t used = pc->handedOut - unused;
        inUse += (size_t)used * cls * 16u;
        /* Charged when the word was handed out, never allocated. */
        jaiHeapAccountFreed((size_t)unused * cls * 16u);
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
    for (unsigned cls = 1; cls <= JAI_SMALL_CLASSES; cls++) {
        JaiPageCursor *pc = &jaiPageCursor[cls];
        JaiPage **link = &pc->pages;
        while (*link != NULL) {
            JaiPage *pg = *link;
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
    for (unsigned cls = 1; cls <= JAI_SMALL_CLASSES; cls++) {
        JaiPageCursor *pc = &jaiPageCursor[cls];
        JaiPage *pg = pc->pages;
        while (pg != NULL) {
            JaiPage *next = pg->next;
            memset(pg->mark, 0, sizeof pg->mark);
            memset(pg->live, 0, sizeof pg->live);
            pg->cls = 0;
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
