/* The f-string memo's tables (value.c), driven the way compiled code drives
 * them: probe the site's entry for (pre, n, post) exactly as
 * emitFmtMemoProbe does, and on a miss file the answer with jaiFmtMemoFill,
 * as the fill stub does after an intern hit. C rather than .jai because what
 * is tested is how often a probe hits, which no program can see -- a memo
 * that misses every time gives the right answer, slower than none at all.
 *
 * The objects are addresses only: nothing here dereferences pre, post or s.
 * Two runs a few bytes apart is how a compiled loop's literals come out of
 * the allocator, and the shared table indexed by n ^ pre >> 4 mapped two
 * such sites over the same ints onto the same slots, so that every probe of
 * either missed (0.86x against no memo at all). */
#include <stdio.h>
#include <stdlib.h>

#include "vm/vm.h"

static int failures = 0;

static void expect(int cond, const char *what) {
    if (!cond) { printf("FAIL: %s\n", what); failures++; }
}

static ObjString *probe(const JaiFmtSite *site, Obj *pre, int64_t n,
                        Obj *post) {
    if (site->entries == NULL) return NULL;
    const JaiFmtMemoEntry *e =
        &site->entries[jaiFmtMemoIndex(pre, n, post, site->mask)];
    if (e->pre == pre && e->post == post && e->n == n) return e->s;
    return NULL;
}

/* A string for (pre, n): any address distinct per key will do. */
static ObjString *answer(Obj *pre, int64_t n) {
    return (ObjString *)(uintptr_t)(((uintptr_t)pre << 20) + (uintptr_t)n * 16u + 0x1000u);
}

/* Through `site` for each pre in turn, n over [0, keys), `passes` times;
 * the hits of the last pass. */
static long run(JaiFmtSite **sites, Obj **pres, int count, int64_t keys,
                int passes) {
    long hits = 0;
    for (int p = 0; p < passes; p++) {
        hits = 0;
        for (int64_t n = 0; n < keys; n++) {
            for (int i = 0; i < count; i++) {
                ObjString *s = probe(sites[i], pres[i], n, NULL);
                if (s != NULL) {
                    if (s != answer(pres[i], n)) {
                        expect(0, "a hit is the answer filed for its key");
                    }
                    hits++;
                } else {
                    jaiFmtMemoFill(sites[i], pres[i], n, NULL,
                                   answer(pres[i], n));
                }
            }
        }
    }
    return hits;
}

int main(void) {
    static char runs[256] __attribute__((aligned(16)));
    Obj *user = (Obj *)(void *)&runs[64];
    Obj *order = (Obj *)(void *)&runs[96];
    Obj *item = (Obj *)(void *)&runs[128];

    /* A new site is off: no table, and the probe reads NULL. */
    JaiFmtSite *fresh = jaiFmtSiteNew();
    expect(fresh != NULL && fresh->entries == NULL, "a new site is off");
    expect(fresh->leaf == (void *)&jaiValueFormatIntLeafMemo,
           "a site calls the memo leaf");

    /* Two and three sites over the same ints, interleaved: every one keeps
     * its own entries. */
    {
        JaiFmtSite *sites[3] = {jaiFmtSiteNew(), jaiFmtSiteNew(),
                                jaiFmtSiteNew()};
        Obj *pres[3] = {user, order, item};
        long hits = run(sites, pres, 2, 5000, 8);
        expect(hits == 2 * 5000, "two sites on one int range hit every time");
        JaiFmtSite *more[3] = {jaiFmtSiteNew(), jaiFmtSiteNew(),
                               jaiFmtSiteNew()};
        hits = run(more, pres, 3, 5000, 8);
        expect(hits == 3 * 5000, "three sites on one int range hit every time");
        expect(sites[0]->entries != NULL && sites[0]->budget > 0,
               "a fill turns a site on with a budget");
        expect(sites[0]->mask + 1u <= 8192u,
               "5000 dense keys need no more than 8192 entries");
    }

    /* One site whose run changes call to call -- `f"{kind}{n}"` -- between
     * two runs at nearby addresses. The CRC puts the two runs' ints at
     * unrelated offsets, so they share slots only where those ranges happen
     * to overlap; averaged over eight pairs most keys keep their entry.
     * n ^ pre >> 4 put them on the same slots every time. Two collections
     * first, to give back the room the sites above hold. */
    jaiFmtMemoClear();
    jaiFmtMemoClear();
    {
        static char many[8 * 64] __attribute__((aligned(16)));
        long hits = 0;
        for (int k = 0; k < 8; k++) {
            JaiFmtSite *one = jaiFmtSiteNew();
            JaiFmtSite *sites[2] = {one, one};
            Obj *pres[2] = {(Obj *)(void *)&many[k * 64],
                            (Obj *)(void *)&many[k * 64 + 16 + 16 * (k % 3)]};
            hits += run(sites, pres, 2, 1000, 12);
        }
        printf("  two runs through one site: %ld of %d hit\n", hits,
               8 * 2 * 1000);
        expect(hits >= 8 * 2 * 1000 * 6 / 10,
               "two runs through one site mostly hit");
    }

    /* The all-absent key of `f"{n}"`, against an empty slot: no match. */
    jaiFmtMemoClear();
    jaiFmtMemoClear();
    {
        JaiFmtSite *bare = jaiFmtSiteNew();
        jaiFmtMemoFill(bare, NULL, 5, NULL, answer(NULL, 5));
        expect(probe(bare, NULL, 5, NULL) == answer(NULL, 5), "f\"{n}\" hits");
        expect(probe(bare, NULL, 0, NULL) == NULL, "n = 0 is not an empty slot");
        expect(probe(bare, NULL, 6, NULL) == NULL, "n = 6 was never filed");
    }

    /* Two collections with nothing filed in between give every table back,
     * and with them the room the sites above took. */
    jaiFmtMemoClear();
    jaiFmtMemoClear();
    {
        JaiFmtSite *sites[1] = {jaiFmtSiteNew()};
        Obj *pres[1] = {item};
        run(sites, pres, 1, 50000, 3);
        expect(sites[0]->mask + 1u >= 32768u,
               "a site grows into the room idle sites gave back");
        jaiFmtMemoClear();
        expect(sites[0]->entries != NULL, "a used site keeps its table");
        jaiFmtMemoClear();
        expect(sites[0]->entries == NULL && sites[0]->table == NULL,
               "an idle site gives its table back and is off");
    }

    /* A collection empties every table, and the sites stay on. */
    {
        JaiFmtSite *sites[1] = {jaiFmtSiteNew()};
        Obj *pres[1] = {user};
        run(sites, pres, 1, 40, 2);
        expect(probe(sites[0], user, 7, NULL) != NULL, "filed before the clear");
        jaiFmtMemoClear();
        expect(probe(sites[0], user, 7, NULL) == NULL, "gone after the clear");
        expect(sites[0]->entries != NULL, "still on after the clear");
    }

    /* A site's table stops growing at the intern soft cap. */
    {
        JaiFmtSite *big = jaiFmtSiteNew();
        for (int64_t n = 0; n < 200000; n++) {
            jaiFmtMemoFill(big, user, n, NULL, answer(user, n));
        }
        expect(big->mask + 1u <= (uint64_t)JAI_INTERN_SOFT_CAP,
               "a table is at most the intern soft cap");
    }

    if (failures == 0) printf("fmt_memo: ok\n");
    return failures == 0 ? 0 : 1;
}
