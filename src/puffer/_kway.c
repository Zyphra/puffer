/* Bounded-memory k-way merge of sorted-unique int64 files -> one sorted-unique
 * file, via a tournament LOSER TREE (Knuth TAOCP Vol.3, replacement selection)
 * over K bounded input buffers.
 *
 * The loser tree keeps, at each internal node, the index of the LOSER of the
 * comparison played there; the overall winner (current global minimum) sits at
 * ls[0]. Replacing the winner's leaf and calling `adjust` re-plays only the
 * log2(K) matches on the path to the root -- O(log K) per emitted key, one
 * comparison per level -> O(N log K) overall, each element consumed once.
 *
 * Exhausted leaves and the build sentinel are represented by a per-leaf class
 * flag (`fin`: 0 finite, 1 +inf, 2 -inf), NOT an in-band INT64 sentinel, so any
 * 64-bit key value (incl. INT64_MIN/MAX) merges correctly.
 *
 * All merge storage is malloc'd and fixed: K input buffers + 1 output buffer
 * (each `bufcap` int64) + O(K) bookkeeping -- a userspace working set bounded by
 * ram_budget and INDEPENDENT of input size (no mmap; the OS page cache for the
 * files is separate and reclaimable), and no
 * hidden stdio buffers (every stream is set _IONBF; our fread/fwrite blocks are
 * the only buffering). `bufcap` is computed and validated by the Python caller
 * (bounded_merge.py) so (K+1)*bufcap*8 + O(K) <= ram_budget; the caller stages
 * larger fan-in. This entrypoint never grows buffers itself.
 *
 * Every fread/fwrite is checked: a short read sets ferror -> error; a short
 * write, or a failed fflush/fclose/rename, aborts and removes the temp so a
 * truncated file is never published as a successful merge.
 *
 * Build: gcc -O3 -shared -fPIC -o _kway.so _kway.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Ordering over value classes: -inf(2) < finite(0) < +inf(1). Returns 1 iff
 * value(a) > value(b); equal classes that are non-finite are never ">" . */
static inline int gt(const int64_t *key, const int *fin, int a, int b) {
    int fa = fin[a], fb = fin[b];
    if (fa == fb) return (fa == 0) ? (key[a] > key[b]) : 0;
    int oa = (fa == 2) ? -1 : (fa == 1) ? 1 : 0;
    int ob = (fb == 2) ? -1 : (fb == 1) ? 1 : 0;
    return oa > ob;
}

/* Leaf s has a new value; re-play matches up to the root. The loser stays at
 * each internal node; the winner bubbles up and lands in ls[0]. */
static inline void adjust(const int64_t *key, const int *fin, int *ls, int K, int s) {
    for (int t = (s + K) / 2; t > 0; t /= 2) {
        if (gt(key, fin, s, ls[t])) { int tmp = s; s = ls[t]; ls[t] = tmp; }
    }
    ls[0] = s;
}

/* Merge with a caller-validated per-buffer capacity `bufcap` (int64 keys): the
 * caller guarantees (K+1)*bufcap*8 + O(K) <= ram_budget. Returns emitted unique
 * count (>=0); -1 on I/O/alloc error; -2 if bufcap < 1 (budget too small for
 * this fan-in -- caller must stage). On success with count>0, out_min and out_max
 * hold the global min/max. */
int64_t kway_merge_unique(const char **paths, int K, const char *out_path,
                          int64_t bufcap, int64_t *out_min, int64_t *out_max) {
    if (K <= 0) return -1;
    if (bufcap < 1) return -2;                   /* no silent floor: caller budgets */

    FILE **f = NULL; int64_t **buf = NULL;
    int64_t *blen = NULL, *bpos = NULL, *key = NULL, *obuf = NULL;
    int *fin = NULL, *ls = NULL;
    FILE *fo = NULL;
    char tmp[8192];
    int64_t total = 0, oc = 0, last = 0, mn = 0, mx = 0;
    int have_last = 0, have_mm = 0, have_tmp = 0;
    int64_t ret = -1;                            /* default: error */

    f    = (FILE **)   calloc((size_t)K, sizeof(FILE *));
    buf  = (int64_t **)calloc((size_t)K, sizeof(int64_t *));
    blen = (int64_t *) calloc((size_t)K, sizeof(int64_t));
    bpos = (int64_t *) calloc((size_t)K, sizeof(int64_t));
    key  = (int64_t *) malloc((size_t)(K + 1) * sizeof(int64_t));
    fin  = (int *)     malloc((size_t)(K + 1) * sizeof(int));
    ls   = (int *)     malloc((size_t)K * sizeof(int));
    obuf = (int64_t *) malloc((size_t)bufcap * sizeof(int64_t));
    if (!f || !buf || !blen || !bpos || !key || !fin || !ls || !obuf) goto done;
    for (int i = 0; i < K; i++) {
        buf[i] = (int64_t *)malloc((size_t)bufcap * sizeof(int64_t));
        if (!buf[i]) goto done;
    }

    for (int i = 0; i < K; i++) {
        f[i] = fopen(paths[i], "rb");
        if (!f[i]) goto done;
        setvbuf(f[i], NULL, _IONBF, 0);          /* no hidden stdio buffer */
        blen[i] = (int64_t)fread(buf[i], sizeof(int64_t), (size_t)bufcap, f[i]);
        if (ferror(f[i])) goto done;
        bpos[i] = 0;
        if (blen[i] > 0) { key[i] = buf[i][0]; fin[i] = 0; }
        else             { key[i] = 0;         fin[i] = 1; }  /* +inf: empty */
    }
    key[K] = 0; fin[K] = 2;                       /* -inf build sentinel */
    for (int i = 0; i < K; i++) ls[i] = K;
    for (int s = K - 1; s >= 0; s--) adjust(key, fin, ls, K, s);

    if (snprintf(tmp, sizeof(tmp), "%s.tmp", out_path) >= (int)sizeof(tmp)) goto done;
    fo = fopen(tmp, "wb");
    if (!fo) goto done;
    have_tmp = 1;
    setvbuf(fo, NULL, _IONBF, 0);

    for (;;) {
        int w = ls[0];
        if (fin[w] != 0) break;                   /* winner is +inf -> all done */
        int64_t v = key[w];
        if (!have_last || v != last) {            /* dedup across the whole stream */
            obuf[oc++] = v; last = v; have_last = 1;
            if (!have_mm) { mn = v; have_mm = 1; }
            mx = v;
            if (oc == bufcap) {
                if ((int64_t)fwrite(obuf, sizeof(int64_t), (size_t)oc, fo) != oc) goto done;
                total += oc; oc = 0;
            }
        }
        bpos[w]++;
        if (bpos[w] >= blen[w]) {                 /* buffer drained -> refill */
            blen[w] = (int64_t)fread(buf[w], sizeof(int64_t), (size_t)bufcap, f[w]);
            if (ferror(f[w])) goto done;
            bpos[w] = 0;
        }
        if (bpos[w] < blen[w]) { key[w] = buf[w][bpos[w]]; fin[w] = 0; }
        else                   { fin[w] = 1; }    /* leaf exhausted -> +inf */
        adjust(key, fin, ls, K, w);
    }
    if (oc > 0) {
        if ((int64_t)fwrite(obuf, sizeof(int64_t), (size_t)oc, fo) != oc) goto done;
        total += oc;
    }
    if (fflush(fo) != 0) goto done;
    if (fclose(fo) != 0) { fo = NULL; goto done; }
    fo = NULL;
    if (rename(tmp, out_path) != 0) goto done;
    have_tmp = 0;
    if (have_mm) { *out_min = mn; *out_max = mx; }
    ret = total;                                  /* success */

done:
    if (fo) fclose(fo);
    if (f)   for (int i = 0; i < K; i++) if (f[i]) fclose(f[i]);
    if (buf) for (int i = 0; i < K; i++) free(buf[i]);
    free(f); free(buf); free(blen); free(bpos);
    free(key); free(fin); free(ls); free(obuf);
    if (have_tmp) remove(tmp);
    return ret;
}
