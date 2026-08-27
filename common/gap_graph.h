// Deterministic synthetic CSR graphs for the GAP-derived kernels.
// Undirected uniform multigraphs: E distinct draws of (u != v), each
// inserted in both directions; CSR built with a stable counting sort,
// so neighbor order is the draw order — identical on every host and
// build. Duplicate parallel edges are allowed (harmless to all
// kernels here); self loops are resampled away. The weighted variant
// stores GAPBS-style INTERLEAVED {v, w} int32 pairs (w uniform
// 1..255, same weight both directions), which is what makes the sssp
// index stream a strided load, as in GAPBS.
#ifndef RVV_INDEX_BENCH_GAP_GRAPH_H
#define RVV_INDEX_BENCH_GAP_GRAPH_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bench_common.h"

typedef struct {
  int32_t n;      // nodes
  int64_t m;      // directed edges (2E)
  int32_t *off;   // n+1 offsets (in edges, or in pairs when weighted)
  int32_t *dat;   // neighbors, or interleaved {v,w} pairs
} GapCsr;

static inline int gap_gen(int32_t n, int64_t e_undirected, int weighted,
                          uint64_t *seed, GapCsr *g) {
  int64_t m = 2 * e_undirected;
  int32_t *eu = (int32_t *)malloc(e_undirected * sizeof(int32_t));
  int32_t *ev = (int32_t *)malloc(e_undirected * sizeof(int32_t));
  int32_t *ew = weighted ? (int32_t *)malloc(e_undirected * sizeof(int32_t))
                         : NULL;
  int32_t *off = (int32_t *)calloc((size_t)n + 2, sizeof(int32_t));
  int32_t *dat =
      (int32_t *)malloc((size_t)m * (weighted ? 2 : 1) * sizeof(int32_t));
  if (!eu || !ev || !off || !dat || (weighted && !ew)) return 0;
  for (int64_t i = 0; i < e_undirected; i++) {
    int32_t u, v;
    do {
      u = (int32_t)sm64_range(seed, (uint32_t)n);
      v = (int32_t)sm64_range(seed, (uint32_t)n);
    } while (u == v);
    eu[i] = u; ev[i] = v;
    if (weighted) ew[i] = 1 + (int32_t)sm64_range(seed, 255);
  }
  for (int64_t i = 0; i < e_undirected; i++) {
    off[eu[i] + 1]++; off[ev[i] + 1]++;
  }
  for (int32_t u = 0; u < n; u++) off[u + 1] += off[u];
  int32_t *cur = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (!cur) return 0;
  for (int32_t u = 0; u < n; u++) cur[u] = off[u];
  const int stride = weighted ? 2 : 1;
  for (int64_t i = 0; i < e_undirected; i++) {
    int64_t p = (int64_t)cur[eu[i]]++ * stride;
    dat[p] = ev[i];
    if (weighted) dat[p + 1] = ew[i];
    p = (int64_t)cur[ev[i]]++ * stride;
    dat[p] = eu[i];
    if (weighted) dat[p + 1] = ew[i];
  }
  free(eu); free(ev); free(ew); free(cur);
  g->n = n; g->m = m; g->off = off; g->dat = dat;
  return 1;
}


/*
 * File loader: RiVec `_bfs` out-CSR images (`.bfs`, produced by
 * rivec/_bfs/el2bfs.c from a SNAP edge list). Layout is
 *     uint64 magic = 0x315253434F534642 ("BFSOCRS1")
 *     uint64 N, NZ
 *     uint64 ia[N+1], ja[NZ]
 * (see rivec/_bfs/src/graph_format.h).
 *
 * The image is a DIRECTED out-CSR; every kernel in this suite treats
 * the graph as undirected (bfs's bottom-up step, cc/cc_sv hooking and
 * pr/bc's pull direction all read `off/dat` as both directions), so
 * each stored edge (u,v) is emitted BOTH ways here — exactly what
 * gap_gen() does for the synthetic graphs, and what GAPBS does for
 * its undirected kernels via the `U` graph variants. Parallel edges
 * are kept (all kernels tolerate them; no dedup pass is needed).
 *
 * Weights (`weighted`, for sssp) are synthesized deterministically
 * from the seed since SNAP graphs carry none, using the same
 * uniform 1..255 convention as gap_gen and GAPBS's converter; both
 * directions of an edge get the same weight.
 *
 * Indices are converted to int32 on load; N is checked against
 * INT32_MAX. Everything happens before ROI_BEGIN().
 */
#define GAP_BFS_OUT_MAGIC 0x315253434F534642ULL

static inline int gap_load(const char *path, int weighted, uint64_t *seed,
                           GapCsr *g) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); return 0; }
    uint64_t magic = 0, hdr[2];
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != GAP_BFS_OUT_MAGIC) {
        fprintf(stderr, "error: '%s' is not a .bfs out-CSR image\n", path);
        fclose(f); return 0;
    }
    if (fread(hdr, sizeof(uint64_t), 2, f) != 2) {
        fprintf(stderr, "error: truncated header in '%s'\n", path);
        fclose(f); return 0;
    }
    const uint64_t N = hdr[0], NZ = hdr[1];
    if (N > 0x7fffffffULL) {
        fprintf(stderr, "error: '%s' has too many vertices for int32\n", path);
        fclose(f); return 0;
    }
    uint64_t *ia = (uint64_t *)malloc((size_t)(N + 1) * sizeof(uint64_t));
    uint64_t *ja = (uint64_t *)malloc((size_t)NZ * sizeof(uint64_t));
    if (!ia || !ja) { fclose(f); return 0; }
    if (fread(ia, sizeof(uint64_t), N + 1, f) != N + 1 ||
        fread(ja, sizeof(uint64_t), NZ, f) != NZ) {
        fprintf(stderr, "error: truncated arrays in '%s'\n", path);
        fclose(f); free(ia); free(ja); return 0;
    }
    fclose(f);

    const int32_t n = (int32_t)N;
    const int64_t m = 2 * (int64_t)NZ;          /* symmetrized */
    const int stride = weighted ? 2 : 1;
    int32_t *off = (int32_t *)calloc((size_t)n + 2, sizeof(int32_t));
    int32_t *dat = (int32_t *)malloc((size_t)m * stride * sizeof(int32_t));
    int32_t *cur = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *w = weighted ? (int32_t *)malloc((size_t)NZ * sizeof(int32_t))
                          : NULL;
    if (!off || !dat || !cur || (weighted && !w)) { return 0; }
    if (weighted)
        for (uint64_t k = 0; k < NZ; k++)
            w[k] = 1 + (int32_t)sm64_range(seed, 255);

    for (uint64_t u = 0; u < N; u++)
        for (uint64_t k = ia[u]; k < ia[u + 1]; k++) {
            off[u + 1]++;
            off[ja[k] + 1]++;
        }
    for (int32_t u = 0; u < n; u++) off[u + 1] += off[u];
    for (int32_t u = 0; u < n; u++) cur[u] = off[u];
    for (uint64_t u = 0; u < N; u++)
        for (uint64_t k = ia[u]; k < ia[u + 1]; k++) {
            const int32_t v = (int32_t)ja[k];
            int64_t p = (int64_t)cur[u]++ * stride;
            dat[p] = v;
            if (weighted) dat[p + 1] = w[k];
            p = (int64_t)cur[v]++ * stride;
            dat[p] = (int32_t)u;
            if (weighted) dat[p + 1] = w[k];
        }
    free(ia); free(ja); free(cur); free(w);
    g->n = n; g->m = m; g->off = off; g->dat = dat;
    return 1;
}

#endif // RVV_INDEX_BENCH_GAP_GRAPH_H
