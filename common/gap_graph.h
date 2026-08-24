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

#endif // RVV_INDEX_BENCH_GAP_GRAPH_H
