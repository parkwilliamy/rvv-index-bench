/*
 * GAP_SSSP — delta-stepping SSSP with a gather-prefiltered relax
 * scan.
 *
 * Semantics follow GAPBS sssp.cc (Meyer & Sanders delta-stepping),
 * serialized: a bucketed frontier, RelaxEdges over each settled
 * vertex, bucket fusion below the size threshold. Upstream indirect
 * access: the distance read `dist[wn.v]` per weighted neighbor —
 * gapbs src/sssp.cc:72 (RelaxEdges, lines 68-85). Edges are stored
 * as GAPBS-style interleaved {v, w} int32 pairs, so the index stream
 * itself is a strided load.
 *
 * RVV formulation (ported from the ~/gapbs vectorization): two
 * stride-8 vlse32 loads pull the ids and weights of a strip, and a
 * vluxei32 gather of dist[] acts as a PREFILTER: dist[] only ever
 * decreases, so a gathered (>= current) distance that already beats
 * new_dist proves the serial body would skip too — never a missed
 * relaxation. Surviving lanes re-read dist[v] and run the exact
 * serial update, so the dist array, the bucket contents, and their
 * order are bit-identical to the scalar build.
 *
 * Why the compiler cannot vectorize it (verified on gapbs with gcc
 * -O3 -ftree-vectorize: zero kernel gathers): RelaxEdges mixes the
 * relax compare with growable-bucket pushes (memory allocation in
 * the loop) and a retry loop — dead on arrival for a vectorizer.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "bench_common.h"
#include "gap_graph.h"
#include "roi.h"

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#endif

#define DIST_INF (INT32_MAX / 2)
#define BIN_SIZE_THRESHOLD 1000

typedef struct { int32_t *v; int64_t n, cap; } Bin;

static void bin_push(Bin *b, int32_t x) {
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 64;
    b->v = (int32_t *)realloc(b->v, (size_t)b->cap * sizeof(int32_t));
  }
  b->v[b->n++] = x;
}

typedef struct { Bin *bins; int64_t nbins, cap; } Bins;

static Bin *bins_get(Bins *bs, int64_t i) {
  if (i >= bs->cap) {
    int64_t ncap = bs->cap ? bs->cap : 16;
    while (ncap <= i) ncap *= 2;
    bs->bins = (Bin *)realloc(bs->bins, (size_t)ncap * sizeof(Bin));
    memset(&bs->bins[bs->cap], 0,
           (size_t)(ncap - bs->cap) * sizeof(Bin));
    bs->cap = ncap;
  }
  if (i >= bs->nbins) bs->nbins = i + 1;
  return &bs->bins[i];
}

static void relax_scalar(const GapCsr *g, int32_t u, int32_t delta,
                         int32_t *dist, Bins *bs) {
  const int32_t *wn = &g->dat[(int64_t)g->off[u] * 2];
  const int32_t deg = g->off[u + 1] - g->off[u];
  const int32_t du = dist[u];
  for (int32_t j = 0; j < deg; j++) {
    int32_t v = wn[2 * j], w = wn[2 * j + 1];
    int32_t nd = du + w;
    if (nd < dist[v]) {
      dist[v] = nd;
      bin_push(bins_get(bs, nd / delta), v);
    }
  }
}

#ifdef USE_RISCV_VECTOR
#define VBUF_LANES 256
static void relax_rvv(const GapCsr *g, int32_t u, int32_t delta,
                      int32_t *dist, Bins *bs) {
  const int32_t *wn = &g->dat[(int64_t)g->off[u] * 2];
  const int32_t deg = g->off[u + 1] - g->off[u];
  const int32_t du = dist[u];
  int32_t vbuf[VBUF_LANES], wbuf[VBUF_LANES], dbuf[VBUF_LANES];
  for (int32_t k = 0; k < deg; ) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
    vint32m1_t v_ids = __riscv_vlse32_v_i32m1(
        &wn[2 * k], 2 * sizeof(int32_t), vl);
    vint32m1_t v_w = __riscv_vlse32_v_i32m1(
        &wn[2 * k + 1], 2 * sizeof(int32_t), vl);
    vuint32m1_t v_off = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(v_ids), 2, vl);
    __riscv_vse32_v_i32m1(vbuf, v_ids, vl);
    __riscv_vse32_v_i32m1(wbuf, v_w, vl);
    __riscv_vse32_v_i32m1(dbuf,
        __riscv_vluxei32_v_i32m1(dist, v_off, vl), vl);
    for (size_t j = 0; j < vl; j++) {
      int32_t nd = du + wbuf[j];
      if (nd >= dbuf[j]) continue;   // stale-filter: provably no-op
      int32_t v = vbuf[j];
      if (nd < dist[v]) {            // current value, exact serial body
        dist[v] = nd;
        bin_push(bins_get(bs, nd / delta), v);
      }
    }
    k += (int32_t)vl;
  }
}
#endif

static void delta_step(const GapCsr *g, int32_t source, int32_t delta,
                       int32_t *dist, int vec) {
  for (int32_t u = 0; u < g->n; u++) dist[u] = DIST_INF;
  dist[source] = 0;
  Bins bs = {0, 0, 0};
  int32_t *frontier = (int32_t *)malloc(sizeof(int32_t));
  int64_t fn = 1;
  frontier[0] = source;
  int64_t curr_bin = 0;
  while (1) {
    for (int64_t i = 0; i < fn; i++) {
      int32_t u = frontier[i];
      if (dist[u] >= delta * (int32_t)curr_bin) {
#ifdef USE_RISCV_VECTOR
        if (vec) relax_rvv(g, u, delta, dist, &bs);
        else relax_scalar(g, u, delta, dist, &bs);
#else
        (void)vec;
        relax_scalar(g, u, delta, dist, &bs);
#endif
      }
    }
    // bucket fusion (GAPBS kBinSizeThreshold)
    while (curr_bin < bs.nbins && bs.bins[curr_bin].n > 0 &&
           bs.bins[curr_bin].n < BIN_SIZE_THRESHOLD) {
      Bin copy = bs.bins[curr_bin];
      bs.bins[curr_bin].v = NULL;
      bs.bins[curr_bin].n = bs.bins[curr_bin].cap = 0;
      for (int64_t i = 0; i < copy.n; i++) {
        int32_t u = copy.v[i];
        if (dist[u] >= delta * (int32_t)curr_bin) {
#ifdef USE_RISCV_VECTOR
          if (vec) relax_rvv(g, u, delta, dist, &bs);
          else relax_scalar(g, u, delta, dist, &bs);
#else
          relax_scalar(g, u, delta, dist, &bs);
#endif
        }
      }
      free(copy.v);
    }
    int64_t next_bin = -1;
    for (int64_t i = curr_bin; i < bs.nbins; i++)
      if (bs.bins[i].n > 0) { next_bin = i; break; }
    if (next_bin < 0) break;
    free(frontier);
    frontier = bs.bins[next_bin].v;
    fn = bs.bins[next_bin].n;
    bs.bins[next_bin].v = NULL;
    bs.bins[next_bin].n = bs.bins[next_bin].cap = 0;
    curr_bin = next_bin + 1;
  }
  free(frontier);
  for (int64_t i = 0; i < bs.cap; i++) free(bs.bins[i].v);
  free(bs.bins);
}

int main(int argc, char **argv) {
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 6) {
    fprintf(stderr,
            "usage: gap_sssp <nodes> <edges> <source> <delta> <seed>\n"
            "       gap_sssp -f <graph.bfs> <source> <delta> <seed>\n");
    return 2;
  }
  // Graph source: synthetic (nodes/edges) or a .bfs image (-f).
  // In file mode argv[2] is the path and the remaining
  // positional args shift down by one.
  int32_t n = 0; int64_t e = 0;
  int32_t source = atoi(argv[3]);
  int32_t delta = atoi(argv[4]);
  uint64_t seed = (uint64_t)strtoull(argv[5], NULL, 0);
  if (!file_mode) {
    n = atoi(argv[1]);
    e = atoll(argv[2]);
    if (n <= 1 || e <= 0) { fprintf(stderr, "error: bad sizes\n"); return 2; }
  }
  GapCsr g;
  if (file_mode) {
    if (!gap_load(argv[2], 1, &seed, &g)) return 2;
  } else {
    if (!gap_gen(n, e, 1, &seed, &g)) { fprintf(stderr, "oom\n"); return 2; }
  }
  n = g.n;  // authoritative after either path
  if (source < 0 || source >= n) {
    fprintf(stderr, "error: source out of range [0,%d)\n", n);
    return 2;
  }
  if (delta <= 0) { fprintf(stderr, "error: bad delta\n"); return 2; }

  int32_t *dist = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (!dist || !ref) { fprintf(stderr, "oom\n"); return 2; }

  ROI_BEGIN();
  delta_step(&g, source, delta, dist, 1);
  ROI_END();

  delta_step(&g, source, delta, ref, 0);
  int pass = memcmp(dist, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_sssp", dist, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
