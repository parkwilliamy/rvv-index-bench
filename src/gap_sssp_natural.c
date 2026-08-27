/*
 * GAP_SSSP_NATURAL — delta-stepping SSSP, shape-faithful
 * vectorization of the original GAPBS serial kernel.
 *
 * Counterpart experiment to gap_sssp.c: that port spills the ids,
 * weights, AND gathered distances of every strip to stack buffers
 * and re-runs the whole compare scalar per lane. This variant
 * vectorizes the ORIGINAL RelaxEdges shape the way an engineer
 * would, with no restructuring in favor of indexed ops:
 *   - the {v, w} pair loads stay the two stride-8 vlse32 (that IS
 *     the natural form of GAPBS's interleaved NodeWeight layout);
 *   - new_dist = dist[u] + w is computed in vector (vadd.vx);
 *   - the relax test nd < dist[v] is a vector compare against the
 *     GATHERED distances — mask in a register, not a buffer;
 *   - vcpop == 0 skips the strip outright (the dominant case late in
 *     the search: dist[] only decreases, so a gathered distance that
 *     already beats nd proves the serial body would skip too);
 *   - only surviving strips spill (ids + nd + the mask bits), and
 *     only surviving LANES run the serial update body, which
 *     re-checks against the live dist[v] — bucket pushes reallocate
 *     memory and lanes may conflict on v, so that part is serial in
 *     any honest vectorization.
 * The finding: the dist[] gather SURVIVES natural vectorization —
 * the relax test is indexed by data — but it is consumed by a vector
 * compare instead of being spilled, and most strips retire with no
 * scalar work at all.
 *
 * Semantics follow GAPBS sssp.cc (Meyer & Sanders delta-stepping),
 * serialized, with bucket fusion — identical driver to gap_sssp.c.
 * Upstream indirect access: `dist[wn.v]` — gapbs src/sssp.cc:72
 * (RelaxEdges, lines 68-85). The prefilter is provably exact (see
 * above), and survivors run the exact serial body in lane order, so
 * dist[], the buckets, and their push order are bit-identical to the
 * scalar build (and to gap_sssp).
 *
 * Why the compiler cannot vectorize it (same blocker as gap_sssp,
 * verified on gapbs with gcc -O3 -ftree-vectorize: zero kernel
 * gathers): RelaxEdges mixes the relax compare with growable-bucket
 * pushes (allocation in the loop) and a data-dependent store.
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
  int32_t vbuf[VBUF_LANES], ndbuf[VBUF_LANES];
  uint8_t mbuf[VBUF_LANES / 8];
  for (int32_t k = 0; k < deg; ) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
    vint32m1_t v_ids = __riscv_vlse32_v_i32m1(
        &wn[2 * k], 2 * sizeof(int32_t), vl);
    vint32m1_t v_nd = __riscv_vadd_vx_i32m1(
        __riscv_vlse32_v_i32m1(&wn[2 * k + 1], 2 * sizeof(int32_t), vl),
        du, vl);
    vint32m1_t v_dist = __riscv_vluxei32_v_i32m1(
        dist,
        __riscv_vsll_vx_u32m1(
            __riscv_vreinterpret_v_i32m1_u32m1(v_ids), 2, vl),
        vl);
    vbool32_t relax = __riscv_vmslt_vv_i32m1_b32(v_nd, v_dist, vl);
    if (__riscv_vcpop_m_b32(relax, vl) == 0) {
      k += (int32_t)vl;          // whole strip provably a no-op
      continue;
    }
    __riscv_vse32_v_i32m1(vbuf, v_ids, vl);
    __riscv_vse32_v_i32m1(ndbuf, v_nd, vl);
    __riscv_vsm_v_b32(mbuf, relax, vl);
    for (size_t j = 0; j < vl; j++) {
      if (!((mbuf[j >> 3] >> (j & 7)) & 1)) continue;
      int32_t v = vbuf[j], nd = ndbuf[j];
      if (nd < dist[v]) {        // live value, exact serial body
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
            "usage: gap_sssp_natural <nodes> <edges> <source> <delta> <seed>\n"
            "       gap_sssp_natural -f <graph.bfs> <source> <delta> <seed>\n");
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
  print_checksum_i32("gap_sssp_natural", dist, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
