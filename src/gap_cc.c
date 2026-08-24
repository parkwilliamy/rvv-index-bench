/*
 * GAP_CC — Afforest connected components with a gather-prefiltered
 * final link phase.
 *
 * Semantics follow GAPBS cc.cc (Sutton et al., "Optimizing Parallel
 * Graph Connectivity Computation via Subgraph Sampling", IPDPS 2018):
 * neighbor-sampling rounds link one edge per node, compression, a
 * sampled most-frequent component is skipped, and a final link phase
 * processes the remaining neighborhoods. Upstream indirect accesses:
 * the label chase inside Link (comp[u]/comp[v]/comp[high]) — gapbs
 * src/cc.cc:42-53 — driven from the final phase's neighbor scans
 * (src/cc.cc:123-147).
 *
 * RVV formulation (ported from the ~/gapbs vectorization, made
 * deterministic): the final phase gathers comp[] for each neighbor
 * BLOCK (vle32 ids -> vsll 2 -> vluxei32) and calls the scalar Link
 * only on lanes whose snapshot label differs from the current
 * comp[u]. Two nodes whose labels were equal at snapshot time are
 * already in the same set (labels only merge), so the skip never
 * loses a union. The snapshot BLOCK is fixed at 64 elements in the
 * algorithm and the scalar mirror buffers the same snapshots, so
 * both builds follow bit-identical trajectories at every VLEN.
 *
 * Why the compiler cannot vectorize it (verified on gapbs with gcc
 * -O3 -ftree-vectorize: zero kernel gathers): the loop body is a
 * call to Link, which contains a data-dependent while-loop label
 * chase with stores — opaque side effects no vectorizer touches.
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

#define SNAP_BLOCK 64
#define NEIGHBOR_ROUNDS 2
#define FREQ_SAMPLES 1024

static void link_nodes(int32_t u, int32_t v, int32_t *comp) {
  int32_t p1 = comp[u];
  int32_t p2 = comp[v];
  while (p1 != p2) {
    int32_t high = p1 > p2 ? p1 : p2;
    int32_t low = p1 + (p2 - high);
    int32_t p_high = comp[high];
    if (p_high == low) break;
    if (p_high == high) { comp[high] = low; break; }
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}

static void compress(const GapCsr *g, int32_t *comp) {
  for (int32_t u = 0; u < g->n; u++)
    while (comp[u] != comp[comp[u]])
      comp[u] = comp[comp[u]];
}

// GAPBS SampleFrequentElement with the suite's RNG (deterministic).
static int32_t sample_frequent(const int32_t *comp, int32_t n) {
  uint64_t s = 42;
  int32_t best = -1, best_cnt = 0;
  // small quadratic tally over the sample set (FREQ_SAMPLES is tiny)
  static int32_t vals[FREQ_SAMPLES], cnts[FREQ_SAMPLES];
  int nv = 0;
  for (int i = 0; i < FREQ_SAMPLES; i++) {
    int32_t c = comp[sm64_range(&s, (uint32_t)n)];
    int j;
    for (j = 0; j < nv; j++)
      if (vals[j] == c) { cnts[j]++; break; }
    if (j == nv) { vals[nv] = c; cnts[nv] = 1; nv++; }
  }
  for (int j = 0; j < nv; j++)
    if (cnts[j] > best_cnt) { best_cnt = cnts[j]; best = vals[j]; }
  return best;
}

static void fill_snapshot_scalar(const int32_t *comp, const int32_t *ids,
                                 int32_t cnt, int32_t *buf) {
  for (int32_t j = 0; j < cnt; j++)
    buf[j] = comp[ids[j]];
}

#ifdef USE_RISCV_VECTOR
static void fill_snapshot_rvv(const int32_t *comp, const int32_t *ids,
                              int32_t cnt, int32_t *buf) {
  for (int32_t i = 0; i < cnt; ) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(cnt - i));
    vuint32m1_t voff = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&ids[i], vl)),
        2, vl);
    __riscv_vse32_v_i32m1(&buf[i],
        __riscv_vluxei32_v_i32m1(comp, voff, vl), vl);
    i += (int32_t)vl;
  }
}
#endif

static void afforest(const GapCsr *g, int32_t *comp, int vec) {
  int32_t cbuf[SNAP_BLOCK];
  for (int32_t u = 0; u < g->n; u++) comp[u] = u;
  for (int r = 0; r < NEIGHBOR_ROUNDS; r++) {
    for (int32_t u = 0; u < g->n; u++) {
      if (g->off[u] + r < g->off[u + 1])
        link_nodes(u, g->dat[g->off[u] + r], comp);
    }
    compress(g, comp);
  }
  int32_t c = sample_frequent(comp, g->n);
  for (int32_t u = 0; u < g->n; u++) {
    if (comp[u] == c) continue;
    const int32_t start = g->off[u] + NEIGHBOR_ROUNDS;
    const int32_t deg = g->off[u + 1] - start;
    if (deg <= 0) continue;
    const int32_t *neigh = &g->dat[start];
    for (int32_t k = 0; k < deg; k += SNAP_BLOCK) {
      int32_t cnt = deg - k < SNAP_BLOCK ? deg - k : SNAP_BLOCK;
#ifdef USE_RISCV_VECTOR
      if (vec) fill_snapshot_rvv(comp, &neigh[k], cnt, cbuf);
      else fill_snapshot_scalar(comp, &neigh[k], cnt, cbuf);
#else
      (void)vec;
      fill_snapshot_scalar(comp, &neigh[k], cnt, cbuf);
#endif
      for (int32_t j = 0; j < cnt; j++) {
        if (cbuf[j] != comp[u])
          link_nodes(u, neigh[k + j], comp);
      }
    }
  }
  compress(g, comp);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: gap_cc <nodes> <edges> <seed>\n");
    return 2;
  }
  int32_t n = atoi(argv[1]);
  int64_t e = atoll(argv[2]);
  uint64_t seed = (uint64_t)strtoull(argv[3], NULL, 0);
  if (n <= 1 || e <= 0) { fprintf(stderr, "error: bad sizes\n"); return 2; }
  GapCsr g;
  if (!gap_gen(n, e, 0, &seed, &g)) { fprintf(stderr, "oom\n"); return 2; }
  int32_t *comp = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (!comp || !ref) { fprintf(stderr, "oom\n"); return 2; }

  ROI_BEGIN();
  afforest(&g, comp, 1);
  ROI_END();

  afforest(&g, ref, 0);
  int pass = memcmp(comp, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_cc", comp, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
