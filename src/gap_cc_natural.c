/*
 * GAP_CC_NATURAL — Afforest connected components, shape-faithful
 * vectorization of the original GAPBS serial kernel.
 *
 * Counterpart experiment to gap_cc.c: that port gathers comp[] into
 * a stack buffer and compares each buffered value scalar before
 * calling Link. This variant vectorizes the ORIGINAL final-phase
 * shape the way an engineer would: the scan's hot work is the
 * comp[v] read and the "already same component?" test —
 *   - gather comp[v] for the block (the label read is indexed by
 *     data, so the gather SURVIVES any vectorization of this loop);
 *   - vector-compare against the block's comp[u] snapshot (vmsne),
 *     vcpop-skip blocks where every neighbor already carries the
 *     same label (the dominant case: after the sampling rounds most
 *     of the graph is already merged — that is Afforest's point);
 *   - only flagged lanes call the ORIGINAL scalar Link (a data-
 *     dependent while-loop label chase with stores — serial in any
 *     honest vectorization, and the reason no compiler touches it).
 * Only 0/1 lane flags cross to the scalar side; gathered values are
 * never spilled.
 *
 * The equal-label skip is exact by the same argument as gap_cc.c:
 * labels only merge downward, so two nodes whose labels were equal
 * at snapshot time are permanently in the same set, and skipping
 * their Link loses no union. Determinism: the filter granularity is
 * fixed at a 64-element BLOCK in the ALGORITHM (all gathers/compares
 * of a block complete against the pre-block state before any lane
 * links), so every VLEN makes identical decisions and the scalar
 * mirror applies the same snapshot filter — bit-identical
 * trajectories in both builds.
 *
 * Semantics follow GAPBS cc.cc (Sutton et al., IPDPS 2018):
 * neighbor-sampling rounds, compression, sampled most-frequent
 * component skipped, final link phase over remaining neighborhoods.
 * Upstream indirect accesses: the comp[] chase inside Link — gapbs
 * src/cc.cc:42-53 — driven from the final phase (src/cc.cc:123-147).
 *
 * Why the compiler cannot vectorize it (same blocker as gap_cc,
 * verified on gapbs with gcc -O3 -ftree-vectorize: zero kernel
 * gathers): the loop body is a call to Link's data-dependent
 * while-chase with stores.
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

// Phase A of a block: flag lanes whose snapshot label differs from
// the block's comp[u] snapshot; returns the number flagged.
static int32_t filter_scalar(const int32_t *comp, const int32_t *ids,
                             int32_t cnt, int32_t comp_u_snap,
                             uint32_t *flags) {
  int32_t nact = 0;
  for (int32_t j = 0; j < cnt; j++) {
    flags[j] = comp[ids[j]] != comp_u_snap;
    nact += (int32_t)flags[j];
  }
  return nact;
}

#ifdef USE_RISCV_VECTOR
static int32_t filter_rvv(const int32_t *comp, const int32_t *ids,
                          int32_t cnt, int32_t comp_u_snap,
                          uint32_t *flags) {
  int32_t nact = 0;
  for (int32_t i = 0; i < cnt; ) {
    size_t vl = __riscv_vsetvl_e32m1((size_t)(cnt - i));
    vint32m1_t vc = __riscv_vluxei32_v_i32m1(
        comp,
        __riscv_vsll_vx_u32m1(
            __riscv_vreinterpret_v_i32m1_u32m1(
                __riscv_vle32_v_i32m1(&ids[i], vl)),
            2, vl),
        vl);
    vbool32_t m = __riscv_vmsne_vx_i32m1_b32(vc, comp_u_snap, vl);
    nact += (int32_t)__riscv_vcpop_m_b32(m, vl);
    __riscv_vse32_v_u32m1(&flags[i],
        __riscv_vmerge_vxm_u32m1(
            __riscv_vmv_v_x_u32m1(0, vl), 1, m, vl), vl);
    i += (int32_t)vl;
  }
  return nact;
}
#endif

static void afforest(const GapCsr *g, int32_t *comp, int vec) {
  uint32_t flags[SNAP_BLOCK];
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
      const int32_t comp_u_snap = comp[u];
      int32_t nact;
#ifdef USE_RISCV_VECTOR
      if (vec) nact = filter_rvv(comp, &neigh[k], cnt, comp_u_snap, flags);
      else nact = filter_scalar(comp, &neigh[k], cnt, comp_u_snap, flags);
#else
      (void)vec;
      nact = filter_scalar(comp, &neigh[k], cnt, comp_u_snap, flags);
#endif
      if (!nact) continue;
      for (int32_t j = 0; j < cnt; j++) {
        if (flags[j])
          link_nodes(u, neigh[k + j], comp);
      }
    }
  }
  compress(g, comp);
}

int main(int argc, char **argv) {
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 4) {
    fprintf(stderr, "usage: gap_cc_natural <nodes> <edges> <seed>\n"
            "       gap_cc_natural -f <graph.bfs> <seed>\n");
    return 2;
  }
  // Graph source: synthetic (nodes/edges) or a .bfs image (-f).
  // In file mode argv[2] is the path and the remaining
  // positional args shift down by one.
  int32_t n = 0; int64_t e = 0;
  uint64_t seed = (uint64_t)strtoull(argv[3], NULL, 0);
  if (!file_mode) {
    n = atoi(argv[1]);
    e = atoll(argv[2]);
    if (n <= 1 || e <= 0) { fprintf(stderr, "error: bad sizes\n"); return 2; }
  }
  GapCsr g;
  if (file_mode) {
    if (!gap_load(argv[2], 0, &seed, &g)) return 2;
  } else {
    if (!gap_gen(n, e, 0, &seed, &g)) { fprintf(stderr, "oom\n"); return 2; }
  }
  n = g.n;  // authoritative after either path

  int32_t *comp = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (!comp || !ref) { fprintf(stderr, "oom\n"); return 2; }

  ROI_BEGIN();
  afforest(&g, comp, 1);
  ROI_END();

  afforest(&g, ref, 0);
  int pass = memcmp(comp, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_cc_natural", comp, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
