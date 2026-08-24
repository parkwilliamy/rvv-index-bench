/*
 * GAP_CC_SV — Shiloach-Vishkin connected components with a
 * gather-formulated edge scan.
 *
 * Semantics follow GAPBS cc_sv.cc (Shiloach & Vishkin 1982 with the
 * Bader/Kothapalli min-max hooking): repeated full edge scans that
 * hook higher component labels under lower ones, with pointer-jumping
 * compression between iterations. Upstream indirect access: the
 * label read `comp[v]` per neighbor — gapbs src/cc_sv.cc:63 (inside
 * the scan at lines 60-73).
 *
 * RVV formulation (ported from the ~/gapbs vectorization, made
 * deterministic): comp[] values for each neighbor BLOCK are gathered
 * (vle32 ids -> vsll 2 -> vluxei32) into a buffer, then the hooking
 * body runs scalar per lane against the buffered snapshot. The
 * snapshot BLOCK is fixed at 64 elements IN THE ALGORITHM (the
 * vector build fills it with as many vsetvl strips as VLMAX needs),
 * so results are identical for every VLEN and for the scalar mirror,
 * which buffers the same 64-element snapshots. Snapshot staleness
 * within a block is benign — SV iterates to a fixpoint — and with
 * both variants using the same block size the trajectories are
 * bit-identical, not merely equivalent.
 *
 * Why the compiler cannot vectorize it (verified on gapbs with gcc
 * -O3 -ftree-vectorize: zero kernel gathers): the body contains a
 * data-dependent store (`comp[high_comp] = low_comp`) that later
 * iterations may read — legal vectorization needs runtime
 * intra-vector conflict detection, which gcc/llvm do not attempt for
 * RVV.
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

static void sv(const GapCsr *g, int32_t *comp, int vec) {
  int32_t cbuf[SNAP_BLOCK];
  for (int32_t u = 0; u < g->n; u++) comp[u] = u;
  int change = 1;
  int num_iter = 0;
  while (change) {
    change = 0;
    num_iter++;
    for (int32_t u = 0; u < g->n; u++) {
      const int32_t *neigh = &g->dat[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
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
          int32_t comp_u = comp[u];
          int32_t comp_v = cbuf[j];
          if (comp_u == comp_v) continue;
          int32_t high = comp_u > comp_v ? comp_u : comp_v;
          int32_t low = comp_u + (comp_v - high);
          if (high == comp[high]) {
            change = 1;
            comp[high] = low;
          }
        }
      }
    }
    for (int32_t u = 0; u < g->n; u++)
      while (comp[u] != comp[comp[u]])
        comp[u] = comp[comp[u]];
  }
  printf("sv iterations: %d\n", num_iter);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: gap_cc_sv <nodes> <edges> <seed>\n");
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
  sv(&g, comp, 1);
  ROI_END();

  sv(&g, ref, 0);
  int pass = memcmp(comp, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_cc_sv", comp, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
