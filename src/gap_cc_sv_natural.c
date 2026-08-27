/*
 * GAP_CC_SV_NATURAL — Shiloach-Vishkin connected components,
 * shape-faithful vectorization of the original GAPBS serial kernel.
 *
 * Counterpart experiment to gap_cc_sv.c: that port gathers comp[]
 * values into a stack buffer and runs the whole hooking body scalar
 * per lane against the buffered snapshot. This variant vectorizes
 * the ORIGINAL scan shape the way an engineer would: the expensive
 * part of the scan is the comp[v] read and the comp_u == comp_v
 * test, and both go to vector —
 *   - gather comp[v] for the block (vle32 ids -> vsll 2 -> vluxei32,
 *     same access as before: the label read is indexed by data, so
 *     the gather SURVIVES any vectorization of this loop);
 *   - vector-compare against the block's comp[u] snapshot (vmsne),
 *     count survivors with vcpop, and skip the block outright when
 *     no neighbor label differs — the dominant case once components
 *     have mostly merged;
 *   - only flagged lanes run the ORIGINAL hooking body, with fresh
 *     comp[] reads (the data-dependent store comp[high] = low may
 *     alias other lanes, so it is serial in any honest
 *     vectorization — the same reason no compiler touches it).
 * The gathered values themselves are never spilled; only 0/1 lane
 * flags cross to the scalar side.
 *
 * Determinism: like gap_cc_sv.c, the filter granularity is fixed at
 * a 64-element BLOCK in the ALGORITHM: all of a block's comp[v]
 * gathers and compares complete against the pre-block state before
 * any lane of the block hooks, so every VLEN partitions the block
 * into identical decisions, and the scalar mirror applies the same
 * snapshot filter. A lane whose label changed between snapshot and
 * body is simply deferred to the next SV iteration; on a pass where
 * nothing hooks, snapshots equal live values, so the filter is exact
 * and the fixpoint (and the change flag) matches the serial
 * algorithm. Both builds follow bit-identical trajectories.
 *
 * Semantics follow GAPBS cc_sv.cc (Shiloach & Vishkin 1982): full
 * edge scans hooking higher labels under lower ones, with pointer-
 * jumping compression between iterations. Upstream indirect access:
 * `comp[v]` — gapbs src/cc_sv.cc:63 (scan, lines 60-73).
 *
 * Why the compiler cannot vectorize it (same blocker as gap_cc_sv,
 * verified on gapbs with gcc -O3 -ftree-vectorize: zero kernel
 * gathers): the data-dependent store needs intra-vector conflict
 * detection.
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

static void sv(const GapCsr *g, int32_t *comp, int vec) {
  uint32_t flags[SNAP_BLOCK];
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
          if (!flags[j]) continue;
          // original GAPBS body, live comp[] reads
          int32_t comp_u = comp[u];
          int32_t comp_v = comp[neigh[k + j]];
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
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 4) {
    fprintf(stderr, "usage: gap_cc_sv_natural <nodes> <edges> <seed>\n"
            "       gap_cc_sv_natural -f <graph.bfs> <seed>\n");
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
  sv(&g, comp, 1);
  ROI_END();

  sv(&g, ref, 0);
  int pass = memcmp(comp, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_cc_sv_natural", comp, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
