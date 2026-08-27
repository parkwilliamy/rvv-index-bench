/*
 * GAP_PR — PageRank, pull-direction Gauss-Seidel, with the gather
 * over outgoing contributions.
 *
 * Semantics follow GAPBS pr.cc (PageRankPullGS: pull updates with
 * immediately-visible scores, damping 0.85, convergence when the
 * total absolute change drops below epsilon):
 *     incoming = sum_{v in neigh(u)} contrib[v]
 *     scores[u] = base + damp * incoming;  contrib[u] updated inline
 * Upstream indirect access: `outgoing_contrib[v]` per in-neighbor —
 * gapbs src/pr.cc:49 (the pull loop, lines 46-54), pinned @ 2972aeb.
 * Note gcc autovectorizes THIS kernel on gapbs (the one GAP kernel
 * that autovecs) — but only as the e64m2 widening chain
 * (vle32 -> vsext.vf2 -> vluxei64); this port is the e32m1 intrinsic
 * form (vluxei32, f32) per the suite conventions, with the scalar
 * implementation in the same file as the reference.
 *
 * RVV formulation: vle32 the neighbor ids -> vsll 2 -> vluxei32
 * contrib[] -> ordered redsum, matching the scalar accumulation
 * order exactly, so scores are bit-identical. Graphs are the suite's
 * deterministic uniform undirected multigraphs (in == out CSR).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "bench_common.h"
#include "gap_graph.h"
#include "roi.h"

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#endif

#define KDAMP 0.85f
#define EPSILON 1e-4

static void pr_scalar(const GapCsr *g, int max_iters, float *scores,
                      float *contrib) {
  const float init_score = 1.0f / (float)g->n;
  const float base_score = (1.0f - KDAMP) / (float)g->n;
  for (int32_t u = 0; u < g->n; u++) scores[u] = init_score;
  for (int32_t u = 0; u < g->n; u++) {
    int32_t deg = g->off[u + 1] - g->off[u];
    contrib[u] = deg ? init_score / (float)deg : 0.0f;
  }
  for (int iter = 0; iter < max_iters; iter++) {
    double error = 0.0;
    for (int32_t u = 0; u < g->n; u++) {
      float incoming = 0.0f;
      for (int32_t j = g->off[u]; j < g->off[u + 1]; j++)
        incoming += contrib[g->dat[j]];
      float old_score = scores[u];
      scores[u] = base_score + KDAMP * incoming;
      error += fabs((double)(scores[u] - old_score));
      int32_t deg = g->off[u + 1] - g->off[u];
      contrib[u] = deg ? scores[u] / (float)deg : 0.0f;
    }
    if (error < EPSILON) break;
  }
}

#ifdef USE_RISCV_VECTOR
static void pr_rvv(const GapCsr *g, int max_iters, float *scores,
                   float *contrib) {
  const float init_score = 1.0f / (float)g->n;
  const float base_score = (1.0f - KDAMP) / (float)g->n;
  for (int32_t u = 0; u < g->n; u++) scores[u] = init_score;
  for (int32_t u = 0; u < g->n; u++) {
    int32_t deg = g->off[u + 1] - g->off[u];
    contrib[u] = deg ? init_score / (float)deg : 0.0f;
  }
  for (int iter = 0; iter < max_iters; iter++) {
    double error = 0.0;
    for (int32_t u = 0; u < g->n; u++) {
      const int32_t *neigh = &g->dat[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
      float incoming = 0.0f;
      if (deg) {
        vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        for (int32_t k = 0; k < deg; ) {
          size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
          vuint32m1_t voff = __riscv_vsll_vx_u32m1(
              __riscv_vreinterpret_v_i32m1_u32m1(
                  __riscv_vle32_v_i32m1(&neigh[k], vl)),
              2, vl);
          acc = __riscv_vfredosum_vs_f32m1_f32m1(
              __riscv_vluxei32_v_f32m1(contrib, voff, vl), acc, vl);
          k += (int32_t)vl;
        }
        incoming = __riscv_vfmv_f_s_f32m1_f32(acc);
      }
      float old_score = scores[u];
      scores[u] = base_score + KDAMP * incoming;
      error += fabs((double)(scores[u] - old_score));
      contrib[u] = deg ? scores[u] / (float)deg : 0.0f;
    }
    if (error < EPSILON) break;
  }
}
#endif

int main(int argc, char **argv) {
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 5) {
    fprintf(stderr, "usage: gap_pr <nodes> <edges> <max_iters> <seed>\n"
            "       gap_pr -f <graph.bfs> <max_iters> <seed>\n");
    return 2;
  }
  // Graph source: synthetic (nodes/edges) or a .bfs image (-f).
  // In file mode argv[2] is the path and the remaining
  // positional args shift down by one.
  int32_t n = 0; int64_t e = 0;
  int max_iters = atoi(argv[3]);
  uint64_t seed = (uint64_t)strtoull(argv[4], NULL, 0);
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
  if (max_iters <= 0) { fprintf(stderr, "error: bad max_iters\n"); return 2; }

  float *scores = (float *)malloc((size_t)n * sizeof(float));
  float *contrib = (float *)malloc((size_t)n * sizeof(float));
  float *ref = (float *)malloc((size_t)n * sizeof(float));
  if (!scores || !contrib || !ref) { fprintf(stderr, "oom\n"); return 2; }

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  pr_rvv(&g, max_iters, scores, contrib);
#else
  pr_scalar(&g, max_iters, scores, contrib);
#endif
  ROI_END();

  pr_scalar(&g, max_iters, ref, contrib);
  int pass = memcmp(scores, ref, (size_t)n * sizeof(float)) == 0;
  print_checksum("gap_pr", scores, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
