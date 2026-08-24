/*
 * GAP_BC — Brandes betweenness centrality with a gather-formulated
 * backward (dependency-accumulation) pass.
 *
 * Semantics follow GAPBS bc.cc (Brandes 2001 via Madduri et al.,
 * IPDPS 2009): a forward BFS records per-edge successor bits, path
 * counts (double), and depth slices of the traversal queue; the
 * backward pass walks depths in reverse accumulating
 *     delta[u] += (path_counts[u]/path_counts[v]) * (1 + delta[v])
 * over successor edges. Upstream indirect accesses: `path_counts[v]`
 * and `deltas[v]` — gapbs src/bc.cc:130 (backward loop, lines
 * 123-136). One source iteration, unnormalized scores (the
 * normalization is a dense epilogue irrelevant to indexing).
 *
 * RVV formulation (ported from the ~/gapbs vectorization): per
 * successor strip, vle32 the neighbor ids and vluxei32 deltas[]
 * (f32, e32m1); the f64 path_counts stay scalar in the consume loop
 * (no widening chains), and the per-edge succ byte is SEQUENTIAL,
 * not gathered. deltas[] of depth d+1 are finalized before depth d
 * runs and writes go to deltas[u]/scores[u] only, so
 * gather-then-consume is exact; every FP op is a single rounding in
 * both builds, so results are bit-identical.
 *
 * Why the compiler cannot vectorize it (verified on gapbs with gcc
 * -O3 -ftree-vectorize: zero kernel gathers): a conditional guards
 * an FP division (possible 0/0 on inactive lanes needs masked
 * division the vectorizer won't speculate), and the f64/f32 mixing
 * adds conversion chains its cost model rejects.
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

// Forward phase (shared, scalar): BFS by levels over the queue,
// recording depths, double path counts, per-edge successor bytes,
// and the queue's depth-slice boundaries.
static int pbfs(const GapCsr *g, int32_t source, double *path_counts,
                uint8_t *succ, int32_t *queue, int64_t *depth_index,
                int *n_depths) {
  int32_t *depths = (int32_t *)malloc((size_t)g->n * sizeof(int32_t));
  if (!depths) return 0;
  for (int32_t u = 0; u < g->n; u++) depths[u] = -1;
  memset(path_counts, 0, (size_t)g->n * sizeof(double));
  depths[source] = 0;
  path_counts[source] = 1.0;
  queue[0] = source;
  int64_t head = 0, tail = 1;
  int depth = 0;
  depth_index[0] = 0;
  while (head < tail) {
    depth++;
    int64_t level_end = tail;
    for (; head < level_end; head++) {
      int32_t u = queue[head];
      for (int32_t j = g->off[u]; j < g->off[u + 1]; j++) {
        int32_t v = g->dat[j];
        if (depths[v] == -1) {
          depths[v] = depth;
          queue[tail++] = v;
        }
        if (depths[v] == depth) {
          succ[j] = 1;
          path_counts[v] += path_counts[u];
        }
      }
    }
    depth_index[depth] = level_end;
  }
  depth_index[depth + 1] = tail;
  *n_depths = depth;
  free(depths);
  return 1;
}

static void backward_scalar(const GapCsr *g, const double *path_counts,
                            const uint8_t *succ, const int32_t *queue,
                            const int64_t *depth_index, int n_depths,
                            float *deltas, float *scores) {
  for (int d = n_depths - 1; d >= 0; d--) {
    for (int64_t i = depth_index[d]; i < depth_index[d + 1]; i++) {
      int32_t u = queue[i];
      float delta_u = 0.0f;
      for (int32_t j = g->off[u]; j < g->off[u + 1]; j++) {
        if (succ[j]) {
          int32_t v = g->dat[j];
          delta_u += (float)((path_counts[u] / path_counts[v]) *
                             (1.0 + (double)deltas[v]));
        }
      }
      deltas[u] = delta_u;
      scores[u] += delta_u;
    }
  }
}

#ifdef USE_RISCV_VECTOR
#define VBUF_LANES 256
static void backward_rvv(const GapCsr *g, const double *path_counts,
                         const uint8_t *succ, const int32_t *queue,
                         const int64_t *depth_index, int n_depths,
                         float *deltas, float *scores) {
  float dbuf[VBUF_LANES];
  for (int d = n_depths - 1; d >= 0; d--) {
    for (int64_t i = depth_index[d]; i < depth_index[d + 1]; i++) {
      int32_t u = queue[i];
      float delta_u = 0.0f;
      const int32_t *neigh = &g->dat[g->off[u]];
      const uint8_t *sbits = &succ[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
      for (int32_t k = 0; k < deg; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
        vuint32m1_t voff = __riscv_vsll_vx_u32m1(
            __riscv_vreinterpret_v_i32m1_u32m1(
                __riscv_vle32_v_i32m1(&neigh[k], vl)),
            2, vl);
        __riscv_vse32_v_f32m1(
            dbuf, __riscv_vluxei32_v_f32m1(deltas, voff, vl), vl);
        for (size_t j = 0; j < vl; j++) {
          if (sbits[k + (int32_t)j]) {
            int32_t v = neigh[k + (int32_t)j];
            delta_u += (float)((path_counts[u] / path_counts[v]) *
                               (1.0 + (double)dbuf[j]));
          }
        }
        k += (int32_t)vl;
      }
      deltas[u] = delta_u;
      scores[u] += delta_u;
    }
  }
}
#endif

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: gap_bc <nodes> <edges> <source> <seed>\n");
    return 2;
  }
  int32_t n = atoi(argv[1]);
  int64_t e = atoll(argv[2]);
  int32_t source = atoi(argv[3]);
  uint64_t seed = (uint64_t)strtoull(argv[4], NULL, 0);
  if (n <= 1 || e <= 0 || source < 0 || source >= n) {
    fprintf(stderr, "error: bad sizes/source\n");
    return 2;
  }
  GapCsr g;
  if (!gap_gen(n, e, 0, &seed, &g)) { fprintf(stderr, "oom\n"); return 2; }
  double *path_counts = (double *)malloc((size_t)n * sizeof(double));
  uint8_t *succ = (uint8_t *)calloc((size_t)g.m, 1);
  int32_t *queue = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int64_t *depth_index = (int64_t *)malloc(((size_t)n + 2) * sizeof(int64_t));
  float *deltas = (float *)calloc((size_t)n, sizeof(float));
  float *scores = (float *)calloc((size_t)n, sizeof(float));
  float *deltas_ref = (float *)calloc((size_t)n, sizeof(float));
  float *scores_ref = (float *)calloc((size_t)n, sizeof(float));
  if (!path_counts || !succ || !queue || !depth_index || !deltas ||
      !scores || !deltas_ref || !scores_ref) {
    fprintf(stderr, "oom\n"); return 2;
  }
  int n_depths;
  if (!pbfs(&g, source, path_counts, succ, queue, depth_index,
            &n_depths)) { fprintf(stderr, "oom\n"); return 2; }

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  backward_rvv(&g, path_counts, succ, queue, depth_index, n_depths,
               deltas, scores);
#else
  backward_scalar(&g, path_counts, succ, queue, depth_index, n_depths,
                  deltas, scores);
#endif
  ROI_END();

  backward_scalar(&g, path_counts, succ, queue, depth_index, n_depths,
                  deltas_ref, scores_ref);
  int pass = memcmp(scores, scores_ref, (size_t)n * sizeof(float)) == 0;
  print_checksum("gap_bc", scores, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
