/*
 * GAP_BC_NATURAL — Brandes betweenness centrality, shape-faithful
 * vectorization of the original GAPBS serial backward pass.
 *
 * Counterpart experiment to gap_bc.c: that port gathers deltas[] and
 * path_counts[] for EVERY lane of every strip (successor or not),
 * spills both to buffers, and re-reads the succ byte scalar per
 * lane. This variant vectorizes the ORIGINAL shape the way an
 * engineer (or a compiler's if-conversion, were it able) would:
 *   - the per-edge succ bytes are SEQUENTIAL, so the "is this edge a
 *     successor edge" test is one unit-stride vle8 + vmsne -> a mask
 *     register, not per-lane scalar byte reads;
 *   - vcpop == 0 retires strips with no successor edges without
 *     touching deltas[]/path_counts[] at all;
 *   - the gathers are MASKED (vluxei32_m under the succ mask):
 *     inactive lanes make no memory access, which is exactly what
 *     if-converting `if (succ[j]) ... deltas[v] ... path_counts[v]`
 *     means — the gathers SURVIVE natural vectorization (both reads
 *     are indexed by data) but only touch the lanes the serial code
 *     would;
 *   - the FP accumulation stays scalar per active lane: matching the
 *     serial result bit-for-bit requires rounding each f64 term to
 *     f32 before the f32 sum, and the narrowing conversion (vfncvt)
 *     is banned in this suite for hitting known gem5 RVV bugs. Lane
 *     order equals serial edge order, so results are bit-identical.
 *
 * Semantics follow GAPBS bc.cc (Brandes 2001 via Madduri et al.,
 * IPDPS 2009); the scalar forward BFS (path counts f64, per-edge
 * successor BYTES standing in for gapbs's edge-indexed Bitmap, depth
 * slices) is shared with gap_bc.c and sits outside the ROI. Upstream
 * indirect accesses: `path_counts[v]` and `deltas[v]` — gapbs
 * src/bc.cc:130 (backward loop, lines 123-136). deltas[]/
 * path_counts[] of depth d+1 are finalized before depth d runs, so
 * the vector step is exact; scores are bit-identical to the scalar
 * build (and to gap_bc).
 *
 * Why the compiler cannot vectorize it (same blocker as gap_bc,
 * verified on gapbs with gcc -O3 -ftree-vectorize: zero kernel
 * gathers): the conditional FP division needs masked-divide
 * speculation, plus the f64/f32 mixing.
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
  double pcbuf[VBUF_LANES];
  uint8_t mbuf[VBUF_LANES / 8];
  for (int d = n_depths - 1; d >= 0; d--) {
    for (int64_t i = depth_index[d]; i < depth_index[d + 1]; i++) {
      int32_t u = queue[i];
      float delta_u = 0.0f;
      const int32_t *neigh = &g->dat[g->off[u]];
      const uint8_t *sbits = &succ[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
      const double pc_u = path_counts[u];
      for (int32_t k = 0; k < deg; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
        vbool32_t sm = __riscv_vmsne_vx_u8mf4_b32(
            __riscv_vle8_v_u8mf4(&sbits[k], vl), 0, vl);
        if (__riscv_vcpop_m_b32(sm, vl) == 0) {
          k += (int32_t)vl;      // no successor edges in this strip
          continue;
        }
        vuint32m1_t vidx = __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&neigh[k], vl));
        // masked gathers: inactive lanes make no memory access
        __riscv_vse32_v_f32m1(
            dbuf, __riscv_vluxei32_v_f32m1_m(
                sm, deltas, __riscv_vsll_vx_u32m1(vidx, 2, vl), vl), vl);
        __riscv_vse64_v_f64m2(
            pcbuf, __riscv_vluxei32_v_f64m2_m(
                sm, path_counts, __riscv_vsll_vx_u32m1(vidx, 3, vl), vl), vl);
        __riscv_vsm_v_b32(mbuf, sm, vl);
        for (size_t j = 0; j < vl; j++) {
          if ((mbuf[j >> 3] >> (j & 7)) & 1) {
            delta_u += (float)((pc_u / pcbuf[j]) *
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
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 5) {
    fprintf(stderr,
            "usage: gap_bc_natural <nodes> <edges> <source> <seed>\n"
            "       gap_bc_natural -f <graph.bfs> <source> <seed>\n");
    return 2;
  }
  // Graph source: synthetic (nodes/edges) or a .bfs image (-f).
  // In file mode argv[2] is the path and the remaining
  // positional args shift down by one.
  int32_t n = 0; int64_t e = 0;
  int32_t source = atoi(argv[3]);
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
  if (source < 0 || source >= n) {
    fprintf(stderr, "error: source out of range [0,%d)\n", n);
    return 2;
  }

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
  print_checksum("gap_bc_natural", scores, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
