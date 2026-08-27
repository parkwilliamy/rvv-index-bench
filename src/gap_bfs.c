/*
 * GAP_BFS — direction-optimizing BFS with a gather-formulated
 * bottom-up step.
 *
 * Semantics follow GAPBS bfs.cc (Beamer et al.; direction-optimizing
 * BFS, SC 2012): top-down steps over a frontier queue, switching to
 * bottom-up when the frontier is large (alpha=15, beta=18), with the
 * GAPBS parent encoding (parent[u] = -out_degree(u) while unvisited).
 * Upstream indirect access: the bottom-up frontier membership test
 * `front.get_bit(v)` over in-neighbors — gapbs src/bfs.cc:54 (the
 * v-indexed read inside BUStep, lines 46-64).
 *
 * RVV formulation (ported from the ~/gapbs vectorization): the bit
 * frontier becomes int32 flags so the test is an indexed gather —
 * vle32 the neighbor ids, vsll 2, vluxei32 the flag array, spill,
 * scan for the first hit. front is read-only during a step and
 * writes go only to parent[u]/next[u], so gather-then-consume equals
 * the serial semantics exactly (bit-identical parent array).
 *
 * Why the compiler cannot vectorize it (verified on gapbs with gcc
 * -O3 -ftree-vectorize: zero kernel gathers emitted): the inner loop
 * has a data-dependent early `break`, and vectorizing it would
 * require speculative gathers past the exit point, which the
 * vectorizer cannot prove safe.
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

static int64_t td_step(const GapCsr *g, int32_t *parent,
                       const int32_t *q, int64_t qn, int32_t *nq,
                       int64_t *nqn) {
  int64_t scout = 0, t = 0;
  for (int64_t i = 0; i < qn; i++) {
    int32_t u = q[i];
    for (int32_t j = g->off[u]; j < g->off[u + 1]; j++) {
      int32_t v = g->dat[j];
      int32_t cv = parent[v];
      if (cv < 0) {
        parent[v] = u;
        nq[t++] = v;
        scout += -cv;
      }
    }
  }
  *nqn = t;
  return scout;
}

static int64_t bu_step_scalar(const GapCsr *g, int32_t *parent,
                              const int32_t *front, int32_t *next) {
  int64_t awake = 0;
  memset(next, 0, (size_t)g->n * sizeof(int32_t));
  for (int32_t u = 0; u < g->n; u++) {
    if (parent[u] < 0) {
      for (int32_t j = g->off[u]; j < g->off[u + 1]; j++) {
        int32_t v = g->dat[j];
        if (front[v]) {
          parent[u] = v;
          awake++;
          next[u] = 1;
          break;
        }
      }
    }
  }
  return awake;
}

#ifdef USE_RISCV_VECTOR
#define VBUF_LANES 256
static int64_t bu_step_rvv(const GapCsr *g, int32_t *parent,
                           const int32_t *front, int32_t *next) {
  int64_t awake = 0;
  memset(next, 0, (size_t)g->n * sizeof(int32_t));
  int32_t fbuf[VBUF_LANES];
  for (int32_t u = 0; u < g->n; u++) {
    if (parent[u] < 0) {
      const int32_t *neigh = &g->dat[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
      int found = 0;
      for (int32_t k = 0; k < deg && !found; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
        vuint32m1_t voff = __riscv_vsll_vx_u32m1(
            __riscv_vreinterpret_v_i32m1_u32m1(
                __riscv_vle32_v_i32m1(&neigh[k], vl)),
            2, vl);
        __riscv_vse32_v_i32m1(
            fbuf, __riscv_vluxei32_v_i32m1(front, voff, vl), vl);
        for (size_t j = 0; j < vl; j++) {
          if (fbuf[j]) {
            parent[u] = neigh[k + (int32_t)j];
            awake++;
            next[u] = 1;
            found = 1;
            break;
          }
        }
        k += (int32_t)vl;
      }
    }
  }
  return awake;
}
#endif

static void dobfs(const GapCsr *g, int32_t source, int32_t *parent,
                  int32_t *q, int32_t *nq, int32_t *front,
                  int32_t *curr, int vec) {
  const int alpha = 15, beta = 18;
  for (int32_t u = 0; u < g->n; u++) {
    int32_t d = g->off[u + 1] - g->off[u];
    parent[u] = d != 0 ? -d : -1;
  }
  parent[source] = source;
  q[0] = source;
  int64_t qn = 1;
  int64_t edges_to_check = g->m;
  int64_t scout = g->off[source + 1] - g->off[source];
  while (qn > 0) {
    if (scout > edges_to_check / alpha) {
      memset(front, 0, (size_t)g->n * sizeof(int32_t));
      for (int64_t i = 0; i < qn; i++) front[q[i]] = 1;
      int64_t awake = qn, old_awake;
      do {
        old_awake = awake;
#ifdef USE_RISCV_VECTOR
        awake = vec ? bu_step_rvv(g, parent, front, curr)
                    : bu_step_scalar(g, parent, front, curr);
#else
        (void)vec;
        awake = bu_step_scalar(g, parent, front, curr);
#endif
        int32_t *tmp = front; front = curr; curr = tmp;
      } while (awake >= old_awake || awake > g->n / beta);
      qn = 0;
      for (int32_t u = 0; u < g->n; u++)
        if (front[u]) q[qn++] = u;
      scout = 1;
    } else {
      edges_to_check -= scout;
      int64_t nqn;
      scout = td_step(g, parent, q, qn, nq, &nqn);
      int32_t *tmp = q; q = nq; nq = tmp;
      qn = nqn;
    }
  }
  for (int32_t u = 0; u < g->n; u++)
    if (parent[u] < -1) parent[u] = -1;
}

int main(int argc, char **argv) {
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if (argc != 5) {
    fprintf(stderr, "usage: gap_bfs <nodes> <edges> <source> <seed>\n"
            "       gap_bfs -f <graph.bfs> <source> <seed>\n");
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

  int32_t *parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *q = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *nq = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *front = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *curr = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  if (!parent || !ref || !q || !nq || !front || !curr) {
    fprintf(stderr, "oom\n"); return 2;
  }

  ROI_BEGIN();
  dobfs(&g, source, parent, q, nq, front, curr, 1);
  ROI_END();

  // Scalar-mirror reference: same DOBFS with the scalar BU step.
  dobfs(&g, source, ref, q, nq, front, curr, 0);

  int pass = memcmp(parent, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_bfs", parent, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
