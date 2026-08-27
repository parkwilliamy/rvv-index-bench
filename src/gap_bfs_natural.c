/*
 * GAP_BFS_NATURAL — direction-optimizing BFS, shape-faithful
 * vectorization of the original GAPBS serial kernel.
 *
 * Counterpart experiment to gap_bfs.c: that port was formulated to
 * maximize indexed loads (the Bitmap frontier was widened to int32
 * flags so the membership test becomes a full-word gather, and
 * gathered strips are spilled to a buffer and scanned scalar). This
 * variant instead vectorizes the ORIGINAL kernel shape the way an
 * engineer would, with no restructuring in favor of indexed ops:
 *   - the frontier stays a GAPBS Bitmap (uint32 words, n/8 bytes);
 *   - the bottom-up membership search over in-neighbors becomes
 *     gather the frontier WORDS (vluxei32 of front[v >> 5]), extract
 *     bit v & 31 with vsrl.vv/vand, vmsne -> vfirst for the first
 *     frontier neighbor — consumed in registers, no spill;
 *   - the top-down step and the queue<->bitmap conversions stay
 *     scalar (compare-and-set with possible duplicate targets, and
 *     bit-granular read-modify-write scatters are not conflict-free).
 * The finding this file exists to demonstrate: the gather SURVIVES
 * natural vectorization — the frontier test is indexed by data, so
 * any vectorization of the search emits it — but it lands on the
 * n/8-byte bitmap instead of a manufactured 4n-byte flag array, and
 * the strip is consumed with mask ops instead of a vse32 spill.
 *
 * Semantics follow GAPBS bfs.cc (Beamer et al., SC 2012; alpha=15,
 * beta=18, parent[u] = -out_degree(u) while unvisited). Upstream
 * indirect access: `front.get_bit(v)` — gapbs src/bfs.cc:54 (BUStep,
 * lines 46-64). The step reads front[] only and writes parent[u]/
 * next[u], and vfirst picks the lowest hit lane, so the vector build
 * follows the serial trajectory exactly: parent[] is bit-identical
 * to the scalar build (and to gap_bfs).
 *
 * Why the compiler cannot vectorize it (same blocker as gap_bfs,
 * verified on gapbs with gcc -O3 -ftree-vectorize: zero kernel
 * gathers): the data-dependent early `break` requires speculative
 * gathers past the exit point; a human knows front[] is in-bounds
 * read-only and speculates the whole strip.
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

static inline int bm_get(const uint32_t *bm, int32_t i) {
  return (bm[i >> 5] >> (i & 31)) & 1u;
}
static inline void bm_set(uint32_t *bm, int32_t i) {
  bm[i >> 5] |= 1u << (i & 31);
}

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
                              const uint32_t *front, uint32_t *next,
                              int32_t nwords) {
  int64_t awake = 0;
  memset(next, 0, (size_t)nwords * sizeof(uint32_t));
  for (int32_t u = 0; u < g->n; u++) {
    if (parent[u] < 0) {
      for (int32_t j = g->off[u]; j < g->off[u + 1]; j++) {
        int32_t v = g->dat[j];
        if (bm_get(front, v)) {
          parent[u] = v;
          awake++;
          bm_set(next, u);
          break;
        }
      }
    }
  }
  return awake;
}

#ifdef USE_RISCV_VECTOR
static int64_t bu_step_rvv(const GapCsr *g, int32_t *parent,
                           const uint32_t *front, uint32_t *next,
                           int32_t nwords) {
  int64_t awake = 0;
  memset(next, 0, (size_t)nwords * sizeof(uint32_t));
  for (int32_t u = 0; u < g->n; u++) {
    if (parent[u] < 0) {
      const int32_t *neigh = &g->dat[g->off[u]];
      const int32_t deg = g->off[u + 1] - g->off[u];
      for (int32_t k = 0; k < deg; ) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(deg - k));
        vuint32m1_t vids = __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&neigh[k], vl));
        // front.get_bit(v): gather word v>>5, test bit v&31
        vuint32m1_t vwords = __riscv_vluxei32_v_u32m1(
            front,
            __riscv_vsll_vx_u32m1(__riscv_vsrl_vx_u32m1(vids, 5, vl), 2, vl),
            vl);
        vbool32_t hitm = __riscv_vmsne_vx_u32m1_b32(
            __riscv_vand_vx_u32m1(
                __riscv_vsrl_vv_u32m1(
                    vwords, __riscv_vand_vx_u32m1(vids, 31, vl), vl),
                1, vl),
            0, vl);
        long hit = __riscv_vfirst_m_b32(hitm, vl);
        if (hit >= 0) {           // first frontier neighbor, serial order
          parent[u] = neigh[k + (int32_t)hit];
          awake++;
          bm_set(next, u);
          break;
        }
        k += (int32_t)vl;
      }
    }
  }
  return awake;
}
#endif

static void dobfs(const GapCsr *g, int32_t source, int32_t *parent,
                  int32_t *q, int32_t *nq, uint32_t *front,
                  uint32_t *curr, int32_t nwords, int vec) {
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
      // QueueToBitmap: bit-granular RMW, not conflict-free — scalar.
      memset(front, 0, (size_t)nwords * sizeof(uint32_t));
      for (int64_t i = 0; i < qn; i++) bm_set(front, q[i]);
      int64_t awake = qn, old_awake;
      do {
        old_awake = awake;
#ifdef USE_RISCV_VECTOR
        awake = vec ? bu_step_rvv(g, parent, front, curr, nwords)
                    : bu_step_scalar(g, parent, front, curr, nwords);
#else
        (void)vec;
        awake = bu_step_scalar(g, parent, front, curr, nwords);
#endif
        uint32_t *tmp = front; front = curr; curr = tmp;
      } while (awake >= old_awake || awake > g->n / beta);
      qn = 0;
      for (int32_t u = 0; u < g->n; u++)
        if (bm_get(front, u)) q[qn++] = u;
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
    fprintf(stderr,
            "usage: gap_bfs_natural <nodes> <edges> <source> <seed>\n"
            "       gap_bfs_natural -f <graph.bfs> <source> <seed>\n");
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
  const int32_t nwords = (n + 31) / 32;

  int32_t *parent = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *ref = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *q = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  int32_t *nq = (int32_t *)malloc((size_t)n * sizeof(int32_t));
  uint32_t *front = (uint32_t *)malloc((size_t)nwords * sizeof(uint32_t));
  uint32_t *curr = (uint32_t *)malloc((size_t)nwords * sizeof(uint32_t));
  if (!parent || !ref || !q || !nq || !front || !curr) {
    fprintf(stderr, "oom\n"); return 2;
  }

  ROI_BEGIN();
  dobfs(&g, source, parent, q, nq, front, curr, nwords, 1);
  ROI_END();

  // Scalar-mirror reference: same DOBFS with the scalar BU step.
  dobfs(&g, source, ref, q, nq, front, curr, nwords, 0);

  int pass = memcmp(parent, ref, (size_t)n * sizeof(int32_t)) == 0;
  print_checksum_i32("gap_bfs_natural", parent, n);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
