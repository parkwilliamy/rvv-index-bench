/*
 * PAGED_ATTN — single-query paged-attention decode microbenchmark.
 *
 * Semantics follow vLLM's paged attention: the K/V cache lives in
 * fixed-size PAGES scattered in memory, and a per-sequence BLOCK
 * TABLE maps logical block -> physical page, so every K/V token row
 * is reached through an indirection:
 *     s[t]  = (1/sqrt(D)) * sum_d q[d] * K[row(t)][d]
 *     p     = softmax(s)
 *     o[d]  = sum_t p[t] * V[row(t)][d]
 *     row(t) = block_table[t / PAGE] * PAGE + t % PAGE
 * Sources for the kernel:
 *   - vLLM / PagedAttention: Kwon et al., "Efficient Memory
 *     Management for Large Language Model Serving with
 *     PagedAttention", SOSP 2023; kernels in
 *     https://github.com/vllm-project/vllm (csrc/attention)
 *   - identified as an RVV-relevant indirect workload in the
 *     2026-08 indirect-access evidence sweep (vLLM paged-attn RVV
 *     ports exist).
 *
 * RVV formulation: caches stay in vLLM's page-row layout; a scalar
 * prologue resolves the two-level indirection into a per-token BYTE
 * OFFSET array (the index stream), and both attention phases then
 * run with the TOKEN as the vector axis: vle32 the offsets,
 * vluxei32 one component d of K (or V) for a strip of tokens.
 * Scores use unfused vfmul+vfadd; the V phase uses the ordered
 * redsum — both match the scalar reference bit-exactly.
 *
 * Why the compiler cannot do this on its own: the token->address map
 * goes through a runtime table lookup plus div/mod (non-affine), so
 * no auto-vectorizer can form the gather; and the softmax between
 * the phases plus the FP reductions (illegal to reassociate without
 * -ffast-math) block vectorizing either phase across tokens.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "bench_common.h"
#include "roi.h"

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#endif

// Shared scalar softmax (identical in both variants; inputs are
// bit-identical so outputs are too).
static void softmax(float *s, int T) {
  float m = s[0];
  for (int t = 1; t < T; t++) if (s[t] > m) m = s[t];
  float sum = 0.0f;
  for (int t = 0; t < T; t++) { s[t] = expf(s[t] - m); sum += s[t]; }
  for (int t = 0; t < T; t++) s[t] /= sum;
}

// Scalar reference / serial timed kernel. Score loop d ascending and
// output loop t ascending match the vector op order exactly.
static void attn_scalar(const float *q, const float *kc, const float *vc,
                        const int64_t *row_off, int T, int D,
                        float inv_sqrt_d, float *s, float *o) {
  for (int t = 0; t < T; t++) {
    const float *krow = (const float *)((const char *)kc + row_off[t]);
    float acc = 0.0f;
    for (int d = 0; d < D; d++)
      acc = acc + q[d] * krow[d];
    s[t] = acc * inv_sqrt_d;
  }
  softmax(s, T);
  for (int d = 0; d < D; d++) {
    float acc = 0.0f;
    for (int t = 0; t < T; t++) {
      const float *vrow = (const float *)((const char *)vc + row_off[t]);
      acc += s[t] * vrow[d];
    }
    o[d] = acc;
  }
}

#ifdef USE_RISCV_VECTOR
static void attn_rvv(const float *q, const float *kc, const float *vc,
                     const int64_t *row_off, int T, int D,
                     float inv_sqrt_d, float *s, float *o) {
  size_t vl;
  // Phase 1: scores, token strips; K gathered per component.
  for (int t = 0; t < T; t += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(T - t));
    vuint64m2_t voff = __riscv_vreinterpret_v_i64m2_u64m2(
        __riscv_vle64_v_i64m2(&row_off[t], vl));
    vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t vk = __riscv_vluxei64_v_f32m1(&kc[d], voff, vl);
      // unfused mul+add: bit-exact vs the -ffp-contract=off scalar
      vs = __riscv_vfadd_vv_f32m1(
          vs, __riscv_vfmul_vf_f32m1(vk, q[d], vl), vl);
    }
    __riscv_vse32_v_f32m1(&s[t],
        __riscv_vfmul_vf_f32m1(vs, inv_sqrt_d, vl), vl);
  }
  softmax(s, T);
  // Phase 2: output, ordered redsum over token strips per component.
  for (int d = 0; d < D; d++) {
    vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int t = 0; t < T; t += (int)vl) {
      vl = __riscv_vsetvl_e32m1((size_t)(T - t));
      vuint64m2_t voff = __riscv_vreinterpret_v_i64m2_u64m2(
          __riscv_vle64_v_i64m2(&row_off[t], vl));
      vfloat32m1_t vv = __riscv_vluxei64_v_f32m1(&vc[d], voff, vl);
      vfloat32m1_t vp = __riscv_vle32_v_f32m1(&s[t], vl);
      acc = __riscv_vfredosum_vs_f32m1_f32m1(
          __riscv_vfmul_vv_f32m1(vv, vp, vl), acc, vl);
    }
    o[d] = __riscv_vfmv_f_s_f32m1_f32(acc);
  }
}
#endif

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: paged_attn <dim(pow2)> <page_size> <num_pages> "
            "<ctx_len> <seed>\n");
    return 2;
  }
  const int D = atoi(argv[1]), PS = atoi(argv[2]);
  const int NP = atoi(argv[3]), T = atoi(argv[4]);
  uint64_t seed = (uint64_t)strtoull(argv[5], NULL, 0);
  const int NB = (T + PS - 1) / PS;
  if (!is_pow2(D) || PS <= 0 || T <= 0 || NB > NP) {
    fprintf(stderr, "error: dim pow2, ctx_len <= num_pages*page_size\n");
    return 2;
  }

  size_t cache_elems = (size_t)NP * PS * D;
  float *kc = (float *)malloc(cache_elems * sizeof(float));
  float *vc = (float *)malloc(cache_elems * sizeof(float));
  float *q = (float *)malloc((size_t)D * sizeof(float));
  int32_t *block_table = (int32_t *)malloc((size_t)NB * sizeof(int32_t));
  int64_t *row_off = (int64_t *)malloc((size_t)T * sizeof(int64_t));
  float *s = (float *)malloc((size_t)T * sizeof(float));
  float *o = (float *)malloc((size_t)D * sizeof(float));
  float *s_ref = (float *)malloc((size_t)T * sizeof(float));
  float *o_ref = (float *)malloc((size_t)D * sizeof(float));
  if (!kc || !vc || !q || !block_table || !row_off || !s || !o ||
      !s_ref || !o_ref) { fprintf(stderr, "oom\n"); return 2; }

  for (size_t i = 0; i < cache_elems; i++) kc[i] = sm64_float(&seed);
  for (size_t i = 0; i < cache_elems; i++) vc[i] = sm64_float(&seed);
  for (int d = 0; d < D; d++) q[d] = sm64_float(&seed);
  // Block table: the first NB entries of a Fisher-Yates shuffle of
  // the physical pages (pages scattered, as in a live vLLM server).
  int32_t *pages = (int32_t *)malloc((size_t)NP * sizeof(int32_t));
  for (int i = 0; i < NP; i++) pages[i] = i;
  for (int i = NP - 1; i > 0; i--) {
    int j = (int)sm64_range(&seed, (uint32_t)(i + 1));
    int32_t tmp = pages[i]; pages[i] = pages[j]; pages[j] = tmp;
  }
  for (int b = 0; b < NB; b++) block_table[b] = pages[b];
  free(pages);
  // Resolve the two-level indirection into byte offsets (the index
  // stream the gathers consume).
  for (int t = 0; t < T; t++)
    row_off[t] = (int64_t)(((size_t)block_table[t / PS] * PS + t % PS) *
                           D * sizeof(float));

  const float inv_sqrt_d = 1.0f / sqrtf((float)D);

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  attn_rvv(q, kc, vc, row_off, T, D, inv_sqrt_d, s, o);
#else
  attn_scalar(q, kc, vc, row_off, T, D, inv_sqrt_d, s, o);
#endif
  ROI_END();

  attn_scalar(q, kc, vc, row_off, T, D, inv_sqrt_d, s_ref, o_ref);
  int pass = memcmp(o, o_ref, (size_t)D * sizeof(float)) == 0;
  print_checksum("paged_attn", o, D);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
