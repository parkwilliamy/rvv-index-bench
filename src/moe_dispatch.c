/*
 * MOE_DISPATCH — Mixture-of-Experts token dispatch/combine
 * microbenchmark (top-1 routing).
 *
 * Semantics follow the permutation phase of sparse MoE layers: tokens
 * are GATHERED into expert-contiguous order (dispatch), a stand-in
 * expert FFN transforms them, and results are SCATTERED back to token
 * order scaled by the router gate (combine):
 *     dispatch: xperm[p][d] = x[src[p]][d]
 *     expert:   yperm = 2*xperm + 1        (stand-in FFN)
 *     combine:  y[src[p]][d] = gate[src[p]] * yperm[p][d]
 * where src[] is the expert-grouped permutation from a counting sort
 * of the per-token expert assignment.
 * Sources for the kernel:
 *   - MegaBlocks: Gale et al., "MegaBlocks: Efficient Sparse Training
 *     with Mixture-of-Experts", MLSys 2023 (grouped permutation
 *     formulation); https://github.com/stanford-futuredata/megablocks
 *   - Switch Transformers: Fedus et al., JMLR 2022 (top-1 routing).
 *   - token permute/unpermute kernels in vLLM / DeepSpeed-MoE.
 *
 * RVV formulation: tokens stay ROW-MAJOR; the vector axis is the
 * PERMUTATION SLOT. Dispatch: vle32 src ids -> vsll -> vluxei32
 * gather of component d across tokens, strided vsse32 into xperm.
 * Combine: same index stream drives a vluxei32 gate gather and a
 * vsuxei32 SCATTER of the scaled results — this benchmark is the
 * suite's indexed-STORE coverage. The scatter is conflict-free by
 * construction (src is a permutation), and every multiply is a
 * single rounding, so scalar and vector results are bit-identical.
 *
 * Why the compiler cannot do this on its own: the dispatch loop's
 * source addresses come from a runtime permutation array — legal
 * vectorization of the combine SCATTER requires proving src[] is
 * duplicate-free (a whole-program property no dependence analysis
 * derives), and gcc/llvm will not emit indexed stores under possible
 * intra-vector aliasing. The dim-outer loop interchange that makes
 * the slot axis vectorizable also defeats their cost models on the
 * natural (slot, d) nest.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "bench_common.h"
#include "roi.h"

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#endif

static void moe_scalar(const float *x, const int32_t *src,
                       const float *gate, int T, int D,
                       float *xperm, float *y) {
  for (int p = 0; p < T; p++)
    for (int d = 0; d < D; d++)
      xperm[(size_t)p * D + d] = x[(size_t)src[p] * D + d];
  for (int p = 0; p < T; p++)
    for (int d = 0; d < D; d++) {
      float yv = 2.0f * xperm[(size_t)p * D + d] + 1.0f;
      y[(size_t)src[p] * D + d] = gate[src[p]] * yv;
    }
}

#ifdef USE_RISCV_VECTOR
static void moe_rvv(const float *x, const int32_t *src,
                    const float *gate, int T, int D,
                    float *xperm, float *y) {
  const int shift = __builtin_ctz((unsigned)D) + 2; // token -> byte offset
  const ptrdiff_t row_bytes = (ptrdiff_t)D * (ptrdiff_t)sizeof(float);
  size_t vl;
  // Dispatch: gather x rows into expert order (indexed loads,
  // strided stores).
  for (int p = 0; p < T; p += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(T - p));
    vuint32m1_t voff = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&src[p], vl)),
        shift, vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(&x[d], voff, vl);
      __riscv_vsse32_v_f32m1(&xperm[(size_t)p * D + d], row_bytes, vx, vl);
    }
  }
  // Expert FFN stand-in + combine: gate gather (indexed load) and
  // scatter back to token order (indexed stores).
  for (int p = 0; p < T; p += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(T - p));
    vuint32m1_t vtok_off = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&src[p], vl)),
        2, vl);
    vfloat32m1_t vgate = __riscv_vluxei32_v_f32m1(gate, vtok_off, vl);
    vuint32m1_t vrow_off = __riscv_vsll_vx_u32m1(vtok_off, shift - 2, vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t vxp = __riscv_vlse32_v_f32m1(
          &xperm[(size_t)p * D + d], row_bytes, vl);
      // unfused: 2*x, +1, *gate — three roundings, same as scalar
      vfloat32m1_t vy = __riscv_vfmul_vv_f32m1(
          vgate,
          __riscv_vfadd_vf_f32m1(
              __riscv_vfmul_vf_f32m1(vxp, 2.0f, vl), 1.0f, vl),
          vl);
      __riscv_vsuxei32_v_f32m1(&y[d], vrow_off, vy, vl);
    }
  }
}
#endif

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr,
            "usage: moe_dispatch <tokens> <dim(pow2)> <experts> <seed>\n");
    return 2;
  }
  const int T = atoi(argv[1]), D = atoi(argv[2]), E = atoi(argv[3]);
  uint64_t seed = (uint64_t)strtoull(argv[4], NULL, 0);
  if (!is_pow2(D) || T <= 0 || E <= 0) {
    fprintf(stderr, "error: dim must be a power of two, sizes > 0\n");
    return 2;
  }

  float *x = (float *)malloc((size_t)T * D * sizeof(float));
  float *gate = (float *)malloc((size_t)T * sizeof(float));
  int32_t *expert = (int32_t *)malloc((size_t)T * sizeof(int32_t));
  int32_t *src = (int32_t *)malloc((size_t)T * sizeof(int32_t));
  int32_t *count = (int32_t *)calloc((size_t)E + 1, sizeof(int32_t));
  float *xperm = (float *)malloc((size_t)T * D * sizeof(float));
  float *y = (float *)malloc((size_t)T * D * sizeof(float));
  float *xperm_ref = (float *)malloc((size_t)T * D * sizeof(float));
  float *y_ref = (float *)malloc((size_t)T * D * sizeof(float));
  if (!x || !gate || !expert || !src || !count || !xperm || !y ||
      !xperm_ref || !y_ref) { fprintf(stderr, "oom\n"); return 2; }

  for (size_t i = 0; i < (size_t)T * D; i++) x[i] = sm64_float(&seed);
  for (int t = 0; t < T; t++)
    gate[t] = 0.5f + 0.5f * sm64_float(&seed);  // (0, 1]
  for (int t = 0; t < T; t++)
    expert[t] = (int32_t)sm64_range(&seed, (uint32_t)E);
  // Counting sort: src[p] = token occupying expert-grouped slot p.
  for (int t = 0; t < T; t++) count[expert[t] + 1]++;
  for (int e = 0; e < E; e++) count[e + 1] += count[e];
  for (int t = 0; t < T; t++) src[count[expert[t]]++] = t;

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  moe_rvv(x, src, gate, T, D, xperm, y);
#else
  moe_scalar(x, src, gate, T, D, xperm, y);
#endif
  ROI_END();

  moe_scalar(x, src, gate, T, D, xperm_ref, y_ref);
  int pass = memcmp(y, y_ref, (size_t)T * D * sizeof(float)) == 0 &&
             memcmp(xperm, xperm_ref, (size_t)T * D * sizeof(float)) == 0;
  print_checksum("moe_dispatch", y, (int64_t)T * D);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
