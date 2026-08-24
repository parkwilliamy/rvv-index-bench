/*
 * RIVEC_SPMV — CSR sparse matrix-vector multiply.
 *
 * Semantics follow the RiVec benchmark suite's _spmv kernel
 * (Ramirez et al., "A RISC-V Simulator and Benchmark Suite for
 * Designing and Evaluating Vector Architectures", TACO 2020):
 *     y[r] = sum_{k in row r} a[k] * x[ja[k]]
 * over a CSR matrix (ia offsets, ja column indices, a values).
 * Upstream indirect access (RALC88/riscv-vectorized-benchmark-suite
 * @ master, `_spmv/src/spmv.c`): serial `sum += a[idx] * x[ja[idx]]`
 * at L50; vectorized gather `vx = _MM_LOAD_INDEX_f64(x, v_idx_row,
 * gvl)` at L37. The upstream kernel is f64 with 64-bit indices; this
 * port is f32 with int32 indices per the suite's e32m1-only rule
 * (and uses raw intrinsics, not RiVec's _MM_* macros).
 *
 * RVV formulation: per row, vle32 the column indices -> vsll 2 ->
 * vluxei32 x[] gather, unit-stride vle32 of the values, unfused
 * vfmul + ordered redsum — the accumulation order matches the scalar
 * loop exactly, so y is bit-identical.
 *
 * Why the compiler is not trusted here: gcc CAN autovectorize plain
 * CSR spmv, but only as the e64m2 widening chain
 * (vle32 -> vsext.vf2 -> vsll -> vluxei64); the suite requires the
 * deterministic e32m1 form, so the kernel is written with intrinsics
 * and autovec stays off (see Makefile).
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

static void spmv_scalar(int32_t nrows, const int32_t *ia,
                        const int32_t *ja, const float *a,
                        const float *x, float *y) {
  for (int32_t r = 0; r < nrows; r++) {
    float sum = 0.0f;
    for (int32_t k = ia[r]; k < ia[r + 1]; k++)
      sum += a[k] * x[ja[k]];
    y[r] = sum;
  }
}

#ifdef USE_RISCV_VECTOR
static void spmv_rvv(int32_t nrows, const int32_t *ia,
                     const int32_t *ja, const float *a,
                     const float *x, float *y) {
  for (int32_t r = 0; r < nrows; r++) {
    const int32_t start = ia[r];
    const int32_t nnz = ia[r + 1] - start;
    vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int32_t k = 0; k < nnz; ) {
      size_t vl = __riscv_vsetvl_e32m1((size_t)(nnz - k));
      vuint32m1_t voff = __riscv_vsll_vx_u32m1(
          __riscv_vreinterpret_v_i32m1_u32m1(
              __riscv_vle32_v_i32m1(&ja[start + k], vl)),
          2, vl);
      vfloat32m1_t vx = __riscv_vluxei32_v_f32m1(x, voff, vl);
      vfloat32m1_t va = __riscv_vle32_v_f32m1(&a[start + k], vl);
      acc = __riscv_vfredosum_vs_f32m1_f32m1(
          __riscv_vfmul_vv_f32m1(va, vx, vl), acc, vl);
      k += (int32_t)vl;
    }
    y[r] = __riscv_vfmv_f_s_f32m1_f32(acc);
  }
}
#endif

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: rivec_spmv <rows> <cols> <nnz_per_row> <seed>\n");
    return 2;
  }
  int32_t nrows = atoi(argv[1]), ncols = atoi(argv[2]);
  int32_t nnz_row = atoi(argv[3]);
  uint64_t seed = (uint64_t)strtoull(argv[4], NULL, 0);
  if (nrows <= 0 || ncols <= 0 || nnz_row <= 0 || nnz_row > ncols) {
    fprintf(stderr, "error: bad sizes\n");
    return 2;
  }
  int64_t nnz = (int64_t)nrows * nnz_row;
  int32_t *ia = (int32_t *)malloc(((size_t)nrows + 1) * sizeof(int32_t));
  int32_t *ja = (int32_t *)malloc((size_t)nnz * sizeof(int32_t));
  float *a = (float *)malloc((size_t)nnz * sizeof(float));
  float *x = (float *)malloc((size_t)ncols * sizeof(float));
  float *y = (float *)malloc((size_t)nrows * sizeof(float));
  float *ref = (float *)malloc((size_t)nrows * sizeof(float));
  if (!ia || !ja || !a || !x || !y || !ref) {
    fprintf(stderr, "oom\n"); return 2;
  }
  for (int32_t r = 0; r <= nrows; r++) ia[r] = r * nnz_row;
  for (int64_t k = 0; k < nnz; k++)
    ja[k] = (int32_t)sm64_range(&seed, (uint32_t)ncols);
  for (int64_t k = 0; k < nnz; k++) a[k] = sm64_float(&seed);
  for (int32_t c = 0; c < ncols; c++) x[c] = sm64_float(&seed);

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  spmv_rvv(nrows, ia, ja, a, x, y);
#else
  spmv_scalar(nrows, ia, ja, a, x, y);
#endif
  ROI_END();

  spmv_scalar(nrows, ia, ja, a, x, ref);
  int pass = memcmp(y, ref, (size_t)nrows * sizeof(float)) == 0;
  print_checksum("rivec_spmv", y, nrows);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
