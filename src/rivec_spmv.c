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
 * gvl)` at L37. This port keeps the upstream element sizes — f64
 * values/x/y and int64 column indices — and uses raw intrinsics, not
 * RiVec's _MM_* macros.
 *
 * RVV formulation (e64m1, matching RiVec's _MMR_VSETVL_E64M1): per
 * row, vle64 the column indices -> vsll 3 -> vluxei64 x[] gather,
 * unit-stride vle64 of the values, unfused vfmul + ordered redsum —
 * the accumulation order matches the scalar loop exactly, so y is
 * bit-identical.
 *
 * Why intrinsics: the FP reduction cannot be reassociated by an
 * autovectorizer without -ffast-math, and the suite requires
 * deterministic codegen, so autovec stays off (see Makefile).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "bench_common.h"
#include "csr_load.h"
#include "roi.h"

#ifdef USE_RISCV_VECTOR
#include <riscv_vector.h>
#endif

static void spmv_scalar(int32_t nrows, const int32_t *ia,
                        const int64_t *ja, const double *a,
                        const double *x, double *y) {
  for (int32_t r = 0; r < nrows; r++) {
    double sum = 0.0;
    for (int32_t k = ia[r]; k < ia[r + 1]; k++)
      sum += a[k] * x[ja[k]];
    y[r] = sum;
  }
}

#ifdef USE_RISCV_VECTOR
static void spmv_rvv(int32_t nrows, const int32_t *ia,
                     const int64_t *ja, const double *a,
                     const double *x, double *y) {
  for (int32_t r = 0; r < nrows; r++) {
    const int32_t start = ia[r];
    const int32_t nnz = ia[r + 1] - start;
    vfloat64m1_t acc = __riscv_vfmv_s_f_f64m1(0.0, 1);
    for (int32_t k = 0; k < nnz; ) {
      size_t vl = __riscv_vsetvl_e64m1((size_t)(nnz - k));
      vuint64m1_t voff = __riscv_vsll_vx_u64m1(
          __riscv_vreinterpret_v_i64m1_u64m1(
              __riscv_vle64_v_i64m1(&ja[start + k], vl)),
          3, vl);
      vfloat64m1_t vx = __riscv_vluxei64_v_f64m1(x, voff, vl);
      vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[start + k], vl);
      acc = __riscv_vfredosum_vs_f64m1_f64m1(
          __riscv_vfmul_vv_f64m1(va, vx, vl), acc, vl);
      k += (int32_t)vl;
    }
    y[r] = __riscv_vfmv_f_s_f64m1_f64(acc);
  }
}
#endif

int main(int argc, char **argv) {
  // Synthetic: <rows> <cols> <nnz_per_row> <seed>
  // File:      -f <matrix.csr> <seed>   (RiVec _spmv binary CSR image)
  const int file_mode = argc > 1 && strcmp(argv[1], "-f") == 0;
  if ((!file_mode && argc != 5) || (file_mode && argc != 4)) {
    fprintf(stderr, "usage: rivec_spmv <rows> <cols> <nnz_per_row> <seed>\n"
            "       rivec_spmv -f <matrix.csr> <seed>\n");
    return 2;
  }
  int32_t nrows = 0, ncols = 0;
  int64_t nnz = 0;
  int32_t *ia = NULL; int64_t *ja = NULL; double *a = NULL;
  uint64_t seed = (uint64_t)strtoull(argv[file_mode ? 3 : 4], NULL, 0);

  if (file_mode) {
    if (!csr_load(argv[2], &nrows, &ncols, &nnz, &ia, &ja, &a)) return 2;
  } else {
    nrows = atoi(argv[1]); ncols = atoi(argv[2]);
    const int32_t nnz_row = atoi(argv[3]);
    if (nrows <= 0 || ncols <= 0 || nnz_row <= 0 || nnz_row > ncols) {
      fprintf(stderr, "error: bad sizes\n");
      return 2;
    }
    nnz = (int64_t)nrows * nnz_row;
    ia = (int32_t *)malloc(((size_t)nrows + 1) * sizeof(int32_t));
    ja = (int64_t *)malloc((size_t)nnz * sizeof(int64_t));
    a = (double *)malloc((size_t)nnz * sizeof(double));
    if (!ia || !ja || !a) { fprintf(stderr, "oom\n"); return 2; }
    for (int32_t r = 0; r <= nrows; r++) ia[r] = r * nnz_row;
    for (int64_t k = 0; k < nnz; k++)
      ja[k] = (int64_t)sm64_range(&seed, (uint32_t)ncols);
    for (int64_t k = 0; k < nnz; k++) a[k] = (double)sm64_float(&seed);
  }

  double *x = (double *)malloc((size_t)ncols * sizeof(double));
  double *y = (double *)malloc((size_t)nrows * sizeof(double));
  double *ref = (double *)malloc((size_t)nrows * sizeof(double));
  if (!x || !y || !ref) { fprintf(stderr, "oom\n"); return 2; }
  for (int32_t c = 0; c < ncols; c++) x[c] = (double)sm64_float(&seed);

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  spmv_rvv(nrows, ia, ja, a, x, y);
#else
  spmv_scalar(nrows, ia, ja, a, x, y);
#endif
  ROI_END();

  spmv_scalar(nrows, ia, ja, a, x, ref);
  int pass = memcmp(y, ref, (size_t)nrows * sizeof(double)) == 0;
  print_checksum_f64("rivec_spmv", y, nrows);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
