/*
 * SLS — SparseLengthsSum / EmbeddingBag(mode='sum') microbenchmark.
 *
 * Semantics follow FBGEMM's EmbeddingSpMDM operator (a.k.a.
 * SparseLengthsSum in Caffe2, EmbeddingBag in PyTorch), the dominant
 * kernel of DLRM-class recommendation inference:
 *     out[b][d] = sum_{i < L} table[idx[b][i]][d]
 * Sources for the kernel and its characterization:
 *   - FBGEMM EmbeddingSpMDM: https://github.com/pytorch/FBGEMM
 *     (wiki "Recent feature additions": JIT'd EmbeddingSpMDM)
 *   - PARAM uBenchmarks (embedding lookup):
 *     https://github.com/facebookresearch/param
 *   - RecNMP, arXiv:1912.12953 (SLS memory-boundedness)
 * Fixed bag length L per the workload evidence that fixed-length
 * pooling dominates production inference; indices are uniform over
 * the table (pass a different seed per run for a different draw).
 *
 * RVV formulation: the table stays ROW-MAJOR and the indices stay
 * INT64 (PyTorch EmbeddingBag's index dtype; FBGEMM's IndexType) —
 * upstream element sizes, not narrowed. The vector axis is the BAG:
 * vle64 the bag's ids, vsll to byte offsets, vluxei64 one f32
 * component d from L different rows (mixed-EEW gather: 64-bit
 * offsets, 32-bit data), ordered redsum, loop d — the canonical
 * index-load -> gather chain (base + (idx << log2(D*4))).
 *
 * Why the compiler cannot do this on its own: on the natural layout
 * the pooling loop nest is (b, i, d) with the indirection in the ROW
 * BASE; auto-vectorizers only vectorize the inner unit-stride d loop
 * (no indexed access). Producing the gather form requires a loop
 * interchange to (b, d, i) plus reassociating the FP pooling
 * reduction across the interchange — a transformation gcc/llvm will
 * not perform (FP reassociation is illegal without -ffast-math, and
 * the interchange across an indirect base defeats their dependence
 * analysis).
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

// Scalar reference (also the serial variant's timed kernel). Loop
// order (b, d, i ascending) matches the RVV ordered-redsum exactly,
// so vector and scalar results are bit-identical.
static void sls_scalar(const float *table, const int64_t *ids,
                       int B, int L, int D, float *out) {
  for (int b = 0; b < B; b++)
    for (int d = 0; d < D; d++) {
      float acc = 0.0f;
      for (int i = 0; i < L; i++)
        acc += table[(size_t)ids[(size_t)b * L + i] * D + d];
      out[(size_t)b * D + d] = acc;
    }
}

#ifdef USE_RISCV_VECTOR
static void sls_rvv(const float *table, const int64_t *ids,
                    int B, int L, int D, float *out) {
  const int shift = __builtin_ctz((unsigned)D) + 2; // id -> byte offset
  for (int b = 0; b < B; b++) {
    const int64_t *bag = &ids[(size_t)b * L];
    for (int d = 0; d < D; d++) {
      vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
      size_t vl;
      for (int i = 0; i < L; i += (int)vl) {
        vl = __riscv_vsetvl_e32m1((size_t)(L - i));
        vuint64m2_t voff = __riscv_vsll_vx_u64m2(
            __riscv_vreinterpret_v_i64m2_u64m2(
                __riscv_vle64_v_i64m2(&bag[i], vl)),
            shift, vl);
        vfloat32m1_t vals = __riscv_vluxei64_v_f32m1(&table[d], voff, vl);
        acc = __riscv_vfredosum_vs_f32m1_f32m1(vals, acc, vl);
      }
      out[(size_t)b * D + d] = __riscv_vfmv_f_s_f32m1_f32(acc);
    }
  }
}
#endif

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: sls <rows> <dim(pow2)> <bags> <bag_len> <seed>\n");
    return 2;
  }
  const int N = atoi(argv[1]), D = atoi(argv[2]);
  const int B = atoi(argv[3]), L = atoi(argv[4]);
  uint64_t seed = (uint64_t)strtoull(argv[5], NULL, 0);
  if (!is_pow2(D) || N <= 0 || B <= 0 || L <= 0) {
    fprintf(stderr, "error: dim must be a power of two, sizes > 0\n");
    return 2;
  }

  float *table = (float *)malloc((size_t)N * D * sizeof(float));
  int64_t *ids = (int64_t *)malloc((size_t)B * L * sizeof(int64_t));
  float *out = (float *)malloc((size_t)B * D * sizeof(float));
  float *ref = (float *)malloc((size_t)B * D * sizeof(float));
  if (!table || !ids || !out || !ref) { fprintf(stderr, "oom\n"); return 2; }

  for (size_t i = 0; i < (size_t)N * D; i++) table[i] = sm64_float(&seed);
  for (size_t i = 0; i < (size_t)B * L; i++)
    ids[i] = (int64_t)sm64_range(&seed, (uint32_t)N);

  ROI_BEGIN();
#ifdef USE_RISCV_VECTOR
  sls_rvv(table, ids, B, L, D, out);
#else
  sls_scalar(table, ids, B, L, D, out);
#endif
  ROI_END();

  sls_scalar(table, ids, B, L, D, ref);
  int pass = memcmp(out, ref, (size_t)B * D * sizeof(float)) == 0;
  print_checksum("sls", out, (int64_t)B * D);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
