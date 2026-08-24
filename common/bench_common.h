// Shared helpers for the rvv-index-bench kernels: deterministic input
// generation (seeded splitmix64 — never wall-clock, so every build and
// host produces identical inputs) and an order-stable checksum.
//
// All kernels are built with -ffp-contract=off and use unfused
// mul/add in their vector paths, so the scalar reference, the RVV
// kernel, and a native host build produce BIT-IDENTICAL results; the
// checksum is printed in %a (hex float) so goldens are exact string
// matches, not tolerance bands.
#ifndef RVV_INDEX_BENCH_COMMON_H
#define RVV_INDEX_BENCH_COMMON_H

#include <stdint.h>
#include <stdio.h>

static inline uint64_t sm64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// float in [-1, 1)
static inline float sm64_float(uint64_t *state) {
  return (float)((int64_t)(sm64_next(state) >> 40) - (1 << 23)) /
         (float)(1 << 23);
}

// uniform integer in [0, n)
static inline uint32_t sm64_range(uint64_t *state, uint32_t n) {
  return (uint32_t)(sm64_next(state) % n);
}

// Sequential double-accumulation checksum: order-stable by
// construction, printed exactly.
static inline void print_checksum(const char *tag, const float *v,
                                  int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; i++)
    acc += (double)v[i];
  printf("checksum %s: %.17g (%a)\n", tag, acc, acc);
}

static inline int is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

#endif // RVV_INDEX_BENCH_COMMON_H
