/*
 * FASTTEXT_SG — fastText skipgram-with-negative-sampling training
 * step microbenchmark.
 *
 * Semantics follow fastText's Model::update path (computeHidden +
 * binaryLogistic + row updates): for each training example, the
 * input word is a bag of L subword/ngram ids into the input matrix
 * wi[Nin][D]; the hidden vector is their AVERAGE; one target plus K
 * negative samples index the output matrix wo[Nout][D]:
 *     hidden[d] = (1/L) * sum_i wi[bag[i]][d]              (gather)
 *     for each sample s (target l=1, negatives l=0):
 *         g[s]   = lr * (l - sigmoid(hidden . wo[s]))      (gather)
 *         grad[d] += g[s] * wo_pre[s][d]
 *         wo[s][d] = wo_pre[s][d] + g[s] * hidden[d]       (scatter)
 *     wi[bag[i]][d] += grad[d] / L                         (scatter)
 * The sigmoid uses fastText's 512-bin lookup table (initSigmoid).
 * Sources for the kernel:
 *   - fastText: Bojanowski et al., "Enriching Word Vectors with
 *     Subword Information", TACL 2017;
 *     https://github.com/facebookresearch/fastText
 *     (src/model.cc computeHidden/binaryLogistic/update,
 *      sigmoid table in src/utils).
 *   - negative sampling: Mikolov et al., NeurIPS 2013.
 * Simplifications, stated: fixed bag length L; ids drawn uniformly
 * and DISTINCT within each bag / sample set (fastText's hash
 * collisions and unigram^0.75 negative table are input-distribution
 * details; distinctness makes the scatters conflict-free so vector
 * and scalar results are bit-identical).
 *
 * RVV formulation: both matrices stay ROW-MAJOR; the vector axis is
 * the BAG (input side) or the SAMPLE SET (output side), dim-outer:
 * vle32 ids -> vsll -> vluxei32 gathers for hidden/dots/grad, and
 * vsuxei32 SCATTERS for both weight updates. Unfused mul/add and
 * ordered redsums keep everything bit-exact vs the scalar reference.
 *
 * Why the compiler cannot do this on its own: examples are
 * sequential SGD steps (each reads rows the previous one wrote), the
 * weight updates are indirect scatter-adds whose legality needs the
 * ids to be duplicate-free (a data property no dependence analysis
 * derives), the sigmoid lookup breaks the dot-product/update loop,
 * and the pooling/gradient reductions cannot be reassociated without
 * -ffast-math. Autovectorizers vectorize none of it.
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

#define SIGMOID_TABLE_SIZE 512
#define MAX_SIGMOID 8.0f
static float sigmoid_table[SIGMOID_TABLE_SIZE + 1];

// fastText utils::initSigmoid
static void init_sigmoid(void) {
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    float x = ((float)i * 2.0f * MAX_SIGMOID) / SIGMOID_TABLE_SIZE -
              MAX_SIGMOID;
    sigmoid_table[i] = 1.0f / (1.0f + expf(-x));
  }
}

static float sigmoid_lut(float x) {
  if (x < -MAX_SIGMOID) return 0.0f;
  if (x > MAX_SIGMOID) return 1.0f;
  int i = (int)((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE /
                (2.0f * MAX_SIGMOID));
  return sigmoid_table[i];
}

// One SGD step, scalar. Loop orders mirror the vector kernel's op
// order exactly (d-outer, index ascending), so results are
// bit-identical. Also the serial variant's timed kernel.
static void step_scalar(float *wi, float *wo, const int32_t *bag, int L,
                        const int32_t *samples, int NS, float lr,
                        int D, float *hidden, float *g, float *grad) {
  const float invL = 1.0f / (float)L;
  for (int d = 0; d < D; d++) {
    float acc = 0.0f;
    for (int i = 0; i < L; i++)
      acc += wi[(size_t)bag[i] * D + d];
    hidden[d] = acc * invL;
  }
  for (int s = 0; s < NS; s++) {
    float dot = 0.0f;
    for (int d = 0; d < D; d++)
      dot = dot + hidden[d] * wo[(size_t)samples[s] * D + d];
    float label = (s == 0) ? 1.0f : 0.0f;
    g[s] = lr * (label - sigmoid_lut(dot));
  }
  for (int d = 0; d < D; d++) {
    float gr = 0.0f;
    for (int s = 0; s < NS; s++) {
      float pre = wo[(size_t)samples[s] * D + d];
      gr += g[s] * pre;
      wo[(size_t)samples[s] * D + d] = pre + g[s] * hidden[d];
    }
    grad[d] = gr;
  }
  for (int d = 0; d < D; d++) {
    float upd = grad[d] * invL;
    for (int i = 0; i < L; i++)
      wi[(size_t)bag[i] * D + d] += upd;
  }
}

#ifdef USE_RISCV_VECTOR
static void step_rvv(float *wi, float *wo, const int32_t *bag, int L,
                     const int32_t *samples, int NS, float lr,
                     int D, float *hidden, float *g, float *grad) {
  const int shift = __builtin_ctz((unsigned)D) + 2;
  const float invL = 1.0f / (float)L;
  size_t vl;
  // hidden = average of bag rows (gather + ordered redsum per d)
  for (int d = 0; d < D; d++) {
    vfloat32m1_t acc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    for (int i = 0; i < L; i += (int)vl) {
      vl = __riscv_vsetvl_e32m1((size_t)(L - i));
      vuint32m1_t voff = __riscv_vsll_vx_u32m1(
          __riscv_vreinterpret_v_i32m1_u32m1(
              __riscv_vle32_v_i32m1(&bag[i], vl)),
          shift, vl);
      acc = __riscv_vfredosum_vs_f32m1_f32m1(
          __riscv_vluxei32_v_f32m1(&wi[d], voff, vl), acc, vl);
    }
    hidden[d] = __riscv_vfmv_f_s_f32m1_f32(acc) * invL;
  }
  // per-sample dots: sample set as the vector axis (NS lanes)
  float dots[SIGMOID_TABLE_SIZE]; // NS << table size; scratch
  for (int s0 = 0; s0 < NS; s0 += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(NS - s0));
    vuint32m1_t voff = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&samples[s0], vl)),
        shift, vl);
    vfloat32m1_t vdot = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t vw = __riscv_vluxei32_v_f32m1(&wo[d], voff, vl);
      vdot = __riscv_vfadd_vv_f32m1(
          vdot, __riscv_vfmul_vf_f32m1(vw, hidden[d], vl), vl);
    }
    __riscv_vse32_v_f32m1(&dots[s0], vdot, vl);
  }
  for (int s = 0; s < NS; s++) {
    float label = (s == 0) ? 1.0f : 0.0f;
    g[s] = lr * (label - sigmoid_lut(dots[s]));
  }
  // grad from pre-update wo rows + wo scatter update, per d
  for (int s0 = 0; s0 < NS; s0 += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(NS - s0));
    vuint32m1_t voff = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&samples[s0], vl)),
        shift, vl);
    vfloat32m1_t vg = __riscv_vle32_v_f32m1(&g[s0], vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t pre = __riscv_vluxei32_v_f32m1(&wo[d], voff, vl);
      vfloat32m1_t contrib = __riscv_vfmul_vv_f32m1(vg, pre, vl);
      float gr = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredosum_vs_f32m1_f32m1(
              contrib, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl));
      grad[d] = (s0 == 0) ? gr : grad[d] + gr;
      vfloat32m1_t upd = __riscv_vfadd_vv_f32m1(
          pre, __riscv_vfmul_vf_f32m1(vg, hidden[d], vl), vl);
      __riscv_vsuxei32_v_f32m1(&wo[d], voff, upd, vl);
    }
  }
  // input update: scatter-add grad/L into the bag rows, per d
  for (int i0 = 0; i0 < L; i0 += (int)vl) {
    vl = __riscv_vsetvl_e32m1((size_t)(L - i0));
    vuint32m1_t voff = __riscv_vsll_vx_u32m1(
        __riscv_vreinterpret_v_i32m1_u32m1(
            __riscv_vle32_v_i32m1(&bag[i0], vl)),
        shift, vl);
    for (int d = 0; d < D; d++) {
      vfloat32m1_t cur = __riscv_vluxei32_v_f32m1(&wi[d], voff, vl);
      __riscv_vsuxei32_v_f32m1(
          &wi[d], voff,
          __riscv_vfadd_vf_f32m1(cur, grad[d] * invL, vl), vl);
    }
  }
}
#endif

// draw n distinct ids in [0, range) (rejection; n << range)
static void draw_distinct(uint64_t *seed, int32_t *out, int n,
                          uint32_t range) {
  for (int i = 0; i < n; i++) {
    int32_t cand;
    int dup;
    do {
      cand = (int32_t)sm64_range(seed, range);
      dup = 0;
      for (int j = 0; j < i; j++)
        if (out[j] == cand) { dup = 1; break; }
    } while (dup);
    out[i] = cand;
  }
}

int main(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: fasttext_sg <nin> <nout> <dim(pow2)> <examples> "
            "<bag_len> <negatives> <seed>\n");
    return 2;
  }
  const int NIN = atoi(argv[1]), NOUT = atoi(argv[2]), D = atoi(argv[3]);
  const int S = atoi(argv[4]), L = atoi(argv[5]), K = atoi(argv[6]);
  uint64_t seed = (uint64_t)strtoull(argv[7], NULL, 0);
  const int NS = K + 1;
  const float lr = 0.05f;
  if (!is_pow2(D) || NIN <= L || NOUT <= NS || S <= 0 || L <= 0 ||
      K < 1 || NS > SIGMOID_TABLE_SIZE) {
    fprintf(stderr, "error: dim pow2; nin > bag_len; nout > negatives+1; "
            "negatives+1 <= 512\n");
    return 2;
  }

  init_sigmoid();
  float *wi = (float *)malloc((size_t)NIN * D * sizeof(float));
  float *wo = (float *)malloc((size_t)NOUT * D * sizeof(float));
  float *wi_ref = (float *)malloc((size_t)NIN * D * sizeof(float));
  float *wo_ref = (float *)malloc((size_t)NOUT * D * sizeof(float));
  int32_t *bags = (int32_t *)malloc((size_t)S * L * sizeof(int32_t));
  int32_t *samps = (int32_t *)malloc((size_t)S * NS * sizeof(int32_t));
  float *hidden = (float *)malloc((size_t)D * sizeof(float));
  float *g = (float *)malloc((size_t)NS * sizeof(float));
  float *grad = (float *)malloc((size_t)D * sizeof(float));
  if (!wi || !wo || !wi_ref || !wo_ref || !bags || !samps || !hidden ||
      !g || !grad) { fprintf(stderr, "oom\n"); return 2; }

  // fastText init: wi uniform in [-1/D, 1/D], wo zero
  for (size_t i = 0; i < (size_t)NIN * D; i++)
    wi[i] = sm64_float(&seed) / (float)D;
  memset(wo, 0, (size_t)NOUT * D * sizeof(float));
  for (int e = 0; e < S; e++) {
    draw_distinct(&seed, &bags[(size_t)e * L], L, (uint32_t)NIN);
    draw_distinct(&seed, &samps[(size_t)e * NS], NS, (uint32_t)NOUT);
  }
  memcpy(wi_ref, wi, (size_t)NIN * D * sizeof(float));
  memcpy(wo_ref, wo, (size_t)NOUT * D * sizeof(float));

  ROI_BEGIN();
  for (int e = 0; e < S; e++) {
#ifdef USE_RISCV_VECTOR
    step_rvv(wi, wo, &bags[(size_t)e * L], L, &samps[(size_t)e * NS],
             NS, lr, D, hidden, g, grad);
#else
    step_scalar(wi, wo, &bags[(size_t)e * L], L, &samps[(size_t)e * NS],
                NS, lr, D, hidden, g, grad);
#endif
  }
  ROI_END();

  for (int e = 0; e < S; e++)
    step_scalar(wi_ref, wo_ref, &bags[(size_t)e * L], L,
                &samps[(size_t)e * NS], NS, lr, D, hidden, g, grad);
  int pass = memcmp(wi, wi_ref, (size_t)NIN * D * sizeof(float)) == 0 &&
             memcmp(wo, wo_ref, (size_t)NOUT * D * sizeof(float)) == 0;
  print_checksum("fasttext_sg_wi", wi, (int64_t)NIN * D);
  print_checksum("fasttext_sg_wo", wo, (int64_t)NOUT * D);
  printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
