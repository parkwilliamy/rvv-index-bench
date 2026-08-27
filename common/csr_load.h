/*
 * Loader for RiVec `_spmv` binary CSR images (`.csr`, produced by
 * rivec/_spmv/mtx2csr.c from a SuiteSparse .mtx). Layout is
 *     uint64 magic = 0x31525343564D5053 ("SPMVCSR1")
 *     uint64 M, N, NZ
 *     uint64 ia[M+1], ja[NZ]
 *     double a[NZ]
 * (see rivec/_spmv/src/csr_format.h).
 *
 * TWO upstream quirks are normalized here, both deliberate:
 *   - ja[] in the image is 1-BASED (the .mtx parser it mirrors keeps
 *     the raw column field). It is converted to 0-based on load, so
 *     the gather is `x[ja[k]]` over a correctly sized x[N] instead of
 *     the upstream off-by-one that reads x[N].
 *   - the .mtx fields are consumed as "col row value", i.e. the image
 *     is the transpose of the mathematical matrix. That is irrelevant
 *     here: the kernel's access pattern (row-segment walk + gather of
 *     x by column index) is identical either way, and the in-binary
 *     scalar reference defines correctness.
 *
 * The image carries no x vector; the caller generates one from the
 * seeded RNG. All of this runs before ROI_BEGIN().
 */
#ifndef RVV_INDEX_BENCH_CSR_LOAD_H
#define RVV_INDEX_BENCH_CSR_LOAD_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SPMV_CSR_MAGIC 0x31525343564D5053ULL

// On success sets *nrows/*ncols/*nnz and allocates ia/ja/a.
static inline int csr_load(const char *path, int32_t *nrows, int32_t *ncols,
                           int64_t *nnz, int32_t **ia_out, int64_t **ja_out,
                           double **a_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "error: cannot open '%s'\n", path); return 0; }
    uint64_t magic = 0, hdr[3];
    if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != SPMV_CSR_MAGIC) {
        fprintf(stderr, "error: '%s' is not a binary .csr image\n", path);
        fclose(f); return 0;
    }
    if (fread(hdr, sizeof(uint64_t), 3, f) != 3) {
        fprintf(stderr, "error: truncated CSR header in '%s'\n", path);
        fclose(f); return 0;
    }
    const uint64_t M = hdr[0], N = hdr[1], NZ = hdr[2];
    if (M > 0x7fffffffULL || N > 0x7fffffffULL) {
        fprintf(stderr, "error: '%s' exceeds int32 dimensions\n", path);
        fclose(f); return 0;
    }
    uint64_t *ia64 = (uint64_t *)malloc((size_t)(M + 1) * sizeof(uint64_t));
    uint64_t *ja64 = (uint64_t *)malloc((size_t)NZ * sizeof(uint64_t));
    double *a = (double *)malloc((size_t)NZ * sizeof(double));
    if (!ia64 || !ja64 || !a) { fclose(f); return 0; }
    if (fread(ia64, sizeof(uint64_t), M + 1, f) != M + 1 ||
        fread(ja64, sizeof(uint64_t), NZ, f) != NZ ||
        fread(a, sizeof(double), NZ, f) != NZ) {
        fprintf(stderr, "error: truncated CSR arrays in '%s'\n", path);
        fclose(f); return 0;
    }
    fclose(f);

    int32_t *ia = (int32_t *)malloc((size_t)(M + 1) * sizeof(int32_t));
    int64_t *ja = (int64_t *)malloc((size_t)NZ * sizeof(int64_t));
    if (!ia || !ja) return 0;
    for (uint64_t r = 0; r <= M; r++) ia[r] = (int32_t)ia64[r];
    for (uint64_t k = 0; k < NZ; k++) {
        // 1-based -> 0-based; clamp defensively so a stray 0 column
        // (absent from well-formed .mtx) cannot index out of bounds.
        int64_t c = (int64_t)ja64[k] - 1;
        ja[k] = c < 0 ? 0 : (c >= (int64_t)N ? (int64_t)N - 1 : c);
    }
    free(ia64); free(ja64);
    *nrows = (int32_t)M; *ncols = (int32_t)N; *nnz = (int64_t)NZ;
    *ia_out = ia; *ja_out = ja; *a_out = a;
    return 1;
}

#endif // RVV_INDEX_BENCH_CSR_LOAD_H
