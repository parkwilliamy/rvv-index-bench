# rvv-index-bench

A dedicated test suite for **indexed load/store kernels in RVV**
(`vluxei`/`vsuxei` and friends), drawn from AI inference workloads
whose hot loops are indirect. Each benchmark keeps its upstream
memory layout and semantics, is written with **raw `__riscv_*`
intrinsics** (no macro layers), carries gem5 ROI markers, and
verifies bit-exactly against an in-binary scalar reference plus
host-generated golden checksums.

Layout: one source per benchmark under `src/`, per-kernel binaries under
`bin/<kernel>/`, shared headers in `common/` (`roi.h` ROI markers ported from
RiVec's `common/roi.h`), golden checksums in `golden/`; build
conventions follow the RiVec suite's gcc Makefile.

## Build

```
make            # all benchmarks, serial + vector (RISC-V, m5ops ROI)
make host       # native builds, used to (re)generate golden/
make RDCYCLE=1  # rdcycle ROI prints instead of m5ops
make NO_M5OPS=1 # ROI markers compiled away
```

Toolchain: `riscv64-unknown-elf-gcc` at `$RVTOOLS` (default
`~/opt/riscv/bin`); m5ops need `libm5.a` under
`$GEM5_ROOT/util/m5/build/riscv/out`. Everything is built with
`-ffp-contract=off` and the vector kernels use unfused mul/add and
*ordered* reductions, so **serial, vector, and host builds produce
bit-identical results** — goldens are exact string matches, not
tolerance bands.

## Benchmarks

### `src/sls.c` — SparseLengthsSum / EmbeddingBag-sum

`out[b][d] = Σ_i table[idx[b][i]][d]` — the embedding-pooling kernel
that dominates DLRM-class recommendation inference. Table stays
row-major (as in FBGEMM); the **bag** is the vector axis: `vle32` the
bag's ids → `vsll` → `vluxei32` one component from L rows → ordered
redsum, looped over the dim. Fixed bag length `L` (production
inference is dominated by fixed-length pooling).

*Args:* `rows dim(pow2) bags bag_len seed`

*Kernel sources:* FBGEMM `EmbeddingSpMDM`
(<https://github.com/pytorch/FBGEMM>), PARAM embedding uBenchmarks
(<https://github.com/facebookresearch/param>), RecNMP
(arXiv:1912.12953) for the memory-boundedness characterization.

*Why intrinsics:* on the natural layout the loop nest is `(b, i, d)`
with the indirection in the row base; autovectorizers only vectorize
the inner unit-stride `d` loop. The gather form needs a loop
interchange to `(b, d, i)` **plus reassociation of the FP pooling
reduction across that interchange** — illegal without `-ffast-math`
and beyond gcc/llvm dependence analysis through an indirect base.

### `src/paged_attn.c` — vLLM-style paged attention (single-query decode)

K/V caches live in fixed-size pages; a block table maps logical
blocks to scattered physical pages. Scores, softmax, and the
V-weighted sum all reach token rows through that indirection. A
scalar prologue resolves block-table + page-offset into a per-token
byte-offset array (the index stream); both attention phases then run
token-strips: `vle32` offsets → `vluxei32` K (or V) per component.

*Args:* `dim(pow2) page_size num_pages ctx_len seed`

*Kernel sources:* PagedAttention — Kwon et al., SOSP 2023; reference
kernels in <https://github.com/vllm-project/vllm> (`csrc/attention`).

*Why intrinsics:* the token→address map goes through a runtime table
lookup plus div/mod — non-affine, so no autovectorizer can form the
gather — and the two FP reductions (dot product, weighted sum) can't
be vectorized without reassociation permissions; softmax between the
phases splits the loop besides.

### `src/moe_dispatch.c` — MoE token dispatch + combine (top-1 routing)

The permutation phase of sparse Mixture-of-Experts layers: tokens are
gathered into expert-contiguous order (`xperm[p] = x[src[p]]`), a
stand-in expert FFN transforms them, and results scatter back to
token order scaled by the router gate. Dispatch = `vluxei32` gathers
+ `vsse32` strided stores; combine = a `vluxei32` gate gather and a
**`vsuxei32` scatter** — the suite's indexed-store coverage. The
scatter is conflict-free by construction (src is a permutation).

*Args:* `tokens dim(pow2) experts seed`

*Kernel sources:* MegaBlocks — Gale et al., MLSys 2023
(<https://github.com/stanford-futuredata/megablocks>); Switch
Transformers — Fedus et al., JMLR 2022 (top-1 routing); token
permute/unpermute kernels in vLLM and DeepSpeed-MoE.

*Why intrinsics:* vectorizing the combine scatter requires proving
`src[]` is duplicate-free — a whole-program property no dependence
analysis derives — and compilers will not emit indexed stores under
possible intra-vector aliasing. The dim-outer interchange that makes
the slot axis vectorizable additionally defeats their cost models.

## Verification & goldens

Every run executes the scalar reference after the ROI and compares
**bit-exactly** (`memcmp`), printing `RESULT: PASS|FAIL` and an
order-stable checksum in `%.17g` and `%a` (hex). `golden/` holds
host-generated outputs for the default configs; RISC-V builds (and
gem5 runs) must reproduce them exactly:

| benchmark | default args | checksum |
|---|---|---|
| sls | `100000 64 512 64 1` | `-167.62071907520294` (`-0x1.4f3dcee4p+7`) |
| paged_attn | `128 16 4096 2048 1` | `0.22962264859961579` (`0x1.d64466314p-3`) |
| moe_dispatch | `4096 64 8 1` | `131969.35790285049` (`0x1.01c0adcfc2b7p+17`) |

Regenerate with `make host` and running the `_host` binaries with the
args above.

## ROI convention

`ROI_BEGIN()`/`ROI_END()` (from `common/roi.h`, ported from RiVec)
wrap the timed kernel only — input generation, the index-resolution
prologue (paged_attn), the routing sort (moe_dispatch), and
verification are all outside the ROI. Under `-DUSE_M5OPS` each run
produces one `m5_reset_stats`→`m5_dump_reset_stats` pair, so gem5
stats section #1 is the kernel ROI.
