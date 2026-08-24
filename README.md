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

*Kernel sources:* FBGEMM `EmbeddingSpMDM` — reference semantics in
[`src/RefImplementations.cc:1551-1565` @ v0.6.0](https://github.com/pytorch/FBGEMM/blob/v0.6.0/src/RefImplementations.cc#L1551-L1565)
(the indirect row read is `input + input_stride * idx + j`, L1559);
PARAM embedding uBenchmarks
(<https://github.com/facebookresearch/param>); RecNMP
(arXiv:1912.12953) for the memory-boundedness characterization.

*Indirect access sites (this repo):* index load `src/sls.c:73`
(`vle32` of the bag ids), gather `src/sls.c:75` (`vluxei32` of table
component `d` across the bag's rows).

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
kernel in vLLM
[`csrc/attention/attention_kernels.cu` @ v0.4.2](https://github.com/vllm-project/vllm/blob/v0.4.2/csrc/attention/attention_kernels.cu):
block-table indirection at L247/L370
(`physical_block_number = block_table[block_idx]`) and the derived
K/V pointers at L254/L372.

*Indirect access sites (this repo):* index loads
`src/paged_attn.c:90` and `:108` (`vle32` of the resolved per-token
byte offsets), K gather `src/paged_attn.c:93` (`vluxei32`, score
phase), V gather `src/paged_attn.c:109` (`vluxei32`, output phase).

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

*Kernel sources:* MegaBlocks — Gale et al., MLSys 2023; the token
gather op in
[`megablocks/ops/gather.py:13` @ v0.5.1](https://github.com/stanford-futuredata/megablocks/blob/v0.5.1/megablocks/ops/gather.py#L13)
(`kernels.gather(x, indices, bin_ids, ...)`), with the scatter as its
backward (L20); Switch Transformers — Fedus et al., JMLR 2022 (top-1
routing); token permute/unpermute kernels in vLLM and DeepSpeed-MoE.

*Indirect access sites (this repo):* index loads
`src/moe_dispatch.c:77` and `:90` (`vle32` of the permutation),
dispatch gather `src/moe_dispatch.c:80` (`vluxei32` of `x`) with
strided store `:81` (`vsse32` into `xperm`), gate gather `:92`
(`vluxei32`), strided reload `:95` (`vlse32` of `xperm`), combine
scatter `src/moe_dispatch.c:103` (`vsuxei32` into `y`).

*Why intrinsics:* vectorizing the combine scatter requires proving
`src[]` is duplicate-free — a whole-program property no dependence
analysis derives — and compilers will not emit indexed stores under
possible intra-vector aliasing. The dim-outer interchange that makes
the slot axis vectorizable additionally defeats their cost models.

### `src/fasttext_sg.c` — fastText skipgram + negative sampling (training step)

The SGD update at the heart of fastText training: each example's input
word is a bag of L subword/ngram ids into `wi[Nin][D]`, pooled by
average into the hidden vector; one target plus K negative samples
index `wo[Nout][D]` for sigmoid dot-products, then **both matrices are
updated in place** (grad accumulation + row updates). Four indexed
phases per step: hidden gather-average, per-sample dot gathers, the
wo gather+`vsuxei32` scatter update, and the wi scatter-add — the
suite's densest gather/scatter mix (5 `vluxei32` + 3 `vsuxei32`
sites). The sigmoid uses fastText's own 512-bin lookup table. Ids are
drawn distinct within each bag/sample set so the scatters are
conflict-free and results stay bit-exact; the unigram^0.75 negative
distribution and hash collisions are simplified to uniform-distinct.

*Args:* `nin nout dim(pow2) examples bag_len negatives seed`

*Kernel sources:* fastText — Bojanowski et al., TACL 2017; upstream
kernel at v0.9.2:
[`src/model.cc:36-42`](https://github.com/facebookresearch/fastText/blob/v0.9.2/src/model.cc#L36-L42)
(`computeHidden`: `hidden.addRow(*wi_, *it)` over the input ids, L41)
and
[`src/loss.cc:90-104`](https://github.com/facebookresearch/fastText/blob/v0.9.2/src/loss.cc#L90-L104)
(`binaryLogistic`: `wo_->dotRow` L93, `grad.addRow(*wo_, target)` L95,
`wo_->addVectorToRow` L96), negatives loop `src/loss.cc:144-146`;
negative sampling — Mikolov et al., NeurIPS 2013.

*Indirect access sites (this repo):* index loads
`src/fasttext_sg.c:128`, `:141`, `:160`, `:180` (`vle32` of bag /
sample ids), hidden gather `src/fasttext_sg.c:131` (`vluxei32` of
`wi`), dot gather `:145` (`vluxei32` of `wo`), update gather+scatter
`:164`/`:172` (`vluxei32` pre-values, `vsuxei32` updated `wo` rows),
input scatter-add `:183`/`:184` (`vluxei32` + `vsuxei32` on `wi`).

*Why intrinsics:* examples are sequential SGD steps (each reads rows
the previous wrote), the weight updates are indirect scatter-adds
whose legality requires duplicate-free ids (a data property invisible
to dependence analysis), the sigmoid LUT breaks the loop between the
dot products and the updates, and the pooling/gradient reductions
cannot be reassociated without `-ffast-math`.

Local line references are to the current sources — regenerate with
`grep -n '__riscv_vluxei32\|__riscv_vsuxei32\|__riscv_vlse32\|__riscv_vsse32\|__riscv_vle32_v_i32m1' src/*.c`
after editing; upstream references are pinned to release tags.

### GAP-derived graph kernels (`src/gap_*.c`)

Standalone C ports of the GAP Benchmark Suite kernels (Beamer et al.,
arXiv:1508.03619; <https://github.com/sbeamer/gapbs>, pinned @
`2972aeb`) with the gather formulations developed in the local gapbs
RVV port. Graphs are deterministic seeded uniform undirected
multigraphs built in-binary (`common/gap_graph.h`); triangle counting
is excluded (its merge-based intersection has no indexed-load form).
For cc/cc_sv, whose gather-then-consume snapshots would otherwise be
vector-length-dependent, the snapshot block is fixed at 64 elements
IN THE ALGORITHM, so scalar, vector, and host builds follow
bit-identical trajectories at every VLEN. None of the five kernels
auto-vectorizes (verified on gapbs with gcc -O3 -ftree-vectorize:
zero kernel gathers emitted); the per-kernel blockers are noted
below.

#### `src/gap_bfs.c` — direction-optimizing BFS

Bottom-up frontier test as an int32-flag gather; top-down steps and
control (alpha=15/beta=18) scalar per GAPBS. Exact (frontier
read-only per step). Blocker: data-dependent early `break` — needs
unprovably-safe speculative gathers. Upstream indirect access:
[`bfs.cc:54`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/bfs.cc#L54) (`front.get_bit(v)` in BUStep).
Local sites: index load `src/gap_bfs.c:95`, flag gather `:98`.
*Args:* `nodes edges source seed`

#### `src/gap_cc_sv.c` — Shiloach-Vishkin components

comp[] gathered per 64-block, hooking scalar per lane; fixpoint
iteration tolerates the snapshot. Blocker: data-dependent store
(`comp[high]=low`) needs intra-vector conflict detection. Upstream:
[`cc_sv.cc:63`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/cc_sv.cc#L63) (`comp[v]` in the scan). Local
sites: index load `src/gap_cc_sv.c:59`, comp gather `:62`.
*Args:* `nodes edges seed`

#### `src/gap_cc.c` — Afforest components

Final link phase gathers comp[] per 64-block and calls Link only on
lanes whose snapshot label differs from comp[u] (equal-at-snapshot
provably means same set). Blocker: the loop body is a call to Link's
data-dependent while-chase. Upstream: [`cc.cc:42-53`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/cc.cc#L42-L53)
(Link's comp chase) driven from [`cc.cc:123-147`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/cc.cc#L123-L147).
Local sites: index load `src/gap_cc.c:98`, comp gather `:101`.
*Args:* `nodes edges seed`

#### `src/gap_sssp.c` — delta-stepping SSSP

Serialized GAPBS delta-stepping (bucket fusion included); RelaxEdges
pulls {v,w} pairs with two stride-8 `vlse32` loads and prefilters
with a `vluxei32` dist gather — provably exact via dist
monotonicity, so dist, buckets, and push order are bit-identical.
Blocker: growable-bucket pushes (allocation) in the loop. Upstream:
[`sssp.cc:72`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/sssp.cc#L72) (`dist[wn.v]` in RelaxEdges). Local
sites: strided index/weight loads `src/gap_sssp.c:93`/`:95`, dist
gather `:102`.
*Args:* `nodes edges source delta seed`

#### `src/gap_bc.c` — Brandes betweenness centrality

Scalar forward BFS (path counts f64, per-edge successor bytes, depth
slices); the backward dependency accumulation gathers `deltas[]`
(f32) per successor strip — exact (depth d+1 finalized before d).
Only the backward pass is inside the ROI. Blocker: conditional FP
division (masked-divide speculation) plus f64/f32 mixing. Upstream:
[`bc.cc:130`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/bc.cc#L130) (`path_counts[v]`, `deltas[v]`). Local
sites: index load `src/gap_bc.c:123`, deltas gather `:126`.
*Args:* `nodes edges source seed`

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
| fasttext_sg | `100000 20000 64 512 8 5 1` | wi `-9.62623206884561` (`-0x1.340a17d5ecbfp+3`), wo `0.057688921787500735` (`0x1.d896700d66f4p-5`) |
| gap_bfs | `65536 262144 3 1` | `2151931909` (`0x1.0087c00ap+31`) |
| gap_cc | `65536 262144 1` | `698770` (`0x1.55324p+19`) |
| gap_cc_sv | `65536 262144 1` | `698770` (`0x1.55324p+19`) |
| gap_sssp | `65536 262144 3 16 1` | `6563865358780` (`0x1.7e11373c6fp+42`) |
| gap_bc | `65536 262144 3 1` | `376576.9998091124` (`0x1.6fc03ffcdf5cp+18`) |

Regenerate with `make host` and running the `_host` binaries with the
args above.

## ROI convention

`ROI_BEGIN()`/`ROI_END()` (from `common/roi.h`, ported from RiVec)
wrap the timed kernel only — input generation, the index-resolution
prologue (paged_attn), the routing sort (moe_dispatch), and
verification are all outside the ROI. Under `-DUSE_M5OPS` each run
produces one `m5_reset_stats`→`m5_dump_reset_stats` pair, so gem5
stats section #1 is the kernel ROI.
