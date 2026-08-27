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
row-major and the indices int64 (PyTorch EmbeddingBag's index dtype;
FBGEMM's `IndexType`); the **bag** is the vector axis: `vle64` the
bag's ids → `vsll` → `vluxei64` one f32 component from L rows
(mixed-EEW gather) → ordered redsum, looped over the dim. Fixed bag
length `L` (production inference is dominated by fixed-length
pooling).

*Args:* `rows dim(pow2) bags bag_len seed`

> **Known simulator issue:** at large configs (e.g. the golden args)
> this kernel currently returns NaN under gem5 with vector chaining
> enabled — gcc allocates its gather as dest==index (`vluxei64
> v2,(a2),v2`), which trips an open pinned-destination write-ordering
> hazard in the chaining machinery (see gem5 DOCUMENTATION.MD,
> "Chaining: pinned-destination write-ordering hazard"). It passes with
> `--disable-chaining` and on the host. The golden below is the host
> value; the other ten benchmarks reproduce their goldens bit-exactly
> in gem5 with chaining on.

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
int64 byte-offset array (the index stream; upstream address math is
64-bit); both attention phases then run token-strips: `vle64`
offsets → `vluxei64` K (or V) f32 per component (mixed-EEW).

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
(f32) AND `path_counts[]` (f64, mixed-EEW `vluxei32` with <<3
offsets) per successor strip, matching upstream's two indirect reads
— exact (depth d+1 finalized before d).
Only the backward pass is inside the ROI. Blocker: conditional FP
division (masked-divide speculation) plus f64/f32 mixing. Upstream:
[`bc.cc:130`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/bc.cc#L130) (`path_counts[v]`, `deltas[v]`). Local
sites: index load `src/gap_bc.c:123`, deltas gather `:126`.
*Args:* `nodes edges source seed`

#### `src/gap_pr.c` — PageRank (pull, Gauss-Seidel)

GAPBS PageRankPullGS semantics: pull-direction updates with
immediately-visible scores, damping 0.85, epsilon 1e-4. The one GAP
kernel gcc CAN autovectorize — but only as the e64m2 widening chain;
this is the e32m1 intrinsic form (scalar implementation in the same
file as the bit-exact reference). Upstream indirect access:
[`pr.cc:49`](https://github.com/sbeamer/gapbs/blob/2972aeb2703165bafd921222f4ed7196f542d3a8/src/pr.cc#L49) (`outgoing_contrib[v]`, pull loop
46-54). Local sites: index load `src/gap_pr.c:87`, contrib gather
`:90`.
*Args:* `nodes edges max_iters seed`

#### `src/rivec_spmv.c` — CSR SpMV

RiVec `_spmv` semantics: `y[r] = Σ a[k] * x[ja[k]]` over CSR
(synthetic uniform column indices, fixed nnz/row). Upstream
([RALC88/riscv-vectorized-benchmark-suite `_spmv/src/spmv.c`](https://github.com/RALC88/riscv-vectorized-benchmark-suite/blob/master/_spmv/src/spmv.c)):
serial indirect read `sum += a[idx] * x[ja[idx]]` at L50, vector
gather `_MM_LOAD_INDEX_f64(x, v_idx_row, gvl)` at L37. This port
keeps the upstream element sizes — f64 values and int64 indices,
e64m1 like RiVec's `_MMR_VSETVL_E64M1` — with raw intrinsics and
autovec off (deterministic codegen; FP reduction reassociation). Local sites: index load `src/rivec_spmv.c:62`,
x gather `:64`.
*Args:* `rows cols nnz_per_row seed`

## Natural-vectorization GAP variants (`src/gap_*_natural.c`)

Counterpart experiment to the gather-formulated GAP ports above.
Those ports were written indexed-ops-first: data structures were
widened to make tests gather-able (bfs's Bitmap → int32 flags) and
gathered strips were spilled to stack buffers and re-scanned scalar,
lane by lane. Each `*_natural` file instead starts from the ORIGINAL
GAPBS serial kernel and changes **only what vectorization itself
requires** — the question they exist to answer being whether
gathers/scatters still get emitted when nobody is optimizing for
them.

**How the original shapes are preserved:**

- **Original data structures, element sizes, and layouts.** The
  Bitmap frontier stays a bitmap, dist/comp stay int32, sssp keeps
  GAPBS's interleaved `{v, w}` pairs, bc keeps f64 path counts next
  to f32 deltas. Nothing is widened, split, or re-laid-out to
  make an access indexable.
- **Original loop nests and trajectories.** The vector axis is the
  inner neighbor/edge loop in every kernel — the axis the serial
  code already iterates — and lane order equals serial iteration
  order, so bfs picks the same first-hit parent, sssp pushes buckets
  in the same order, and bc rounds in the same order. Three of the
  five (`bfs`, `sssp`, `bc`) are exactly serial-equivalent, verified
  by `memcmp` against the *unmodified* serial algorithm; cc/cc_sv
  need the same fixed 64-block snapshot device as their
  gather-formulated counterparts (the hooking store makes any
  vectorization VLEN-sensitive otherwise), mirrored in the scalar
  reference.
- **Serial parts stay serial.** Compare-and-set parent updates,
  growable bucket pushes, Link's label chase, and bit-granular
  bitmap writes are not conflict-free — the natural versions leave
  them scalar rather than restructure them into scatters.

**What "enabling vectorization" added** — three moves, all
register-side, none touching the memory layout:

1. **Speculate past data-dependent exits/branches.** The compiler's
   blocker in every kernel is an early `break` or conditional it
   cannot speculate across; a human knows the arrays are in-bounds
   and read-only during the step, gathers the whole strip, and
   resolves the condition afterwards with mask ops (`vmsne`/`vmslt`
   → `vfirst`/`vcpop`).
2. **Consume gathered values in registers.** Membership tests,
   relax compares, and label compares happen in vector; at most 0/1
   flags or mask bits cross to the scalar side. `vcpop == 0` retires
   a strip with no scalar work at all — the dominant case late in
   sssp/cc/cc_sv.
3. **Mask instead of restructure.** bc's `if (succ[j])` becomes a
   mask from one unit-stride `vle8` and the gathers execute under it
   (`vluxei32 ..., v0.t`), so inactive lanes make no memory access —
   if-conversion, not loop surgery.

**The finding:** the gathers SURVIVE in all five kernels. Every hot
loop's central read (`front`/`comp`/`dist`/`deltas`+`path_counts`)
is indexed by data, so any vectorization of the loop emits it —
these workloads are indexed-load benchmarks by nature, not by
formulation. What disappears is the *manufactured* traffic: bfs
gathers the original n/8-byte bitmap words instead of a widened
4n-byte flag array, and the buffer spill/re-scan of every strip is
gone. `gap_pr` has no `_natural` variant on purpose: its existing
kernel (gather + ordered redsum straight off the serial pull loop)
already **is** the natural vectorization.

All five verify bit-exactly against their scalar mirrors, and all
five default-config checksums equal the original variants' goldens.
Args and upstream citations are identical to the counterparts.
Per-file notes and local access sites:

#### `src/gap_bfs_natural.c` — direction-optimizing BFS

The bottom-up membership search keeps GAPBS's Bitmap: gather the
frontier **words** (`front[v >> 5]`), extract bit `v & 31` with
`vsrl.vv`/`vand`, `vmsne` → `vfirst` for the first frontier
neighbor. Serial-exact (first-hit order preserved); top-down step
and queue↔bitmap conversions stay scalar (compare-and-set;
bit-granular RMW is not conflict-free). Sites: index load
`src/gap_bfs_natural.c:113`, bitmap-word gather `:115`.

**Diff vs upstream `bfs.cc`:** upstream is OpenMP-parallel —
SlidingQueue with thread-local QueueBuffers, `compare_and_swap`
parent claims in TDStep (`bfs.cc:79`), parallel queue↔bitmap
conversions. This port serializes all of that: plain `q`/`nq`
arrays, direct parent stores, scalar conversions — same alpha=15/
beta=18 control and negative-degree parent encoding. The only
kernel change is vectorizing the BU membership search, whose one
memory-visible delta is speculation: the serial loop reads
neighbors only up to the first frontier hit, the vector strip reads
all `vl` neighbor ids and frontier words before `vfirst` resolves
the break. Data structures are upstream's exactly (uint32 Bitmap,
int32 parent).

#### `src/gap_cc_sv_natural.c` — Shiloach-Vishkin components

comp[] gathered per 64-block and vector-compared against the
block's `comp[u]` snapshot; `vcpop`-skip for settled blocks; flagged
lanes run the ORIGINAL hooking body with live comp[] reads. Sites:
index load `src/gap_cc_sv_natural.c:85`, comp gather `:81`.

**Diff vs upstream `cc_sv.cc`:** upstream scans all edges each SV
iteration in parallel, hooking `comp[high] = low` with benign
OpenMP races. This port serializes the scan and adds the ONE
semantic device upstream lacks: the fixed 64-block snapshot filter
— a block's comp[] gathers and compares all complete against the
pre-block state before any lane hooks, so a lane whose label
changed mid-block is deferred to the next SV iteration instead of
seen immediately. Same fixpoint, and the change flag matches at the
fixpoint; the hooking body and the pointer-jump compression pass
are upstream's unchanged. The device exists to make the trajectory
VLEN-independent and is mirrored in the scalar reference.

#### `src/gap_cc_natural.c` — Afforest components

Same vector filter for the final link phase; flagged lanes call the
original scalar `Link`. The `vcpop`-skip is the dominant case
post-sampling — which is Afforest's own premise. Sites: index load
`src/gap_cc_natural.c:123`, comp gather `:119`.

**Diff vs upstream `cc.cc`:** upstream Afforest is parallel (CAS
inside Link, `cc.cc:50`), samples the frequent component with
`std::mt19937`, and its finish phase calls Link on EVERY remaining
neighbor unconditionally (plus in-neighbors for directed graphs).
This port serializes Link (plain stores), samples with the suite's
deterministic RNG (seed 42, `FREQ_SAMPLES`), is undirected-only,
and adds the 64-block snapshot filter as a work-elision device
absent upstream: lanes whose snapshot label equals `comp[u]` are
provably in the same set already, so their Link calls are skipped.
neighbor_rounds=2 and the compress pass match upstream.

#### `src/gap_sssp_natural.c` — delta-stepping SSSP

`nd = dist[u] + w` and the relax test move to vector (`vmslt`
against the gathered distances — a provably exact prefilter, since
dist[] only decreases); `vcpop == 0` strips retire with no spill;
surviving strips spill ids/nd plus mask bits (`vsm`), and surviving
lanes run the exact serial update against live dist[]. Serial-exact
including bucket push order. Sites: strided id/weight loads
`src/gap_sssp_natural.c:108`/`:111`, dist gather `:113`.

**Diff vs upstream `sssp.cc`:** upstream delta-stepping is parallel
— thread-local bins, a shared frontier swap, and a
`compare_and_swap` relax loop (`sssp.cc:130`). This port serializes
the driver (single bin set, plain compare-and-store relax) with the
same delta binning, bucket fusion (threshold 1000), and interleaved
`{v, w}` NodeWeight layout. The vector relax prefilter reads
`dist[v]` for the whole strip up front, but the serial body reads
every `dist[wn.v]` anyway, so the memory footprint is unchanged —
the prefilter is exact by dist monotonicity, and dist, buckets, and
push order are bit-identical to the serial run.

#### `src/gap_bc_natural.c` — Brandes betweenness centrality

Successor test from one unit-stride `vle8` (e8mf4) + `vmsne`;
`vcpop`-skip for strips with no successor edges; MASKED `vluxei32`
gathers of deltas (f32) and path_counts (f64, mixed-EEW) touch only
active lanes. The FP accumulation stays scalar per active lane:
matching the serial result bit-for-bit needs each f64 term rounded
to f32 before the f32 sum, and `vfncvt` is banned in this suite for
hitting known gem5 RVV bugs. Sites: succ mask load
`src/gap_bc_natural.c:137`, index load `:143`, masked deltas gather
`:146`, masked path_counts gather `:149`.

**Diff vs upstream `bc.cc`:** upstream Brandes is parallel — CAS on
depths and atomic path-count adds in the forward BFS, SlidingQueue
depth slices, parallel backward accumulation per depth. This port
serializes both passes (plain queue + depth_index array) and makes
the suite's one data-structure widening: upstream stores successor
flags as a per-EDGE Bitmap (`bc.cc:115`, m/8 bytes); here succ is a
byte array (m bytes) so the successor test is a single unit-stride
`vle8` per strip instead of a bit-granular word dance. path_counts
f64 / deltas f32 / scores f32 match upstream's CountT/ScoreT, the
per-term f64→f32 rounding order is serial-exact, and (as in the
counterpart) only the backward pass is inside the ROI.

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
| gap_pr | `65536 262144 20 1` | `0.99965455636402112` (`0x1.ffd2b8d5c3p-1`) |
| rivec_spmv | `65536 65536 16 1` | `-253.21490919923698` (`-0x1.fa6e08941caa3p+7`) |
| gap_bfs_natural | `65536 262144 3 1` | `2151931909` (`0x1.0087c00ap+31`) |
| gap_cc_natural | `65536 262144 1` | `698770` (`0x1.55324p+19`) |
| gap_cc_sv_natural | `65536 262144 1` | `698770` (`0x1.55324p+19`) |
| gap_sssp_natural | `65536 262144 3 16 1` | `6563865358780` (`0x1.7e11373c6fp+42`) |
| gap_bc_natural | `65536 262144 3 1` | `376576.9998091124` (`0x1.6fc03ffcdf5cp+18`) |

Regenerate with `make host` and running the `_host` binaries with the
args above.

## ROI convention

`ROI_BEGIN()`/`ROI_END()` (from `common/roi.h`, ported from RiVec)
wrap the timed kernel only — input generation, the index-resolution
prologue (paged_attn), the routing sort (moe_dispatch), and
verification are all outside the ROI. Under `-DUSE_M5OPS` each run
produces one `m5_reset_stats`→`m5_dump_reset_stats` pair, so gem5
stats section #1 is the kernel ROI.
