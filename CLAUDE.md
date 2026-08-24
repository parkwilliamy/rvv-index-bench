# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

A dedicated test suite for **indexed load/store kernels in RVV**
(`vluxei`/`vsuxei`, plus strided ops where the kernel needs them),
drawn from real workloads whose hot loops are indirect. It exists to
evaluate indexed-access hardware (prefetchers, memory-system changes)
in gem5, so the indexed accesses ARE the point — a kernel belongs here
only if its ROI is dominated by indexed loads/stores. More benchmarks
will be added over time; follow the conventions below for every
addition.

## Hard rules

- **Raw `__riscv_*` intrinsics only.** Never use RiVec's `_MM_*`
  macro layer or any other wrapper. (Writing intrinsic kernels is
  permitted in THIS repo without asking; that permission does not
  extend to other repos.)
- **Cite kernel sources exactly.** Every benchmark's README section
  names the upstream implementation with file and line references
  **pinned to a release tag** (verified against the actual source,
  not from memory), and lists this repo's indirect access sites as
  `src/<file>.c:<line>` with the role of each access. Regenerate the
  local lines after editing:
  `grep -n '__riscv_vluxei32\|__riscv_vsuxei32\|__riscv_vlse32\|__riscv_vsse32\|__riscv_vle32_v_i32m1' src/*.c`
- **README.md carries, per benchmark:** a short description, the
  args, the source citations, and if applicable, a justification of **why the
  kernel is not auto-vectorizable** (so readers understand why
  intrinsics were required). Test the claim before writing it: build
  the scalar code with `-ftree-vectorize` and check the dump.
- **Golden outputs are exact, not tolerances.** Keep the
  bit-exactness discipline: `-ffp-contract=off` everywhere, unfused
  `vfmul`+`vfadd` (never `vfmacc`), *ordered* reductions
  (`vfredosum`) with the strip-carried accumulator, seeded splitmix64
  inputs (never wall-clock), and a scalar reference whose loop order
  mirrors the vector op order. Then serial, vector, and native host
  builds produce identical bits; goldens in `golden/` are generated
  with `make host` and checked by exact string match (`%a` hex
  floats). Every binary also self-verifies (`memcmp` vs the in-binary
  scalar reference) and prints `RESULT: PASS|FAIL`.

## Layout & build

- One source per benchmark: `src/<name>.c`. Binaries per kernel:
  `bin/<name>/<name>_{serial,vector,host}` (no `.exe` suffix) plus
  the vector objdump. Shared headers in `common/` (`roi.h` is the
  ROI-marker header ported from RiVec; `bench_common.h` has the RNG
  and checksum helpers). Goldens in `golden/`.
- Top-level gcc Makefile (RiVec `Makefile.gcc` conventions):
  `riscv64-unknown-elf-gcc`, `-march=rv64gcv_zifencei -static`,
  autovec off in both RISC-V variants (the serial control must stay
  scalar; the vector build's indexed ops must be exactly the
  hand-written ones). Knobs: `RDCYCLE=1`, `NO_M5OPS=1`. m5ops need
  `libm5.a` under `$GEM5_ROOT/util/m5/build/riscv/out`. New
  benchmarks: add the name to `BENCHES` — the pattern rules do the
  rest.

## Kernel-writing conventions

- **Keep the upstream memory layout** (row-major tables etc.); make
  the INDEX the vector axis (dim-outer loops). That is what turns a
  row-granular indirect workload into an indexed-load benchmark.
- Sizes that feed shift-based addressing must be powers of two
  (`dim` in the existing kernels) — offsets are `id << log2(row
  bytes)`, which keeps the chain in `base + (idx << shift)` form.
- **e32m1 only**: no widening/narrowing chains, no fractional LMUL,
  no segment ops, no partial-lane accumulator idioms, no
  `vcompress` — these hit known gem5 RVV model bugs.
- Scatters must be **conflict-free by construction** (distinct ids /
  permutations), stated in the source header, so vector and scalar
  results stay bit-identical.
- CLI-parameterized sizes with a trailing `seed`; input generation,
  index-resolution prologues, and verification stay OUTSIDE the ROI —
  `ROI_BEGIN()`/`ROI_END()` wrap the kernel only, giving one gem5
  stats-section pair per run (section #1 = the ROI).

## Definition of done for a new benchmark

1. Builds clean: host + serial + vector.
2. Objdump of the vector binary shows the intended indexed ops (and
   the serial one shows none).
3. Host run at the default config produces the golden (`golden/` file
   + README table row) and `RESULT: PASS`.
4. gem5 run of the vector binary at the same config: `RESULT: PASS`
   with the checksum matching the host golden bit-for-bit, M5_OPS
   cycles printed. (Run via gem5's
   `rvv/riscv-rvv-se-ara-prefetcher.py` with `--parms="<args>"`.)
5. README section added: description, args, pinned source citations,
   local access-site lines, why-not-autovectorizable.
