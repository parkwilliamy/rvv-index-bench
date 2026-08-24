# rvv-index-bench — indexed load/store kernels in RVV.
# One source per benchmark under src/, binaries under bin/ (GAPBS-style
# flat layout); gcc conventions follow rivec/Makefile.gcc, but kernels
# use RAW __riscv_* intrinsics only.
#
#   make                  # all benchmarks, serial + vector
#   make sls_vector       # one target
#   make host             # native x86 builds (golden generation)
#   make RDCYCLE=1        # rdcycle ROI prints, no gem5 dependency
#   make NO_M5OPS=1       # ROI markers compiled away
#
# libm5 must exist: cd $(GEM5_ROOT)/util/m5 && scons build/riscv/out/m5
# -ffp-contract=off everywhere: vector, serial, and host results are
# bit-identical, so the checksums in golden/ are exact references.

BENCHES = sls paged_attn moe_dispatch fasttext_sg gap_bfs gap_cc gap_cc_sv gap_sssp gap_bc gap_pr rivec_spmv

RVTOOLS ?= $(HOME)/opt/riscv/bin
RVGCC ?= $(RVTOOLS)/riscv64-unknown-elf-gcc
OBJDUMP ?= $(RVTOOLS)/riscv64-unknown-elf-objdump
GEM5_ROOT ?= $(HOME)/gem5

FLAGS = -std=gnu99 -mcmodel=medany -mabi=lp64d -static -O3 \
        -ffp-contract=off -fno-common -fno-builtin-printf \
        -ffunction-sections -fdata-sections -Icommon
LIBS = -lm

ifdef RDCYCLE
FLAGS += -DRDCYCLE
else ifndef NO_M5OPS
FLAGS += -DUSE_M5OPS -I$(GEM5_ROOT)/include
LIBS += -L$(GEM5_ROOT)/util/m5/build/riscv/out -lm5
endif

# Serial keeps the same march (toolchain multilib is rv64gcv-only);
# scalar comes from omitting USE_RISCV_VECTOR, autovec off everywhere
# so the serial control stays scalar and the vector build's indexed
# ops are exactly the hand-written ones.
MARCH = -march=rv64gcv_zifencei -fno-tree-vectorize

HOSTCC ?= gcc
HOSTFLAGS = -std=gnu99 -O2 -ffp-contract=off -Icommon

all: $(foreach b,$(BENCHES),$(b)_serial $(b)_vector)
host: $(foreach b,$(BENCHES),$(b)_host)

define BENCH_RULES
$(1)_vector: src/$(1).c common/roi.h common/bench_common.h
	@mkdir -p bin/$(1)
	$(RVGCC) $(FLAGS) $(MARCH) -DUSE_RISCV_VECTOR $$< -o bin/$(1)/$(1)_vector $(LIBS)
	$(OBJDUMP) -d bin/$(1)/$(1)_vector > bin/$(1)/$(1)_vector.dump

$(1)_serial: src/$(1).c common/roi.h common/bench_common.h
	@mkdir -p bin/$(1)
	$(RVGCC) $(FLAGS) $(MARCH) $$< -o bin/$(1)/$(1)_serial $(LIBS)

$(1)_host: src/$(1).c common/bench_common.h
	@mkdir -p bin/$(1)
	$(HOSTCC) $(HOSTFLAGS) $$< -o bin/$(1)/$(1)_host -lm
endef

$(foreach b,$(BENCHES),$(eval $(call BENCH_RULES,$(b))))

.PHONY: all host clean $(foreach b,$(BENCHES),$(b)_serial $(b)_vector $(b)_host)
clean:
	rm -f bin/*
