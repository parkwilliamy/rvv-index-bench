// gem5 region-of-interest markers around each benchmark's kernel (same
// scheme as tsvc-gem5/src/common.h and ligra/ligra.h). Build with
// -DUSE_M5OPS to reset gem5 stats on ROI entry and dump+reset on exit (one
// stats.txt section for the kernel), or -DRDCYCLE to only print raw rdcycle
// counts. Both are RISC-V-only; builds without either flag compile the
// markers away. Self-contained on purpose: some benchmarks define their own
// get_time()/timing helpers, so this header must not depend on riscv_util.h.
#ifndef RIVEC_ROI_H
#define RIVEC_ROI_H

#ifdef RDCYCLE
#undef USE_M5OPS
#endif

#if defined(USE_M5OPS) || defined(RDCYCLE)
#include <stdio.h>
#include <stdint.h>
static inline uint64_t roi_read_cycles_(void) {
  uint64_t val;
  __asm__ volatile ("rdcycle %0" : "=r" (val));
  return val;
}
static uint64_t roi_start_cycles_;
#endif

#ifdef USE_M5OPS
#include <gem5/m5ops.h>
#define ROI_BEGIN() do { m5_reset_stats(0, 0); \
                         roi_start_cycles_ = roi_read_cycles_(); } while (0)
#define ROI_END()   do { uint64_t roi_end_cycles_ = roi_read_cycles_(); \
                         m5_dump_reset_stats(0, 0); \
                         printf("cycles: %llu [M5_OPS]\n", \
                           (unsigned long long)(roi_end_cycles_ - roi_start_cycles_)); } while (0)
#elif defined(RDCYCLE)
#define ROI_BEGIN() do { roi_start_cycles_ = roi_read_cycles_(); \
                         printf("start cycles: %llu [RDCYCLE]\n", \
                           (unsigned long long)roi_start_cycles_); } while (0)
#define ROI_END()   do { uint64_t roi_end_cycles_ = roi_read_cycles_(); \
                         printf("end cycles: %llu [RDCYCLE]\n", \
                           (unsigned long long)roi_end_cycles_); \
                         printf("cycles: %llu [RDCYCLE]\n", \
                           (unsigned long long)(roi_end_cycles_ - roi_start_cycles_)); } while (0)
#else
#define ROI_BEGIN()
#define ROI_END()
#endif

#endif // RIVEC_ROI_H
