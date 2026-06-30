// Scaffold for p109_row_prefix_sum. Save as `row_prefix_sum.metal`.
//
// y[m, j] = sum_{i=0..j} x[m, i]   — inclusive prefix sum along the
// row dim. Same shape in and out: (M, N) = (1024, 4096).
//
// One TG per row. 256 threads/TG. Each thread handles PER_THREAD =
// N/256 = 16 CONTIGUOUS elements of its row.
//
// Three-stage parallel scan:
//
//   STAGE 1 — per-thread sequential scan
//     Each thread sequentially scans its 16 elements, keeping the
//     running prefix in `local_buf[i]` and the per-thread total
//     (= local_buf[15]) in a register.
//
//   STAGE 2 — within-SG inclusive scan via simd_prefix_inclusive_sum
//     simd_prefix_inclusive_sum(thread_total) returns, in each lane
//     k, the sum of thread_totals from lanes 0..k of this SG. The
//     EXCLUSIVE prefix is (inclusive - own_total).
//
//   STAGE 3 — across-SG scan via SG 0
//     Each SG's last lane (31) publishes its SG total to TG memory.
//     One barrier. SG 0 then runs simd_prefix_inclusive_sum on the
//     8 SG totals and writes back the EXCLUSIVE scan (each entry =
//     sum of preceding SGs' totals).
//
// Combine: global_offset = sg_prefix[sg_in_tg] + my_sg_exclusive.
// Each thread adds global_offset to its 16 local results and writes
// them.
//
// Dispatch:
//   grid        = (M * TG_THREADS, 1, 1)
//   threadgroup = (TG_THREADS,     1, 1)

#include <metal_stdlib>
using namespace metal;

constant uint N           = 4096;
constant uint TG_THREADS  = 256;
constant uint SGS_PER_TG  = 8;
constant uint PER_THREAD  = N / TG_THREADS;   // 16

kernel void row_prefix_sum_kernel(
    device const float* x   [[buffer(0)]],
    device float*       y   [[buffer(1)]],
    uint tid_in_tg                  [[thread_position_in_threadgroup]],
    uint tg_in_grid                 [[threadgroup_position_in_grid]],
    uint sg_in_tg                   [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    threadgroup float sg_prefix[SGS_PER_TG];

    uint m = tg_in_grid;
    uint thread_base = m * N + tid_in_tg * PER_THREAD;

    // ============================================================
    // TODO STAGE 1: per-thread sequential scan
    //
    //   float local_buf[PER_THREAD];
    //   float running = 0.0f;
    //   for (uint i = 0; i < PER_THREAD; i++) {
    //       running += x[thread_base + i];
    //       local_buf[i] = running;
    //   }
    //   float thread_total = running;
    // ============================================================



    // ============================================================
    // TODO STAGE 2: simd_prefix_inclusive_sum on thread totals
    //
    //   float my_inc = simd_prefix_inclusive_sum(thread_total);
    //   float my_sg_exclusive = my_inc - thread_total;
    //
    // Lane 31's `my_inc` now equals the SG's full total. Publish it:
    //
    //   if (lane == 31) {
    //       sg_prefix[sg_in_tg] = my_inc;
    //   }
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO STAGE 3: SG 0 produces the exclusive scan of SG totals
    //
    //   if (sg_in_tg == 0) {
    //       float v = (lane < SGS_PER_TG) ? sg_prefix[lane] : 0.0f;
    //       float exc = simd_prefix_inclusive_sum(v) - v;
    //       if (lane < SGS_PER_TG) {
    //           sg_prefix[lane] = exc;
    //       }
    //   }
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    //
    // (Using "inclusive - own_value" instead of the dedicated
    // simd_prefix_exclusive_sum keeps the kernel portable to older
    // Metal versions that may not have the exclusive intrinsic.)
    // ============================================================



    // ============================================================
    // TODO combine + writeback
    //
    //   float global_offset = sg_prefix[sg_in_tg] + my_sg_exclusive;
    //   for (uint i = 0; i < PER_THREAD; i++) {
    //       y[thread_base + i] = local_buf[i] + global_offset;
    //   }
    // ============================================================
}
