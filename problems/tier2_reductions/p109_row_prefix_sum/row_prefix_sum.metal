#include <metal_stdlib>
using namespace metal;

// Row-wise inclusive prefix sum (cumsum) using a three-stage scan:
//   1. Each thread sequentially scans its 16 contiguous row elements
//      and records both the running prefix and its total.
//   2. Within each SIMD-group, simd_prefix_inclusive_sum on the
//      thread totals gives every thread the running sum of all
//      preceding threads' totals in the SG.
//   3. The 8 SG totals are published to TG memory; SG 0 runs an
//      exclusive scan on them and writes the SG-prefix back.
// Each thread then adds (its SG's prefix + within-SG exclusive
// prefix) to its 16 local results and writes them.
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
    // sg_prefix[i] = inclusive-sum-of-SG-totals[0..i-1] after stage 3.
    threadgroup float sg_prefix[SGS_PER_TG];

    uint m = tg_in_grid;
    uint row_base = m * N;
    uint thread_base = row_base + tid_in_tg * PER_THREAD;

    // ---- stage 1: per-thread sequential scan ----
    // local[i] = inclusive sum of x[thread_base..thread_base+i].
    // Use a register array; PER_THREAD=16 is small enough to unroll.
    float local_buf[PER_THREAD];
    float running = 0.0f;
    for (uint i = 0; i < PER_THREAD; i++) {
        running += x[thread_base + i];
        local_buf[i] = running;
    }
    float thread_total = running;  // = local_buf[PER_THREAD - 1]

    // ---- stage 2: within-SG inclusive scan of thread totals ----
    // simd_prefix_inclusive_sum returns, in each lane k, the sum of
    // thread_totals from lanes 0..k of this SG. The EXCLUSIVE prefix
    // (sum from lanes 0..k-1) is what we want to ADD to local_buf.
    float my_inc = simd_prefix_inclusive_sum(thread_total);
    float my_sg_exclusive = my_inc - thread_total;

    // The last lane (31) of each SG now holds the SG's total in my_inc.
    // Publish it (only lane 31 writes) so stage 3 can run.
    if (lane == 31) {
        sg_prefix[sg_in_tg] = my_inc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 3: SG 0 turns sg_prefix into the EXCLUSIVE scan ----
    // Only the first 8 lanes of SG 0 have meaningful work.
    if (sg_in_tg == 0) {
        float v = (lane < SGS_PER_TG) ? sg_prefix[lane] : 0.0f;
        float exc = simd_prefix_inclusive_sum(v) - v;  // simd has no _exclusive_sum prior to 3.1
        if (lane < SGS_PER_TG) {
            sg_prefix[lane] = exc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- combine: add the two prefixes and write back ----
    float global_offset = sg_prefix[sg_in_tg] + my_sg_exclusive;
    for (uint i = 0; i < PER_THREAD; i++) {
        y[thread_base + i] = local_buf[i] + global_offset;
    }
}
