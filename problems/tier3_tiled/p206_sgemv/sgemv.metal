#include <metal_stdlib>
using namespace metal;

// One TG per output row of y. 256 threads collectively compute the
// length-N dot product A[m, :] · x[:] via strip-mining + two-stage
// simd_sum reduction.
// M (= 4096) is implicit in the grid: there are M TGs, one per output row.
// Only N is referenced in the kernel body, so M is omitted to avoid an
// unused-constant warning.
constant uint N           = 4096;
constant uint TG_THREADS  = 256;
constant uint SGS_PER_TG  = 8;     // 256 / 32
constant uint PER_THREAD  = N / TG_THREADS;   // 16

kernel void sgemv_kernel(
    device const float* a   [[buffer(0)]],
    device const float* x   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    uint tid_in_tg                  [[thread_position_in_threadgroup]],
    uint tg_in_grid                 [[threadgroup_position_in_grid]],
    uint sg_in_tg                   [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    // Scratch for the second reduction stage: one float per SIMD-group.
    threadgroup float sg_partials[SGS_PER_TG];

    uint m = tg_in_grid;
    uint row_base = m * N;

    // ---- per-thread strip-mined dot product ----
    // Thread `tid_in_tg` walks elements tid_in_tg, tid_in_tg + 256,
    // tid_in_tg + 512, ...  Adjacent threads (varying tid_in_tg)
    // hit adjacent A and x addresses → coalesced loads on both.
    float acc = 0.0f;
    for (uint i = 0; i < PER_THREAD; i++) {
        uint n = tid_in_tg + i * TG_THREADS;
        acc += a[row_base + n] * x[n];
    }

    // ---- stage 1: collapse 32 lanes within each SG to one float ----
    // simd_sum is a SIMD-group-synchronous lane reduction; no barrier
    // or scratch needed for this within-SG step.
    float sg_sum = simd_sum(acc);

    // ---- publish each SG's partial to TG memory ----
    // Only lane 0 of each SG writes, so the 8 writes go to 8 distinct
    // addresses with no contention.
    if (lane == 0) {
        sg_partials[sg_in_tg] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- stage 2: SG 0 reduces the 8 partials to a scalar ----
    // The other 7 SGs are done at this point. Could exit them early
    // with a return, but no benefit since the TG can't dissolve
    // until SG 0 finishes anyway.
    if (sg_in_tg == 0) {
        float v = (lane < SGS_PER_TG) ? sg_partials[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0) {
            y[m] = total;
        }
    }
}
