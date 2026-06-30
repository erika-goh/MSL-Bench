#include <metal_stdlib>
using namespace metal;

// Row-wise RMSNorm with learnable scale:
//   rms[m]   = sqrt(mean_j(x[m, j]^2) + eps)
//   y[m, j]  = x[m, j] * g[j] / rms[m]
//
// One TG per row. 256 threads strip-mine the row in steps of
// TG_THREADS — every iteration of the inner loop reads 256
// contiguous floats from x (coalesced) while the inner offset
// keeps thread t looking at column (t + iter * TG_THREADS).
constant uint  N           = 4096;
constant uint  TG_THREADS  = 256;
constant uint  SGS_PER_TG  = 8;
constant uint  PER_THREAD  = N / TG_THREADS;   // 16
constant float EPS         = 1e-6f;
constant float INV_N       = 1.0f / float(N);

kernel void rmsnorm_kernel(
    device const float* x   [[buffer(0)]],
    device const float* g   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    uint tid_in_tg                  [[thread_position_in_threadgroup]],
    uint tg_in_grid                 [[threadgroup_position_in_grid]],
    uint sg_in_tg                   [[simdgroup_index_in_threadgroup]],
    uint lane                       [[thread_index_in_simdgroup]])
{
    threadgroup float sg_partials[SGS_PER_TG];

    uint m = tg_in_grid;
    uint row_base = m * N;

    // ---- phase 1: per-thread partial sum of squares ----
    // Strip-mined: thread t walks columns t, t+256, ..., t+15*256.
    // Each iteration's 32 lanes hit 32 contiguous x addresses →
    // coalesced.
    float partial = 0.0f;
    for (uint i = 0; i < PER_THREAD; i++) {
        uint j = tid_in_tg + i * TG_THREADS;
        float v = x[row_base + j];
        partial += v * v;
    }

    // ---- phase 2: two-stage reduction to row sum of squares ----
    float sg_sum = simd_sum(partial);
    if (lane == 0) {
        sg_partials[sg_in_tg] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread sums the 8 published SG partials and computes
    // inv_rms independently. Same value in every thread, no race,
    // no second barrier, and the compiler can't warn about an
    // uninitialized broadcast cell because there isn't one.
    float total = 0.0f;
    for (uint i = 0; i < SGS_PER_TG; i++) {
        total += sg_partials[i];
    }
    float inv_rms = rsqrt(total * INV_N + EPS);

    // ---- phase 3: normalized + scaled writeback ----
    // Same strip-mined pattern → coalesced y stores and g reads.
    for (uint i = 0; i < PER_THREAD; i++) {
        uint j = tid_in_tg + i * TG_THREADS;
        y[row_base + j] = x[row_base + j] * g[j] * inv_rms;
    }
}
