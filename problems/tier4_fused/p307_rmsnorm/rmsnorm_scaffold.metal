// Scaffold for p307_rmsnorm. Save as `rmsnorm.metal`.
//
// RMSNorm — the simpler cousin of LayerNorm used in LLaMA, Mistral,
// Gemma. Computes:
//
//   rms[m]   = sqrt(mean_j(x[m, j]^2) + eps)
//   y[m, j]  = x[m, j] * g[j] / rms[m]
//
// Note: no mean subtraction (that's the difference vs LayerNorm).
// Just the per-row RMS as the normalization statistic.
//
// Shape: x is (M, N) = (1024, 4096); g is (N,); y is (M, N).
// One TG per row, 256 threads (8 SIMD-groups × 32 lanes).
//
// Three phases:
//
//   PHASE 1: per-thread partial sum of squares.
//     Strip-mined: thread t walks columns t, t+256, ..., t+15*256.
//     Adjacent threads → adjacent x addresses → coalesced.
//
//   PHASE 2: two-stage reduction to row sum-of-squares + inv_rms.
//     Stage A: simd_sum within each SG. Lane 0 of each SG writes
//     to sg_partials[sg_in_tg]. One threadgroup_barrier.
//     Stage B: every thread independently sums the 8 sg_partials
//     and computes inv_rms = rsqrt(mean_sq + EPS). Same value in
//     every thread → no race, no broadcast cell needed.
//
//   PHASE 3: normalized + scaled writeback.
//     y[m, j] = x[m, j] * g[j] * inv_rms.

#include <metal_stdlib>
using namespace metal;

constant uint  N           = 4096;
constant uint  TG_THREADS  = 256;
constant uint  SGS_PER_TG  = 8;
constant uint  PER_THREAD  = N / TG_THREADS;
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

    // ============================================================
    // TODO PHASE 1: per-thread sum of squares
    //
    //   float partial = 0.0f;
    //   for (uint i = 0; i < PER_THREAD; i++) {
    //       uint j = tid_in_tg + i * TG_THREADS;
    //       float v = x[row_base + j];
    //       partial += v * v;
    //   }
    // ============================================================



    // ============================================================
    // TODO PHASE 2: two-stage reduction, then inv_rms in every thread
    //
    //   // stage A: collapse 32 lanes within each SG
    //   float sg_sum = simd_sum(partial);
    //   if (lane == 0) {
    //       sg_partials[sg_in_tg] = sg_sum;
    //   }
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    //
    //   // stage B: every thread sums the 8 SG partials, computes inv_rms
    //   float total = 0.0f;
    //   for (uint i = 0; i < SGS_PER_TG; i++) {
    //       total += sg_partials[i];
    //   }
    //   float inv_rms = rsqrt(total * INV_N + EPS);
    //
    // Why all threads compute inv_rms: avoids needing a TG
    // broadcast cell (which the compiler warns about as possibly
    // uninitialized) and removes the second barrier.
    // ============================================================



    // ============================================================
    // TODO PHASE 3: normalized + scaled writeback
    //
    //   for (uint i = 0; i < PER_THREAD; i++) {
    //       uint j = tid_in_tg + i * TG_THREADS;
    //       y[row_base + j] = x[row_base + j] * g[j] * inv_rms;
    //   }
    // ============================================================
}
