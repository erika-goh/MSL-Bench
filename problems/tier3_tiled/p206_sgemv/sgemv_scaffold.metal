// Scaffold for p206_sgemv. Save as `sgemv.metal`.
//
// y = A @ x where A is (M, N) and x is (N,) → y is (M,).
//
// Layout: one TG per output row (M TGs of TG_THREADS=256 threads).
// Each thread strip-mines PER_THREAD = N/256 = 16 multiply-adds of
// its slice of A's row against x. Then the TG reduces 256 partial
// sums to one scalar via TWO-STAGE simd_sum.
//
// Why two stages: simd_sum is a SIMD-group lane reduction — it
// collapses 32 lanes to a single value but cannot reduce across
// SGs by itself. So we first run simd_sum within each of the 8 SGs,
// then publish the 8 SG partials through TG memory and have one SG
// run simd_sum a second time on those.
//
// Memory pattern is fully coalesced: adjacent threads (varying
// tid_in_tg) read adjacent A and x addresses.
//
// Dispatch:
//   grid        = (M * TG_THREADS, 1, 1) = (1048576, 1, 1)
//   threadgroup = (TG_THREADS,     1, 1) = (256,     1, 1)

#include <metal_stdlib>
using namespace metal;

constant uint N           = 4096;
constant uint TG_THREADS  = 256;
constant uint SGS_PER_TG  = 8;
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
    threadgroup float sg_partials[SGS_PER_TG];

    uint m = tg_in_grid;

    // ============================================================
    // TODO 1: strip-mined per-thread dot product.
    //
    //   float acc = 0.0f;
    //   for (uint i = 0; i < PER_THREAD; i++) {
    //       uint n = tid_in_tg + i * TG_THREADS;
    //       acc += a[m * N + n] * x[n];
    //   }
    //
    // Adjacent threads at each i read adjacent A and x addresses
    // → coalesced.
    // ============================================================



    // ============================================================
    // TODO 2: stage 1 — collapse the 32 lanes within each SG.
    //
    //   float sg_sum = simd_sum(acc);
    //
    // simd_sum is SIMD-group-synchronous; no barrier needed.
    // ============================================================



    // ============================================================
    // TODO 3: publish each SG's partial to TG memory.
    //
    //   if (lane == 0) {
    //       sg_partials[sg_in_tg] = sg_sum;
    //   }
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 4: stage 2 — SG 0 reduces the 8 partials.
    //
    //   if (sg_in_tg == 0) {
    //       float v = (lane < SGS_PER_TG) ? sg_partials[lane] : 0.0f;
    //       float total = simd_sum(v);
    //       if (lane == 0) {
    //           y[m] = total;
    //       }
    //   }
    // ============================================================
}
