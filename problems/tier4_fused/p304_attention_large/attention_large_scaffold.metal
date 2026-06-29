// Scaffold for p304_attention_large. Save as `attention_large.metal`.
//
// This problem uses the same algorithm as p303_attention_head — only
// the shape changes (M = D = 512 instead of 64). The point of the
// problem is to measure how that same kernel performs once compute,
// not dispatch overhead, dominates.
//
// Buffer bindings:
//   buffer(0) = q   (queries,  M*D floats, row-major)
//   buffer(1) = k   (keys,     M*D floats, row-major)
//   buffer(2) = v   (values,   M*D floats, row-major)
//   buffer(3) = out (output,   M*D floats, same shape as Q)
//
// Dispatch geometry:
//   grid        = (M*M, 1, 1)         (262144 total threads)
//   threadgroup = (M,   1, 1)         (512 threads/TG, 512 TGs)
//
// Algorithm (per query row m):
//   scores[j]  = (Q[m, :] · K[j, :]) / sqrt(D)     for j in 0..M
//   probs[j]   = softmax(scores)[j]                stable: subtract row max
//   out[m, d]  = sum_j probs[j] * V[j, d]          for d in 0..D
//
// Implementation strategy: ONE shared scratch[M] array reused across
// all phases — scores → reduction workspace → probabilities — with
// barriers at every transition. Q row is staged into a small q_row[D]
// once at the top.
//
// Differences from p303 to watch:
//   - dot products are length 512 instead of 64 (8× per-thread compute)
//   - tree reduction is 9 levels deep instead of 6
//   - phase 1 K reads are uncoalesced across threads at a stride of 512
//     (vs 64 in p303) → the actual bandwidth cost. Phase 3 V reads are
//     coalesced (stride-1 across threads), same as p303.

#include <metal_stdlib>
using namespace metal;

constant uint  M     = 512;
constant uint  D     = 512;
constant float SCALE = 0.04419417382415922f;  // 1.0 / sqrt(D=512)

kernel void attention_large(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[M];
    threadgroup float q_row[D];

    uint m   = tg_in_grid;
    uint tid = tid_in_tg;

    // ============================================================
    // TODO 0: stage Q[m, :] into q_row[].
    //
    //   q_row[tid] = q[m * D + tid];
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 1: compute this thread's score against K[tid, :].
    //
    //   float s = 0.0f;
    //   for (uint d = 0; d < D; d++) {
    //       s += q_row[d] * k[tid * D + d];
    //   }
    //   s *= SCALE;
    //
    //   scratch[tid] = s;
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 2a: tree-reduce-max over scratch.
    //
    //   for (uint stride = M / 2; stride > 0; stride >>= 1) {
    //       if (tid < stride) {
    //           scratch[tid] = max(scratch[tid], scratch[tid + stride]);
    //       }
    //       threadgroup_barrier(mem_flags::mem_threadgroup);
    //   }
    //   float row_max = scratch[0];
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 2b: compute exp(s - row_max), tree-reduce-sum.
    //
    //   float e = exp(s - row_max);
    //   scratch[tid] = e;
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    //
    //   for (uint stride = M / 2; stride > 0; stride >>= 1) {
    //       if (tid < stride) {
    //           scratch[tid] += scratch[tid + stride];
    //       }
    //       threadgroup_barrier(mem_flags::mem_threadgroup);
    //   }
    //   float row_sum = scratch[0];
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 2c: normalize and publish probabilities to scratch.
    //
    //   float p = e / row_sum;
    //   scratch[tid] = p;
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 3: compute one output dim. Thread `tid` produces out[m, tid].
    //
    //   float o = 0.0f;
    //   for (uint j = 0; j < M; j++) {
    //       o += scratch[j] * v[j * D + tid];
    //   }
    //   out[m * D + tid] = o;
    //
    // V access at fixed j is stride-1 across threads → COALESCED.
    // (The uncoalesced reads in this kernel are phase 1's K reads,
    // not these V reads.)
    // ============================================================



}
