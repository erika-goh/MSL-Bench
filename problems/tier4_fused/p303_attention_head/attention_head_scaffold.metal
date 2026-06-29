// Scaffold for p303_attention_head. Save as `attention_head.metal`.
//
// Buffer bindings:
//   buffer(0) = q   (queries,  M*D floats, row-major)
//   buffer(1) = k   (keys,     M*D floats, row-major)
//   buffer(2) = v   (values,   M*D floats, row-major)
//   buffer(3) = out (output,   M*D floats, same shape as Q)
//
// Dispatch geometry:
//   grid        = (M*M, 1, 1)
//   threadgroup = (M,   1, 1)
//
// One TG per query row, M threads per TG. With M == D == 64, every
// thread has a clean 1:1 role in every phase.
//
// Algorithm (per query row m):
//   scores[j]  = (Q[m, :] · K[j, :]) / sqrt(D)     for j in 0..M
//   probs[j]   = softmax(scores)[j]                stable: subtract row max
//   out[m, d]  = sum_j probs[j] * V[j, d]          for d in 0..D
//
// Implementation strategy: ONE shared scratch[M] array reused across
// all phases — scores → reduction workspace → probabilities — with
// barriers at every transition. Q row is staged into a small q_row[D]
// once at the top so phase 1's dot product reads it from threadgroup
// memory rather than re-reading device memory.
//
// K is read from device with an UNCOALESCED across-threads pattern
// in phase 1 (stride D between adjacent threads) — the real bandwidth
// cost of this kernel. V reads in phase 3 are coalesced (adjacent
// threads read adjacent columns of the same row).

#include <metal_stdlib>
using namespace metal;

constant uint  M     = 64;
constant uint  D     = 64;
constant float SCALE = 0.125f;  // 1.0 / sqrt(D=64)

kernel void attention_head(
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
    // TODO 0: stage Q[m, :] into q_row[]. One thread per element
    // (M == D == 64 so the mapping is direct).
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
    // Then publish s to scratch[tid] for the upcoming max-reduce.
    // Keep s in a register too — needed for the exp step in phase 2b.
    //   scratch[tid] = s;
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 2a: tree-reduce-max over scratch (same shape as p102 row_max).
    // After the loop, scratch[0] holds the row max. Capture it into
    // a register, then barrier before phase 2b overwrites scratch.
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
    // TODO 2b: compute exp(s - row_max), tree-reduce-sum (same shape
    // as p101 row_sum). Each thread also keeps its own exp value `e`
    // for the upcoming normalize.
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
    //
    // Why publish to scratch when we already have `p` in a register?
    // Phase 3 needs EVERY thread to read EVERY other thread's prob —
    // so they must live in shared memory, not per-thread registers.
    // ============================================================



    // ============================================================
    // TODO 3: compute one output dim. Thread `tid` produces out[m, tid]
    // by weight-summing the M=64 V rows by probability.
    //
    //   float o = 0.0f;
    //   for (uint j = 0; j < M; j++) {
    //       o += scratch[j] * v[j * D + tid];
    //   }
    //   out[m * D + tid] = o;
    //
    // V access at fixed j: stride-1 between adjacent threads → COALESCED
    // (thread t reads v[j*D + t], adjacent columns of the same row).
    // No bandwidth issue here. The uncoalesced pattern in this kernel
    // is phase 1's K reads, not phase 3's V reads.
    // ============================================================



}
