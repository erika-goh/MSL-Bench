// Scaffold for p101_row_sum. Save your version as `row_sum.metal` (or
// anywhere — you'll pass the path to run_problem.py). Don't edit this scaffold
// file in place; keep it as a reference for what was given vs. what you wrote.
//
// Buffer bindings (from the spec's input/output order):
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, B floats)
//
// Dispatch geometry (from the spec's `launch` override):
//   grid        = (B*K, 1, 1)   total threads
//   threadgroup = (K,   1, 1)   one threadgroup per row, K=256 threads per group
//
// You have:
//   threadgroup_position_in_grid.x  → row index `b` in [0, B)
//   thread_position_in_threadgroup.x → thread index `tid` within row in [0, K)

#include <metal_stdlib>
using namespace metal;

constant uint K = 256;  // row width; matches the spec's K and threadgroup size

kernel void row_sum(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    // Threadgroup-shared scratch — every thread in this group sees the same array.
    // One slot per thread. ~1 KB total (256 * 4 bytes).
    threadgroup float scratch[K];

    uint b   = tg_in_grid;   // which row this threadgroup is summing
    uint tid = tid_in_tg;    // which column-slot within the row this thread owns

    // ============================================================
    // TODO 1: Load one element from global memory into your slot.
    //
    // Each thread loads exactly one float: x[b][tid], which in row-major
    // flat storage is at index b * K + tid.
    //
    // Write it to scratch[tid].
    // ============================================================



    // ============================================================
    // TODO 2: Barrier so every thread sees every other thread's load
    // before the reduction starts.
    //
    // Use: threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 3: Tree reduction. For stride = K/2, K/4, K/8, ..., 1:
    //   - if tid < stride: scratch[tid] += scratch[tid + stride]
    //   - barrier (so the next stage sees this stage's writes)
    //
    // Why the `if (tid < stride)` guard: at stride=128, only threads
    // 0..127 do work; threads 128..255 sit idle. At stride=64, only
    // threads 0..63 work. Etc. Without the guard, threads 128..255
    // would read scratch[256..383] — out of bounds.
    //
    // After log2(K) = 8 stages, scratch[0] holds the row sum.
    // ============================================================



    // ============================================================
    // TODO 4: Only thread 0 writes the result.
    //
    // If tid == 0: out[b] = scratch[0]. Other threads do nothing.
    // (No barrier needed before this — the loop's last barrier already
    // ensured scratch[0] is final.)
    // ============================================================



}
