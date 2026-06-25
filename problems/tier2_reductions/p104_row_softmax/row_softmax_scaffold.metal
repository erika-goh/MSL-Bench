// Scaffold for p104_row_softmax. Save your version as `row_softmax.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, B*K floats — same shape as input)
//
// Dispatch geometry (from the spec):
//   grid        = (B*K, 1, 1)
//   threadgroup = (K,   1, 1)
//
// Algorithm (all within one kernel, one TG per row):
//
//   Phase 1 — max reduction
//     scratch[tid] = x[idx]            // each thread loads its column
//     barrier
//     tree-reduce-max into scratch[0]
//     if (tid == 0) row_max = scratch[0]
//     barrier                          // <-- load-bearing! see below
//
//   Phase 2 — sum-of-exp reduction
//     float e = exp(value - row_max)   // value is the load from phase 1
//     scratch[tid] = e
//     barrier
//     tree-reduce-sum into scratch[0]
//     if (tid == 0) row_sum = scratch[0]
//     barrier
//
//   Phase 3 — per-element divide and write
//     out[idx] = e / row_sum
//
// Why the barrier after each broadcast (row_max = ..., row_sum = ...) is
// load-bearing: thread 0 does the assignment, but threads 1..K-1 are
// about to read row_max / row_sum. Without the barrier, threads 1..K-1
// can race ahead while thread 0 hasn't yet committed the write to
// threadgroup memory. Result: garbage output that varies run to run.
//
// Each thread should keep its loaded `value` in a register across the
// barriers — it's needed in phase 2 (to compute exp) and phase 3
// (implicitly, via `e`). Don't re-read from device memory.

#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_softmax(device const float* x   [[buffer(0)]],
                        device float* out       [[buffer(1)]],
                        uint tid_in_tg  [[thread_position_in_threadgroup]],
                        uint tg_in_grid [[threadgroup_position_in_grid]])
{
    // Shared scratch — reused across the max-reduce and the sum-reduce.
    threadgroup float scratch[K];
    // Per-row broadcast scalars: thread 0 writes, all threads read.
    threadgroup float row_max;
    threadgroup float row_sum;

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * K + tid;

    // ============================================================
    // TODO 1: load this thread's column into a local variable `value`
    // AND into scratch[tid]. We need `value` later in phase 2.
    // ============================================================



    // ============================================================
    // TODO 2: barrier, then tree-reduce-max into scratch[0].
    // (Same tree shape as p102, with max as the combining op.)
    // ============================================================



    // ============================================================
    // TODO 3: thread 0 broadcasts: row_max = scratch[0];
    //         barrier (LOAD-BEARING — see header comment).
    // ============================================================



    // ============================================================
    // TODO 4: compute float e = exp(value - row_max);
    //         scratch[tid] = e;
    //         barrier;
    //         tree-reduce-sum into scratch[0].
    // ============================================================



    // ============================================================
    // TODO 5: thread 0 broadcasts: row_sum = scratch[0];
    //         barrier.
    // ============================================================



    // ============================================================
    // TODO 6: out[idx] = e / row_sum;
    // (Every thread writes its own slot. No barrier needed afterward —
    // the kernel ends here and the command buffer's completion handles
    // the device-memory visibility.)
    // ============================================================



}
