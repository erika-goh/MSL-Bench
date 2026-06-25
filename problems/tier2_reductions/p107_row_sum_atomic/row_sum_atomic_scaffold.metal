// Scaffold for p107_row_sum_atomic. Save as `row_sum_atomic.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, B atomic_floats — same bytes as plain float)
//
// Dispatch geometry (from the spec):
//   grid        = (K, B, 1)       = (256, 262144, 1)
//   threadgroup = (K_CHUNK, 1, 1) = (64,  1,      1)
//
// So there are K/K_CHUNK = 4 TGs covering the K dimension of each row,
// and B TGs covering the rows. Total: 4 * B = 1,048,576 TGs.
//
// This is the row-reduction analog of p106. Compared to p101's design
// (one TG per row, 256 cooperating threads, tree-reduce across all K=256):
//
//   - p101: 262,144 TGs, 256 threads each, no atomics
//   - p107: 1,048,576 TGs (4x more), 64 threads each, atomic combine
//
// The experiment: does atomics-with-more-TGs beat cooperation-with-tree-
// reduce on row_sum the way it did on col_sum? The hypothesis from
// p106 is that high TG count plus light per-TG work beats lower TG
// count plus heavy cooperative reduction.
//
// Memory pattern: thread tid of TG (tg_x, row) reads
//   x[row * K + tg_x * K_CHUNK + tid]
// 32 threads of a SIMD-group share tg_x and row, vary tid 0..31 →
// 32 consecutive addresses → coalesced.

#include <metal_stdlib>
using namespace metal;

constant uint K       = 256;
constant uint K_CHUNK = 64;

kernel void row_sum_atomic(
    device const float*   x   [[buffer(0)]],
    device atomic_float*  out [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[K_CHUNK];

    // Metal requires thread/threadgroup position attributes to share
    // vector width — uint vs uint2 mixed declarations fail to compile.
    // Since we have a 2D grid, both must be uint2; tid.y is always 0
    // because TG_Y = 1.
    uint tid = tid_in_tg.x;
    uint row = tg_in_grid.y;
    uint col = tg_in_grid.x * K_CHUNK + tid;

    // ============================================================
    // TODO 1: each thread loads its assigned (row, col) into scratch[tid].
    // Coalesced because adjacent tid → adjacent col → adjacent address.
    // ============================================================



    // ============================================================
    // TODO 2: barrier so every load is visible to the reduction.
    // ============================================================



    // ============================================================
    // TODO 3: tree reduction over the K_CHUNK threads.
    // log2(K_CHUNK) = log2(64) = 6 stages. Same shape as p101's
    // reduction but over fewer threads.
    // ============================================================



    // ============================================================
    // TODO 4: thread 0 atomic-adds scratch[0] to out[row].
    //
    //   atomic_fetch_add_explicit(&out[row], scratch[0],
    //                             memory_order_relaxed);
    //
    // K/K_CHUNK = 4 TGs contribute per row → 4 atomics per output slot.
    // ============================================================



}
