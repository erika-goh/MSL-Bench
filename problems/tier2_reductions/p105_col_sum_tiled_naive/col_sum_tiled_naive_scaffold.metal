// Scaffold for p105_col_sum_tiled_naive. Save as `col_sum_tiled_naive.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, K floats — one sum per column)
//
// Dispatch geometry (from the spec):
//   grid        = (K,    TG_Y, 1)   = (256, 8, 1)
//   threadgroup = (TG_X, TG_Y, 1)   = (32,  8, 1)
//
// What's new vs p101/p102/p103: the threadgroup is 2D.
//   - thread_position_in_threadgroup is now a uint2: .x in [0, 32), .y in [0, 8)
//   - threadgroup_position_in_grid is also uint2: .x in [0, 8) (K/TG_X TGs), .y = 0
//
// Why this fixes p103's coalescing problem:
//   Metal linearizes threads inside a TG along x FIRST when forming SIMD-groups
//   of 32. So in a (32, 8) TG:
//     SIMD-group 0 = threads (0..31, 0)
//     SIMD-group 1 = threads (0..31, 1)
//     ...
//     SIMD-group 7 = threads (0..31, 7)
//   At any instant, the 32 threads in one SIMD-group share the same y and have
//   consecutive x. If we make their column index `col = tg.x * 32 + tid.x`,
//   then the 32 threads in a SIMD-group read 32 consecutive columns of the
//   same row → 32 consecutive float addresses → ONE coalesced transaction.
//
//   In p103 we had a 1D (256, 1) TG with all 256 threads working on the SAME
//   column (different rows) → 32 threads in a SIMD-group read addresses K=256
//   floats apart → 32 SEPARATE transactions.

#include <metal_stdlib>
using namespace metal;

constant uint K     = 256;
constant uint TG_X  = 32;
constant uint TG_Y  = 8;
constant uint B     = 262144;

kernel void col_sum_tiled_naive(
    device const float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    // Scratch shaped (TG_X, TG_Y) so each thread has its own slot.
    // We'll combine partials along the y axis at the end.
    threadgroup float scratch[TG_X][TG_Y];

    uint tx = tid_in_tg.x;        // 0..31  — picks the column within the TG
    uint ty = tid_in_tg.y;        // 0..7   — picks the row stripe
    uint col = tg_in_grid.x * TG_X + tx;   // global column index

    // ============================================================
    // TODO 1: each thread accumulates its column over its row stripe.
    //
    // Thread (tx, ty) handles rows ty, ty + TG_Y, ty + 2*TG_Y, ...
    // For each such row r, add x[r * K + col] to a local partial.
    //
    // float partial = 0.0f;
    // for (uint r = ty; r < B; r += TG_Y) {
    //     partial += x[r * K + col];
    // }
    //
    // Coalescing check: at fixed ty and r, threads tx=0..31 read
    // x[r*K + col_0], x[r*K + col_0 + 1], ..., x[r*K + col_0 + 31].
    // 32 consecutive floats → one coalesced load.
    // ============================================================



    // ============================================================
    // TODO 2: each thread stashes its partial in scratch[tx][ty];
    // then a threadgroup barrier so every partial is visible.
    // ============================================================



    // ============================================================
    // TODO 3: combine the TG_Y partials per column.
    //
    // Only the threads with ty == 0 do this — one survivor per column.
    // They each sum scratch[tx][0..TG_Y-1] and write out[col].
    //
    // if (ty == 0) {
    //     float total = 0.0f;
    //     for (uint i = 0; i < TG_Y; i++) total += scratch[tx][i];
    //     out[col] = total;
    // }
    //
    // Why no tree reduction here? TG_Y = 8 is tiny — a sequential
    // 8-add loop is faster than 3 stages of tree + barriers.
    // ============================================================



}
