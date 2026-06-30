// Scaffold for p208_conv2d_5x5_tiled. Save as `conv2d_5x5_tiled.metal`.
//
// 5×5 valid convolution with tile-with-halo input staging — the
// canonical CUDA stencil-optimization pattern.
//
// Each TG covers a TILE×TILE = 16×16 block of output. That block
// depends on a (TILE + K - 1)×(TILE + K - 1) = 20×20 input region
// (the output footprint plus a HALO of K-1=4 extra pixels covering
// the kernel's reach). The 256 threads in the TG cooperatively
// stage that 400-pixel input region into threadgroup memory ONCE,
// barrier, then every output thread reads its 5×5 window from
// threadgroup memory — zero redundant device reads.
//
// Contrast with the naïve thread-per-output kernel (think p207
// with K=5): there, each input pixel is potentially read 25 times
// (once per output thread that includes it in its window), and
// the L1 cache has to absorb the redundancy. The halo pattern
// makes the deduplication explicit.
//
// Cooperative load layout: 256 threads, 400 pixels → some threads
// load 2, others 1. A flat linear loop with stride 256 walks the
// 20×20 tile cleanly.

#include <metal_stdlib>
using namespace metal;

constant uint IN_W    = 1028;
constant uint K       = 5;
constant uint OUT_W   = 1024;
constant uint TILE    = 16;
constant uint IN_TILE = 20;   // TILE + K - 1

kernel void conv2d_5x5_tiled_kernel(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float x_tile[IN_TILE * IN_TILE];

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint bx = tg_in_grid.x;
    uint by = tg_in_grid.y;

    uint tid = ty * TILE + tx;
    uint in_row_base = by * TILE;
    uint in_col_base = bx * TILE;

    // ============================================================
    // TODO PHASE 1: cooperative input load
    //
    //   for (uint i = tid; i < IN_TILE * IN_TILE; i += TILE * TILE) {
    //       uint local_row = i / IN_TILE;
    //       uint local_col = i % IN_TILE;
    //       x_tile[i] = x[(in_row_base + local_row) * IN_W +
    //                     (in_col_base + local_col)];
    //   }
    //
    // The flat linear traversal means most threads do 1 load and
    // 144 of them do 2 loads (because 400 / 256 = 1.56). Adjacent
    // threads at each step access adjacent input columns →
    // coalesced device reads.
    // ============================================================



    // ============================================================
    // TODO barrier between load and compute
    //
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO PHASE 2: per-thread 5x5 dot product from TG memory
    //
    //   float acc = 0.0f;
    //   for (uint ki = 0; ki < K; ki++) {
    //       uint x_row_off = (ty + ki) * IN_TILE;
    //       uint w_row_off = ki * K;
    //       for (uint kj = 0; kj < K; kj++) {
    //           acc += x_tile[x_row_off + tx + kj] * w[w_row_off + kj];
    //       }
    //   }
    //
    //   uint i_out = by * TILE + ty;
    //   uint j_out = bx * TILE + tx;
    //   y[i_out * OUT_W + j_out] = acc;
    // ============================================================
}
