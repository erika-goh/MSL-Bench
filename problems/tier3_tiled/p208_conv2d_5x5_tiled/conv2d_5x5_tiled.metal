#include <metal_stdlib>
using namespace metal;

// 5×5 valid convolution with tile-with-halo input staging.
//
// Each TG covers a TILE×TILE block of output. That block depends
// on a (TILE + K - 1)×(TILE + K - 1) = 20×20 input region (the
// TILE×TILE output footprint plus a HALO=K-1=4 ring around it,
// distributed as 2 pixels on every side because of valid-mode
// indexing). All 256 threads in the TG cooperatively load that
// 400-pixel input region into TG memory once, barrier, then each
// thread computes its 5×5 dot product reading only from TG memory.
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
    threadgroup float x_tile[IN_TILE * IN_TILE];   // 400 floats = 1600 bytes

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint bx = tg_in_grid.x;
    uint by = tg_in_grid.y;

    // Linear thread index within the TG, used by the cooperative load
    // to walk a flat 400-element scratch buffer.
    uint tid = ty * TILE + tx;          // 0..255
    uint in_row_base = by * TILE;       // top-left input row of this TG's region
    uint in_col_base = bx * TILE;       // top-left input col

    // ---------- cooperative input load into TG memory ----------
    // 256 threads, 400 input pixels. Each thread walks the flat
    // x_tile in strides of 256, so threads with tid < 144 load 2
    // pixels and the rest load 1. All loads are device-coalesced
    // because adjacent threads (varying tid % IN_TILE) hit
    // adjacent input columns.
    for (uint i = tid; i < IN_TILE * IN_TILE; i += TILE * TILE) {
        uint local_row = i / IN_TILE;
        uint local_col = i % IN_TILE;
        x_tile[i] = x[(in_row_base + local_row) * IN_W + (in_col_base + local_col)];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---------- per-thread 5x5 dot product from TG memory ----------
    // Thread (tx, ty) produces output (i_out, j_out) reading the
    // 5x5 window x_tile[ty..ty+4][tx..tx+4]. All reads are from
    // threadgroup memory; w is read by every thread but small
    // enough (100 bytes) to stay in L1/constant cache.
    float acc = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
        uint x_row_off = (ty + ki) * IN_TILE;
        uint w_row_off = ki * K;
        for (uint kj = 0; kj < K; kj++) {
            acc += x_tile[x_row_off + tx + kj] * w[w_row_off + kj];
        }
    }

    uint i_out = by * TILE + ty;
    uint j_out = bx * TILE + tx;
    y[i_out * OUT_W + j_out] = acc;
}
