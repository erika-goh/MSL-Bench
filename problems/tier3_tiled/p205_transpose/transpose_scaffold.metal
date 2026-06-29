// Scaffold for p205_transpose. Save as `transpose.metal`.
//
// B[i, j] = A[j, i].  A is (M, N), B is (N, M).  M = N = 2048.
//
// Naïve thread-per-element transpose:
//
//   uint i = thread_position_in_grid.y;  // row in A
//   uint j = thread_position_in_grid.x;  // col in A
//   b[j * M + i] = a[i * N + j];
//
// This reads A coalesced (adjacent threads → adjacent A addresses)
// but writes B UNCOALESCED — adjacent threads write to B addresses
// that are M floats apart, so each SIMD-group store becomes 32
// separate cache-line writes.
//
// Tiled fix: stage a TILE x TILE block of A into threadgroup memory
// (coalesced read), barrier, then write to B by reading the tile
// with swapped indices (coalesced write). The threadgroup-memory
// "non-coalesced" read of tile[tx][ty] is fine because TG memory is
// not bandwidth-bound at TILE=16.
//
// Dispatch:
//   grid        = (N, M, 1)        one thread per A element
//   threadgroup = (TILE, TILE, 1)  16 x 16 = 256 threads per TG

#include <metal_stdlib>
using namespace metal;

constant uint M    = 2048;
constant uint N    = 2048;
constant uint TILE = 16;

kernel void transpose_kernel(
    device const float* a    [[buffer(0)]],
    device float*       b    [[buffer(1)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TILE][TILE];

    uint tx = tid_in_tg.x;
    uint ty = tid_in_tg.y;
    uint bx = tg_in_grid.x;
    uint by = tg_in_grid.y;

    // ============================================================
    // TODO 1: coalesced read of A into the tile.
    //
    //   uint a_row = by * TILE + ty;
    //   uint a_col = bx * TILE + tx;
    //   tile[ty][tx] = a[a_row * N + a_col];
    //
    // Adjacent threads (varying tx) read adjacent A addresses → one
    // cache-line load per SIMD-group row.
    // ============================================================



    // ============================================================
    // TODO 2: barrier — the writes above must complete before any
    // thread reads the tile (a different thread's data) below.
    //
    //   threadgroup_barrier(mem_flags::mem_threadgroup);
    // ============================================================



    // ============================================================
    // TODO 3: coalesced write of B via TG-memory swap-read.
    //
    //   uint b_row = bx * TILE + ty;
    //   uint b_col = by * TILE + tx;
    //   b[b_row * M + b_col] = tile[tx][ty];
    //
    // Note the [tx][ty] index order — that's where the transpose
    // happens. The B write itself uses (ty, tx) the same way as the
    // A read, so adjacent threads (varying tx) write adjacent B
    // addresses → coalesced.
    // ============================================================
}
