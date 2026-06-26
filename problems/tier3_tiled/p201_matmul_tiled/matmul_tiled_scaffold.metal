// Scaffold for p201_matmul_tiled. Save as `matmul_tiled.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = a (input A, M*K floats, row-major)
//   buffer(1) = b (input B, K*N floats, row-major)
//   buffer(2) = c (output C, M*N floats, row-major)
//
// Dispatch geometry (from the spec):
//   grid        = (N,    M,    1)   = (1024, 1024, 1)
//   threadgroup = (TILE, TILE, 1)   = (16,   16,   1)
//
// One thread per output element of C. Each TG computes a TILE×TILE
// block of C. Number of TGs: (N/TILE)×(M/TILE) = 64×64 = 4096.
//
// ============================================================
// The big picture
// ============================================================
//
// C[m, n] = sum_k A[m, k] * B[k, n]
//
// Naive (untiled) approach: each thread reads K elements of A and K
// of B from device memory. For K=1024, that's 2*K = 2048 device-memory
// reads per output element. Awful reuse — every A[m, k] is read by
// every thread on row m (N=1024 times), every B[k, n] by every thread
// on column n (M=1024 times).
//
// Tiled approach (this kernel): load TILE×TILE blocks of A and B into
// threadgroup memory once per K-tile, then have all TILE² threads in
// the TG reuse those blocks for their share of the work. Device-memory
// reads per output element drop from 2K to 2K/TILE = 128 — a 16×
// reduction with TILE=16. The other 15/16 of the reads turn into
// threadgroup-memory reads, which are much cheaper.
//
// ============================================================
// Per-iteration mapping
// ============================================================
//
// For TG (by, bx) — where by = tg_in_grid.y, bx = tg_in_grid.x — and
// thread (ty, tx) within the TG, the output owned by this thread is:
//
//   m = by * TILE + ty
//   n = bx * TILE + tx
//   thread computes C[m, n]
//
// At K-tile iteration `t` ∈ [0, K/TILE):
//   - The TG cooperatively loads A_tile and B_tile from device memory:
//
//       A_tile[ty][tx] = a[(by*TILE + ty) * K + (t*TILE + tx)];   // A[m, t*TILE+tx]
//       B_tile[ty][tx] = b[(t*TILE + ty) * N + (bx*TILE + tx)];   // B[t*TILE+ty, n]
//
//     Every thread loads exactly one element of each — 256 threads
//     cover 256 elements of each tile in parallel.
//
//   - Barrier so all loads land before any thread reads the tile.
//
//   - Inner K-loop over the tile: each thread does TILE multiply-adds.
//
//       for (uint k = 0; k < TILE; k++)
//           acc += A_tile[ty][k] * B_tile[k][tx];
//
//     Thread (ty, tx) reads row ty of A_tile and column tx of B_tile.
//
//   - Barrier so all reads of the tile complete before the next
//     iteration overwrites it.
//
// After all K/TILE iterations, write the accumulator:
//   c[m * N + n] = acc;

#include <metal_stdlib>
using namespace metal;

constant uint M    = 1024;
constant uint N    = 1024;
constant uint K    = 1024;
constant uint TILE = 16;

kernel void matmul_tiled(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float A_tile[TILE][TILE];
    threadgroup float B_tile[TILE][TILE];

    uint ty = tid_in_tg.y;     // row within the tile
    uint tx = tid_in_tg.x;     // col within the tile
    uint by = tg_in_grid.y;    // tile row
    uint bx = tg_in_grid.x;    // tile col

    uint m = by * TILE + ty;   // global row of C this thread owns
    uint n = bx * TILE + tx;   // global col of C this thread owns

    float acc = 0.0f;

    // ============================================================
    // TODO 1: K-loop over the inner dimension in tile-sized steps.
    //
    // for (uint t = 0; t < K / TILE; t++) {
    //     // (a) cooperative tile load
    //     A_tile[ty][tx] = a[m * K + (t * TILE + tx)];
    //     B_tile[ty][tx] = b[(t * TILE + ty) * N + n];
    //     threadgroup_barrier(mem_flags::mem_threadgroup);
    //
    //     // (b) inner accumulation: TILE multiply-adds per thread
    //     for (uint k = 0; k < TILE; k++) {
    //         acc += A_tile[ty][k] * B_tile[k][tx];
    //     }
    //     threadgroup_barrier(mem_flags::mem_threadgroup);
    // }
    //
    // Both barriers are load-bearing:
    //  - First: tile loads must finish before any thread reads the tile.
    //  - Second: every thread's read of the tile must finish before the
    //    NEXT iteration's load starts overwriting it.
    // ============================================================



    // ============================================================
    // TODO 2: write the accumulator to its slot in C.
    //
    // c[m * N + n] = acc;
    // ============================================================



}
