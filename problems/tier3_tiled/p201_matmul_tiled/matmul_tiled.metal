#include <metal_stdlib>
using namespace metal;

// M is implicit in the grid.y dimension; N and K are referenced below.
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

    uint ty = tid_in_tg.y;
    uint tx = tid_in_tg.x;
    uint by = tg_in_grid.y;
    uint bx = tg_in_grid.x;

    uint m = by * TILE + ty;
    uint n = bx * TILE + tx;

    float acc = 0.0f;

    for (uint t = 0; t < K / TILE; t++) {
        // Cooperative load: every thread brings one element of each tile
        // from device memory. 256 threads load 256+256 = 512 floats per
        // iteration, reused 16x by the inner k-loop below.
        A_tile[ty][tx] = a[m * K + (t * TILE + tx)];
        B_tile[ty][tx] = b[(t * TILE + ty) * N + n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner accumulation: each thread reads its row of A_tile and
        // its column of B_tile, doing TILE=16 multiply-adds.
        for (uint k = 0; k < TILE; k++) {
            acc += A_tile[ty][k] * B_tile[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[m * N + n] = acc;
}
