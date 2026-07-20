// C = A @ B^T, tiled with cooperative threadgroup staging.
// B is stored row-major with shape (N, K), so B[n, k] = b[n*K + k].
// The B-tile slot [ty][tx] must hold B^T[row=t*TILE+ty][col=bx*TILE+tx]
//   = B[bx*TILE+tx][t*TILE+ty] = b[(bx*TILE+tx)*K + (t*TILE+ty)].
// A and the inner accumulation are identical to the plain tiled matmul.
#include <metal_stdlib>
using namespace metal;

constant uint N    = 1024;
constant uint K    = 1024;
constant uint TILE = 16;

kernel void matmul_a_bt(
    device const float* a    [[buffer(0)]],
    device const float* b    [[buffer(1)]],
    device float*       c    [[buffer(2)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float A_tile[TILE][TILE];
    threadgroup float B_tile[TILE][TILE];

    uint ty = tid_in_tg.y, tx = tid_in_tg.x;
    uint by = tg_in_grid.y, bx = tg_in_grid.x;

    uint m = by * TILE + ty;
    uint n = bx * TILE + tx;

    float acc = 0.0f;

    for (uint t = 0; t < K / TILE; t++) {
        // A tile: same as plain tiled matmul.
        A_tile[ty][tx] = a[m * K + (t * TILE + tx)];
        // B^T tile: transposed/strided read from the (N, K) B.
        B_tile[ty][tx] = b[n * K + (t * TILE + ty)];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += A_tile[ty][k] * B_tile[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[m * N + n] = acc;
}
