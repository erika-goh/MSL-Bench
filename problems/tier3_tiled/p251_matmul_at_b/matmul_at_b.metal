// C = A^T @ B, tiled with cooperative threadgroup staging.
// A is stored K-major with shape (K, M), so A[k, m] = a[k*M + m].
// The A-tile slot [ty][tx] must hold A^T[row=by*TILE+ty][col=t*TILE+tx]
//   = A[t*TILE+tx][by*TILE+ty] = a[(t*TILE+tx)*M + (by*TILE+ty)].
// B and the inner accumulation are identical to the plain tiled matmul.
#include <metal_stdlib>
using namespace metal;

constant uint M    = 1024;
constant uint N    = 1024;
constant uint K    = 1024;
constant uint TILE = 16;

kernel void matmul_at_b(
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
        // A^T tile: transposed/strided read from the K-major A.
        A_tile[ty][tx] = a[(t * TILE + tx) * M + m];
        // B tile: same as plain tiled matmul.
        B_tile[ty][tx] = b[(t * TILE + ty) * N + n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += A_tile[ty][k] * B_tile[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    c[m * N + n] = acc;
}
