// out = sigmoid(x @ w + b), fused: tiled matmul + per-column bias + activation.
// Matmul body is the plain cooperative-tile-staging kernel (p201); the only
// change is the store epilogue, which adds the bias for this column and
// applies sigmoid before writing to device memory.
#include <metal_stdlib>
using namespace metal;

constant uint N    = 1024;
constant uint K    = 1024;
constant uint TILE = 16;

kernel void fused_linear_sigmoid(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device const float* b   [[buffer(2)]],
    device float*       out [[buffer(3)]],
    uint2 tid_in_tg  [[thread_position_in_threadgroup]],
    uint2 tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float X_tile[TILE][TILE];
    threadgroup float W_tile[TILE][TILE];

    uint ty = tid_in_tg.y, tx = tid_in_tg.x;
    uint by = tg_in_grid.y, bx = tg_in_grid.x;

    uint m = by * TILE + ty;
    uint n = bx * TILE + tx;

    float acc = 0.0f;
    for (uint t = 0; t < K / TILE; t++) {
        X_tile[ty][tx] = x[m * K + (t * TILE + tx)];
        W_tile[ty][tx] = w[(t * TILE + ty) * N + n];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += X_tile[ty][k] * W_tile[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fused epilogue: bias add + sigmoid.
    float z = acc + b[n];
    out[m * N + n] = 1.0f / (1.0f + exp(-z));
}
