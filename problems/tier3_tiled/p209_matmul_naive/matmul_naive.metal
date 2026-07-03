// Naive matmul reference: one thread per output element, K-loop in the kernel.
// Baseline for tiled variants. Correct but leaves cache reuse and threadgroup
// cooperation on the table -- expect MPS to beat this by 3-5x on 1024x1024x1024.
#include <metal_stdlib>
using namespace metal;

constant uint M = 1024;
constant uint N = 1024;
constant uint K = 1024;

kernel void matmul_naive(device const float* a   [[buffer(0)]],
                         device const float* b   [[buffer(1)]],
                         device float* c         [[buffer(2)]],
                         uint2 gid               [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += a[row * K + k] * b[k * N + col];
    }
    c[row * N + col] = acc;
}
