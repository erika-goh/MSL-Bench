#include <metal_stdlib>
using namespace metal;

// One thread per output element. Each thread computes a 9-element
// 2D dot product over its input window. The weight matrix (36
// bytes) is small enough that every thread reads it from L1 / the
// constant cache without explicit staging.
// OUT_H, OUT_W (= 1024) are implicit in the grid — the launch
// dispatches exactly one thread per output element. Only IN_W (the
// input's row stride) and K are referenced in the kernel body.
constant uint IN_W  = 1026;
constant uint K     = 3;
constant uint OUT_W = 1024;

kernel void conv2d_3x3_kernel(
    device const float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    device float*       y   [[buffer(2)]],
    uint2 idx [[thread_position_in_grid]])
{
    uint j_out = idx.x;
    uint i_out = idx.y;

    // The launch is clean (OUT_H, OUT_W both divisible by TILE=16),
    // so no bounds check is needed — every dispatched thread maps to
    // a valid output element.

    float acc = 0.0f;
    for (uint ki = 0; ki < K; ki++) {
        // Pre-multiply ki by row stride once per outer iteration.
        uint x_row_off = (i_out + ki) * IN_W;
        uint w_row_off = ki * K;
        for (uint kj = 0; kj < K; kj++) {
            acc += x[x_row_off + j_out + kj] * w[w_row_off + kj];
        }
    }

    y[i_out * OUT_W + j_out] = acc;
}
