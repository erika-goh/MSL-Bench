#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_kernel(device const float* x [[buffer(0)]],
                           device float* out     [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    out[i] = 1.0f / (1.0f + exp(-x[i]));
}
