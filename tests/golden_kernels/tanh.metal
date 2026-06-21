#include <metal_stdlib>
using namespace metal;

kernel void tanh_kernel(device const float* x [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = tanh(x[i]);
}
