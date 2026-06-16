#include <metal_stdlib>
using namespace metal;

kernel void leaky_relu(device const float* x     [[buffer(0)]],
                       device const float* alpha [[buffer(1)]],
                       device float* out         [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = v > 0.0f ? v : alpha[0] * v;
}
