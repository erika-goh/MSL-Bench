#include <metal_stdlib>
using namespace metal;

kernel void clamp_kernel(device const float* x  [[buffer(0)]],
                         device const float* lo [[buffer(1)]],
                         device const float* hi [[buffer(2)]],
                         device float* out      [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    out[i] = clamp(x[i], lo[0], hi[0]);
}
