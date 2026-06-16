#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul(device const float* x     [[buffer(0)]],
                       device const float* alpha [[buffer(1)]],
                       device float* out         [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = alpha[0] * x[i];
}
