#include <metal_stdlib>
using namespace metal;

kernel void axpby(device const float* a   [[buffer(0)]],
                  device const float* b   [[buffer(1)]],
                  device const float* x   [[buffer(2)]],
                  device const float* y   [[buffer(3)]],
                  device float* out       [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
    out[i] = a[0] * x[i] + b[0] * y[i];
}
