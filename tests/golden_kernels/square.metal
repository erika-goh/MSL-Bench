#include <metal_stdlib>
using namespace metal;

kernel void square(device const float* x   [[buffer(0)]],
                   device float* out       [[buffer(1)]],
                   uint i                  [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = v * v;
}
