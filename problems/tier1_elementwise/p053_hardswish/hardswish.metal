// Element-wise hardswish: x * relu6(x + 3) / 6, where relu6(t) = clamp(t, 0, 6).
#include <metal_stdlib>
using namespace metal;

kernel void hardswish(device const float* x   [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = v * clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
}
