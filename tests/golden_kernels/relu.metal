#include <metal_stdlib>
using namespace metal;

// MKB_GRID: 1048576 1 1
// MKB_TG: 256 1 1
kernel void relu_kernel(device const float* x [[buffer(0)]],
                        device float* out     [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    out[i] = max(x[i], 0.0f);
}
