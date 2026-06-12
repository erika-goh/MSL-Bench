#include <metal_stdlib>
using namespace metal;

// MKB_GRID: 1048576 1 1
// MKB_TG: 256 1 1
kernel void vector_add(device const float* a   [[buffer(0)]],
                       device const float* b   [[buffer(1)]],
                       device float* out       [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}
