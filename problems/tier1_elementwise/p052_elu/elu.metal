// Element-wise ELU (Exponential Linear Unit), alpha = 1.0.
// out = x            if x > 0
//     = exp(x) - 1   otherwise   (alpha * (exp(x) - 1) with alpha = 1)
#include <metal_stdlib>
using namespace metal;

kernel void elu(device const float* x   [[buffer(0)]],
                device float*       out [[buffer(1)]],
                uint i [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = v > 0.0f ? v : (exp(v) - 1.0f);
}
