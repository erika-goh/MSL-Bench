// Element-wise softplus, numerically stable form.
// softplus(x) = log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|)),
// which avoids exp() overflow for large positive x.
#include <metal_stdlib>
using namespace metal;

kernel void softplus(device const float* x   [[buffer(0)]],
                     device float*       out [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    float v = x[i];
    out[i] = max(v, 0.0f) + log(1.0f + exp(-fabs(v)));
}
