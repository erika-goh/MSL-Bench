#include <metal_stdlib>
using namespace metal;

// GELU, tanh approximation:
//   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//
// This is the form used by TensorFlow's gelu and explicitly supported
// by PyTorch's F.gelu(x, approximate='tanh'). The exact form uses
// `erf`, which Apple does NOT expose in metal_stdlib — that omission
// is itself an interesting LLM-eval signal: a CUDA-port that assumes
// erf is available will fail to compile on Metal. The tanh
// approximation is accurate to within ~1e-3 of the exact form.
constant float SQRT_2_OVER_PI = 0.7978845608028654f;   // sqrt(2 / π)
constant float GELU_COEFF     = 0.044715f;

kernel void gelu_kernel(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    float v = x[i];
    float inner = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanh(inner));
}
