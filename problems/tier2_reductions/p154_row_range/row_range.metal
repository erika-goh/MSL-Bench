// row_range: (max - min) per row, via a dual threadgroup tree reduction.
// maxv and minv reduce in parallel over the same tree; thread 0 writes the
// difference. Same barrier discipline as the argmax/argmin goldens.
#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_range(device const float* x   [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      uint tid_in_tg  [[thread_position_in_threadgroup]],
                      uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float maxv[K];
    threadgroup float minv[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    float v = x[b * K + tid];
    maxv[tid] = v;
    minv[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            maxv[tid] = max(maxv[tid], maxv[tid + stride]);
            minv[tid] = min(minv[tid], minv[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[b] = maxv[0] - minv[0];
    }
}
