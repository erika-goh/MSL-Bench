// row_argmin: index of the minimum value per row, via a paired (value, index)
// threadgroup tree reduction. Structurally identical to p108_row_argmax, but
// the survivor test is strict `<` (min), so ties keep the lower-index survivor
// — matching torch.argmin's convention.
#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_argmin(device const float* x   [[buffer(0)]],
                       device int*         out [[buffer(1)]],
                       uint tid_in_tg  [[thread_position_in_threadgroup]],
                       uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float vals[K];
    threadgroup int   idxs[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    vals[tid] = x[b * K + tid];
    idxs[tid] = int(tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = K / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (vals[tid + stride] < vals[tid]) {
                vals[tid] = vals[tid + stride];
                idxs[tid] = idxs[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[b] = idxs[0];
    }
}
