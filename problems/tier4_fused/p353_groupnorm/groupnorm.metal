// GroupNorm over channels (no affine), fused. One threadgroup per sample,
// C threads. Channels split into G contiguous groups of CPG = C/G. Because
// groups are CPG-aligned (CPG is a power of two) and the reduction stride
// stays < CPG, a single tree reduction keeps every partner inside one group
// — a segmented reduction. Each thread then reads its group's sum/sumsq from
// the group's base slot and normalizes its own element.
#include <metal_stdlib>
using namespace metal;

constant uint  C   = 256;
constant uint  CPG = 32;    // channels per group (C / G)
constant float EPS = 1e-5f;

kernel void groupnorm(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    uint tid_in_tg  [[thread_position_in_threadgroup]],
    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float sum_scratch[C];
    threadgroup float sqs_scratch[C];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;
    uint idx = b * C + tid;
    uint local = tid % CPG;              // position within the group

    float v = x[idx];
    sum_scratch[tid] = v;
    sqs_scratch[tid] = v * v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Segmented tree reduction: stride < CPG keeps partners in-group.
    for (uint stride = CPG / 2; stride > 0; stride >>= 1) {
        if (local < stride) {
            sum_scratch[tid] += sum_scratch[tid + stride];
            sqs_scratch[tid] += sqs_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint base = tid - local;             // group's base slot
    float mean    = sum_scratch[base] / float(CPG);
    float var     = sqs_scratch[base] / float(CPG) - mean * mean;
    float inv_std = rsqrt(var + EPS);

    out[idx] = (v - mean) * inv_std;
}
