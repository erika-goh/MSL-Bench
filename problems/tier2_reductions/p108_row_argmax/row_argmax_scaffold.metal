// Scaffold for p108_row_argmax. Save as `row_argmax.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, B int32s — column indices of the row maxima)
//
// Dispatch geometry (from the spec):
//   grid        = (B*K, 1, 1)
//   threadgroup = (K,   1, 1)
//
// This is structurally identical to p102 row_max — same tree shape, same
// one-TG-per-row layout — with one key change: each scratch slot now holds
// a (value, index) PAIR instead of a single float. At each reduction
// stage, the value AND the index propagate together. The thread that
// finally writes out[row] writes the INDEX, not the value.
//
// Paired reductions in Metal: the natural way is two parallel
// threadgroup arrays — one for values, one for indices — kept in
// lockstep. (You could also use a struct, but two arrays compile
// to slightly cleaner code and there is no SIMD-shuffle for structs.)
//
// Ties: use strict `>` not `>=`. With strict-greater, "new beats current"
// only when value is strictly larger; equal values keep the survivor
// (which has the lower index, since reduction starts from low end).
// Matches torch.argmax's "first maximal" convention.

#include <metal_stdlib>
using namespace metal;

constant uint K = 256;

kernel void row_argmax(device const float* x   [[buffer(0)]],
                       device int*         out [[buffer(1)]],
                       uint tid_in_tg  [[thread_position_in_threadgroup]],
                       uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float vals[K];
    threadgroup int   idxs[K];

    uint b   = tg_in_grid;
    uint tid = tid_in_tg;

    // ============================================================
    // TODO 1: each thread loads its (value, index) pair.
    //
    //   vals[tid] = x[b * K + tid];
    //   idxs[tid] = int(tid);
    //
    // The index for thread `tid` is simply `tid` — the column it
    // is responsible for in row b.
    // ============================================================



    // ============================================================
    // TODO 2: barrier so every (value, index) pair is visible.
    // ============================================================



    // ============================================================
    // TODO 3: paired tree reduction.
    //
    // For stride = K/2, K/4, ..., 1:
    //   if (tid < stride) {
    //       if (vals[tid + stride] > vals[tid]) {
    //           vals[tid] = vals[tid + stride];
    //           idxs[tid] = idxs[tid + stride];
    //       }
    //       // strict > → ties keep the existing (lower-index) survivor
    //   }
    //   barrier;
    //
    // Both writes (vals and idxs) happen together inside the if,
    // so the pair stays consistent.
    // ============================================================



    // ============================================================
    // TODO 4: thread 0 writes the final index (not the value!) to out[b].
    // ============================================================



}
