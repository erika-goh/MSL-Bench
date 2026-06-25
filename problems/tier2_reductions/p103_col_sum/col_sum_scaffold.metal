// Scaffold for p103_col_sum. Save your version as `col_sum.metal`.
// Don't edit this scaffold file in place.
//
// Buffer bindings:
//   buffer(0) = x   (input, B*K floats, row-major)
//   buffer(1) = out (output, K floats — one sum per column)
//
// Dispatch geometry (from the spec):
//   grid        = (K * TG, 1, 1)   where TG = threads per threadgroup
//   threadgroup = (TG,    1, 1)    so there are K threadgroups, one per column
//
// Conceptually different from p101/p102 even though the code looks similar:
//   - In row_sum, each thread's load was x[b * K + tid] → adjacent threads
//     read adjacent floats → coalesced.
//   - Here, each thread reads multiple rows of one fixed column. Within a
//     single iteration of the accumulate loop, adjacent threads read
//     addresses K=256 floats apart. UNCOALESCED.
//
// The kernel below uses the same threadgroup-shared scratch + tree reduce
// pattern as p101 for cross-thread combination. The only structural change
// is in TODO 1 / 1b: each thread does B/TG sequential additions before
// stashing its partial sum.

#include <metal_stdlib>
using namespace metal;

constant uint K       = 256;     // number of columns / output size
constant uint TG      = 256;     // threads per threadgroup
constant uint B       = 262144;  // number of rows
constant uint PER_THR = B / TG;  // rows summed by each thread = 1024

kernel void col_sum(device const float* x   [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    uint tid_in_tg  [[thread_position_in_threadgroup]],
                    uint tg_in_grid [[threadgroup_position_in_grid]])
{
    threadgroup float scratch[TG];

    uint col = tg_in_grid;
    uint tid = tid_in_tg;

    // ============================================================
    // TODO 1: accumulate this thread's partial sum.
    //
    // Strategy: thread `tid` handles rows tid, tid+TG, tid+2*TG, ...
    // (interleaved striping). For each row r in that set, add
    // x[r * K + col] to a local accumulator.
    //
    // float partial = 0.0f;
    // for (uint r = tid; r < B; r += TG) {
    //     partial += x[r * K + col];
    // }
    //
    // Why interleaved (stride TG) and not blocked (chunks of PER_THR)?
    // Cache reuse: at iteration 0 of the loop, threads 0..255 read rows
    // 0..255 simultaneously. Those reads span only 256 rows of a single
    // column — the working set hits the L1 with some locality even if
    // the addresses themselves are stride-K. With blocked partitioning,
    // thread t would read row t*PER_THR first, scattering reads across
    // the entire B-row range and getting zero cross-thread locality.
    // ============================================================



    // ============================================================
    // TODO 1b: stash this thread's partial into scratch[tid], then
    // barrier so every partial is visible to the reduction.
    // ============================================================



    // ============================================================
    // TODO 2: tree reduction across the TG partials.
    // Identical to p101: stride = TG/2, TG/4, ..., 1; barrier outside
    // the `if`. After log2(TG) stages, scratch[0] holds the column sum.
    // ============================================================



    // ============================================================
    // TODO 3: thread 0 writes scratch[0] to out[col].
    // ============================================================



}
