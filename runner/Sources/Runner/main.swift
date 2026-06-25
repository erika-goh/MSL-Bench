// MKB Runner — minimal Metal compute kernel executor.
// Usage: Runner <manifest.json>
// Reads raw binary input buffers, dispatches the kernel (warmup + timed runs),
// writes raw binary outputs, prints a JSON result on stdout.
// Keep this dumb on purpose: one kernel, one dispatch shape, no features.

import Foundation
import Metal

struct BufferSpec: Codable {
    let path: String
    let bytes: Int
    let mode: String // "in" or "out"
}

struct Manifest: Codable {
    let metallib: String
    let entry_point: String
    let grid: [Int]        // total threads, [x, y, z]
    let threadgroup: [Int] // threads per threadgroup, [x, y, z]
    let buffers: [BufferSpec]
    let warmup_ms_min: Double
    let warmup_ms_max: Double
    let warmup_iter_max: Int
    let runs: Int
    // When true, output buffers are re-zeroed before every dispatch.
    // Required for atomic-accumulator kernels — without it, repeat
    // dispatches accumulate on top of prior results. Defaults to false
    // since elementwise kernels overwrite outputs and don't need it.
    let zero_output_each_run: Bool?
}

struct RunResult: Codable {
    let ok: Bool
    let error: String?
    let gpu_times_ms: [Double]
}

func fail(_ msg: String) -> Never {
    let res = RunResult(ok: false, error: msg, gpu_times_ms: [])
    if let data = try? JSONEncoder().encode(res), let s = String(data: data, encoding: .utf8) {
        print(s)
    }
    exit(1)
}

guard CommandLine.arguments.count == 2 else { fail("usage: Runner <manifest.json>") }
let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])

let m: Manifest
do {
    let manifestData = try Data(contentsOf: manifestURL)
    m = try JSONDecoder().decode(Manifest.self, from: manifestData)
} catch {
    fail("could not read or parse manifest at \(manifestURL.path): \(error)")
}

guard let device = MTLCreateSystemDefaultDevice() else { fail("no Metal device") }
guard let queue = device.makeCommandQueue() else { fail("could not create command queue") }

let libURL = URL(fileURLWithPath: m.metallib)
guard let library = try? device.makeLibrary(URL: libURL) else { fail("could not load metallib at \(m.metallib)") }
guard let fn = library.makeFunction(name: m.entry_point) else {
    fail("entry point '\(m.entry_point)' not found in metallib (check kernel function name)")
}
guard let pipeline = try? device.makeComputePipelineState(function: fn) else {
    fail("could not create pipeline state for '\(m.entry_point)'")
}

// Create buffers. Inputs are loaded from disk; outputs are zero-filled.
var mtlBuffers: [MTLBuffer] = []
for spec in m.buffers {
    if spec.mode == "in" {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: spec.path)), data.count == spec.bytes else {
            fail("input buffer \(spec.path): missing or wrong size (expected \(spec.bytes) bytes)")
        }
        guard let buf = data.withUnsafeBytes({ raw -> MTLBuffer? in
            device.makeBuffer(bytes: raw.baseAddress!, length: spec.bytes, options: .storageModeShared)
        }) else { fail("could not create input MTLBuffer for \(spec.path)") }
        mtlBuffers.append(buf)
    } else {
        guard let buf = device.makeBuffer(length: spec.bytes, options: .storageModeShared) else {
            fail("could not create output MTLBuffer (\(spec.bytes) bytes)")
        }
        memset(buf.contents(), 0, spec.bytes)
        mtlBuffers.append(buf)
    }
}

let grid = MTLSize(width: m.grid[0], height: m.grid[1], depth: m.grid[2])
let tg = MTLSize(width: m.threadgroup[0], height: m.threadgroup[1], depth: m.threadgroup[2])

if tg.width * tg.height * tg.depth > pipeline.maxTotalThreadsPerThreadgroup {
    fail("threadgroup size \(tg) exceeds max \(pipeline.maxTotalThreadsPerThreadgroup) for this kernel")
}

let zeroEachRun = m.zero_output_each_run ?? false

func dispatchOnce() -> Double? {
    // CPU-side memset of outputs before encode. Happens outside the GPU
    // timing window (we read cmd.gpuStartTime/gpuEndTime), so the cost
    // does not contaminate kernel_ms. Only matters for atomic kernels.
    if zeroEachRun {
        for (i, spec) in m.buffers.enumerated() where spec.mode == "out" {
            memset(mtlBuffers[i].contents(), 0, spec.bytes)
        }
    }
    guard let cmd = queue.makeCommandBuffer(),
          let enc = cmd.makeComputeCommandEncoder() else { return nil }
    enc.setComputePipelineState(pipeline)
    for (i, buf) in mtlBuffers.enumerated() {
        enc.setBuffer(buf, offset: 0, index: i)
    }
    // dispatchThreads supports non-uniform threadgroups on Apple GPUs,
    // so grid does not need to be a multiple of the threadgroup size.
    enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    if cmd.status == .error { return nil }
    return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0 // ms
}
var cumulativeMs: Double = 0.0
var iter: Int = 0
while cumulativeMs < m.warmup_ms_min && cumulativeMs < m.warmup_ms_max && iter < m.warmup_iter_max {
    guard let t = dispatchOnce() else { fail("kernel execution failed during warmup (runtime GPU error)") }
    cumulativeMs += t
    iter += 1
}

var times: [Double] = []
for _ in 0..<m.runs {
    guard let t = dispatchOnce() else { fail("kernel execution failed during timed run") }
    times.append(t)
}

// Write output buffers back to disk.
for (i, spec) in m.buffers.enumerated() where spec.mode == "out" {
    let data = Data(bytes: mtlBuffers[i].contents(), count: spec.bytes)
    do { try data.write(to: URL(fileURLWithPath: spec.path)) }
    catch { fail("could not write output buffer to \(spec.path)") }
}

let result = RunResult(ok: true, error: nil, gpu_times_ms: times)
let out = try! JSONEncoder().encode(result)
print(String(data: out, encoding: .utf8)!)
