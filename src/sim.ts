// sim.ts — host-side WebGPU runtime: creates buffers, binds the LIF kernel,
// runs the step loop, exposes hooks for snapshot export.

import lifWgsl from "./shaders/lif.wgsl?raw";
import accumWgsl from "./shaders/accumulate.wgsl?raw";
import type { Brain } from "./brain";

export interface SimParams {
  dtMs: number;          // timestep in milliseconds, e.g. 1.0 for 1 kHz sim
  tauMs: number;         // membrane time constant, e.g. 20 ms
  vThresh: number;       // mV
  vReset: number;        // mV
  vRest: number;         // mV
  refractoryMs: number;  // ms (rounded to integer steps)
  extGain: number;       // multiplier on ext_input buffer
  wSyn: number;          // mV per synapse — Shiu et al 2024 free parameter (0.275 mV)
}

// LIF state constants are from Shiu et al. 2024 (Nature 632:210–217;
// github.com/philshiu/Drosophila_brain_model/model.py):
//   v_0 / v_rst = -52 mV   v_th = -45 mV   t_mbr = 20 ms   t_rfc = 2.2 ms
//   tau_syn = 5 ms (alpha synapse)
//
// The kernel runs a real two-state alpha synapse, A_SYN = exp(-1/5).
// w_syn is tuned EMPIRICALLY (not from peak-matching) to land KC at the
// canonical 5-15% on Mixed sensory. Alpha synapse integrates each spike
// over ~5ms so the cascade amplifies non-linearly vs old single-step
// direct injection — peak-matching gives way too hot a brain (73% KC).
// 0.005 keeps the dataset's natural cascade strength visible without
// runaway.
export const DEFAULT_PARAMS: SimParams = {
  dtMs: 1.0,
  tauMs: 20.0,
  vThresh: -45.0,
  vReset: -52.0,
  vRest: -52.0,
  refractoryMs: 2.2,
  extGain: 2.0,
  wSyn: 0.005,
};

const PARAMS_BYTES = 36; // matches struct Params in lif.wgsl (9 × 4 bytes)

export class FlySim {
  readonly device: GPUDevice;
  readonly brain: Brain;
  readonly params: SimParams;

  private pipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
  private accumPipeline!: GPUComputePipeline;
  private clearAccumPipeline!: GPUComputePipeline;
  private bindGroup!: GPUBindGroup;
  private paramsBuf!: GPUBuffer;
  private rowPtrBuf!: GPUBuffer;
  private colIdxBuf!: GPUBuffer;
  private weightBuf!: GPUBuffer;
  private spikesA!: GPUBuffer;     // ping
  private spikesB!: GPUBuffer;     // pong
  private vmBuf!: GPUBuffer;
  private refracBuf!: GPUBuffer;
  private extBuf!: GPUBuffer;
  private gxBuf!: GPUBuffer;       // alpha-synapse state x
  private gyBuf!: GPUBuffer;       // alpha-synapse state y (= synaptic current)
  private accumBuf!: GPUBuffer;    // per-neuron spike count over current window

  private bindAtoB!: GPUBindGroup; // gather reads A, writes B
  private bindBtoA!: GPUBindGroup; // gather reads B, writes A
  private accumBindA!: GPUBindGroup; // accumulate reads spikesA
  private accumBindB!: GPUBindGroup; // accumulate reads spikesB

  private step_ = 0;
  private prevIsA_ = true;

  static async create(brain: Brain, params: SimParams = DEFAULT_PARAMS): Promise<FlySim> {
    if (!("gpu" in navigator)) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");

    // Request the adapter's maxes for storage. The LIF kernel binds 10
    // storage buffers (csr×3, spikes×2, vm, refrac, ext, g_x, g_y);
    // WebGPU's default cap is 8, so without this request the kernel
    // dispatches silently with no work — we hit that bug last time.
    const required = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    };
    const device = await adapter.requestDevice({ requiredLimits: required });

    return new FlySim(device, brain, params);
  }

  private constructor(device: GPUDevice, brain: Brain, params: SimParams) {
    this.device = device;
    this.brain = brain;
    this.params = params;
    this.initBuffers();
    this.initPipeline();
  }

  private initBuffers() {
    const { device, brain } = this;
    const N = brain.header.numNeurons;
    const words = (N + 31) >>> 5;

    const make = (data: ArrayBufferView, usage: GPUBufferUsageFlags, label?: string) => {
      const buf = device.createBuffer({
        size: data.byteLength,
        usage,
        mappedAtCreation: true,
        label,
      });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buf.unmap();
      return buf;
    };

    this.paramsBuf = device.createBuffer({
      size: PARAMS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "params",
    });

    this.rowPtrBuf = make(brain.rowPtr, GPUBufferUsage.STORAGE, "row_ptr");
    this.colIdxBuf = make(brain.colIdx, GPUBufferUsage.STORAGE, "col_idx");
    this.weightBuf = make(brain.weight, GPUBufferUsage.STORAGE, "weight");

    this.spikesA = device.createBuffer({
      size: words * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "spikes_A",
    });
    this.spikesB = device.createBuffer({
      size: words * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "spikes_B",
    });

    // Vm initialised to v_rest
    const vm0 = new Float32Array(N);
    vm0.fill(this.params.vRest);
    this.vmBuf = make(vm0, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, "vm");

    this.refracBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "refrac",
    });

    this.extBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "ext_input",
    });

    // Alpha-synapse states. Zero-initialized by WebGPU spec; reset()
    // also explicitly zeros them on stim restart so transients don't
    // leak across runs.
    this.gxBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "g_x",
    });
    this.gyBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: "g_y",
    });

    this.accumBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: "accum",
    });
  }

  private initPipeline() {
    const { device } = this;
    const module = device.createShaderModule({ code: lifWgsl, label: "lif" });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

    this.pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "step_lif" },
      label: "lif.step",
    });
    this.clearPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "clear_spikes" },
      label: "lif.clear",
    });

    const mkBind = (prev: GPUBuffer, curr: GPUBuffer) =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: this.rowPtrBuf } },
          { binding: 2, resource: { buffer: this.colIdxBuf } },
          { binding: 3, resource: { buffer: this.weightBuf } },
          { binding: 4, resource: { buffer: prev } },
          { binding: 5, resource: { buffer: curr } },
          { binding: 6, resource: { buffer: this.vmBuf } },
          { binding: 7, resource: { buffer: this.refracBuf } },
          { binding: 8, resource: { buffer: this.extBuf } },
          { binding: 9, resource: { buffer: this.gxBuf } },
          { binding: 10, resource: { buffer: this.gyBuf } },
        ],
      });
    this.bindAtoB = mkBind(this.spikesA, this.spikesB);
    this.bindBtoA = mkBind(this.spikesB, this.spikesA);
    this.bindGroup = this.bindAtoB; // initial; flipped each step

    // --- Accumulator pipeline (separate WGSL, separate layout) ---
    const accumModule = device.createShaderModule({ code: accumWgsl, label: "accum" });
    const accumLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const accumPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [accumLayout] });
    this.accumPipeline = device.createComputePipeline({
      layout: accumPipelineLayout,
      compute: { module: accumModule, entryPoint: "accumulate_spikes" },
      label: "accum.add",
    });
    this.clearAccumPipeline = device.createComputePipeline({
      layout: accumPipelineLayout,
      compute: { module: accumModule, entryPoint: "clear_accum" },
      label: "accum.clear",
    });
    const mkAccumBind = (spikes: GPUBuffer) =>
      device.createBindGroup({
        layout: accumLayout,
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: spikes } },
          { binding: 2, resource: { buffer: this.accumBuf } },
        ],
      });
    this.accumBindA = mkAccumBind(this.spikesA);
    this.accumBindB = mkAccumBind(this.spikesB);
  }

  private writeParams() {
    const alpha = Math.exp(-this.params.dtMs / this.params.tauMs);
    const refrSteps = Math.max(0, Math.round(this.params.refractoryMs / this.params.dtMs));
    const ab = new ArrayBuffer(PARAMS_BYTES);
    const dv = new DataView(ab);
    dv.setUint32(0, this.brain.header.numNeurons, true);
    dv.setFloat32(4, alpha, true);
    dv.setFloat32(8, this.params.vThresh, true);
    dv.setFloat32(12, this.params.vReset, true);
    dv.setFloat32(16, this.params.vRest, true);
    dv.setUint32(20, refrSteps, true);
    dv.setFloat32(24, this.params.extGain, true);
    dv.setUint32(28, this.step_, true);
    dv.setFloat32(32, this.params.wSyn, true);
    this.device.queue.writeBuffer(this.paramsBuf, 0, ab);
  }

  /** Hard reset: Vm to v_rest, refrac to 0, alpha-synapse state to 0,
   * both spike bitsets to 0. */
  reset() {
    const N = this.brain.header.numNeurons;
    const words = (N + 31) >>> 5;
    const vm0 = new Float32Array(N);
    vm0.fill(this.params.vRest);
    this.device.queue.writeBuffer(this.vmBuf, 0, vm0);
    const zerosN = new Uint32Array(N);
    this.device.queue.writeBuffer(this.refracBuf, 0, zerosN);
    const zerosF = new Float32Array(N);
    this.device.queue.writeBuffer(this.gxBuf, 0, zerosF);
    this.device.queue.writeBuffer(this.gyBuf, 0, zerosF);
    const zerosW = new Uint32Array(words);
    this.device.queue.writeBuffer(this.spikesA, 0, zerosW);
    this.device.queue.writeBuffer(this.spikesB, 0, zerosW);
    this.step_ = 0;
    this.prevIsA_ = true;
  }

  /** Set per-neuron external input (will be multiplied by ext_gain in shader). */
  setExternalInput(values: Float32Array) {
    if (values.length !== this.brain.header.numNeurons) {
      throw new Error(`ext_input length ${values.length} != ${this.brain.header.numNeurons}`);
    }
    this.device.queue.writeBuffer(this.extBuf, 0, values);
  }

  /** Advance the sim by `nSteps` timesteps. */
  step(nSteps = 1) {
    const N = this.brain.header.numNeurons;
    const words = (N + 31) >>> 5;
    const nWg = Math.ceil(N / 64);
    const nWgClear = Math.ceil(words / 64);

    for (let s = 0; s < nSteps; s++) {
      this.writeParams();
      this.bindGroup = this.prevIsA_ ? this.bindAtoB : this.bindBtoA;

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.bindGroup);
      pass.setPipeline(this.clearPipeline);
      pass.dispatchWorkgroups(nWgClear);
      pass.setPipeline(this.pipeline);
      pass.dispatchWorkgroups(nWg);
      pass.end();
      this.device.queue.submit([enc.finish()]);

      this.prevIsA_ = !this.prevIsA_;
      this.step_++;
    }
  }

  /** Read back current spike bitset (one step) — async. */
  async readSpikes(): Promise<Uint32Array> {
    const N = this.brain.header.numNeurons;
    const words = (N + 31) >>> 5;
    const src = this.prevIsA_ ? this.spikesA : this.spikesB; // last-written
    const stage = this.device.createBuffer({
      size: words * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, stage, 0, words * 4);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return out;
  }

  /**
   * Capture a snapshot: per-neuron spike counts over `windowSteps` steps,
   * normalised to spikes-per-step in [0, 1].
   *
   * All work runs in a single command encoder: clear accumulator → loop
   * (clear_spikes, step_lif, accumulate_spikes) × windowSteps → copy accum
   * to staging → submit → mapAsync. One queue submit, one readback.
   */
  async captureRollingRate(windowSteps: number): Promise<Float32Array> {
    const N = this.brain.header.numNeurons;
    const nWg = Math.ceil(N / 64);
    const nWgClear = Math.ceil(((N + 31) >>> 5) / 64);

    this.writeParams();

    const stage = this.device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: "accum_stage",
    });

    const enc = this.device.createCommandEncoder({ label: "captureRollingRate" });

    // Clear accumulator using its own pipeline (works for any layout)
    {
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.accumBindA); // any accum bind group; clear ignores spikes binding
      pass.setPipeline(this.clearAccumPipeline);
      pass.dispatchWorkgroups(nWg);
      pass.end();
    }

    for (let s = 0; s < windowSteps; s++) {
      const stepBind = this.prevIsA_ ? this.bindAtoB : this.bindBtoA;
      // After step, spikes_curr is the buffer that was just written to.
      // If prev=A and we write to B, then after the step the new spikes are in B.
      const accumBind = this.prevIsA_ ? this.accumBindB : this.accumBindA;

      const stepPass = enc.beginComputePass();
      stepPass.setBindGroup(0, stepBind);
      stepPass.setPipeline(this.clearPipeline);
      stepPass.dispatchWorkgroups(nWgClear);
      stepPass.setPipeline(this.pipeline);
      stepPass.dispatchWorkgroups(nWg);
      stepPass.end();

      const accumPass = enc.beginComputePass();
      accumPass.setBindGroup(0, accumBind);
      accumPass.setPipeline(this.accumPipeline);
      accumPass.dispatchWorkgroups(nWg);
      accumPass.end();

      this.prevIsA_ = !this.prevIsA_;
      this.step_++;
    }

    enc.copyBufferToBuffer(this.accumBuf, 0, stage, 0, N * 4);
    this.device.queue.submit([enc.finish()]);

    await stage.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();

    const rate = new Float32Array(N);
    const inv = 1 / windowSteps;
    for (let i = 0; i < N; i++) rate[i] = counts[i] * inv;
    return rate;
  }

  /** Read back per-neuron Vm — async. */
  async readVm(): Promise<Float32Array> {
    const N = this.brain.header.numNeurons;
    const stage = this.device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.vmBuf, 0, stage, 0, N * 4);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return out;
  }

  get currentStep() { return this.step_; }
}
