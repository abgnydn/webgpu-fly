// sim.ts — host-side WebGPU runtime: creates buffers, binds the LIF kernel,
// runs the step loop, exposes hooks for snapshot export.

import lifWgsl from "./shaders/lif.wgsl?raw";
import type { Brain } from "./brain";

export interface SimParams {
  dtMs: number;          // timestep in milliseconds, e.g. 1.0 for 1 kHz sim
  tauMs: number;         // membrane time constant, e.g. 20 ms
  vThresh: number;       // mV
  vReset: number;        // mV
  vRest: number;         // mV
  refractoryMs: number;  // ms (rounded to integer steps)
  extGain: number;       // multiplier on ext_input buffer
}

export const DEFAULT_PARAMS: SimParams = {
  dtMs: 1.0,
  tauMs: 20.0,
  vThresh: -50.0,
  vReset: -70.0,
  vRest: -65.0,
  refractoryMs: 2.0,
  extGain: 1.0,
};

const PARAMS_BYTES = 32; // matches struct Params in lif.wgsl (8 × 4 bytes)

export class FlySim {
  readonly device: GPUDevice;
  readonly brain: Brain;
  readonly params: SimParams;

  private pipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
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

  private bindAtoB!: GPUBindGroup; // gather reads A, writes B
  private bindBtoA!: GPUBindGroup; // gather reads B, writes A

  private step_ = 0;
  private prevIsA_ = true;

  static async create(brain: Brain, params: SimParams = DEFAULT_PARAMS): Promise<FlySim> {
    if (!("gpu" in navigator)) throw new Error("WebGPU not available");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");

    // We need enough headroom for col_idx + weight which can each be ~20 MB
    // for the aggregated v1 — well under default limits — but request adapter
    // max in case raw-synapse mode is enabled later.
    const required = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
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
    this.vmBuf = make(vm0, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, "vm");

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
        ],
      });
    this.bindAtoB = mkBind(this.spikesA, this.spikesB);
    this.bindBtoA = mkBind(this.spikesB, this.spikesA);
    this.bindGroup = this.bindAtoB; // initial; flipped each step
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
    this.device.queue.writeBuffer(this.paramsBuf, 0, ab);
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
   * Capture a snapshot: per-neuron spike counts accumulated across the next
   * `windowSteps` steps. Call before stepping, then advance, then await the
   * returned promise.
   *
   * Implementation: clear an accumulator buffer, run the steps, after each
   * step OR the spike bitset into the accumulator (still on GPU), then
   * read the accumulator back. Cheaper than reading per-step.
   *
   * Currently this just polls readSpikes() each step on the host — fine for
   * snapshot intervals of ~10 steps. If we want per-step later, move the
   * accumulation into a small WGSL kernel.
   */
  async captureRollingRate(windowSteps: number): Promise<Float32Array> {
    const N = this.brain.header.numNeurons;
    const rate = new Float32Array(N);
    for (let s = 0; s < windowSteps; s++) {
      this.step(1);
      const bits = await this.readSpikes();
      for (let i = 0; i < N; i++) {
        if ((bits[i >>> 5] >>> (i & 31)) & 1) rate[i] += 1;
      }
    }
    // normalise to spikes per step (0..1); host can convert to Hz with /dt
    for (let i = 0; i < N; i++) rate[i] /= windowSteps;
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
