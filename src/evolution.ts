// evolution.ts — WebGPU CMA-ES-flavored evolution of a CPG policy.
//
// Population of N gait policies, each evaluated in parallel by
// shaders/evolve.wgsl. After each generation we take the top-k%
// performers, compute their mean, and resample around it with shrinking
// sigma. The winner gets handed to physics.ts so the live mujoco_wasm
// fly walks under evolved parameters — sim-to-real for browsers.
//
// 8-dim policy: log_freq, coxa_amp, femur_amp, tibia_amp, swing_ratio,
// lift_offset, stride_gain, drag_coeff. Decoded inside evolve.wgsl.

import evolveWgsl from "./shaders/evolve.wgsl?raw";

export const POLICY_DIM = 8;

export interface EvolvedGait {
  freq: number;        // Hz (decoded from log_freq)
  coxaAmp: number;
  femurAmp: number;
  tibiaAmp: number;
  swingRatio: number;
  liftOffset: number;
  strideGain: number;
  dragCoeff: number;
}

export interface EvolutionConfig {
  population: number;
  generations: number;
  topFrac: number;     // fraction of population kept as elite
  initSigma: number;
  sigmaDecay: number;  // multiplier on sigma per generation
  onProgress?: (gen: number, bestReward: number, meanReward: number) => void;
}

const DEFAULT_CONFIG: EvolutionConfig = {
  population: 128,
  generations: 60,
  topFrac: 0.2,
  initSigma: 0.5,
  sigmaDecay: 0.96,
};

const PARAMS_BYTES = 16; // 4 × u32, 16-byte aligned

export class GaitEvolver {
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private bindGroup!: GPUBindGroup;
  private uniformBuf!: GPUBuffer;
  private paramsBuf!: GPUBuffer;
  private rewardsBuf!: GPUBuffer;
  private rewardsStage!: GPUBuffer;
  private populationSize = 0;

  static async create(): Promise<GaitEvolver> {
    if (!("gpu" in navigator)) throw new Error("WebGPU unavailable for gait evolver");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice();
    const e = new GaitEvolver();
    e.device = device;
    e.initPipeline();
    return e;
  }

  private initPipeline() {
    const module = this.device.createShaderModule({ code: evolveWgsl, label: "evolve" });
    const layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "evaluate" },
    });
    this.uniformBuf = this.device.createBuffer({
      size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private resizeBuffers(N: number) {
    if (N === this.populationSize) return;
    this.populationSize = N;
    this.paramsBuf?.destroy();
    this.rewardsBuf?.destroy();
    this.rewardsStage?.destroy();
    this.paramsBuf = this.device.createBuffer({
      size: N * POLICY_DIM * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.rewardsBuf = this.device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.rewardsStage = this.device.createBuffer({
      size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const layout = this.pipeline.getBindGroupLayout(0);
    this.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.paramsBuf } },
        { binding: 2, resource: { buffer: this.rewardsBuf } },
      ],
    });
    const u = new ArrayBuffer(PARAMS_BYTES);
    const dv = new DataView(u);
    dv.setUint32(0, N, true);
    dv.setUint32(4, POLICY_DIM, true);
    dv.setUint32(8, 200, true);  // sim_steps must match shader's SIM_STEPS const
    dv.setUint32(12, 0, true);
    this.device.queue.writeBuffer(this.uniformBuf, 0, u);
  }

  private async evaluate(params: Float32Array): Promise<Float32Array> {
    const N = params.length / POLICY_DIM;
    this.device.queue.writeBuffer(this.paramsBuf, 0, params);

    // We cannot map a buffer that already has unmapped pending state,
    // so we need to make sure rewardsStage is unmapped. New buffers
    // start unmapped; after readback we explicitly unmap below.
    const enc = this.device.createCommandEncoder({ label: "evolve" });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(N / 64));
    pass.end();
    enc.copyBufferToBuffer(this.rewardsBuf, 0, this.rewardsStage, 0, N * 4);
    this.device.queue.submit([enc.finish()]);

    await this.rewardsStage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.rewardsStage.getMappedRange().slice(0));
    this.rewardsStage.unmap();
    return out;
  }

  /** Run the full evolution loop and return the best gait found. */
  async evolve(cfg: Partial<EvolutionConfig> = {}): Promise<EvolvedGait> {
    const c = { ...DEFAULT_CONFIG, ...cfg };
    this.resizeBuffers(c.population);

    // Initial mean: a hand-seeded baseline gait so generation 0 isn't
    // wholly random. Numbers are roughly the Physics.driveLegs defaults
    // mapped to the policy's parameter space.
    const mean = new Float32Array([
      0.69,    // log_freq → 5 * e^0.69 ≈ 10 Hz
      0.85,    // coxa_amp
      0.65,    // femur_amp
      0.35,    // tibia_amp
      0.5,     // swing_ratio
      0.5,     // lift_offset
      1.0,     // stride_gain
      0.3,     // drag_coeff
    ]);
    let sigma = c.initSigma;

    const params = new Float32Array(c.population * POLICY_DIM);
    let best = { reward: -Infinity, params: new Float32Array(POLICY_DIM) };

    for (let g = 0; g < c.generations; g++) {
      // Sample N candidates around mean. First candidate keeps the
      // mean unperturbed so we always have a clean baseline (elitism).
      for (let d = 0; d < POLICY_DIM; d++) params[d] = mean[d];
      for (let i = 1; i < c.population; i++) {
        for (let d = 0; d < POLICY_DIM; d++) {
          params[i * POLICY_DIM + d] = mean[d] + sigma * gauss();
        }
      }

      const rewards = await this.evaluate(params);

      // Sort candidates by reward, descending.
      const ranked = Array.from({ length: c.population }, (_, i) => i);
      ranked.sort((a, b) => rewards[b] - rewards[a]);

      const k = Math.max(1, Math.floor(c.population * c.topFrac));
      const newMean = new Float32Array(POLICY_DIM);
      for (let r = 0; r < k; r++) {
        const idx = ranked[r];
        for (let d = 0; d < POLICY_DIM; d++) {
          newMean[d] += params[idx * POLICY_DIM + d];
        }
      }
      for (let d = 0; d < POLICY_DIM; d++) newMean[d] /= k;

      // Track absolute best.
      const bestIdx = ranked[0];
      if (rewards[bestIdx] > best.reward) {
        best.reward = rewards[bestIdx];
        for (let d = 0; d < POLICY_DIM; d++) {
          best.params[d] = params[bestIdx * POLICY_DIM + d];
        }
      }

      // Stats for progress callback.
      let sum = 0;
      for (let i = 0; i < c.population; i++) sum += rewards[i];
      c.onProgress?.(g, rewards[bestIdx], sum / c.population);

      mean.set(newMean);
      sigma *= c.sigmaDecay;
    }

    return decode(best.params);
  }

  dispose() {
    this.paramsBuf?.destroy();
    this.rewardsBuf?.destroy();
    this.rewardsStage?.destroy();
    this.uniformBuf?.destroy();
  }
}

function gauss(): number {
  // Box-Muller. Drops the second sample; good enough for evolution.
  const u = Math.random() || 1e-9;
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function decode(p: Float32Array): EvolvedGait {
  const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
  return {
    freq: 5.0 * Math.exp(clamp(p[0], -1.5, 1.5)),
    coxaAmp: clamp(p[1], 0, 1.5),
    femurAmp: clamp(p[2], 0, 1.5),
    tibiaAmp: clamp(p[3], 0, 1.5),
    swingRatio: clamp(p[4], 0.1, 0.9),
    liftOffset: clamp(p[5], 0, 1),
    strideGain: clamp(p[6], 0, 2),
    dragCoeff: clamp(p[7], 0, 1),
  };
}
