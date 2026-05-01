// physics.ts — loads the TuragaLab flybody MJCF (fruitfly.xml + 85 OBJ
// meshes) into MuJoCo via a VFS, steps physics each frame, and exposes
// body poses for rendering. Scale: flybody is in cm; we scale up for
// visibility in the arena (one fly ≈ 30 cm in the visual arena).
//
// Phase 2 scope: load + render. Actuator-driven gait control under DN
// drive is Phase 2.1 — flybody has 111 actuators and the gait-control
// problem is research-grade.

import loadMujoco from "@mujoco/mujoco";
import type { MainModule, MjModel, MjData, MjVFS } from "@mujoco/mujoco";

export interface BodyInfo {
  id: number;
  name: string;
  parentId: number;
  /** Mesh filenames attached to this body's geoms (visual). */
  meshFiles: string[];
}

export interface FlybodyPose {
  /** xpos: [x0,y0,z0, x1,y1,z1, ...] in MuJoCo cm. Length = 3 * nbody. */
  xpos: Float32Array;
  /** xquat: [w0,x0,y0,z0, w1,x1,y1,z1, ...]. Length = 4 * nbody. */
  xquat: Float32Array;
}

export class Physics {
  private mujoco!: MainModule;
  private model!: MjModel;
  private data!: MjData;
  private vfs!: MjVFS;
  private nbody = 0;

  bodies: BodyInfo[] = [];
  /** Reusable scratch buffers populated by step(). */
  private xposScratch!: Float32Array;
  private xquatScratch!: Float32Array;

  static async create(onProgress?: (msg: string) => void): Promise<Physics> {
    const p = new Physics();
    onProgress?.("loading mujoco_wasm");
    p.mujoco = await loadMujoco();

    onProgress?.("fetching fruitfly.xml");
    const xmlText = await (await fetch("/flybody/fruitfly.xml")).text();

    // Discover mesh files referenced by the XML so we can preload them
    // into the VFS. Anything in `file="…"` attributes counts.
    const meshFiles = Array.from(
      new Set(
        Array.from(xmlText.matchAll(/file="([^"]+)"/g), (m) => m[1])
          .filter((f) => f.endsWith(".obj")),
      ),
    );
    onProgress?.(`fetching ${meshFiles.length} meshes`);

    p.vfs = new p.mujoco.MjVFS();
    // Parallel fetch with bounded concurrency. Lower than 8 so vite's
    // HTTP/1.1 dev server doesn't queue. Progress is reported per
    // completed file so the brain log shows the fetch is alive.
    const CONCURRENCY = 4;
    let inFlight = 0, idx = 0, completed = 0, totalBytes = 0;
    const t0 = performance.now();
    await new Promise<void>((resolve, reject) => {
      const next = () => {
        while (inFlight < CONCURRENCY && idx < meshFiles.length) {
          const file = meshFiles[idx++];
          inFlight++;
          fetch(`/flybody/${file}`)
            .then((r) => {
              if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
              return r.arrayBuffer();
            })
            .then((buf) => {
              p.vfs.addBuffer(file, new Uint8Array(buf));
              completed++;
              totalBytes += buf.byteLength;
              if (completed % 20 === 0 || completed === meshFiles.length) {
                onProgress?.(`fetched ${completed}/${meshFiles.length} meshes (${(totalBytes / 1e6).toFixed(0)} MB)`);
              }
              inFlight--;
              if (idx >= meshFiles.length && inFlight === 0) resolve();
              else next();
            })
            .catch(reject);
        }
      };
      next();
    });
    onProgress?.(`fetched all meshes in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

    onProgress?.("compiling MJCF (synchronous; tab may freeze ~5-15s)");
    const tCompile = performance.now();
    p.model = p.mujoco.MjModel.from_xml_string(xmlText, p.vfs);
    onProgress?.(`MJCF compiled in ${((performance.now() - tCompile) / 1000).toFixed(1)} s`);
    p.data = new p.mujoco.MjData(p.model);
    p.nbody = p.model.nbody;

    p.xposScratch = new Float32Array(p.nbody * 3);
    p.xquatScratch = new Float32Array(p.nbody * 4);

    // Walk the body tree to collect names + parent links + mesh
    // attachments so the renderer can build a matching scene graph.
    p.bodies = p.collectBodies(xmlText);

    // Apply the model's keyframe (rest pose) if it has one — keeps the
    // fly from collapsing in t=0 before stepping.
    if (p.model.nkey > 0) {
      p.mujoco.mj_resetDataKeyframe(p.model, p.data, 0);
    } else {
      p.mujoco.mj_resetData(p.model, p.data);
    }

    onProgress?.(`flybody ready (${p.nbody} bodies)`);
    return p;
  }

  /** Step physics once. Reads MuJoCo body poses into scratch buffers. */
  step(substeps = 8): FlybodyPose {
    for (let s = 0; s < substeps; s++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    // xpos/xquat are live Float64 views — copy to Float32 once per frame.
    const xpos = this.data.xpos as Float64Array;
    const xquat = this.data.xquat as Float64Array;
    for (let i = 0; i < this.nbody * 3; i++) this.xposScratch[i] = xpos[i];
    for (let i = 0; i < this.nbody * 4; i++) this.xquatScratch[i] = xquat[i];
    return { xpos: this.xposScratch, xquat: this.xquatScratch };
  }

  reset() {
    if (this.model.nkey > 0) {
      this.mujoco.mj_resetDataKeyframe(this.model, this.data, 0);
    } else {
      this.mujoco.mj_resetData(this.model, this.data);
    }
  }

  dispose() {
    this.data.delete();
    this.model.delete();
    this.vfs.delete();
  }

  /** Parse the XML to extract body names + their visual mesh attachments. */
  private collectBodies(xmlText: string): BodyInfo[] {
    // Simple stack-based parse — just enough to associate <geom mesh="…"/>
    // entries with their enclosing <body name="…">. Brittle if the XML
    // ever stops being well-formed but works for fruitfly.xml.
    const bodies: BodyInfo[] = [];
    // worldbody is body 0 in MuJoCo.
    const worldbodyMatch = xmlText.match(/<worldbody>/);
    if (!worldbodyMatch) return bodies;

    // Use the model's mj_id2name to canonicalise names from MuJoCo.
    for (let id = 0; id < this.nbody; id++) {
      const name = this.mujoco.mj_id2name(
        this.model,
        this.mujoco.mjtObj.mjOBJ_BODY.value,
        id,
      ) ?? `body_${id}`;
      // body_parentid is a flat int array on the model
      const parentArr = this.model.body_parentid as Int32Array;
      const parentId = parentArr[id];
      bodies.push({ id, name, parentId, meshFiles: [] });
    }

    // Now scan the XML for <body name="X"><geom … mesh="Y"/></body> nesting.
    // For each body name, collect the mesh names referenced. Then look up
    // the mesh→file map from <mesh name="Y" file="Z.obj"/> declarations.
    const meshFile = new Map<string, string>();
    for (const m of xmlText.matchAll(/<mesh\s+name="([^"]+)"\s+file="([^"]+)"\s*\/?>/g)) {
      meshFile.set(m[1], m[2]);
    }

    // Per-body mesh attachments via simple body-stack walk.
    const bodyByName = new Map(bodies.map((b) => [b.name, b]));
    const stack: string[] = [];
    const tagRE = /<(\/?)(body|geom)\b([^>]*)>/g;
    let mt: RegExpExecArray | null;
    while ((mt = tagRE.exec(xmlText))) {
      const closing = mt[1] === "/";
      const tag = mt[2];
      const attrs = mt[3];
      if (tag === "body") {
        if (closing) {
          stack.pop();
        } else {
          const nm = /name="([^"]+)"/.exec(attrs)?.[1];
          stack.push(nm ?? "");
        }
      } else if (tag === "geom" && stack.length > 0) {
        const meshName = /mesh="([^"]+)"/.exec(attrs)?.[1];
        if (meshName) {
          const file = meshFile.get(meshName);
          const bodyName = stack[stack.length - 1];
          const body = bodyByName.get(bodyName);
          if (file && body) body.meshFiles.push(file);
        }
      }
    }
    return bodies;
  }

  /** Number of bodies (incl. worldbody at id=0). */
  get bodyCount() { return this.nbody; }
}
