// vncMeta.ts — pure host-side shape validation for vnc.meta.json.
// Kept separate from main.ts so unit tests can import it without pulling in
// DOM / WebGPU dependencies.

export interface VncMeta {
  dn_inputs: Record<string, number[]>;
  motor: Record<string, { all: number[] } & Record<string, number[]>>;
  motor_by_subclass: Record<string, number[]>;
  motor_by_target: Record<string, number[]>;
  num_neurons: number;
  num_edges: number;
}

export function assertVncMetaShape(v: unknown): asserts v is VncMeta {
  if (typeof v !== "object" || v === null || Array.isArray(v)) {
    throw new Error("vnc.meta.json: missing root object");
  }
  const o = v as Record<string, unknown>;

  if (typeof o.dn_inputs !== "object" || o.dn_inputs === null) {
    throw new Error("vnc.meta.json: missing dn_inputs");
  }
  if (typeof o.motor !== "object" || o.motor === null) {
    throw new Error("vnc.meta.json: missing motor");
  }
  if (typeof o.motor_by_subclass !== "object" || o.motor_by_subclass === null) {
    throw new Error("vnc.meta.json: missing motor_by_subclass");
  }

  // Loose one-level validation: motor groups must be objects (arrays live
  // inside them). We deliberately do not walk every nested array deeply.
  for (const [key, group] of Object.entries(o.motor)) {
    if (typeof group !== "object" || group === null) {
      throw new Error(`vnc.meta.json: motor.${key} is not an object`);
    }
  }
}
