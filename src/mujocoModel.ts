// src/mujocoModel.ts — typed bridge for MuJoCo MjModel array fields.
//
// The @mujoco/mujoco package declares every MjModel array property as
// `any`, so room.ts casts through this interface. Extend it if new model
// properties are read in room.ts.
export interface FlyMjModel {
  ngeom: number;
  nbody: number;

  geom_group: Int32Array;
  geom_bodyid: Int32Array;
  geom_type: Int32Array;
  geom_dataid: Int32Array;
  geom_size: Float32Array;
  geom_rgba: Float32Array;
  geom_pos: Float32Array;
  geom_quat: Float32Array;

  mesh_vert: Float32Array;
  mesh_normal: Float32Array;
  mesh_texcoord: Float32Array;
  mesh_face: Int32Array;
  mesh_facetexcoord: Int32Array;
  mesh_facenormal: Int32Array;
  mesh_vertadr: Int32Array;
  mesh_vertnum: Int32Array;
  mesh_normaladr: Int32Array;
  mesh_normalnum: Int32Array;
  mesh_texcoordadr: Int32Array;
  mesh_faceadr: Int32Array;
  mesh_facenum: Int32Array;
}
