import type { MujocoSim } from "./mujocoSim";

/** The minimal surface the sim loop needs from a renderer, so the loop is
 * decoupled from any concrete (WebGL/WebGPU) implementation. */
export interface RendererLike {
  /** Rebuild the visible robot from a (possibly new) model. */
  setModel(sim: MujocoSim): void;
  /** Copy the latest body transforms from the simulation. */
  update(sim: MujocoSim): void;
  /** Draw the current frame. May be async for WebGPU backends. */
  render(): void | Promise<void>;
}
