export type Theme = "dark" | "light";

/** Colors for the 3D scene per theme (the UI chrome is themed via CSS variables
 * in globals.css; these drive the WebGPU scene, which can't read CSS). */
export interface SceneTheme {
  bg: number;
  ground: number;
  grid1: number;
  grid2: number;
  hemiSky: number;
  hemiGround: number;
  /** Accent color applied to the agent so it always contrasts the backdrop
   * (MuJoCo's default body colors blend into a light scene). */
  robot: number;
}

export const SCENE_THEMES: Record<Theme, SceneTheme> = {
  dark: {
    bg: 0x0b0e14,
    ground: 0x141924,
    grid1: 0x2a3346,
    grid2: 0x1a2130,
    hemiSky: 0xbcd0ff,
    hemiGround: 0x20242e,
    robot: 0xff8a5b,
  },
  light: {
    bg: 0xe9eef6,
    ground: 0xd2dae8,
    grid1: 0xaab6ca,
    grid2: 0xc6cfdd,
    hemiSky: 0xffffff,
    hemiGround: 0xc2cad6,
    robot: 0xe1542a,
  },
};

