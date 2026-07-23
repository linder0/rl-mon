// Import the full three API from the WebGPU build so every object (lights,
// materials, geometries) is an instance the WebGPURenderer's node pipeline
// recognizes. Mixing plain "three" objects into a "three/webgpu" renderer
// breaks instanceof checks (e.g. lights get ignored), so we deliberately use a
// single source here.
import * as THREE from "three/webgpu";
import { pass, screenUV, uv, smoothstep } from "three/tsl";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { MujocoSim } from "@/lib/mujocoSim";
import type { RendererLike } from "@/lib/rendererLike";
import { SCENE_THEMES, type Theme } from "@/lib/theme";

// MuJoCo geom type ids (mjtGeom).
const GEOM_PLANE = 0;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;

const GROUND_SIZE = 240; // large enough to fill the view; edges fade to backdrop
const GRID_CELL = 1; // meters per grid cell
const SNAP = GRID_CELL; // recenter step so world-locked grid never appears to slide

// Per-body external force arrows, drawn from MuJoCo's cfrc_ext — the 6D spatial
// force [torque(3), force(3)] on each body (the same quantity Ant feeds into its
// observation). We draw one arrow per body for the translational force and,
// optionally, one for the rotational torque, each in its own color. Real forces
// here are only a few newtons, so length = magnitude * a user "scale", clamped,
// with tiny values culled so the agent isn't buried in jitter arrows.
const FORCE_MIN = 0.05; // cull arrows whose magnitude is below this (N or N·m)
const FORCE_MAX_LEN = 3; // m: clamp very large forces so they stay on-screen
const ARROW_SHAFT_R = 0.022; // shaft radius (m)
const ARROW_HEAD_R = 0.07; // arrowhead base radius (m)
const ARROW_HEAD_LEN = 0.16; // arrowhead length at full scale (m)

const _forceUp = new THREE.Vector3(0, 1, 0);
const _forceDir = new THREE.Vector3();

/** User-tunable force visualization (Scene panel). Colors are 0xRRGGBB. */
export interface ForceVizCfg {
  /** Draw the translational force component (cfrc_ext[3..5]). */
  force: boolean;
  /** Draw the rotational torque component (cfrc_ext[0..2]) along its axis. */
  torque: boolean;
  forceColor: number;
  torqueColor: number;
  /** Meters of arrow length per unit magnitude (N or N·m). */
  scale: number;
}

export const DEFAULT_FORCE_VIZ: ForceVizCfg = {
  force: true,
  torque: false,
  forceColor: 0xff4d3d, // warm red — the conventional "force" hue
  torqueColor: 0x38bdf8, // cyan — clearly distinct from the force red
  scale: 0.15,
};

/** One reusable arrow: a group (positioned + oriented per body) holding a
 * stretched shaft and a fixed-proportion head, both along the group's local +Y. */
interface ForceArrow {
  group: THREE.Group;
  shaft: THREE.Mesh;
  head: THREE.Mesh;
}

export type Backend = "webgpu" | "webgl";

function hex(n: number): string {
  return `#${n.toString(16).padStart(6, "0")}`;
}

/** Vertical gradient backdrop (lighter up top), as a canvas texture. Gives the
 * scene depth versus a flat fill without a full skybox. */
function gradientBackground(top: number, bottom: number): THREE.Texture {
  const cv = document.createElement("canvas");
  cv.width = 2;
  cv.height = 256;
  const c = cv.getContext("2d")!;
  const g = c.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, hex(top));
  g.addColorStop(1, hex(bottom));
  c.fillStyle = g;
  c.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** A single grid cell tiled across the floor: filled ground color with a hairline
 * border. Tiled + anisotropically filtered, it reads as a clean studio grid. */
function gridTexture(ground: number, line: number): THREE.CanvasTexture {
  const S = 256;
  const cv = document.createElement("canvas");
  cv.width = cv.height = S;
  const c = cv.getContext("2d")!;
  c.fillStyle = hex(ground);
  c.fillRect(0, 0, S, S);
  c.strokeStyle = hex(line);
  c.lineWidth = 3;
  c.strokeRect(0, 0, S, S);
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(GROUND_SIZE / GRID_CELL, GROUND_SIZE / GRID_CELL);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/** three.js WebGPU scene that renders a MuJoCo model with a "clean studio" look:
 * image-based lighting, a gradient backdrop, a soft reflective floor whose grid
 * fades with distance, and bloom + vignette post-processing. Implements
 * RendererLike so it drops straight into the shared SimLoop. */
export class WebGPURenderer implements RendererLike {
  private readonly canvas: HTMLCanvasElement;
  private renderer!: THREE.WebGPURenderer;
  private post: THREE.PostProcessing | null = null;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly root: THREE.Group;
  private bodies: THREE.Group[] = [];
  private robotMats: THREE.MeshStandardNodeMaterial[] = [];
  private foodMarker: THREE.Mesh | null = null;
  // Force visualization: a container under `root` holding two lazily-grown pools
  // of arrows (one per force-bearing body) — translational force and rotational
  // torque — plus shared geometry and a per-type material.
  private forcesGroup!: THREE.Group;
  private forceArrows: ForceArrow[] = [];
  private torqueArrows: ForceArrow[] = [];
  private forceShaftGeo: THREE.CylinderGeometry | null = null;
  private forceHeadGeo: THREE.ConeGeometry | null = null;
  private forceMat: THREE.MeshBasicNodeMaterial | null = null;
  private torqueMat: THREE.MeshBasicNodeMaterial | null = null;
  private showForces = false;
  private forceViz: ForceVizCfg = { ...DEFAULT_FORCE_VIZ };
  private torsoIndex = 1;
  private ready = false;
  private resizeObserver: ResizeObserver | null = null;

  private hemi!: THREE.HemisphereLight;
  private ground!: THREE.Mesh;
  private keyLight!: THREE.DirectionalLight;
  private envTexture: THREE.Texture | null = null;
  private theme: Theme = "dark";

  // Live-editable scene appearance (Scene panel). Seeded from the theme, but
  // overridable at runtime; reset back to theme defaults on a theme switch.
  private sceneBg = 0x0b0e14;
  private sceneGround = 0x141924;
  private sceneGrid = 0x2a3346;
  // Agent accent override; null means "use the current theme's default accent".
  // Kept separately so it survives model reloads (which rebuild the materials).
  private sceneAgent: number | null = null;
  private gridOn = true;
  private bloomPass: { strength: { value: number } } | null = null;
  private bloomStrength = 0.5;

  follow = true;
  backend: Backend = "webgpu";
  private readonly camOffset = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement, theme: Theme = "dark") {
    this.canvas = canvas;
    this.theme = theme;
    const t = SCENE_THEMES[theme];
    this.sceneBg = t.bg;
    this.sceneGround = t.ground;
    this.sceneGrid = t.grid1;

    this.scene = new THREE.Scene();
    this.scene.background = gradientBackground(this.bgTop(), this.sceneBg);
    this.scene.fog = new THREE.Fog(this.sceneBg, 18, 55);

    const { width, height } = this.size();
    this.camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 400);
    this.camera.position.set(3.2, 2.2, 3.6);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 1.0, 0);
    this.controls.maxPolarAngle = Math.PI * 0.495;

    this.addLightsAndGround();

    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2; // Z-up (MuJoCo) -> Y-up (three.js)
    this.scene.add(this.root);

    // Foraging target for the AntFood task: a glowing sphere (bloom picks up the
    // emissive) placed in MuJoCo world coords under `root`. Hidden unless the
    // active sim has a food target. Positioned/scaled each frame in update().
    const foodMat = new THREE.MeshStandardNodeMaterial({
      color: 0xffca3a,
      roughness: 0.3,
      metalness: 0.0,
    });
    foodMat.emissive = new THREE.Color(0xffb703);
    foodMat.emissiveIntensity = 1.6;
    this.foodMarker = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), foodMat);
    this.foodMarker.castShadow = true;
    this.foodMarker.visible = false;
    this.root.add(this.foodMarker);

    // Force arrows live under `root` too, so their MuJoCo-frame positions and
    // directions get the same Z-up -> Y-up rotation as the bodies. Hidden until
    // the user turns on the Scene panel's "Show forces" toggle.
    this.forcesGroup = new THREE.Group();
    this.forcesGroup.visible = false;
    this.root.add(this.forcesGroup);
  }

  /** A slightly lighter tint of the current backdrop for the top of the
   * gradient (derived from whatever background color is active). */
  private bgTop(): number {
    const bg = new THREE.Color(this.sceneBg);
    bg.lerp(new THREE.Color(0xffffff), this.theme === "light" ? 0.5 : 0.06);
    return bg.getHex();
  }

  /** Create and initialize the GPU backend. Prefers WebGPU, falling back to
   * WebGL when navigator.gpu is unavailable (older browsers, some Linux, etc).
   * Must be awaited before the first render. Returns the active backend. */
  async init(): Promise<Backend> {
    const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;
    this.backend = hasWebGPU ? "webgpu" : "webgl";

    this.renderer = new THREE.WebGPURenderer({
      canvas: this.canvas,
      antialias: true,
      forceWebGL: !hasWebGPU,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    await this.renderer.init();

    // The renderer may transparently fall back to WebGL if the WebGPU device
    // request fails even when navigator.gpu exists.
    const b = this.renderer.backend as unknown as { isWebGPUBackend?: boolean };
    if (b && b.isWebGPUBackend === false) this.backend = "webgl";

    this.setupEnvironment();
    this.setupPostProcessing();

    const { width, height } = this.size();
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();

    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(this.canvas);

    this.ready = true;
    return this.backend;
  }

  /** Soft image-based lighting from a procedural room, for believable material
   * shading and gentle floor/robot reflections. Best-effort: on failure the
   * scene still works with the analytic lights below. */
  private setupEnvironment(): void {
    try {
      const pmrem = new THREE.PMREMGenerator(this.renderer);
      this.envTexture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      this.scene.environment = this.envTexture;
      this.scene.environmentIntensity = this.theme === "light" ? 0.55 : 0.35;
      pmrem.dispose();
    } catch {
      /* no IBL: analytic lights still provide shading */
    }
  }

  /** Bloom (on bright highlights + activation glows) + a soft vignette. Wrapped
   * so any pipeline issue falls back to direct rendering instead of a blank
   * canvas. */
  private setupPostProcessing(): void {
    try {
      const post = new THREE.PostProcessing(this.renderer);
      const scenePass = pass(this.scene, this.camera);
      // High threshold: only genuinely bright pixels bloom, keeping it subtle.
      const bloomPass = bloom(scenePass, this.bloomStrength, 0.6, 0.85);
      const d = screenUV.sub(0.5).length();
      const vignette = d.mul(d).mul(0.9).oneMinus().clamp(0, 1);
      post.outputNode = scenePass.add(bloomPass).mul(vignette);
      this.post = post;
      this.bloomPass = bloomPass as unknown as { strength: { value: number } };
    } catch {
      this.post = null;
    }
  }

  private size(): { width: number; height: number } {
    const width = this.canvas.clientWidth || window.innerWidth;
    const height = this.canvas.clientHeight || window.innerHeight;
    return { width, height };
  }

  private addLightsAndGround(): void {
    const t = SCENE_THEMES[this.theme];
    const hemi = new THREE.HemisphereLight(t.hemiSky, t.hemiGround, 0.35);
    this.hemi = hemi;
    this.scene.add(hemi);

    // Warm key light with soft shadows — the main sculpting + grounding light.
    const key = new THREE.DirectionalLight(0xfff2e0, 2.1);
    key.position.set(4, 8, 5);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 40;
    const d = 8;
    key.shadow.camera.left = -d;
    key.shadow.camera.right = d;
    key.shadow.camera.top = d;
    key.shadow.camera.bottom = -d;
    key.shadow.bias = -0.0002;
    key.shadow.normalBias = 0.02;
    key.shadow.radius = 6; // softer PCF penumbra
    this.scene.add(key);
    this.scene.add(key.target);
    this.keyLight = key;

    const rim = new THREE.DirectionalLight(0x9ec1ff, 0.4);
    rim.position.set(-6, 4, -4);
    this.scene.add(rim);

    this.ground = this.buildGround();
    this.scene.add(this.ground);
  }

  /** A large floor plane (three.js XZ plane at y=0) with a tiled grid that fades
   * radially into the backdrop, plus a faint environment reflection. */
  private buildGround(): THREE.Mesh {
    const mat = new THREE.MeshStandardNodeMaterial({
      map: gridTexture(this.sceneGround, this.gridOn ? this.sceneGrid : this.sceneGround),
      roughness: 0.72,
      metalness: 0.0,
      transparent: true,
    });
    mat.envMapIntensity = this.theme === "light" ? 0.25 : 0.4;
    // Dissolve the floor into the backdrop with distance from its center (which
    // recenters under the agent), so the grid never ends in a hard edge.
    const r = uv().sub(0.5).length(); // 0 at center .. ~0.707 at corners
    mat.opacityNode = smoothstep(0.5, 0.06, r); // 1 near center -> 0 by the rim

    const ground = new THREE.Mesh(new THREE.PlaneGeometry(GROUND_SIZE, GROUND_SIZE), mat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    return ground;
  }

  /** Keep the floor + shadow light centered under the agent (snapped to the grid
   * cell so world-locked lines never appear to slide). The floor is huge and
   * fades at the rim, so this reads as an infinite studio ground. */
  private recenterWorld(tx: number, tz: number): void {
    const sx = Math.round(tx / SNAP) * SNAP;
    const sz = Math.round(tz / SNAP) * SNAP;
    // Cancel the plane offset in texture space so the grid stays world-locked.
    const mat = this.ground.material as THREE.MeshStandardNodeMaterial;
    const map = mat.map as THREE.Texture | null;
    if (map) {
      map.offset.set(sx / GRID_CELL, -sz / GRID_CELL);
      map.needsUpdate = true;
    }
    this.ground.position.set(sx, 0, sz);
    this.keyLight.target.position.set(tx, 0, tz);
    this.keyLight.position.set(tx + 4, 8, tz + 5);
  }

  /** Swap the scene palette (backdrop, fog, ground, grid, lights) to match the
   * UI theme. Resets any live Scene-panel overrides back to the theme defaults. */
  setTheme(theme: Theme): void {
    this.theme = theme;
    const t = SCENE_THEMES[theme];
    this.sceneBg = t.bg;
    this.sceneGround = t.ground;
    this.sceneGrid = t.grid1;

    this.applyBackground();
    this.hemi.color.set(t.hemiSky);
    this.hemi.groundColor.set(t.hemiGround);
    this.scene.environmentIntensity = theme === "light" ? 0.55 : 0.35;

    const mat = this.ground.material as THREE.MeshStandardNodeMaterial;
    mat.envMapIntensity = theme === "light" ? 0.25 : 0.4;
    this.applyGround();

    // Re-tint the agent to the active accent (an override if set, else the new
    // theme's default) so it stays legible on both themes.
    const accent = this.sceneAgent ?? t.robot;
    for (const rm of this.robotMats) rm.color.set(accent);
  }

  private applyBackground(): void {
    (this.scene.background as THREE.Texture | null)?.dispose?.();
    this.scene.background = gradientBackground(this.bgTop(), this.sceneBg);
    if (this.scene.fog) (this.scene.fog as THREE.Fog).color.set(this.sceneBg);
  }

  private applyGround(): void {
    const mat = this.ground.material as THREE.MeshStandardNodeMaterial;
    (mat.map as THREE.Texture | null)?.dispose();
    mat.map = gridTexture(this.sceneGround, this.gridOn ? this.sceneGrid : this.sceneGround);
    mat.needsUpdate = true;
  }

  // -- Live Scene-panel setters (hex color = 0xRRGGBB) ----------------------
  setBackgroundColor(hex: number): void {
    this.sceneBg = hex;
    this.applyBackground();
  }
  setGroundColor(hex: number): void {
    this.sceneGround = hex;
    this.applyGround();
  }
  setGridColor(hex: number): void {
    this.sceneGrid = hex;
    this.applyGround();
  }
  setGridVisible(on: boolean): void {
    this.gridOn = on;
    this.applyGround();
  }
  /** Set the agent accent, or pass null to fall back to the theme default. */
  setAgentColor(hex: number | null): void {
    this.sceneAgent = hex;
    const accent = hex ?? SCENE_THEMES[this.theme].robot;
    for (const rm of this.robotMats) rm.color.set(accent);
  }
  setBloomStrength(v: number): void {
    this.bloomStrength = v;
    if (this.bloomPass) this.bloomPass.strength.value = v;
  }
  /** Master toggle for the per-body force arrows (MuJoCo cfrc_ext). When off the
   * whole container is hidden; the pooled arrows are kept for reuse. */
  setShowForces(on: boolean): void {
    this.showForces = on;
    this.forcesGroup.visible = on;
    if (!on) {
      for (const a of this.forceArrows) a.group.visible = false;
      for (const a of this.torqueArrows) a.group.visible = false;
    }
  }

  /** Update the force-viz detail options (which components to draw, their colors,
   * and the magnitude->length scale). */
  setForceViz(cfg: ForceVizCfg): void {
    this.forceViz = { ...cfg };
    this.ensureForceAssets();
    this.forceMat!.color.set(cfg.forceColor);
    this.torqueMat!.color.set(cfg.torqueColor);
  }

  /** Current effective scene colors (so the Scene panel can seed its inputs). */
  sceneColors(): { bg: number; ground: number; grid: number; agent: number } {
    return {
      bg: this.sceneBg,
      ground: this.sceneGround,
      grid: this.sceneGrid,
      agent: this.sceneAgent ?? SCENE_THEMES[this.theme].robot,
    };
  }

  /** Rebuild the visible robot from a model's geoms. */
  setModel(sim: MujocoSim): void {
    for (const b of this.bodies) {
      this.root.remove(b);
      b.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | undefined;
        if (mat && typeof mat.dispose === "function") mat.dispose();
      });
    }
    this.bodies = [];
    this.robotMats = [];

    const m = sim.model as unknown as {
      nbody: number;
      ngeom: number;
      geom_type: Int32Array;
      geom_bodyid: Int32Array;
      geom_size: Float64Array;
      geom_pos: Float64Array;
      geom_quat: Float64Array;
      geom_rgba: Float32Array;
    };

    for (let b = 0; b < m.nbody; b++) {
      const group = new THREE.Group();
      this.root.add(group);
      this.bodies.push(group);
    }

    for (let g = 0; g < m.ngeom; g++) {
      const type = m.geom_type[g];
      if (type === GEOM_PLANE) continue; // we draw our own ground
      const sx = m.geom_size[g * 3 + 0];
      const sy = m.geom_size[g * 3 + 1];
      const sz = m.geom_size[g * 3 + 2];

      const geometry = makeGeometry(type, sx, sy, sz);
      if (!geometry) continue;

      // Use a theme accent so the agent always contrasts the backdrop, but keep
      // a hint of the model's own hue so different geoms stay distinguishable.
      const mjcf = new THREE.Color(
        m.geom_rgba[g * 4 + 0],
        m.geom_rgba[g * 4 + 1],
        m.geom_rgba[g * 4 + 2],
      );
      const accent = this.sceneAgent ?? SCENE_THEMES[this.theme].robot;
      const color = new THREE.Color(accent).lerp(mjcf, 0.15);
      // Soft, lightly polished plastic — env reflections make it read nicely.
      const material = new THREE.MeshStandardNodeMaterial({
        color,
        roughness: 0.42,
        metalness: 0.05,
      });
      material.envMapIntensity = 0.6;
      this.robotMats.push(material);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      // Local geom offset within its body frame (raw MuJoCo coords/quat).
      mesh.position.set(
        m.geom_pos[g * 3 + 0], m.geom_pos[g * 3 + 1], m.geom_pos[g * 3 + 2],
      );
      mesh.quaternion.set(
        m.geom_quat[g * 4 + 1], m.geom_quat[g * 4 + 2],
        m.geom_quat[g * 4 + 3], m.geom_quat[g * 4 + 0],
      );
      if (type === GEOM_ELLIPSOID) mesh.scale.set(sx, sy, sz);

      const body = m.geom_bodyid[g];
      this.bodies[body].add(mesh);
    }

    // Pick a torso body to follow: first non-world body.
    this.torsoIndex = m.nbody > 1 ? 1 : 0;
  }

  /** Copy the latest body world transforms from the simulation. Reads the
   * typed-array views fresh (MuJoCo returns a new view per access; a cached one
   * can detach when the WASM heap grows) and skips any non-finite value so a
   * transient bad read never throws or hides the model permanently. */
  update(sim: MujocoSim): void {
    const d = sim.data as unknown as { xpos: Float64Array; xquat: Float64Array };
    const xpos = d.xpos;
    const xquat = d.xquat;
    if (!xpos || !xquat || xpos.length < this.bodies.length * 3) return;

    for (let b = 0; b < this.bodies.length; b++) {
      const px = xpos[b * 3 + 0], py = xpos[b * 3 + 1], pz = xpos[b * 3 + 2];
      if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
      const g = this.bodies[b];
      g.position.set(px, py, pz);
      g.quaternion.set(
        xquat[b * 4 + 1], xquat[b * 4 + 2], xquat[b * 4 + 3], xquat[b * 4 + 0],
      );
    }

    // Track the agent's ABSOLUTE ground position each frame (not accumulated
    // deltas) so the camera always re-centers after a reset/model switch, and
    // preserve the user's orbit offset. MuJoCo is Z-up under a root rotated to
    // three.js Y-up, so world (mx,my,mz) -> three (mx, mz, -my): the ground
    // plane is three (x, z) = (mx, -my). Tracking both handles Ant, which
    // roams in 2D (Hopper/Walker only move in x).
    // Foraging target marker (AntFood only): sits on the ground at the food's
    // world position. MuJoCo coords go straight in since `root` does the Z-up
    // -> Y-up rotation for everything it holds.
    if (this.foodMarker) {
      if (sim.hasFood) {
        const r = sim.meta.food?.marker_radius ?? 0.3;
        this.foodMarker.visible = true;
        this.foodMarker.scale.setScalar(r);
        this.foodMarker.position.set(sim.foodX, sim.foodY, r);
      } else {
        this.foodMarker.visible = false;
      }
    }

    this.updateForceArrows(sim);

    const mx = xpos[this.torsoIndex * 3 + 0];
    const my = xpos[this.torsoIndex * 3 + 1];
    if (Number.isFinite(mx) && Number.isFinite(my)) {
      const tx = mx;
      const tz = -my;
      // The world (floor/grid/shadows) follows even when the camera doesn't,
      // so an agent that roams far never walks off the rendered floor.
      this.recenterWorld(tx, tz);
      if (this.follow) {
        this.camOffset.copy(this.camera.position).sub(this.controls.target);
        const ex = tx - this.controls.target.x;
        const ez = tz - this.controls.target.z;
        const snap = Math.hypot(ex, ez) > 3; // reset/switch jump
        this.controls.target.x += snap ? ex : ex * 0.12;
        this.controls.target.z += snap ? ez : ez * 0.12;
        this.camera.position.copy(this.controls.target).add(this.camOffset);
      }
    }
  }

  /** Lazily create the shared arrow geometry (unit shaft + head, both along +Y
   * with their base at the local origin) and the per-type materials. */
  private ensureForceAssets(): void {
    if (!this.forceMat) {
      const mat = new THREE.MeshBasicNodeMaterial({ color: this.forceViz.forceColor });
      mat.toneMapped = false; // keep the color punchy regardless of exposure
      this.forceMat = mat;
    }
    if (!this.torqueMat) {
      const mat = new THREE.MeshBasicNodeMaterial({ color: this.forceViz.torqueColor });
      mat.toneMapped = false;
      this.torqueMat = mat;
    }
    if (!this.forceShaftGeo) {
      const geo = new THREE.CylinderGeometry(ARROW_SHAFT_R, ARROW_SHAFT_R, 1, 12);
      geo.translate(0, 0.5, 0); // base at origin, extends up +Y (scale.y = length)
      this.forceShaftGeo = geo;
    }
    if (!this.forceHeadGeo) {
      const geo = new THREE.ConeGeometry(ARROW_HEAD_R, ARROW_HEAD_LEN, 16);
      geo.translate(0, ARROW_HEAD_LEN / 2, 0); // base at origin, tip up +Y
      this.forceHeadGeo = geo;
    }
  }

  /** Add a fresh arrow (of the given material) to a pool and the container. */
  private makeArrow(mat: THREE.MeshBasicNodeMaterial, pool: ForceArrow[]): ForceArrow {
    this.ensureForceAssets();
    const group = new THREE.Group();
    const shaft = new THREE.Mesh(this.forceShaftGeo!, mat);
    const head = new THREE.Mesh(this.forceHeadGeo!, mat);
    group.add(shaft, head);
    this.forcesGroup.add(group);
    const arrow: ForceArrow = { group, shaft, head };
    pool.push(arrow);
    return arrow;
  }

  /** Size an arrow to total length `len` (m): stretch the shaft, keep the head's
   * proportions but shrink it for very short arrows so the tip never overruns. */
  private setArrowLength(a: ForceArrow, len: number): void {
    const headLen = Math.min(ARROW_HEAD_LEN, len * 0.4);
    const shaftLen = Math.max(len - headLen, 1e-4);
    a.shaft.scale.y = shaftLen;
    a.head.position.y = shaftLen;
    a.head.scale.setScalar(headLen / ARROW_HEAD_LEN);
  }

  /** Draw an arrow at each body's origin for its external force and/or torque,
   * reading MuJoCo's cfrc_ext (the [torque(3), force(3)] spatial force per body,
   * world frame). No-op unless "Show forces" is on. Recomputes cfrc_ext each
   * frame so the arrows track the current substep, not the last control step. */
  private updateForceArrows(sim: MujocoSim): void {
    if (!this.showForces) return;
    this.ensureForceAssets(); // materials/geometry must exist before pooling arrows

    sim.computeContactForces();
    const d = sim.data as unknown as { xpos: Float64Array; cfrc_ext: Float64Array };
    const xpos = d.xpos;
    const cfrc = d.cfrc_ext;
    const nbody = this.bodies.length;
    if (!xpos || !cfrc || cfrc.length < nbody * 6) return;

    const { force, torque, scale } = this.forceViz;
    let nF = 0;
    let nT = 0;
    // Skip body 0 (worldbody): it carries the reaction forces, not the agent's.
    for (let b = 1; b < nbody; b++) {
      const px = xpos[b * 3 + 0], py = xpos[b * 3 + 1], pz = xpos[b * 3 + 2];
      if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;

      // cfrc_ext row layout: [torque(0..2), force(3..5)].
      if (force) {
        const fx = cfrc[b * 6 + 3], fy = cfrc[b * 6 + 4], fz = cfrc[b * 6 + 5];
        const mag = Math.hypot(fx, fy, fz);
        if (mag > FORCE_MIN) {
          const a = this.forceArrows[nF] ?? this.makeArrow(this.forceMat!, this.forceArrows);
          a.group.visible = true;
          a.group.position.set(px, py, pz);
          _forceDir.set(fx / mag, fy / mag, fz / mag);
          a.group.quaternion.setFromUnitVectors(_forceUp, _forceDir);
          this.setArrowLength(a, Math.min(mag * scale, FORCE_MAX_LEN));
          nF++;
        }
      }
      if (torque) {
        const tx = cfrc[b * 6 + 0], ty = cfrc[b * 6 + 1], tz = cfrc[b * 6 + 2];
        const mag = Math.hypot(tx, ty, tz);
        if (mag > FORCE_MIN) {
          const a = this.torqueArrows[nT] ?? this.makeArrow(this.torqueMat!, this.torqueArrows);
          a.group.visible = true;
          a.group.position.set(px, py, pz);
          _forceDir.set(tx / mag, ty / mag, tz / mag);
          a.group.quaternion.setFromUnitVectors(_forceUp, _forceDir);
          this.setArrowLength(a, Math.min(mag * scale, FORCE_MAX_LEN));
          nT++;
        }
      }
    }

    // Hide arrows left over from a busier frame (or a now-disabled component).
    for (let i = nF; i < this.forceArrows.length; i++) this.forceArrows[i].group.visible = false;
    for (let i = nT; i < this.torqueArrows.length; i++) this.torqueArrows[i].group.visible = false;
  }

  render(): Promise<void> | void {
    if (!this.ready) return;
    this.controls.update();
    // renderAsync/render return a Promise on the WebGPU backend once init() has
    // been awaited; the loop awaits it. Post-processing path applies bloom +
    // vignette; fall back to a direct render if it wasn't set up.
    if (this.post) {
      return this.post.renderAsync() as unknown as Promise<void>;
    }
    return this.renderer.render(this.scene, this.camera) as unknown as Promise<void>;
  }

  private onResize(): void {
    if (!this.ready) return;
    const { width, height } = this.size();
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  dispose(): void {
    this.resizeObserver?.disconnect();
    this.controls.dispose();
    this.envTexture?.dispose();
    this.forceShaftGeo?.dispose();
    this.forceHeadGeo?.dispose();
    this.forceMat?.dispose();
    this.torqueMat?.dispose();
    try {
      (this.renderer as unknown as { dispose?: () => void }).dispose?.();
    } catch {
      /* noop */
    }
  }
}

/** Build three.js geometry for a MuJoCo geom, aligning capsule/cylinder long
 * axes to MuJoCo's local Z (three primitives are Y-aligned by default). */
function makeGeometry(
  type: number, sx: number, sy: number, sz: number,
): THREE.BufferGeometry | null {
  switch (type) {
    case GEOM_SPHERE:
      return new THREE.SphereGeometry(sx, 24, 16);
    case GEOM_ELLIPSOID:
      return new THREE.SphereGeometry(1, 24, 16); // scaled by caller
    case GEOM_CAPSULE: {
      const geo = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 24);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    case GEOM_CYLINDER: {
      const geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 24);
      geo.rotateX(Math.PI / 2);
      return geo;
    }
    case GEOM_BOX:
      return new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    default:
      return null;
  }
}
