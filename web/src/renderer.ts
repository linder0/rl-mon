import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import type { MujocoSim } from "./mujocoSim";

// MuJoCo geom type ids (mjtGeom).
const GEOM_PLANE = 0;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;

/** three.js scene that renders a MuJoCo model. MuJoCo is Z-up; we parent every
 * body under a root rotated -90 deg about X so we can feed raw MuJoCo
 * coordinates straight through (three.js is Y-up). */
export class Renderer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly root: THREE.Group;
  private bodies: THREE.Group[] = [];
  private torsoIndex = 1;
  private ground!: THREE.Mesh;
  private grid!: THREE.GridHelper;
  private keyLight!: THREE.DirectionalLight;

  follow = true;
  private readonly camOffset = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e14);
    this.scene.fog = new THREE.Fog(0x0b0e14, 14, 40);

    // Soft image-based lighting for believable material shading.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    this.camera = new THREE.PerspectiveCamera(
      50, window.innerWidth / window.innerHeight, 0.01, 200,
    );
    this.camera.position.set(3.2, 2.2, 3.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 1.0, 0);
    this.controls.maxPolarAngle = Math.PI * 0.495;

    this.addLightsAndGround();

    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2; // Z-up (MuJoCo) -> Y-up (three.js)
    this.scene.add(this.root);

    window.addEventListener("resize", this.onResize);
  }

  private addLightsAndGround(): void {
    const hemi = new THREE.HemisphereLight(0xbcd0ff, 0x20242e, 0.6);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 2.6);
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
    this.scene.add(key);
    this.scene.add(key.target);
    this.keyLight = key;

    const rim = new THREE.DirectionalLight(0x88aaff, 0.7);
    rim.position.set(-6, 4, -4);
    this.scene.add(rim);

    // Ground plane (three.js XZ plane at y=0, matching MuJoCo's z=0 floor).
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({ color: 0x141924, roughness: 0.95, metalness: 0.0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);
    this.ground = ground;

    const grid = new THREE.GridHelper(200, 200, 0x2a3346, 0x1a2130);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    grid.position.y = 0.001;
    this.scene.add(grid);
    this.grid = grid;
  }

  /** Keep the (finite) floor, grid, and shadow-casting light centered under
   * the agent. MuJoCo's plane is infinite, but our meshes aren't — a good Ant
   * covers hundreds of meters and would walk straight off them. Positions
   * snap to the 1 m grid cells so the pattern never appears to slide. */
  private recenterWorld(tx: number, tz: number): void {
    const sx = Math.round(tx);
    const sz = Math.round(tz);
    this.ground.position.set(sx, 0, sz);
    this.grid.position.set(sx, 0.001, sz);
    this.keyLight.target.position.set(tx, 0, tz);
    this.keyLight.position.set(tx + 4, 8, tz + 5);
  }

  /** Rebuild the visible robot from a model's geoms. */
  setModel(sim: MujocoSim): void {
    for (const b of this.bodies) this.root.remove(b);
    this.bodies = [];

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

      const color = new THREE.Color(
        m.geom_rgba[g * 4 + 0],
        m.geom_rgba[g * 4 + 1],
        m.geom_rgba[g * 4 + 2],
      );
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.55,
        metalness: 0.15,
      });
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

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private onResize = (): void => {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  };
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
