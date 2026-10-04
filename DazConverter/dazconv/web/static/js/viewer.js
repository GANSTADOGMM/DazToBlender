// Three.js preview of the *converted* model (loads the exported glTF), so the
// viewport shows exactly what the output file contains: geometry, materials,
// textures, the rig and the morph targets.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0e13);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 500);
    this.camera.position.set(1.6, 1.3, 2.6);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.target.set(0, 0.9, 0);

    this._buildEnvironment();

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.skeletonHelper = null;
    this.spin = false;
    this.wireframe = false;
    this.texturesOn = true;
    this.morphMeshes = [];
    this._loader = new GLTFLoader();
    this._raf = null;

    const ro = new ResizeObserver(() => this.resize());
    ro.observe(container);
    this.resize();
    this._animate();
  }

  _buildEnvironment() {
    const hemi = new THREE.HemisphereLight(0xd8e6ff, 0x2a2418, 1.15);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xfff4e6, 2.1);
    key.position.set(2.5, 3.5, 2.2);
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x9fc4ff, 0.75);
    fill.position.set(-3, 1.6, -1.5);
    this.scene.add(fill);

    const rim = new THREE.DirectionalLight(0xffffff, 1.0);
    rim.position.set(0, 2.0, -3.4);
    this.scene.add(rim);

    this.grid = new THREE.GridHelper(8, 32, 0x2a3240, 0x1a2029);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.65;
    this.scene.add(this.grid);
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _animate() {
    this._raf = requestAnimationFrame(() => this._animate());
    if (this.spin) this.root.rotation.y += 0.0055;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  clear() {
    this.root.clear();
    if (this.skeletonHelper) {
      this.scene.remove(this.skeletonHelper);
      this.skeletonHelper = null;
    }
    this.morphMeshes = [];
    this.root.rotation.y = 0;
  }

  /** Load a .glb from a URL.  Resolves with simple scene statistics. */
  async load(url) {
    const gltf = await this._loader.loadAsync(url);
    this.clear();
    this.root.add(gltf.scene);

    let vertices = 0, triangles = 0, meshes = 0, bones = 0;
    const materials = new Set();
    const morphNames = [];
    let firstSkinned = null;

    gltf.scene.traverse((o) => {
      if (o.isMesh) {
        meshes++;
        const g = o.geometry;
        vertices += g.attributes.position ? g.attributes.position.count : 0;
        triangles += g.index ? g.index.count / 3
          : (g.attributes.position ? g.attributes.position.count / 3 : 0);
        (Array.isArray(o.material) ? o.material : [o.material])
          .forEach((m) => m && materials.add(m.uuid));
        o.frustumCulled = false;
        if (o.morphTargetDictionary) {
          this.morphMeshes.push(o);
          for (const name of Object.keys(o.morphTargetDictionary)) {
            if (!morphNames.includes(name)) morphNames.push(name);
          }
        }
      }
      if (o.isSkinnedMesh && !firstSkinned) firstSkinned = o;
      if (o.isBone) bones++;
    });

    if (firstSkinned) {
      this.skeletonHelper = new THREE.SkeletonHelper(gltf.scene);
      this.skeletonHelper.visible = false;
      this.skeletonHelper.material.linewidth = 2;
      this.scene.add(this.skeletonHelper);
    }

    this._applyWireframe();
    this._applyTextures();
    this.frame();

    return { vertices, triangles: Math.round(triangles), meshes, bones,
             materials: materials.size, morphNames };
  }

  frame() {
    const box = new THREE.Box3().setFromObject(this.root);
    if (!box.isEmpty()) {
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const dist = maxDim / (2 * Math.tan((Math.PI * this.camera.fov) / 360)) * 1.55;
      this.controls.target.copy(center);
      this.camera.position.set(center.x + dist * 0.55, center.y + size.y * 0.12,
                               center.z + dist * 0.9);
      this.camera.near = Math.max(0.001, dist / 1000);
      this.camera.far = dist * 50;
      this.camera.updateProjectionMatrix();
      this.grid.position.y = box.min.y;
      this.controls.update();
    }
  }

  setWireframe(on) { this.wireframe = on; this._applyWireframe(); }
  _applyWireframe() {
    this.root.traverse((o) => {
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { if (m) m.wireframe = this.wireframe; });
    });
  }

  setTextures(on) { this.texturesOn = on; this._applyTextures(); }
  _applyTextures() {
    this.root.traverse((o) => {
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => {
        if (!m) return;
        if (m.userData._maps === undefined) {
          m.userData._maps = { map: m.map, normalMap: m.normalMap,
                               roughnessMap: m.roughnessMap,
                               emissiveMap: m.emissiveMap };
        }
        const s = m.userData._maps;
        m.map = this.texturesOn ? s.map : null;
        m.normalMap = this.texturesOn ? s.normalMap : null;
        m.roughnessMap = this.texturesOn ? s.roughnessMap : null;
        m.emissiveMap = this.texturesOn ? s.emissiveMap : null;
        m.needsUpdate = true;
      });
    });
  }

  setSkeleton(on) { if (this.skeletonHelper) this.skeletonHelper.visible = on; }
  hasSkeleton() { return !!this.skeletonHelper; }
  setSpin(on) { this.spin = on; }

  setMorph(name, value) {
    for (const mesh of this.morphMeshes) {
      const i = mesh.morphTargetDictionary[name];
      if (i !== undefined) mesh.morphTargetInfluences[i] = value;
    }
  }

  resetMorphs() {
    for (const mesh of this.morphMeshes) {
      if (mesh.morphTargetInfluences) mesh.morphTargetInfluences.fill(0);
    }
  }
}
