import * as THREE from 'https://unpkg.com/three@0.152.2/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.152.2/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'https://unpkg.com/three@0.152.2/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'https://unpkg.com/three@0.152.2/examples/jsm/loaders/OBJLoader.js';

const container = document.getElementById('canvas-container');
// remove status indicator if present
const statusEl = document.getElementById('status');
if (statusEl) statusEl.textContent = 'JS loaded';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(2, 2, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const light = new THREE.DirectionalLight(0xffffff, 1);
light.position.set(5, 10, 7.5);
scene.add(light);
scene.add(new THREE.AmbientLight(0x888888));

let meshObject = null;
let vertices = [];

function addMesh(geometry) {
  if (meshObject) scene.remove(meshObject);
  geometry.computeBoundingBox();
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ color: 0x888888, metalness: 0.1, roughness: 0.6 });
  meshObject = new THREE.Mesh(geometry, material);
  scene.add(meshObject);

  // extract vertices array
  vertices = [];
  const pos = geometry.attributes.position.array;
  for (let i = 0; i < pos.length; i += 3) {
    vertices.push(new THREE.Vector3(pos[i], pos[i+1], pos[i+2]));
  }
}

function computeNearestVertex(p) {
  let minD = Infinity;
  let nearest = null;
  for (let v of vertices) {
    const d = p.distanceToSquared(v);
    if (d < minD) {
      minD = d;
      nearest = v;
    }
  }
  return { nearest, d: Math.sqrt(minD) };
}

function runStreamlines(opts) {
  if (!meshObject) { alert('Load a mesh first'); return; }
  const bbox = meshObject.geometry.boundingBox;
  const xmin = bbox.min.x; const xmax = bbox.max.x;
  const ymin = bbox.min.y; const ymax = bbox.max.y;
  const zmin = bbox.min.z; const zmax = bbox.max.z;

  const seeds = [];
  const np = parseInt(opts.nparts);
  const ny = Math.ceil(Math.sqrt(np));
  const nz = Math.ceil(np / ny);
  for (let i = 0; i < ny; i++) {
    for (let j = 0; j < nz; j++) {
      const yy = ymin + (i + 0.5) * (ymax - ymin) / ny;
      const zz = zmin + (j + 0.5) * (zmax - zmin) / nz;
      seeds.push(new THREE.Vector3(xmin - 0.5*(xmax-xmin), yy, zz));
    }
  }

  const streamGroup = new THREE.Group();
  scene.add(streamGroup);

  const U = parseFloat(opts.U);
  const d0 = parseFloat(opts.d0);
  const kappa = parseFloat(opts.kappa);
  const dt = 0.05;
  const steps = 400;

  for (let s of seeds) {
    const pts = [];
    let p = s.clone();
    for (let t = 0; t < steps; t++) {
      const { nearest, d } = computeNearestVertex(p);
      const f = Math.exp(- (d/d0) * (d/d0));
      // push away from nearest vertex
      const away = new THREE.Vector3(0,0,0);
      if (nearest) {
        away.copy(p).sub(nearest).normalize();
      }
      const lateral = away.multiplyScalar(kappa * f);
      const vx = U * (1 - f);
      const v = new THREE.Vector3(vx, lateral.y, lateral.z);
      p = p.clone().addScaledVector(v, dt);
      pts.push(p.clone());
      // stop if far downstream
      if (p.x > xmax + (xmax-xmin)) break;
    }
    if (pts.length > 1) {
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({ color: 0xff3333, linewidth: 2 });
      const line = new THREE.Line(g, mat);
      streamGroup.add(line);
    }
  }
}

// UI wiring
document.getElementById('file').addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const name = f.name.toLowerCase();
  const reader = new FileReader();
  reader.addEventListener('load', (e) => {
    const arrayBuffer = e.target.result;
    if (name.endsWith('.stl')) {
      const loader = new STLLoader();
      const geom = loader.parse(arrayBuffer);
      addMesh(geom);
      // center and scale
      const bbox = geom.boundingBox;
      const center = new THREE.Vector3();
      bbox.getCenter(center);
      meshObject.position.sub(center);
      const size = new THREE.Vector3(); bbox.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z);
      const scale = 1.0 / maxDim;
      meshObject.scale.set(scale, scale, scale);
    } else if (name.endsWith('.obj')) {
      const loader = new OBJLoader();
      const text = new TextDecoder().decode(arrayBuffer);
      const obj = loader.parse(text);
      // merge geometry
      let geom = null;
      obj.traverse((c) => { if (c.isMesh) { geom = c.geometry; } });
      if (geom) addMesh(geom);
    } else {
      alert('Unsupported file type');
    }
  });
  // read as arraybuffer for STL, but FileReader will do text for OBJ
  if (name.endsWith('.stl')) reader.readAsArrayBuffer(f);
  else reader.readAsArrayBuffer(f);
});

document.getElementById('run').addEventListener('click', () => {
  const opts = {
    nparts: document.getElementById('nparts').value,
    U: document.getElementById('U').value,
    d0: document.getElementById('d0').value,
    kappa: document.getElementById('kappa').value,
  };
  runStreamlines(opts);
});

document.getElementById('clear').addEventListener('click', () => {
  // remove all lines
  scene.children = scene.children.filter(c => c.type !== 'Group');
  if (meshObject) scene.add(meshObject);
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
