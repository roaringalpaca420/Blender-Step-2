/* Face Tracking Avatar - MediaPipe + Three.js */
/* Uses Watchdog Rigged.glb with raccoon-style face tracking */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import {
  FilesetResolver,
  FaceLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9';

// --- Logging System ---
const logs = [];
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;

function logEntry(level, ...args) {
  const msg = args.map(a => (typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a))).join(' ');
  const entry = `[${new Date().toISOString()}] [${level}] ${msg}`;
  logs.push(entry);
  if (level === 'ERROR') originalConsoleError.apply(console, args);
  else if (level === 'WARN') originalConsoleWarn.apply(console, args);
}

console.error = (...args) => logEntry('ERROR', ...args);
console.warn = (...args) => logEntry('WARN', ...args);

function addLog(level, msg) {
  logEntry(level, msg);
}

// --- Status System ---
function setStatus(text, state = 'loading') {
  const el = document.getElementById('status-text');
  const bar = document.getElementById('status-bar');
  if (el) el.textContent = text;
  if (bar) {
    bar.className = '';
    if (state) bar.classList.add(state);
  }
}

// --- DOM Elements ---
let faceLandmarker = null;
let video = null;
let scene = null;
let avatar = null;

function getViewportSizeAtDepth(camera, depth) {
  const viewportHeightAtDepth = 2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * camera.fov));
  const viewportWidthAtDepth = viewportHeightAtDepth * camera.aspect;
  return new THREE.Vector2(viewportWidthAtDepth, viewportHeightAtDepth);
}

function createCameraPlaneMesh(camera, depth, material) {
  const viewportSize = getViewportSizeAtDepth(camera, depth);
  const geometry = new THREE.PlaneGeometry(viewportSize.width, viewportSize.height);
  geometry.translate(0, 0, -depth);
  return new THREE.Mesh(geometry, material);
}

class BasicScene {
  constructor() {
    this.height = window.innerHeight;
    this.width = (this.height * 1280) / 720;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.01, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.width, this.height);
    if (this.renderer.outputColorSpace !== undefined) {
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    } else {
      this.renderer.outputEncoding = THREE.sRGBEncoding;
    }
    document.querySelector('.container').appendChild(this.renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    this.scene.add(directionalLight);

    this.camera.position.z = 0;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    const orbitTarget = this.camera.position.clone();
    orbitTarget.z -= 5;
    this.controls.target = orbitTarget;
    this.controls.update();

    const vid = document.getElementById('video');
    const inputFrameTexture = new THREE.VideoTexture(vid);
    const inputFramesDepth = 500;
    const inputFramesPlane = createCameraPlaneMesh(
      this.camera,
      inputFramesDepth,
      new THREE.MeshBasicMaterial({ map: inputFrameTexture })
    );
    this.scene.add(inputFramesPlane);

    this.lastTime = 0;
    this.callbacks = [];
    this.render();

    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.render(this.scene, this.camera);
  }

  render(time = this.lastTime) {
    const delta = (time - this.lastTime) / 1000;
    this.lastTime = time;
    for (const cb of this.callbacks) cb(delta);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((t) => this.render(t));
  }
}

class Avatar {
  constructor(url, scene) {
    this.url = url;
    this.scene = scene;
    this.loader = new GLTFLoader();
    this.gltf = null;
    this.root = null;
    this.morphTargetMeshes = [];
    this.loadModel(url);
  }

  loadModel(url) {
    this.loader.load(
      url,
      (gltf) => {
        if (this.gltf) {
          this.scene.remove(this.gltf.scene);
          this.morphTargetMeshes = [];
        }
        this.gltf = gltf;
        this.scene.add(gltf.scene);
        this.init(gltf);
        addLog('INFO', 'Avatar model loaded successfully');
      },
      (progress) => {
        const pct = progress.total ? 100 * (progress.loaded / progress.total) : 0;
        addLog('INFO', `Loading model... ${pct.toFixed(1)}%`);
      },
      (err) => {
        addLog('ERROR', `Failed to load model: ${err.message || err}`);
        setStatus('Model load failed', 'error');
      }
    );
  }

  init(gltf) {
    gltf.scene.traverse((obj) => {
      if (obj.isBone && !this.root) this.root = obj;
      if (!obj.isMesh) return;
      const mesh = obj;
      mesh.frustumCulled = false;
      if (mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
        this.morphTargetMeshes.push(mesh);
      }
    });
  }

  updateBlendshapes(blendshapes) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      for (const [name, value] of blendshapes) {
        if (!(name in mesh.morphTargetDictionary)) continue;
        const idx = mesh.morphTargetDictionary[name];
        mesh.morphTargetInfluences[idx] = value;
      }
    }
  }

  applyMatrix(matrix, opts = {}) {
    const { scale = 1 } = opts;
    if (!this.gltf) return;
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(matrix);
  }

  offsetRoot(offset, rotation) {
    if (this.root) {
      this.root.position.copy(offset);
      if (rotation) {
        const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(rotation.x, rotation.y, rotation.z));
        this.root.quaternion.copy(q);
      }
    }
  }
}

function retarget(blendshapes) {
  const categories = blendshapes[0].categories;
  const coefsMap = new Map();
  for (let i = 0; i < categories.length; i++) {
    const b = categories[i];
    let score = b.score;
    switch (b.categoryName) {
      case 'browOuterUpLeft':
      case 'browOuterUpRight':
      case 'eyeBlinkLeft':
      case 'eyeBlinkRight':
        score *= 1.2;
        break;
    }
    coefsMap.set(b.categoryName, score);
  }
  return coefsMap;
}

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !avatar) return;
  try {
    const landmarks = faceLandmarker.detectForVideo(video, time);
    const matrices = landmarks.facialTransformationMatrixes;
    if (matrices && matrices.length > 0) {
      const matrix = new THREE.Matrix4().fromArray(matrices[0].data);
      avatar.applyMatrix(matrix, { scale: 40 });
    }
    const blends = landmarks.faceBlendshapes;
    if (blends && blends.length > 0) {
      avatar.updateBlendshapes(retarget(blends));
    }
  } catch (e) {
    addLog('ERROR', `Face detection error: ${e.message}`);
  }
}

function onVideoFrame(time) {
  detectFaceLandmarks(time);
  video.requestVideoFrameCallback(onVideoFrame);
}

async function streamWebcam() {
  setStatus('Requesting camera...', 'loading');
  video = document.getElementById('video');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: 'user', width: 1280, height: 720 }
    });
    video.srcObject = stream;
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = () => video.play();
      video.onplaying = resolve;
      video.onerror = () => reject(new Error('Video failed to play'));
    });
    setStatus('Calibrating camera...', 'loading');
    addLog('INFO', 'Camera acquired');
  } catch (e) {
    addLog('ERROR', `Camera failed: ${e.message}`);
    setStatus('Camera permission denied or failed', 'error');
    throw e;
  }
}

async function initMediaPipe() {
  setStatus('Loading MediaPipe...', 'loading');
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.9/wasm'
  );
  faceLandmarker = await FaceLandmarker.createFromModelPath(
    vision,
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
  );
  await faceLandmarker.setOptions({
    baseOptions: { delegate: 'GPU' },
    runningMode: 'VIDEO',
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true
  });
  addLog('INFO', 'MediaPipe FaceLandmarker loaded');
}

async function runDemo() {
  try {
    setStatus('Initializing...', 'loading');
    scene = new BasicScene();
    const modelPath = 'assets/Watchdog Rigged.glb';
    avatar = new Avatar(modelPath, scene.scene);

    await streamWebcam();
    await initMediaPipe();

    video.requestVideoFrameCallback(onVideoFrame);
    addLog('INFO', 'Face tracking started');
    setStatus('Ready - Face the camera', 'success');
  } catch (e) {
    addLog('ERROR', `Demo failed: ${e.message}`);
    setStatus(`Failed: ${e.message}`, 'error');
  }
}

// --- UI ---
document.getElementById('gear-btn').addEventListener('click', () => {
  const panel = document.getElementById('settings-panel');
  panel.classList.toggle('hidden');
});

document.getElementById('logs-btn').addEventListener('click', () => {
  document.getElementById('settings-panel').classList.add('hidden');
  document.getElementById('logs-modal').classList.remove('hidden');
  document.getElementById('logs-output').textContent = logs.length ? logs.join('\n') : 'No logs yet.';
});

document.getElementById('close-settings-btn').addEventListener('click', () => {
  document.getElementById('settings-panel').classList.add('hidden');
});

document.getElementById('close-logs-btn').addEventListener('click', () => {
  document.getElementById('logs-modal').classList.add('hidden');
});

document.getElementById('clear-logs-btn').addEventListener('click', () => {
  logs.length = 0;
  document.getElementById('logs-output').textContent = 'Logs cleared.';
});

document.getElementById('logs-modal').addEventListener('click', (e) => {
  if (e.target.id === 'logs-modal') {
    document.getElementById('logs-modal').classList.add('hidden');
  }
});

runDemo();
