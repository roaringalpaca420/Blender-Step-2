/* Face Tracking Avatar - MediaPipe + Three.js */
/* Uses Watchdog Rigged.glb with raccoon-style face tracking */

import * as THREE from 'three';
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

class BasicScene {
  constructor() {
    this.height = window.innerHeight;
    this.width = (this.height * 1280) / 720;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.01, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setClearColor(0x000000);
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

    this.camera.position.set(0, 0, 0);
    this.camera.lookAt(0, 0, -3);

    // No OrbitControls - fixed view, no dragging/panning

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
    let lastLogPct = -1;
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
        if (!progress.total) return;
        const pct = Math.floor(100 * (progress.loaded / progress.total));
        if (pct - lastLogPct >= 25 || pct >= 100) {
          lastLogPct = pct;
          addLog('INFO', `Loading model... ${Math.min(pct, 100)}%`);
        }
      },
      (err) => {
        addLog('ERROR', `Failed to load model: ${err.message || err}`);
        setStatus('Model load failed', 'error');
      }
    );
  }

  init(gltf) {
    this.gltf.scene.visible = false;
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

  setVisible(visible) {
    if (this.gltf?.scene) this.gltf.scene.visible = visible;
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
    const { fixedScale = 4, fixedDepth = -2.5 } = opts;
    if (!this.gltf) return;
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();
    matrix.decompose(pos, quat, scl);
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.position.set(0, 0, fixedDepth);
    this.gltf.scene.quaternion.copy(quat);
    this.gltf.scene.scale.setScalar(fixedScale);
    this.gltf.scene.updateMatrix();
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

let calibrationActive = true;

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !avatar) return;
  try {
    const landmarks = faceLandmarker.detectForVideo(video, time);
    const matrices = landmarks.facialTransformationMatrixes;
    const blends = landmarks.faceBlendshapes;
    if (calibrationActive) return;
    if (matrices && matrices.length > 0 && blends && blends.length > 0) {
      avatar.setVisible(true);
      const matrix = new THREE.Matrix4().fromArray(matrices[0].data);
      avatar.applyMatrix(matrix, { fixedScale: 4, fixedDepth: -2.5 });
      avatar.updateBlendshapes(retarget(blends));
    } else {
      avatar.setVisible(false);
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

const CALIBRATION_SECONDS = 3;

async function runCalibration() {
  document.body.classList.add('calibrating');
  const overlay = document.getElementById('calibration-overlay');
  const textEl = document.getElementById('calibration-text');
  const countdownEl = document.getElementById('calibration-countdown');
  overlay.classList.remove('hidden');

  for (let i = CALIBRATION_SECONDS; i >= 0; i--) {
    textEl.textContent = i > 0 ? 'Position your face in the frame' : 'Calibrated!';
    countdownEl.textContent = i > 0 ? String(i) : '';
    await new Promise((r) => setTimeout(r, 1000));
  }

  overlay.classList.add('hidden');
  document.body.classList.remove('calibrating');
  calibrationActive = false;
}

async function runDemo() {
  try {
    setStatus('Initializing...', 'loading');
    scene = new BasicScene();
    const modelPath = 'assets/Watchdog Rigged.glb';
    avatar = new Avatar(modelPath, scene.scene);

    await streamWebcam();
    await initMediaPipe();

    setStatus('Calibrating - position your face', 'loading');
    video.requestVideoFrameCallback(onVideoFrame);
    await runCalibration();

    addLog('INFO', 'Face tracking started');
    setStatus('Ready - Face the camera', 'success');
  } catch (e) {
    addLog('ERROR', `Demo failed: ${e.message}`);
    setStatus(`Failed: ${e.message}`, 'error');
    document.getElementById('calibration-overlay')?.classList.add('hidden');
    document.body.classList.remove('calibrating');
    calibrationActive = false;
  }
}

// --- UI ---
async function copyLogsToClipboard() {
  const text = logs.length ? logs.join('\n') : 'No logs yet.';
  try {
    await navigator.clipboard.writeText(text);
    const statusEl = document.getElementById('status-text');
    const origStatus = statusEl?.textContent || '';
    if (statusEl) {
      statusEl.textContent = 'Logs copied!';
      setTimeout(() => { statusEl.textContent = origStatus; }, 2000);
    }
  } catch (e) {
    addLog('ERROR', `Copy failed: ${e.message}`);
  }
}

document.getElementById('gear-btn').addEventListener('click', async () => {
  await copyLogsToClipboard();
  const panel = document.getElementById('settings-panel');
  panel.classList.toggle('hidden');
});

document.getElementById('logs-btn').addEventListener('click', () => {
  document.getElementById('settings-panel').classList.add('hidden');
  document.getElementById('logs-modal').classList.remove('hidden');
  document.getElementById('logs-output').textContent = logs.length ? logs.join('\n') : 'No logs yet.';
});

document.getElementById('copy-logs-btn').addEventListener('click', async () => {
  const text = document.getElementById('logs-output').textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById('copy-logs-btn');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = orig; }, 2000);
  } catch (e) {
    addLog('ERROR', `Copy failed: ${e.message}`);
  }
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
