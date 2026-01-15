import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { DATASETS, DEFAULT_DATASET_KEY } from "./datasets.js";
import { loadDataset } from "./data.js";
import {
  basisCoordsOfVector,
  mixAndNormalize,
  normalizeInPlace,
  orthogonalize,
  orthonormalize3,
  projectPointsTo3,
  randomUnitVector,
  xorshift32,
} from "./linalg.js";

const els = {
  datasetName: document.getElementById("datasetName"),
  status: document.getElementById("status"),
  viewport: document.getElementById("viewport"),

  primaryClass: document.getElementById("primaryClass"),
  primaryStrength: document.getElementById("primaryStrength"),
  primaryStrengthLabel: document.getElementById("primaryStrengthLabel"),
  primaryStrengthValue: document.getElementById("primaryStrengthValue"),

  secondaryEnabled: document.getElementById("secondaryEnabled"),
  secondaryControls: document.getElementById("secondaryControls"),
  secondaryClass: document.getElementById("secondaryClass"),
  secondaryStrength: document.getElementById("secondaryStrength"),
  secondaryStrengthLabel: document.getElementById("secondaryStrengthLabel"),
  secondaryStrengthValue: document.getElementById("secondaryStrengthValue"),

  showArrow: document.getElementById("showArrow"),
  showAxes: document.getElementById("showAxes"),
  pointSize: document.getElementById("pointSize"),
  pointSizeValue: document.getElementById("pointSizeValue"),
  randomTriad: document.getElementById("randomTriad"),
  resetCamera: document.getElementById("resetCamera"),

  legend: document.getElementById("legend"),
};

const state = {
  datasetKey: DEFAULT_DATASET_KEY,
  dataset: null,

  // UI
  primaryLabel: null,
  primaryStrength: 0,
  secondaryEnabled: false,
  secondaryLabel: null,
  secondaryStrength: 0,
  showArrow: true,
  showAxes: true,
  pointSize: 3,

  // three.js
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  pointsGroup: null,
  classPoints: new Map(),
  axesHelper: null,
  arrowGroup: null,
  primaryArrow: null,
  secondaryArrow: null,
  baseCameraPos: null,
  viewBasisBase: null,

  // buffers
  positions: null,

  classColors: new Map(),
  classVisibility: new Map(),
  rng: xorshift32(20240101),
  baseRadius: 1.0,
};

function setStatus(msg) {
  els.status.textContent = msg;
}

function hslColorForIndex(i, n) {
  // evenly spaced hues
  const h = (i / Math.max(1, n)) % 1;
  const c = new THREE.Color();
  c.setHSL(h, 0.62, 0.55);
  return c;
}

function buildLegend(dataset) {
  els.legend.innerHTML = "";
  dataset.classLabels.forEach((label, i) => {
    const c = state.classColors.get(label);
    const item = document.createElement("button");
    const isVisible = state.classVisibility.get(label) ?? true;
    item.className = "legend__item";
    item.type = "button";
    item.dataset.label = label;
    item.setAttribute("aria-pressed", String(isVisible));
    if (!isVisible) item.classList.add("is-hidden");
    if (isVisible) item.classList.add("is-active");

    const sw = document.createElement("div");
    sw.className = "legend__swatch";
    sw.style.background = `#${c.getHexString()}`;

    const txt = document.createElement("div");
    txt.className = "legend__label";
    txt.textContent = label;

    const stateLabel = document.createElement("div");
    stateLabel.className = "legend__state";
    stateLabel.textContent = isVisible ? "On" : "Off";

    item.appendChild(sw);
    item.appendChild(txt);
    item.appendChild(stateLabel);
    els.legend.appendChild(item);

    item.addEventListener("click", () => {
      setClassVisibility(label, !(state.classVisibility.get(label) ?? true));
    });
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setClassVisibility(label, !(state.classVisibility.get(label) ?? true));
      }
    });
  });
}

function populateSelect(selectEl, labels, selected) {
  selectEl.innerHTML = "";
  for (const l of labels) {
    const opt = document.createElement("option");
    opt.value = l;
    opt.textContent = l;
    if (l === selected) opt.selected = true;
    selectEl.appendChild(opt);
  }
}

function buildUI(dataset) {
  els.datasetName.textContent = `dataset: ${dataset.name}`;

  // Colors
  dataset.classLabels.forEach((label, i) => {
    state.classColors.set(label, hslColorForIndex(i, dataset.classLabels.length));
    state.classVisibility.set(label, true);
  });

  // Primary controls
  state.primaryLabel = dataset.classLabels[0];
  populateSelect(els.primaryClass, dataset.classLabels, state.primaryLabel);

  // Secondary controls
  state.secondaryLabel = dataset.classLabels.length > 1 ? dataset.classLabels[1] : dataset.classLabels[0];
  populateSelect(els.secondaryClass, dataset.classLabels, state.secondaryLabel);

  // Labels that make the "linked" relationship obvious
  updatePrimaryStrengthLabel();
  updateSecondaryStrengthLabel();

  // Default: secondary off
  els.secondaryEnabled.checked = false;
  toggleSecondaryControls(false);

  // Wire events
  els.primaryClass.addEventListener("change", () => {
    state.primaryLabel = els.primaryClass.value;
    updatePrimaryStrengthLabel();
    // If secondary is on and same class, bump secondary to next available
    if (state.secondaryEnabled && state.secondaryLabel === state.primaryLabel) {
      const alt = dataset.classLabels.find((c) => c !== state.primaryLabel) ?? state.primaryLabel;
      state.secondaryLabel = alt;
      els.secondaryClass.value = alt;
      updateSecondaryStrengthLabel();
    }
    updateSceneFromUI();
  });

  els.primaryStrength.addEventListener("input", () => {
    state.primaryStrength = Number(els.primaryStrength.value) / 100;
    els.primaryStrengthValue.textContent = `${Math.round(state.primaryStrength * 100)}%`;
    updateSceneFromUI();
  });

  els.secondaryEnabled.addEventListener("change", () => {
    state.secondaryEnabled = els.secondaryEnabled.checked;
    toggleSecondaryControls(state.secondaryEnabled);
    updateSceneFromUI();
  });

  els.secondaryClass.addEventListener("change", () => {
    state.secondaryLabel = els.secondaryClass.value;
    updateSecondaryStrengthLabel();
    updateSceneFromUI();
  });

  els.secondaryStrength.addEventListener("input", () => {
    state.secondaryStrength = Number(els.secondaryStrength.value) / 100;
    els.secondaryStrengthValue.textContent = `${Math.round(state.secondaryStrength * 100)}%`;
    updateSceneFromUI();
  });

  els.showArrow.addEventListener("change", () => {
    state.showArrow = els.showArrow.checked;
    if (state.arrowGroup) state.arrowGroup.visible = state.showArrow;
    renderOnce();
  });

  els.showAxes.addEventListener("change", () => {
    state.showAxes = els.showAxes.checked;
    if (state.axesHelper) state.axesHelper.visible = state.showAxes;
    renderOnce();
  });

  els.pointSize.value = Math.round(state.pointSize).toString();
  els.pointSizeValue.textContent = `${els.pointSize.value}px`;
  els.pointSize.addEventListener("input", () => {
    state.pointSize = Number(els.pointSize.value);
    els.pointSizeValue.textContent = `${els.pointSize.value}px`;
    updatePointSizes();
  });

  els.resetCamera.addEventListener("click", () => resetCamera());
  els.randomTriad.addEventListener("click", () => randomizeTriad());

  // Legend
  buildLegend(dataset);

  // Status note if missing Ws
  if (dataset.missingWs?.length) {
    setStatus(
      `Loaded ${dataset.N} points. Missing W vectors for: ${dataset.missingWs.join(", ")}`
    );
  } else {
    setStatus(`Loaded ${dataset.N} points across ${dataset.classLabels.length} classes.`);
  }
}

function updatePrimaryStrengthLabel() {
  els.primaryStrengthLabel.textContent = `Alignment strength (to “${state.primaryLabel}”)`;
}

function updateSecondaryStrengthLabel() {
  els.secondaryStrengthLabel.textContent = `Secondary strength (to “${state.secondaryLabel}”)`;
}

function toggleSecondaryControls(enabled) {
  els.secondaryControls.classList.toggle("is-disabled", !enabled);
  els.secondaryEnabled.checked = enabled;
}

function initThree() {
  const width = els.viewport.clientWidth;
  const height = els.viewport.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height);
  renderer.setClearColor(0x0b0f15, 1);

  els.viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(60, width / height, 0.01, 2000);
  camera.position.set(0, 0, 4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.addEventListener("change", renderOnce);

  const axesHelper = new THREE.AxesHelper(1.2);
  axesHelper.visible = state.showAxes;
  scene.add(axesHelper);

  // Arrow group
  const arrowGroup = new THREE.Group();
  arrowGroup.visible = state.showArrow;
  scene.add(arrowGroup);

  // Primary and secondary arrows (directions in view space, updated each tick)
  const primaryArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1.0, 0xffffff);
  arrowGroup.add(primaryArrow);

  const secondaryArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 0.85, 0xffffff);
  secondaryArrow.visible = false;
  arrowGroup.add(secondaryArrow);

  // Save
  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.controls = controls;
  state.axesHelper = axesHelper;
  state.arrowGroup = arrowGroup;
  state.primaryArrow = primaryArrow;
  state.secondaryArrow = secondaryArrow;

  state.baseCameraPos = camera.position.clone();

  window.addEventListener("resize", onResize);
}

function setClassVisibility(label, visible) {
  state.classVisibility.set(label, visible);
  const meta = state.classPoints.get(label);
  if (meta) meta.points.visible = visible;

  const item = els.legend.querySelector(`[data-label="${label}"]`);
  if (item) {
    item.classList.toggle("is-hidden", !visible);
    item.classList.toggle("is-active", visible);
    item.setAttribute("aria-pressed", String(visible));
    const stateLabel = item.querySelector(".legend__state");
    if (stateLabel) stateLabel.textContent = visible ? "On" : "Off";
  }
  renderOnce();
}

function updatePointSizes() {
  for (const meta of state.classPoints.values()) {
    meta.material.size = state.pointSize;
    meta.material.needsUpdate = true;
  }
  renderOnce();
}

function onResize() {
  if (!state.renderer || !state.camera) return;

  const width = els.viewport.clientWidth;
  const height = els.viewport.clientHeight;

  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(width, height);
  renderOnce();
}

function resetCamera() {
  if (!state.controls || !state.camera) return;
  state.controls.reset();
  state.camera.position.copy(state.baseCameraPos);
  state.camera.lookAt(0, 0, 0);
  renderOnce();
}

function randomizeTriad() {
  if (!state.dataset) return;
  state.viewBasisBase = randomTriad(state.dataset.dim);
  state.primaryStrength = 0;
  state.secondaryStrength = 0;
  els.primaryStrength.value = "0";
  els.primaryStrengthValue.textContent = "0%";
  els.secondaryStrength.value = "0";
  els.secondaryStrengthValue.textContent = "0%";
  updateSceneFromUI();
}

function randomTriad(dim) {
  const u1 = randomUnitVector(dim, state.rng);
  const u2 = randomUnitVector(dim, state.rng);
  const u3 = randomUnitVector(dim, state.rng);
  return orthonormalize3(u1, u2, u3, state.rng);
}

function createPointsObject(dataset) {
  const N = dataset.N;

  state.positions = new Float32Array(N * 3);
  const group = new THREE.Group();

  dataset.classLabels.forEach((label) => {
    const indices = dataset.classToIndices.get(label) ?? [];
    const positions = new Float32Array(indices.length * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingSphere();

    const color = state.classColors.get(label) ?? new THREE.Color(0xffffff);
    const material = new THREE.PointsMaterial({
      size: state.pointSize,
      sizeAttenuation: false,
      color,
      transparent: true,
      opacity: 0.95,
    });

    const points = new THREE.Points(geometry, material);
    points.visible = state.classVisibility.get(label) ?? true;
    group.add(points);
    state.classPoints.set(label, { points, geometry, material, positions, indices });
  });

  state.pointsGroup = group;
  state.scene.add(group);
}

function computeViewBasis(dataset) {
  const base = state.viewBasisBase ?? dataset.baseBasis; // [b1,b2,b3]
  const dim = dataset.dim;

  const b1 = base[0], b2 = base[1], b3 = base[2];

  // Primary axis: mix PCA axis1 with class w1
  const w1 = dataset.classWs.get(state.primaryLabel);
  let u1 = b1;
  if (w1) {
    const mixed = mixAndNormalize(b1, w1, state.primaryStrength);
    u1 = mixed ?? w1 ?? b1;
  }

  // Make b2 orthogonal to u1
  let b2o = orthogonalize(b2, [u1]);
  if (!normalizeInPlace(b2o)) {
    // fallback: random orthogonal
    const rng = state.rng;
    b2o = randomUnitVector(dim, rng);
    b2o = orthogonalize(b2o, [u1]);
    normalizeInPlace(b2o);
  }

  // Secondary axis: optional mix toward w2 (orthogonalized to u1)
  let u2 = b2o;
  if (state.secondaryEnabled) {
    const w2 = dataset.classWs.get(state.secondaryLabel);
    if (w2) {
      let t2 = orthogonalize(w2, [u1]);
      if (!normalizeInPlace(t2)) {
        // if w2 nearly collinear with u1, keep b2o
        t2 = b2o;
      }
      const mixed2 = mixAndNormalize(b2o, t2, state.secondaryStrength);
      u2 = mixed2 ?? t2 ?? b2o;
      // ensure orthogonal again
      u2 = orthogonalize(u2, [u1]);
      normalizeInPlace(u2);
    }
  }

  // Third axis from b3, orthogonal to u1,u2
  let u3 = orthogonalize(b3, [u1, u2]);
  if (!normalizeInPlace(u3)) {
    u3 = randomUnitVector(dim, state.rng);
    u3 = orthogonalize(u3, [u1, u2]);
    normalizeInPlace(u3);
  }

  // Final orthonormalization for safety
  return orthonormalize3(u1, u2, u3, state.rng);
}

function updatePointsAndArrows() {
  const dataset = state.dataset;
  if (!dataset || !state.pointsGroup) return;

  const basis = computeViewBasis(dataset);

  // Project points
  projectPointsTo3(dataset.Xc, dataset.N, dataset.dim, basis, state.positions);

  let maxSq = 0;
  for (let i = 0; i < state.positions.length; i += 3) {
    const x = state.positions[i];
    const y = state.positions[i + 1];
    const z = state.positions[i + 2];
    const d = x * x + y * y + z * z;
    if (d > maxSq) maxSq = d;
  }

  for (const meta of state.classPoints.values()) {
    const { indices, positions, geometry } = meta;
    for (let i = 0; i < indices.length; i++) {
      const src = indices[i] * 3;
      const dst = i * 3;
      positions[dst] = state.positions[src];
      positions[dst + 1] = state.positions[src + 1];
      positions[dst + 2] = state.positions[src + 2];
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeBoundingSphere();
  }

  // Compute a stable-ish radius to set arrow lengths
  const r = Math.sqrt(maxSq);
  state.baseRadius = Number.isFinite(r) && r > 0 ? r : 1.0;

  // Update primary arrow direction to show where w1 lies in the current view basis.
  const w1 = dataset.classWs.get(state.primaryLabel);
  if (w1) {
    const [x, y, z] = basisCoordsOfVector(w1, basis);
    const dir = new THREE.Vector3(x, y, z);
    if (dir.length() > 1e-6) dir.normalize();
    state.primaryArrow.setDirection(dir);
    state.primaryArrow.setLength(Math.max(0.6, state.baseRadius * 0.9), 0.14, 0.08);

    const c = state.classColors.get(state.primaryLabel);
    if (c) state.primaryArrow.setColor(c);
  }

  // Secondary arrow
  if (state.secondaryEnabled) {
    const w2 = dataset.classWs.get(state.secondaryLabel);
    state.secondaryArrow.visible = Boolean(w2);
    if (w2) {
      const [x2, y2, z2] = basisCoordsOfVector(w2, basis);
      const dir2 = new THREE.Vector3(x2, y2, z2);
      if (dir2.length() > 1e-6) dir2.normalize();
      state.secondaryArrow.setDirection(dir2);
      state.secondaryArrow.setLength(Math.max(0.5, state.baseRadius * 0.75), 0.12, 0.07);

      const c2 = state.classColors.get(state.secondaryLabel);
      if (c2) state.secondaryArrow.setColor(c2);
    }
  } else {
    state.secondaryArrow.visible = false;
  }
}

function updateSceneFromUI() {
  updatePointsAndArrows();
  renderOnce();
}

let _needsRender = true;
function renderOnce() { _needsRender = true; }

function animate() {
  requestAnimationFrame(animate);
  if (!state.renderer) return;

  state.controls?.update();

  if (_needsRender) {
    state.renderer.render(state.scene, state.camera);
    _needsRender = false;
  }
}

async function main() {
  try {
    setStatus("Loading dataset…");
    const config = DATASETS[state.datasetKey];
    state.dataset = await loadDataset(config);

    state.viewBasisBase = state.dataset.baseBasis;
    buildUI(state.dataset);
    initThree();
    createPointsObject(state.dataset);

    // Initial projection (PCA view)
    updatePointsAndArrows();

    // Set a reasonable camera distance based on bounding sphere
    const r = state.baseRadius || 1.0;
    state.camera.position.set(0, 0, Math.max(2.5, r * 3.2));
    state.baseCameraPos = state.camera.position.clone();
    state.controls.update();

    renderOnce();
    animate();
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
  }
}

main();
