import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { DATASETS, DEFAULT_DATASET_KEY } from "./datasets.js";
import { loadDataset } from "./data.js";
import {
  basisCoordsOfVector,
  computeCovariance,
  mixAndNormalize,
  normalizeInPlace,
  orthogonalize,
  orthonormalize3,
  powerIterationTopK,
  projectPointsTo3,
  randomUnitVector,
  xorshift32,
} from "./linalg.js";

const els = {
  datasetName: document.getElementById("datasetName"),
  status: document.getElementById("status"),
  viewport: document.getElementById("viewport"),

  axis1Select: document.getElementById("axis1Select"),
  axis2Select: document.getElementById("axis2Select"),
  axis3Select: document.getElementById("axis3Select"),
  axis1Note: document.getElementById("axis1Note"),
  axis2Note: document.getElementById("axis2Note"),
  axis3Note: document.getElementById("axis3Note"),

  randomTriad: document.getElementById("randomTriad"),
  randomAxes: document.getElementById("randomAxes"),
  showPca: document.getElementById("showPca"),
  viewBasisNote: document.getElementById("viewBasisNote"),

  axisSlidersEnabled: document.getElementById("axisSlidersEnabled"),
  axisSlidersPanel: document.getElementById("axisSlidersPanel"),
  axis1Target: document.getElementById("axis1Target"),
  axis2Target: document.getElementById("axis2Target"),
  axis1Strength: document.getElementById("axis1Strength"),
  axis2Strength: document.getElementById("axis2Strength"),
  axis1StrengthValue: document.getElementById("axis1StrengthValue"),
  axis2StrengthValue: document.getElementById("axis2StrengthValue"),

  expertMode: document.getElementById("expertMode"),
  showAdditionalPcs: document.getElementById("showAdditionalPcs"),

  showArrow: document.getElementById("showArrow"),
  showAxes: document.getElementById("showAxes"),
  pointSize: document.getElementById("pointSize"),
  pointSizeValue: document.getElementById("pointSizeValue"),
  resetCamera: document.getElementById("resetCamera"),

  legend: document.getElementById("legend"),
};

const state = {
  datasetKey: DEFAULT_DATASET_KEY,
  dataset: null,
  pcaDirections: null,

  axisDefs: [],
  sliderEnabled: false,
  sliderTargets: [],
  sliderStrengths: [0, 0],
  expertMode: false,
  showAdditionalPcs: false,

  showArrow: true,
  showAxes: true,
  pointSize: 3,

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

function makeConceptDef(label) {
  return { kind: "concept", label };
}

function makePcaDef(index) {
  return { kind: "pca", index };
}

function makeRawDef(index) {
  return { kind: "raw", index };
}

function makeRandomDef(vector) {
  return { kind: "random", vector };
}

function axisDefLabel(def) {
  if (!def) return "Unknown";
  switch (def.kind) {
    case "concept":
      return `${def.label} (concept)`;
    case "pca":
      return `PC${def.index + 1}`;
    case "raw":
      return `axis ${def.index} (raw)`;
    case "random":
      return "random orthonormal";
    default:
      return "Unknown";
  }
}

function axisDefValue(def) {
  if (!def) return null;
  switch (def.kind) {
    case "concept":
      return `concept:${def.label}`;
    case "pca":
      return `pca:${def.index}`;
    case "raw":
      return `raw:${def.index}`;
    default:
      return null;
  }
}

function parseAxisDef(value) {
  if (!value) return null;
  const [kind, rest] = value.split(":");
  if (kind === "concept") return makeConceptDef(rest);
  if (kind === "pca") return makePcaDef(Number(rest));
  if (kind === "raw") return makeRawDef(Number(rest));
  return null;
}

function ensurePcaDirections(count) {
  if (!state.dataset) return;
  if (state.pcaDirections && state.pcaDirections.length >= count) return;
  const { Xc, N, dim } = state.dataset;
  const C = computeCovariance(Xc, N, dim);
  const vecs = powerIterationTopK(C, dim, count, 60);
  const dirs = vecs.map((v) => {
    normalizeInPlace(v);
    return v;
  });
  if (state.dataset.baseBasis?.length >= 3) {
    dirs[0] = state.dataset.baseBasis[0];
    dirs[1] = state.dataset.baseBasis[1];
    dirs[2] = state.dataset.baseBasis[2];
  }
  state.pcaDirections = dirs;
}

function getPcaDirection(index) {
  if (!state.dataset) return null;
  if (index < 3 && state.dataset.baseBasis?.[index]) {
    return state.dataset.baseBasis[index];
  }
  ensurePcaDirections(Math.max(index + 1, state.showAdditionalPcs ? state.dataset.dim : 3));
  return state.pcaDirections?.[index] ?? null;
}

function unitAxis(dim, index) {
  const v = new Float32Array(dim);
  if (index >= 0 && index < dim) v[index] = 1;
  return v;
}

function resolveAxisVector(def, fallback) {
  if (!state.dataset || !def) return fallback ?? null;
  const { dim, classWs } = state.dataset;
  if (def.kind === "concept") {
    return classWs.get(def.label) ?? fallback ?? null;
  }
  if (def.kind === "pca") {
    return getPcaDirection(def.index) ?? fallback ?? null;
  }
  if (def.kind === "raw") {
    return unitAxis(dim, def.index);
  }
  if (def.kind === "random") {
    return def.vector ?? fallback ?? null;
  }
  return fallback ?? null;
}

function getAxisDropdownOptions() {
  if (!state.dataset) return [];
  const options = [];
  for (const label of state.dataset.classLabels) {
    options.push({ value: axisDefValue(makeConceptDef(label)), text: axisDefLabel(makeConceptDef(label)) });
  }

  const pcaCount = state.showAdditionalPcs ? state.dataset.dim : 3;
  for (let i = 0; i < pcaCount; i++) {
    options.push({ value: axisDefValue(makePcaDef(i)), text: axisDefLabel(makePcaDef(i)) });
  }

  if (state.expertMode) {
    for (let i = 0; i < state.dataset.dim; i++) {
      options.push({ value: axisDefValue(makeRawDef(i)), text: axisDefLabel(makeRawDef(i)) });
    }
  }

  return options;
}

function getSliderTargetOptions() {
  if (!state.dataset) return [];
  const options = [];
  for (const label of state.dataset.classLabels) {
    options.push({ value: axisDefValue(makeConceptDef(label)), text: axisDefLabel(makeConceptDef(label)) });
  }

  const pcaCount = state.showAdditionalPcs ? state.dataset.dim : 3;
  for (let i = 0; i < pcaCount; i++) {
    options.push({ value: axisDefValue(makePcaDef(i)), text: axisDefLabel(makePcaDef(i)) });
  }

  return options;
}

function populateSelect(selectEl, options, selectedDef) {
  selectEl.innerHTML = "";
  const selectedValue = axisDefValue(selectedDef);
  let hasSelected = false;

  for (const opt of options) {
    const optionEl = document.createElement("option");
    optionEl.value = opt.value;
    optionEl.textContent = opt.text;
    if (opt.value === selectedValue) {
      optionEl.selected = true;
      hasSelected = true;
    }
    selectEl.appendChild(optionEl);
  }

  if (!hasSelected && selectedDef) {
    const fallback = document.createElement("option");
    fallback.value = selectedValue ?? "";
    fallback.textContent = axisDefLabel(selectedDef);
    fallback.disabled = true;
    fallback.selected = true;
    selectEl.prepend(fallback);
  }
}

function updateAxisDropdowns() {
  const options = getAxisDropdownOptions();
  populateSelect(els.axis1Select, options, state.axisDefs[0]);
  populateSelect(els.axis2Select, options, state.axisDefs[1]);
  populateSelect(els.axis3Select, options, state.axisDefs[2]);
}

function updateSliderDropdowns() {
  const options = getSliderTargetOptions();
  populateSelect(els.axis1Target, options, state.sliderTargets[0]);
  populateSelect(els.axis2Target, options, state.sliderTargets[1]);
}

function setViewBasisNote(text) {
  if (els.viewBasisNote) {
    els.viewBasisNote.textContent = text;
  }
}

function updateAxisNotes() {
  const base1 = axisDefLabel(state.axisDefs[0]);
  const base2 = axisDefLabel(state.axisDefs[1]);
  const base3 = axisDefLabel(state.axisDefs[2]);

  if (state.sliderEnabled) {
    const target1 = axisDefLabel(state.sliderTargets[0]);
    const target2 = axisDefLabel(state.sliderTargets[1]);
    const pct1 = Math.round(state.sliderStrengths[0] * 100);
    const pct2 = Math.round(state.sliderStrengths[1] * 100);

    els.axis1Note.textContent = `Axis 1: ${base1} → ${target1} (via slider, ${pct1}%)`;
    els.axis2Note.textContent = `Axis 2: ${base2} → ${target2} (via slider, ${pct2}%)`;
    els.axis3Note.textContent = `Axis 3: orthonormal completion (from ${base3})`;
  } else {
    els.axis1Note.textContent = `Axis 1: ${base1}`;
    els.axis2Note.textContent = `Axis 2: ${base2}`;
    els.axis3Note.textContent = `Axis 3: ${base3}`;
  }
}

function setAxisDefs(newDefs) {
  state.axisDefs = newDefs;
  updateAxisDropdowns();
  updateAxisNotes();
  updateSceneFromUI();
}

function buildUI(dataset) {
  els.datasetName.textContent = `dataset: ${dataset.name}`;

  dataset.classLabels.forEach((label, i) => {
    state.classColors.set(label, hslColorForIndex(i, dataset.classLabels.length));
    state.classVisibility.set(label, true);
  });

  state.axisDefs = [makePcaDef(0), makePcaDef(1), makePcaDef(2)];

  const defaultTarget1 = dataset.classLabels[0];
  const defaultTarget2 = dataset.classLabels[1] ?? dataset.classLabels[0];
  state.sliderTargets = [makeConceptDef(defaultTarget1), makeConceptDef(defaultTarget2)];
  state.sliderStrengths = [0, 0];

  updateAxisDropdowns();
  updateSliderDropdowns();
  updateAxisNotes();
  toggleSliderControls(false);

  els.axis1Select.addEventListener("change", () => {
    const def = parseAxisDef(els.axis1Select.value);
    if (def) state.axisDefs[0] = def;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis2Select.addEventListener("change", () => {
    const def = parseAxisDef(els.axis2Select.value);
    if (def) state.axisDefs[1] = def;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis3Select.addEventListener("change", () => {
    const def = parseAxisDef(els.axis3Select.value);
    if (def) state.axisDefs[2] = def;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axisSlidersEnabled.addEventListener("change", () => {
    state.sliderEnabled = els.axisSlidersEnabled.checked;
    toggleSliderControls(state.sliderEnabled);
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis1Target.addEventListener("change", () => {
    const def = parseAxisDef(els.axis1Target.value);
    if (def) state.sliderTargets[0] = def;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis2Target.addEventListener("change", () => {
    const def = parseAxisDef(els.axis2Target.value);
    if (def) state.sliderTargets[1] = def;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis1Strength.addEventListener("input", () => {
    state.sliderStrengths[0] = Number(els.axis1Strength.value) / 100;
    els.axis1StrengthValue.textContent = `${Math.round(state.sliderStrengths[0] * 100)}%`;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.axis2Strength.addEventListener("input", () => {
    state.sliderStrengths[1] = Number(els.axis2Strength.value) / 100;
    els.axis2StrengthValue.textContent = `${Math.round(state.sliderStrengths[1] * 100)}%`;
    updateAxisNotes();
    updateSceneFromUI();
  });

  els.expertMode.addEventListener("change", () => {
    state.expertMode = els.expertMode.checked;
    updateAxisDropdowns();
  });

  els.showAdditionalPcs.addEventListener("change", () => {
    state.showAdditionalPcs = els.showAdditionalPcs.checked;
    if (state.showAdditionalPcs && state.dataset) {
      ensurePcaDirections(state.dataset.dim);
    }
    updateAxisDropdowns();
    updateSliderDropdowns();
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
  els.randomAxes.addEventListener("click", () => randomizeAxes());
  els.showPca.addEventListener("click", () => showPcaTriad());
  setViewBasisNote("Showing PCA axes 1–3");

  buildLegend(dataset);

  if (dataset.missingWs?.length) {
    setStatus(`Loaded ${dataset.N} points. Missing W vectors for: ${dataset.missingWs.join(", ")}`);
  } else {
    setStatus(`Loaded ${dataset.N} points across ${dataset.classLabels.length} classes.`);
  }
}

function toggleSliderControls(enabled) {
  els.axisSlidersPanel.classList.toggle("is-disabled", !enabled);
  els.axisSlidersEnabled.checked = enabled;
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

  const arrowGroup = new THREE.Group();
  arrowGroup.visible = state.showArrow;
  scene.add(arrowGroup);

  const primaryArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1.0, 0xffffff);
  arrowGroup.add(primaryArrow);

  const secondaryArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 0.85, 0xffffff);
  arrowGroup.add(secondaryArrow);

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
  const u1 = randomUnitVector(state.dataset.dim, state.rng);
  const u2 = randomUnitVector(state.dataset.dim, state.rng);
  const u3 = randomUnitVector(state.dataset.dim, state.rng);
  const triad = orthonormalize3(u1, u2, u3, state.rng);
  setAxisDefs([makeRandomDef(triad[0]), makeRandomDef(triad[1]), makeRandomDef(triad[2])]);
  setViewBasisNote("Showing random triad");
}

function randomizeAxes() {
  if (!state.dataset) return;
  setAxisDefs([makeRawDef(0), makeRawDef(1), makeRawDef(2)]);
  setViewBasisNote("Showing raw axes 0–2");
}

function showPcaTriad() {
  if (!state.dataset) return;
  setAxisDefs([makePcaDef(0), makePcaDef(1), makePcaDef(2)]);
  setViewBasisNote("Showing PCA axes 1–3");
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

function computeBaselineBasis() {
  if (!state.dataset) return null;
  const fallback = randomUnitVector(state.dataset.dim, state.rng);
  const v1 = resolveAxisVector(state.axisDefs[0], fallback) ?? fallback;
  const v2 = resolveAxisVector(state.axisDefs[1], fallback) ?? fallback;
  const v3 = resolveAxisVector(state.axisDefs[2], fallback) ?? fallback;
  return orthonormalize3(v1, v2, v3, state.rng);
}

function computeSliderBasis(baseline) {
  const fallback = randomUnitVector(state.dataset.dim, state.rng);
  const t1 = resolveAxisVector(state.sliderTargets[0], fallback) ?? baseline[0];
  const t2 = resolveAxisVector(state.sliderTargets[1], fallback) ?? baseline[1];

  let u1 = baseline[0];
  if (state.sliderStrengths[0] > 0) {
    const mixed = mixAndNormalize(baseline[0], t1, state.sliderStrengths[0]);
    u1 = mixed ?? baseline[0];
  }

  let u2 = baseline[1];
  if (state.sliderStrengths[1] > 0) {
    const mixed2 = mixAndNormalize(baseline[1], t2, state.sliderStrengths[1]);
    u2 = mixed2 ?? baseline[1];
  }

  u2 = orthogonalize(u2, [u1]);
  if (!normalizeInPlace(u2)) {
    u2 = orthogonalize(baseline[1], [u1]);
    normalizeInPlace(u2);
  }

  let u3 = orthogonalize(baseline[2], [u1, u2]);
  if (!normalizeInPlace(u3)) {
    u3 = orthogonalize(randomUnitVector(state.dataset.dim, state.rng), [u1, u2]);
    normalizeInPlace(u3);
  }

  return orthonormalize3(u1, u2, u3, state.rng);
}

function computeViewBasis() {
  const baseline = computeBaselineBasis();
  if (!baseline) return null;
  if (!state.sliderEnabled) return baseline;
  return computeSliderBasis(baseline);
}

function getArrowDefForAxis(index) {
  if (state.sliderEnabled && index < 2) {
    return state.sliderTargets[index];
  }
  return state.axisDefs[index];
}

function updatePointsAndArrows() {
  const dataset = state.dataset;
  if (!dataset || !state.pointsGroup) return;

  const basis = computeViewBasis();
  if (!basis) return;

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

  const r = Math.sqrt(maxSq);
  state.baseRadius = Number.isFinite(r) && r > 0 ? r : 1.0;
  if (state.axesHelper) {
    const cameraDistance = state.camera ? state.camera.position.length() : 0;
    const axesScale = Math.max(1.2, state.baseRadius * 0.9, cameraDistance * 0.25);
    state.axesHelper.scale.setScalar(axesScale);
  }

  const arrowDefs = [getArrowDefForAxis(0), getArrowDefForAxis(1)];
  const arrows = [state.primaryArrow, state.secondaryArrow];

  arrowDefs.forEach((def, idx) => {
    const arrow = arrows[idx];
    const vec = resolveAxisVector(def, null);
    if (!vec) {
      arrow.visible = false;
      return;
    }
    const [x, y, z] = basisCoordsOfVector(vec, basis);
    const dir = new THREE.Vector3(x, y, z);
    if (dir.length() > 1e-6) dir.normalize();
    arrow.setDirection(dir);
    arrow.setLength(Math.max(0.6, state.baseRadius * 0.9), 0.14, 0.08);
    const color = def?.kind === "concept" ? state.classColors.get(def.label) : null;
    if (color) {
      arrow.setColor(color);
    } else {
      arrow.setColor(new THREE.Color(0xffffff));
    }
    arrow.visible = true;
  });
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

    buildUI(state.dataset);
    initThree();
    createPointsObject(state.dataset);

    updatePointsAndArrows();

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
