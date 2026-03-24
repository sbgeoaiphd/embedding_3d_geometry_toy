import { computeMean, centerData, computePcaBasis, normalize, normalizeInPlace } from "./linalg.js";

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

export async function loadDataset(config) {
  const { name, embeddingsUrl, classWsUrl } = config;

  const embRes = await fetch(embeddingsUrl);
  assert(embRes.ok, `Failed to fetch embeddings: ${embeddingsUrl} (${embRes.status})`);
  const rawPoints = await embRes.json();

  assert(Array.isArray(rawPoints) && rawPoints.length > 0, "Embeddings JSON must be a non-empty array.");

  const N = rawPoints.length;
  const dim = rawPoints[0].embedding?.length;
  assert(Number.isInteger(dim), "Each item must have an 'embedding' array.");
  const expectedDim = config.dim ?? 64;
  assert(dim === expectedDim, `Expected embedding dim=${expectedDim}, got ${dim}.`);

  const ids = new Array(N);
  const labels = new Array(N);
  const X = new Float32Array(N * dim);

  for (let i = 0; i < N; i++) {
    const p = rawPoints[i];
    const emb = p.embedding;
    assert(Array.isArray(emb) && emb.length === dim, `Point ${i} has missing or wrong-length embedding.`);
    const label = p.class ?? p.label ?? p.y;
    assert(typeof label === "string" && label.length > 0, `Point ${i} is missing a string 'class' label.`);
    labels[i] = label;
    ids[i] = (p.id != null) ? String(p.id) : String(i);
    const off = i * dim;
    for (let j = 0; j < dim; j++) X[off + j] = emb[j];
  }

  const classLabels = Array.from(new Set(labels)).sort();
  const classToIndices = new Map();
  for (const c of classLabels) classToIndices.set(c, []);
  for (let i = 0; i < N; i++) classToIndices.get(labels[i]).push(i);

  const mean = computeMean(X, N, dim);
  const Xc = centerData(X, mean, N, dim);

  // Load class Ws (linear directions), normalized.
  const wsRes = await fetch(classWsUrl);
  assert(wsRes.ok, `Failed to fetch class Ws: ${classWsUrl} (${wsRes.status})`);
  const rawWs = await wsRes.json();
  assert(rawWs && typeof rawWs === "object", "class_ws.json must be an object mapping class label -> vector array.");

  const classWs = new Map();
  const missing = [];

  // Load ALL concept vectors from the JSON (not just those matching point classes).
  for (const c of Object.keys(rawWs)) {
    const arr = rawWs[c];
    assert(Array.isArray(arr) && arr.length === dim, `W for class '${c}' must be length ${dim}.`);
    const w = new Float32Array(dim);
    for (let j = 0; j < dim; j++) w[j] = arr[j];
    if (!normalizeInPlace(w)) {
      throw new Error(`W for class '${c}' has near-zero norm.`);
    }
    classWs.set(c, w);
  }

  // Track which point classes are missing concept vectors.
  for (const c of classLabels) {
    if (!classWs.has(c)) missing.push(c);
  }
  if (missing.length) {
    console.warn("Missing W vectors for classes:", missing);
  }

  // All concept vector labels (superset of classLabels).
  const conceptLabels = Array.from(classWs.keys()).sort();

  // Base 3D view basis: PCA on centered data.
  const baseBasis = computePcaBasis(Xc, N, dim, 3);

  return {
    name,
    N,
    dim,
    ids,
    labels,
    classLabels,
    classToIndices,
    X,    // original
    mean, // mean of X
    Xc,   // centered
    classWs,
    conceptLabels,
    baseBasis,
    missingWs: missing,
  };
}
