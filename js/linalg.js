// Small linear algebra helpers (no big dependencies).
// Everything is tuned for d=64 and k=3.

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function normalize(vec, eps = 1e-12) {
  const n = norm(vec);
  if (n < eps) return null;
  const out = new Float32Array(vec.length);
  const inv = 1 / n;
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv;
  return out;
}

export function normalizeInPlace(vec, eps = 1e-12) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const n = Math.sqrt(s);
  if (n < eps) return false;
  const inv = 1 / n;
  for (let i = 0; i < vec.length; i++) vec[i] *= inv;
  return true;
}

export function orthogonalize(vec, basisList) {
  // vec is Float32Array (or array-like). Returns Float32Array.
  const out = new Float32Array(vec.length);
  out.set(vec);
  for (const b of basisList) {
    const c = dot(out, b);
    for (let i = 0; i < out.length; i++) out[i] -= c * b[i];
  }
  return out;
}

export function mixAndNormalize(a, b, t, eps = 1e-12) {
  // returns normalize((1-t)*a + t*b)
  const out = new Float32Array(a.length);
  const ta = 1 - t;
  for (let i = 0; i < a.length; i++) out[i] = ta * a[i] + t * b[i];
  if (!normalizeInPlace(out, eps)) return null;
  return out;
}

export function xorshift32(seed = 123456789) {
  let x = seed >>> 0;
  return function rand() {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x >>>= 0;
    x ^= x << 5; x >>>= 0;
    return (x >>> 0) / 4294967296;
  };
}

export function randomUnitVector(d, rng) {
  // Box-Muller for approximate normal, then normalize.
  const v = new Float32Array(d);
  for (let i = 0; i < d; i += 2) {
    const u1 = Math.max(1e-12, rng());
    const u2 = Math.max(1e-12, rng());
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    v[i] = r * Math.cos(theta);
    if (i + 1 < d) v[i + 1] = r * Math.sin(theta);
  }
  normalizeInPlace(v);
  return v;
}

export function orthonormalize3(v1, v2, v3, rng = xorshift32(42), eps = 1e-8) {
  // Gram-Schmidt for 3 vectors (Float32Array), with fallbacks.
  const out1 = new Float32Array(v1.length); out1.set(v1);
  if (!normalizeInPlace(out1, eps)) out1.set(randomUnitVector(v1.length, rng));

  let out2 = orthogonalize(v2, [out1]);
  if (!normalizeInPlace(out2, eps)) out2 = randomUnitVector(v1.length, rng);
  out2 = orthogonalize(out2, [out1]);
  if (!normalizeInPlace(out2, eps)) out2 = randomUnitVector(v1.length, rng);

  let out3 = orthogonalize(v3, [out1, out2]);
  if (!normalizeInPlace(out3, eps)) out3 = randomUnitVector(v1.length, rng);
  out3 = orthogonalize(out3, [out1, out2]);
  if (!normalizeInPlace(out3, eps)) out3 = randomUnitVector(v1.length, rng);

  return [out1, out2, out3];
}

export function computeMean(X, N, d) {
  const mean = new Float32Array(d);
  for (let i = 0; i < N; i++) {
    const off = i * d;
    for (let j = 0; j < d; j++) mean[j] += X[off + j];
  }
  const invN = 1 / N;
  for (let j = 0; j < d; j++) mean[j] *= invN;
  return mean;
}

export function centerData(X, mean, N, d) {
  const Xc = new Float32Array(N * d);
  for (let i = 0; i < N; i++) {
    const off = i * d;
    for (let j = 0; j < d; j++) Xc[off + j] = X[off + j] - mean[j];
  }
  return Xc;
}

export function computeCovariance(Xc, N, d) {
  // C = (1/N) sum x x^T
  const C = new Float64Array(d * d);
  for (let i = 0; i < N; i++) {
    const off = i * d;
    for (let a = 0; a < d; a++) {
      const xa = Xc[off + a];
      const row = a * d;
      for (let b = 0; b < d; b++) {
        C[row + b] += xa * Xc[off + b];
      }
    }
  }
  const invN = 1 / N;
  for (let i = 0; i < C.length; i++) C[i] *= invN;
  return C;
}

function matVecMul(C, v, d, out) {
  out.fill(0);
  for (let i = 0; i < d; i++) {
    const row = i * d;
    let s = 0;
    for (let j = 0; j < d; j++) s += C[row + j] * v[j];
    out[i] = s;
  }
  return out;
}

export function powerIterationTopK(Cin, d, k = 3, iters = 50, seed = 1337) {
  // Returns k eigenvectors (Float32Array), approximate, using deflation.
  const C = new Float64Array(Cin); // working copy
  const rng = xorshift32(seed);

  const vecs = [];
  const tmp = new Float64Array(d);
  const tmp2 = new Float64Array(d);

  for (let comp = 0; comp < k; comp++) {
    // init random vector
    let v = new Float64Array(d);
    for (let i = 0; i < d; i++) v[i] = rng() * 2 - 1;
    // normalize
    let n = 0;
    for (let i = 0; i < d; i++) n += v[i] * v[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < d; i++) v[i] /= n;

    for (let it = 0; it < iters; it++) {
      matVecMul(C, v, d, tmp);
      // normalize tmp -> v
      let nn = 0;
      for (let i = 0; i < d; i++) nn += tmp[i] * tmp[i];
      nn = Math.sqrt(nn) || 1;
      for (let i = 0; i < d; i++) v[i] = tmp[i] / nn;
    }

    // eigenvalue lambda = v^T C v
    matVecMul(C, v, d, tmp2);
    let lambda = 0;
    for (let i = 0; i < d; i++) lambda += v[i] * tmp2[i];

    // store as float32
    const vf = new Float32Array(d);
    for (let i = 0; i < d; i++) vf[i] = v[i];
    // normalize defensively
    normalizeInPlace(vf);
    vecs.push(vf);

    // Deflate: C = C - lambda * v v^T
    for (let i = 0; i < d; i++) {
      const row = i * d;
      const vi = v[i];
      for (let j = 0; j < d; j++) C[row + j] -= lambda * vi * v[j];
    }
  }

  return vecs;
}

export function computePcaBasis(Xc, N, d, k = 3) {
  const C = computeCovariance(Xc, N, d);
  const vecs = powerIterationTopK(C, d, k, 60);
  // Orthonormalize just in case numerical drift accumulated.
  return orthonormalize3(vecs[0], vecs[1], vecs[2]);
}

export function projectPointsTo3(Xc, N, d, basis3, outPositions) {
  const [u1, u2, u3] = basis3;
  for (let i = 0; i < N; i++) {
    const off = i * d;
    let x1 = 0, x2 = 0, x3 = 0;
    for (let j = 0; j < d; j++) {
      const v = Xc[off + j];
      x1 += v * u1[j];
      x2 += v * u2[j];
      x3 += v * u3[j];
    }
    const p = i * 3;
    outPositions[p] = x1;
    outPositions[p + 1] = x2;
    outPositions[p + 2] = x3;
  }
}

export function basisCoordsOfVector(vec, basis3) {
  // vec in R^d, basis is orthonormal-ish. Returns [x,y,z].
  const [u1, u2, u3] = basis3;
  return [
    dot(vec, u1),
    dot(vec, u2),
    dot(vec, u3),
  ];
}
