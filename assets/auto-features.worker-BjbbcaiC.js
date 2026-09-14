/* global importScripts */
// This file is copied as a raw classic worker (?url). OpenCV's unmodified UMD
// artifact requires classic script semantics; all photo processing stays here.
let working = false;

function progress(message, completed, total) { self.postMessage({ type: 'progress', message, completed, total }); }

async function loadEngine(url) {
  const address = new URL(url, self.location.href);
  if (address.origin !== self.location.origin) throw new Error('Vaizdų variklis turi būti įkeliamas iš šios svetainės.');
  importScripts(address.href);
  await new Promise((resolve, reject) => {
    const runtime = self.cv;
    if (!runtime || typeof runtime.then !== 'function') { reject(new Error('Nepavyko inicializuoti vaizdų atpažinimo variklio.')); return; }
    runtime.onAbort = () => reject(new Error('Vaizdų atpažinimo variklis sustojo.'));
    runtime.then(() => resolve());
  });
  const cv = self.cv;
  if (!cv?.Mat || !cv.ORB || !cv.BFMatcher || !cv.findHomography) throw new Error('Šiame variklyje trūksta nuotraukų sujungimo funkcijų.');
  // OpenCV is a self-resolving thenable, so returning it directly from an async
  // function would keep Promise assimilation running forever.
  return { cv };
}

async function extract(cv, frame, detector) {
  let bitmap;
  let rgba;
  const gray = new cv.Mat();
  const mask = new cv.Mat();
  const keypoints = new cv.KeyPointVector();
  const descriptors = new cv.Mat();
  let retained = false;
  try {
    bitmap = await createImageBitmap(frame.image);
    if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > 40_000_000) throw new Error('Nuotraukos raiška netinkama.');
    const scale = Math.min(1, 960 / Math.max(bitmap.width, bitmap.height));
    const width = Math.max(2, Math.round(bitmap.width * scale));
    const height = Math.max(2, Math.round(bitmap.height * scale));
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Nepavyko perskaityti nuotraukos taškų.');
    context.drawImage(bitmap, 0, 0, width, height);
    rgba = cv.matFromImageData(context.getImageData(0, 0, width, height));
    cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
    detector.detectAndCompute(gray, mask, keypoints, descriptors);
    const points = [];
    for (let i = 0; i < keypoints.size(); i++) {
      const point = keypoints.get(i).pt;
      points.push({ x: point.x, y: point.y, u: point.x / (width - 1), v: point.y / (height - 1) });
    }
    retained = true;
    return { id: frame.id, width, height, points, descriptors };
  } finally {
    bitmap?.close(); rgba?.delete(); gray.delete(); mask.delete(); keypoints.delete();
    if (!retained) descriptors.delete();
  }
}

function nearest(cv, matcher, source, target) {
  const rows = new cv.DMatchVectorVector();
  const good = new Map();
  try {
    if (source.rows < 2 || target.rows < 2) return good;
    matcher.knnMatch(source, target, rows, 2);
    for (let i = 0; i < rows.size(); i++) {
      const row = rows.get(i);
      try {
        if (row.size() < 2) continue;
        const first = row.get(0), second = row.get(1);
        // Both a distinctive nearest neighbour and an absolute distance limit
        // are necessary in rooms with repeated handles, corners and patterns.
        if (first.distance <= 64 && first.distance < second.distance * .75) good.set(first.queryIdx, first.trainIdx);
      } finally { row.delete(); }
    }
    return good;
  } finally { rows.delete(); }
}

function matchPair(cv, matcher, source, target) {
  const forward = nearest(cv, matcher, source.descriptors, target.descriptors);
  const backward = nearest(cv, matcher, target.descriptors, source.descriptors);
  const candidates = [], sourcePixels = [], targetPixels = [];
  for (const [a, b] of forward) {
    if (backward.get(b) !== a) continue;
    const p = source.points[a], q = target.points[b];
    if (!p || !q) continue;
    candidates.push({ source: { u: p.u, v: p.v }, target: { u: q.u, v: q.v } });
    sourcePixels.push(p.x, p.y); targetPixels.push(q.x, q.y);
  }
  const result = { sourceId: source.id, targetId: target.id, candidates, inliers: [] };
  if (candidates.length < 20) return result;
  const sourceMat = cv.matFromArray(candidates.length, 1, cv.CV_32FC2, sourcePixels);
  const targetMat = cv.matFromArray(candidates.length, 1, cv.CV_32FC2, targetPixels);
  const mask = new cv.Mat();
  let homography;
  try {
    cv.setRNGSeed(7391);
    homography = cv.findHomography(sourceMat, targetMat, cv.RANSAC, 3, mask, 2000, .995);
    if (!homography.empty() && [...homography.data64F].every(Number.isFinite)) {
      result.inliers = candidates.filter((_, index) => mask.data[index] === 1);
    }
    return result;
  } finally { sourceMat.delete(); targetMat.delete(); mask.delete(); homography?.delete(); }
}

self.onmessage = async event => {
  if (working) return;
  working = true;
  const prepared = [];
  let detector;
  let matcher;
  try {
    const { frames, opencvUrl } = event.data;
    if (!Array.isArray(frames) || frames.length < 2 || frames.length > 12
      || frames.some(frame => !frame || typeof frame.id !== 'string' || !(frame.image instanceof Blob) || frame.image.size > 15 * 1024 * 1024)) {
      throw new Error('Automatiniam sujungimui reikia nuo 2 iki 12 nuotraukų.');
    }
    if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas !== 'function') throw new Error('Ši naršyklė nepalaiko nuotraukų analizės fone. Atnaujinkite „Safari“, „Chrome“ arba „Edge“.');
    progress('Įkeliamas vietinis vaizdų atpažinimo variklis (apie 11 MB)…', 0, 0);
    const { cv } = await loadEngine(opencvUrl);
    detector = new cv.ORB(); detector.setMaxFeatures(1800); detector.setFastThreshold(12);
    matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const statuses = [];
    for (let index = 0; index < frames.length; index++) {
      const frame = frames[index];
      progress(`Ieškoma detalių: ${frame.title || `kampas ${index + 1}`}…`, index, frames.length);
      try {
        const feature = await extract(cv, frame, detector);
        prepared.push(feature);
        statuses.push({ frameId: frame.id, features: feature.points.length });
      } catch {
        statuses.push({ frameId: frame.id, features: 0, error: 'Nepavyko perskaityti šios nuotraukos bendrų detalių.' });
      }
    }
    const matches = [];
    const total = prepared.length * (prepared.length - 1) / 2;
    let completed = 0;
    for (let i = 0; i < prepared.length; i++) for (let j = i + 1; j < prepared.length; j++) {
      progress(`Lyginami kampai ${i + 1} ir ${j + 1} (${completed + 1}/${total})…`, completed, total);
      matches.push(matchPair(cv, matcher, prepared[i], prepared[j]));
      completed++;
    }
    self.postMessage({ type: 'result', matches, frames: statuses });
  } catch (cause) {
    self.postMessage({ type: 'error', message: cause instanceof Error ? cause.message : 'Nuotraukų atpažinimo variklis nebaigė darbo. Bandykite dar kartą.' });
  } finally {
    prepared.forEach(feature => feature.descriptors.delete()); detector?.delete(); matcher?.delete();
  }
};
