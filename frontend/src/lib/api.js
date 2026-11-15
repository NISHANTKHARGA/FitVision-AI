const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8017';

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function parseError(res) {
  let detail = `Request failed (${res.status})`;
  try {
    const data = await res.json();
    detail = data.detail || data.message || JSON.stringify(data);
  } catch { /* ignore */ }
  throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), res.status);
}

export async function apiHealth() {
  const res = await fetch(`${API}/api/health`);
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function analyzePerson(file, heightCm) {
  const fd = new FormData();
  fd.append('file', file);
  if (heightCm) fd.append('height_cm', String(heightCm));
  const res = await fetch(`${API}/api/analyze/person`, { method: 'POST', body: fd });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function analyzeGarment(file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(`${API}/api/analyze/garment`, { method: 'POST', body: fd });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function generateTryOn({ person, garment, category, fit, steps, seed, garmentPhotoType, saveHistory, body, garmentM, guidance, segmentationFree }) {
  const fd = new FormData();
  fd.append('person', person);
  fd.append('garment', garment);
  fd.append('category', category);
  fd.append('fit', fit);
  fd.append('steps', String(steps));
  fd.append('seed', String(seed));
  fd.append('garment_photo_type', garmentPhotoType);
  fd.append('guidance_scale', String(guidance ?? 1.5));
  fd.append('segmentation_free', segmentationFree === false ? 'false' : 'true');
  fd.append('save_history', saveHistory ? 'true' : 'false');
  fd.append('body_measurements', JSON.stringify(body || {}));
  fd.append('garment_measurements', JSON.stringify(garmentM || {}));
  const res = await fetch(`${API}/api/tryon/generate`, { method: 'POST', body: fd });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`${API}/api/jobs/${jobId}`);
  if (!res.ok) return parseError(res);
  return res.json();
}

export function resultImageUrl(jobId, variant) {
  return `${API}/api/results/${jobId}/image/${variant}`;
}

export async function getProfile() {
  const res = await fetch(`${API}/api/profile`);
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function putProfile(profile) {
  const res = await fetch(`${API}/api/profile`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function getHistory() {
  const res = await fetch(`${API}/api/history`);
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function deleteHistory(jobId) {
  const res = await fetch(`${API}/api/history/${jobId}`, { method: 'DELETE' });
  if (!res.ok) return parseError(res);
  return res.json();
}

export async function saveToHistory(jobId) {
  const res = await fetch(`${API}/api/history/save/${jobId}`, { method: 'POST' });
  if (!res.ok) return parseError(res);
  return res.json();
}

export { API };
