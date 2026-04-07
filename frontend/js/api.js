// api.js
// Shared API client — all fetch calls to the FastAPI backend go through here.

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://127.0.0.1:8000'   // local dev
  : 'https://your-app.onrender.com'; // production (update before deploy)

/**
 * POST /analyze/synthetic
 * @param {File} videoFile
 * @returns {Promise<{label, confidence, prob_ai, prob_deepfake}>}
 */
async function analyzeSynthetic(videoFile) {
  const form = new FormData();
  form.append('video', videoFile);

  const res = await fetch(`${API_BASE}/analyze/synthetic`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Synthetic detection failed.');
  }

  return res.json();
}

/**
 * POST /analyze/virality
 * @param {File} videoFile
 * @param {{title, post_hour, post_day, tag_count}} meta
 * @returns {Promise<{virality_score, label, probability, top_features, features}>}
 */
async function analyzeVirality(videoFile, meta) {
  const form = new FormData();
  form.append('video',      videoFile);
  form.append('title',      meta.title      ?? '');
  form.append('post_hour',  meta.post_hour  ?? 12);
  form.append('post_day',   meta.post_day   ?? 1);
  form.append('tag_count',  meta.tag_count  ?? 5);

  const res = await fetch(`${API_BASE}/analyze/virality`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Virality prediction failed.');
  }

  return res.json();
}
