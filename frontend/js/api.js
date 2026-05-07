// api.js
// Shared API client — all fetch calls to the FastAPI backend go through here.

const _h = window.location.hostname;
const API_BASE = (_h === 'localhost' || _h === '127.0.0.1' || _h === '::1' || _h === '')
  ? 'http://127.0.0.1:8000'   // local dev
  : 'https://parina13-synthsenses-api.hf.space'; // production

/**
 * Wakes the Render server if it has spun down (free tier cold start ~30-90s).
 * Updates the loading message while waiting.
 */
async function _wakeServer() {
  const TIMEOUT_MS = 90_000;
  const POLL_MS    = 3_000;
  const deadline   = Date.now() + TIMEOUT_MS;

  // Try once immediately — if server is up this returns fast
  try {
    const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (res.ok) return;
  } catch (_) { /* server is sleeping, start polling */ }

  // Show "waking up" message if server didn't respond immediately
  const loadingEl = document.getElementById('loading');
  const loadingP  = loadingEl ? loadingEl.querySelector('p') : null;
  if (loadingP) loadingP.textContent = 'Waking up server (first request may take ~30s)…';

  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, POLL_MS));
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
      if (res.ok) {
        if (loadingP) loadingP.textContent = 'Analysing video…';
        return;
      }
    } catch (_) { /* still booting */ }
  }

  throw new Error('Server took too long to wake up. Please try again in a moment.');
}

/**
 * POST /analyze/synthetic
 * @param {File} videoFile
 * @returns {Promise<{label, confidence, prob_ai, prob_deepfake}>}
 */
async function analyzeSynthetic(videoFile) {
  await _wakeServer();

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
 * @param {{title, post_hour, post_day, tag_count, user_caption, user_hashtags}} meta
 * @returns {Promise<{virality_score, label, probability, top_features, features, explanation}>}
 */
async function analyzeVirality(videoFile, meta) {
  await _wakeServer();

  const form = new FormData();
  form.append('video',         videoFile);
  form.append('title',         meta.title         ?? '');
  form.append('post_hour',     meta.post_hour      ?? 12);
  form.append('post_day',      meta.post_day       ?? 1);
  form.append('tag_count',     meta.tag_count      ?? 5);
  form.append('user_caption',  meta.user_caption   ?? '');
  form.append('user_hashtags', meta.user_hashtags  ?? '');

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
