// api.js
// Shared API client — all fetch calls to the FastAPI backend go through here.
// Also provides startLogStream() / stopLogStream() for the live log panel.

const _h = window.location.hostname;
const API_BASE = (_h === 'localhost' || _h === '127.0.0.1' || _h === '::1' || _h === '')
  ? 'http://127.0.0.1:8000'   // local dev
  : 'https://socialmedia-content-analysis.onrender.com'; // production

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
 * @param {{title, post_hour, post_day, tag_count, user_caption, user_hashtags}} meta
 * @returns {Promise<{virality_score, label, probability, top_features, features, explanation}>}
 */
async function analyzeVirality(videoFile, meta) {
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

// ── Live log streaming ─────────────────────────────────────────────────────

let _logSource = null;

function startLogStream() {
  const panel    = document.getElementById('log-panel');
  const logLines = document.getElementById('log-lines');
  if (!panel) return;

  panel.hidden   = false;
  logLines.innerHTML = '';

  _logSource = new EventSource(`${API_BASE}/logs`);

  _logSource.onmessage = (e) => {
    const msg = e.data;
    if (msg === 'connected' || msg.startsWith(':')) return;

    const line = document.createElement('div');
    line.className = 'log-line';

    if (msg.includes('ERROR') || msg.includes('error'))       line.classList.add('error');
    else if (msg.includes('WARNING') || msg.includes('warn')) line.classList.add('warn');
    else line.classList.add('info');

    line.textContent = msg;
    logLines.appendChild(line);
    panel.scrollTop = panel.scrollHeight;
  };
}

function stopLogStream() {
  if (_logSource) {
    _logSource.close();
    _logSource = null;
  }
}
