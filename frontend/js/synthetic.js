// synthetic.js
// Handles the synthetic detection page: file upload, API call, result rendering.

const form        = document.getElementById('synthetic-form');
const uploadZone  = document.getElementById('upload-zone');
const videoInput  = document.getElementById('video-input');
const browseBtn   = document.getElementById('browse-btn');
const filename    = document.getElementById('upload-filename');
const submitBtn   = document.getElementById('submit-btn');
const resultCard  = document.getElementById('result-card');
const errorBox    = document.getElementById('error-box');
const loading     = document.getElementById('loading');

// ── File selection ─────────────────────────────────────────────────────────

// Clicking the "browse" text opens the file picker
browseBtn.addEventListener('click', () => videoInput.click());

// Clicking anywhere on the zone also opens the picker
uploadZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn) videoInput.click();
});

// Drag and drop support
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

videoInput.addEventListener('change', () => {
  if (videoInput.files[0]) setFile(videoInput.files[0]);
});

function setFile(file) {
  filename.textContent = file.name;
  uploadZone.classList.add('has-file');
  submitBtn.disabled = false;
}

// ── Form submission ────────────────────────────────────────────────────────

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = videoInput.files[0];
  if (!file) return;

  // Reset UI state
  resultCard.hidden = true;
  errorBox.hidden   = true;
  loading.hidden    = false;
  submitBtn.disabled = true;

  try {
    const result = await analyzeSynthetic(file);
    renderResult(result);
  } catch (err) {
    showError(err.message);
  } finally {
    loading.hidden    = true;
    submitBtn.disabled = false;
  }
});

// ── Result rendering ───────────────────────────────────────────────────────

function renderResult(data) {
  document.getElementById('result-verdict').textContent    = data.label;
  document.getElementById('result-confidence').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const probAI   = (data.prob_ai       * 100).toFixed(1);
  const probDF   = (data.prob_deepfake * 100).toFixed(1);
  const probReal = Math.max(0, (100 - parseFloat(probAI) - parseFloat(probDF))).toFixed(1);

  setBar('bar-ai',   'pct-ai',   probAI);
  setBar('bar-df',   'pct-df',   probDF);
  setBar('bar-real', 'pct-real', probReal);

  // LLM explanation
  const expEl = document.getElementById('explanation');
  if (data.explanation) {
    expEl.textContent = data.explanation;
    expEl.hidden = false;
  } else {
    expEl.hidden = true;
  }

  resultCard.hidden = false;
}

function setBar(barId, pctId, value) {
  document.getElementById(pctId).textContent    = `${value}%`;
  // Small delay so CSS transition fires after element is visible
  setTimeout(() => {
    document.getElementById(barId).style.width = `${value}%`;
  }, 50);
}

function showError(message) {
  errorBox.textContent = `Error: ${message}`;
  errorBox.hidden      = false;
}
