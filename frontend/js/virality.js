// virality.js
// Handles the virality prediction page: file upload, metadata, API call, result rendering.

const form        = document.getElementById('virality-form');
const uploadZone  = document.getElementById('upload-zone');
const videoInput  = document.getElementById('video-input');
const browseBtn   = document.getElementById('browse-btn');
const filename    = document.getElementById('upload-filename');
const submitBtn   = document.getElementById('submit-btn');
const resultCard  = document.getElementById('result-card');
const errorBox    = document.getElementById('error-box');
const loading     = document.getElementById('loading');

// ── File selection ─────────────────────────────────────────────────────────

browseBtn.addEventListener('click', () => videoInput.click());

uploadZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn) videoInput.click();
});

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

  resultCard.hidden  = true;
  errorBox.hidden    = true;
  loading.hidden     = false;
  submitBtn.disabled = true;

  const meta = {
    title:         document.getElementById('title').value.trim(),
    post_hour:     parseInt(document.getElementById('post-hour').value),
    post_day:      parseInt(document.getElementById('post-day').value),
    tag_count:     parseInt(document.getElementById('tag-count').value),
    user_caption:  document.getElementById('caption').value.trim(),
    user_hashtags: document.getElementById('hashtags').value.trim(),
  };

  try {
    const result = await analyzeVirality(file, meta);
    renderResult(result);
  } catch (err) {
    showError(err.message);
  } finally {
    loading.hidden     = true;
    submitBtn.disabled = false;
  }
});

// ── Result rendering ───────────────────────────────────────────────────────

function renderResult(data) {
  // Score ring — circumference of r=50 circle is 2π×50 ≈ 314
  const CIRCUMFERENCE = 314;
  const offset = CIRCUMFERENCE - (data.virality_score / 100) * CIRCUMFERENCE;
  document.getElementById('ring-fill').style.strokeDashoffset = offset;

  // Animate the score number counting up
  animateNumber('score-number', 0, data.virality_score, 800);

  document.getElementById('result-verdict').textContent    = data.label;
  document.getElementById('result-confidence').textContent =
    `Probability: ${(data.probability * 100).toFixed(1)}%`;

  // LLM explanation
  const expEl = document.getElementById('explanation');
  if (data.explanation) {
    expEl.textContent = data.explanation;
    expEl.hidden = false;
  } else {
    expEl.hidden = true;
  }

  // Top feature importance bars
  renderFeatureBars(data.top_features);

  resultCard.hidden = false;
}

function renderFeatureBars(topFeatures) {
  const container = document.getElementById('feature-bars');
  container.innerHTML = '';

  // Normalise importance values to percentages relative to the max
  const values = Object.values(topFeatures);
  const max    = Math.max(...values);

  Object.entries(topFeatures).forEach(([key, val]) => {
    const pct   = max > 0 ? ((val / max) * 100).toFixed(1) : 0;
    const label = key.replace(/_/g, ' ');  // brisque_score → brisque score

    const row = document.createElement('div');
    row.className = 'feat-row';
    row.innerHTML = `
      <span class="feat-label">${label}</span>
      <div class="feat-track">
        <div class="feat-fill" style="width: 0%" data-pct="${pct}"></div>
      </div>
      <span class="feat-val">${val.toFixed(3)}</span>
    `;
    container.appendChild(row);
  });

  // Trigger CSS transitions after the DOM is painted
  setTimeout(() => {
    container.querySelectorAll('.feat-fill').forEach((el) => {
      el.style.width = `${el.dataset.pct}%`;
    });
  }, 50);
}

// Counts a number element from start to end over `duration` ms
function animateNumber(id, start, end, duration) {
  const el    = document.getElementById(id);
  const range = end - start;
  const step  = 16;  // ~60fps
  const steps = duration / step;
  let current = start;

  const timer = setInterval(() => {
    current += range / steps;
    el.textContent = Math.round(Math.min(current, end));
    if (current >= end) {
      el.textContent = end;
      clearInterval(timer);
    }
  }, step);
}

function showError(message) {
  errorBox.textContent = `Error: ${message}`;
  errorBox.hidden      = false;
}
