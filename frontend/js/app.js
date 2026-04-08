// app.js
// Handles the background video loop — clamps playback to the first 19 seconds only.

const video = document.getElementById('bg-video');
const hero  = document.querySelector('.hero');

if (video) {
  // Skip past the first frame (often black) — start at second 1
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = 1;
  });

  // Once video is actually playing, fade in the hero content
  video.addEventListener('playing', () => {
    if (hero) hero.classList.add('hero-visible');
  });

  // Fallback: if video doesn't fire 'playing' within 800ms, show content anyway
  setTimeout(() => {
    if (hero) hero.classList.add('hero-visible');
  }, 800);

  // Every time the video ticks past 19 seconds, snap back to second 1
  video.addEventListener('timeupdate', () => {
    if (video.currentTime >= 19) {
      video.currentTime = 1;
    }
  });
}
