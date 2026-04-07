// app.js
// Handles the background video loop — clamps playback to the first 20 seconds only.

const video = document.getElementById('bg-video');

if (video) {
  // Once enough data is loaded, start from 0 and enforce the 20s cap
  video.addEventListener('loadedmetadata', () => {
    video.currentTime = 0;
  });

  // Every time the video ticks past 20 seconds, snap back to the start
  video.addEventListener('timeupdate', () => {
    if (video.currentTime >= 20) {
      video.currentTime = 0;
    }
  });
}
