// app.js
// Fades in the hero content once the background video starts playing.

const video = document.getElementById('bg-video');
const hero  = document.querySelector('.hero');

if (video) {
  // Once video is actually playing, fade in the hero content
  video.addEventListener('playing', () => {
    if (hero) hero.classList.add('hero-visible');
  });

  // Fallback: show content after 800ms regardless (e.g. autoplay blocked)
  setTimeout(() => {
    if (hero) hero.classList.add('hero-visible');
  }, 800);
}
