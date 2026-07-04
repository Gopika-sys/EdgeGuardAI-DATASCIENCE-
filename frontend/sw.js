// EdgeGuard AI — service worker
// Strategy: cache-first for the app shell, network-first for the backend API.
// Falls back to the offline page if both fail.
const CACHE = 'edgeguard-v2';
const SHELL = [
  './',
  './index.html',
  './app.js',
  './index.css',
  './digital_twin.html',
  './manifest.json',
  './vendor-three.module.min.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Always go to network for backend API and WebSocket
  if (url.pathname.startsWith('/api/') || url.host !== location.host) {
    return; // bypass SW
  }
  // Network-first for HTML so updates ship fast
  if (e.request.mode === 'navigate' || url.pathname.endsWith('.html')) {
    e.respondWith(
      fetch(e.request).then((r) => {
        const copy = r.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return r;
      }).catch(() => caches.match(e.request).then((r) => r || caches.match('./index.html')))
    );
    return;
  }
  // Cache-first for static assets
  e.respondWith(
    caches.match(e.request).then((cached) => cached || fetch(e.request).then((r) => {
      const copy = r.clone();
      if (r.status === 200) caches.open(CACHE).then((c) => c.put(e.request, copy));
      return r;
    }).catch(() => cached))
  );
});
