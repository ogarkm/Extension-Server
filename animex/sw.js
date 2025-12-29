/* sw.js
   PWA offline service worker
   - precaches core app shell + important pages
   - runtime caches images/videos/pdf with limits
   - navigation fallback to /offline.html
   - supports skipWaiting and downloadOffline messages
*/

const CACHE_VERSION = '2025-11-20-8'; // bump when you change assets
const PRECACHE = `precache-${CACHE_VERSION}`;
const RUNTIME = `runtime-${CACHE_VERSION}`;

// Offline fallback page
const OFFLINE_URL = '/offline.html';
const PRECACHE_URLS = [
  '/',                      // index
  '/index.html',
  '/Launch.html',
  '/about.html',
  '/in.html',
  '/intro.html',
  '/login.html',
  '/manga.html',
  '/library.html',
  '/offline.html',
  '/portal.html',
  '/reader.html',
  '/video_player.html',
  '/pdf.html',
  '/view.html',
  '/search.html',
  '/lists.html',
  '/settings.html',
  '/Resources/manifest.json',
  '/Resources/styles.css',
  '/Resources/manga.css',
  '/Resources/series.css',
  '/Resources/favicon.png',
  '/Resources/Images/Launch.png',
  '/Resources/Images/Launch_screen.png',
  '/Resources/Images/logo-256.png',
  '/Resources/Images/logo-512.png',
  '/sw.js'
];

// Runtime cache limits
const MAX_IMAGE_ITEMS = 80;
const MAX_VIDEO_ITEMS = 15;
const MAX_PDF_ITEMS = 20;

/* Utility: trim cache to max items (LRU-ish by deleting oldest) */
async function trimCache(cacheName, maxItems) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();
  if (keys.length <= maxItems) return;
  const removeCount = keys.length - maxItems;
  for (let i = 0; i < removeCount; i++) {
    await cache.delete(keys[i]);
  }
}

/* Install: precache app shell */
self.addEventListener('install', event => {
  self.skipWaiting(); // activate worker immediately (be careful with breaking changes)
  event.waitUntil(
    caches.open(PRECACHE)
      .then(cache => cache.addAll(PRECACHE_URLS))
      .catch(err => {
        console.error('Precache failed:', err);
      })
  );
});

/* Activate: clean old caches */
self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(
      names.filter(name => name !== PRECACHE && name !== RUNTIME)
           .map(name => caches.delete(name))
    );
    // Immediately take control of the pages
    await self.clients.claim();
  })());
});

/* Fetch handler */
self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);

  // Only handle same-origin requests (adjust if assets are on a CDN)
  const sameOrigin = url.origin === self.location.origin;

  // 1) Navigation requests -> network-first, fallback to cache -> offline page
  if (req.mode === 'navigate') {
    event.respondWith(networkFirstFallbackToCache(req));
    return;
  }

  // 2) API / JSON/XHR requests -> network-first (don't cache large dynamic responses)
  if (sameOrigin && url.pathname.startsWith('/api')) {
    event.respondWith(networkFirst(req));
    return;
  }

  // 3) Images -> cache-first with size limit
  if (req.destination === 'image' || /\.(?:png|jpg|jpeg|gif|webp|svg)$/i.test(url.pathname)) {
    event.respondWith(cacheFirstWithRuntime(req, 'images-cache', MAX_IMAGE_ITEMS));
    return;
  }

  // 4) Video files -> cache-first but avoid precaching; runtime cache with small limit
  if (/\.(?:mp4|webm|m4v|mov)$/i.test(url.pathname)) {
    event.respondWith(cacheFirstWithRuntime(req, 'videos-cache', MAX_VIDEO_ITEMS));
    return;
  }

  // 5) PDFs and other documents -> cache-first with limit
  if (/\.(?:pdf|epub|mobi)$/i.test(url.pathname) || req.destination === 'document') {
    event.respondWith(cacheFirstWithRuntime(req, 'docs-cache', MAX_PDF_ITEMS));
    return;
  }

  // 6) CSS/JS/font -> stale-while-revalidate strategy
  if (/\.(?:js|css|woff2?|ttf|otf)$/i.test(url.pathname) || req.destination === 'script' || req.destination === 'style' || req.destination === 'font') {
    event.respondWith(staleWhileRevalidate(req));
    return;
  }

  // Default: try cache first then network
  event.respondWith(
    caches.match(req).then(cached => cached || fetch(req).catch(() => {
      // if fetch failed and request is a navigation or HTML, show offline page
      if (req.headers.get('accept') && req.headers.get('accept').includes('text/html')) {
        return caches.match(OFFLINE_URL);
      }
      return new Response(null, { status: 503, statusText: 'Service Unavailable' });
    }))
  );
});

/* Strategies */

async function networkFirstFallbackToCache(request) {
  try {
    const response = await fetch(request);
    // Put navigation responses in runtime cache for offline use
    const cache = await caches.open(RUNTIME);
    cache.put(request, response.clone()).catch(() => {});
    return response;
  } catch (err) {
    // network failed -> try cache
    const cached = await caches.match(request);
    if (cached) return cached;
    // finally fallback to offline page
    return caches.match(OFFLINE_URL);
  }
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    // Update runtime cache
    const cache = await caches.open(RUNTIME);
    cache.put(request, response.clone()).catch(() => {});
    return response;
  } catch (err) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw err;
  }
}

async function cacheFirstWithRuntime(request, cacheName, maxItems = 50) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  try {
    const response = await fetch(request);
    // Only cache successful responses (200)
    if (response && response.status === 200) {
      cache.put(request, response.clone()).catch(() => {});
      // trim if necessary
      trimCache(cacheName, maxItems).catch(() => {});
    }
    return response;
  } catch (err) {
    // fallback to precache or offline
    const fallback = await caches.match(request);
    if (fallback) return fallback;
    if (request.headers.get('accept') && request.headers.get('accept').includes('text/html')) {
      return caches.match(OFFLINE_URL);
    }
    return new Response(null, { status: 503, statusText: 'Service Unavailable' });
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(RUNTIME);
  const cached = await cache.match(request);
  const networkFetch = fetch(request).then(response => {
    if (response && response.status === 200) {
      cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  }).catch(() => null);
  return cached || networkFetch;
}

/* Listen for messages from the page to trigger SW actions (skipWaiting, downloadOffline) */
self.addEventListener('message', event => {
  const data = event.data;
  if (!data) return;

  if (data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (data.type === 'DOWNLOAD_OFFLINE') {
    // Cache urls that are missing from PRECACHE
    downloadOffline();
  }
});

/* Pre-cache any resources that aren't yet cached */
async function downloadOffline() {
  const cache = await caches.open(PRECACHE);
  const cachedRequests = await cache.keys();
  const cachedUrls = cachedRequests.map(r => new URL(r.url).pathname);
  const toCache = PRECACHE_URLS.filter(url => !cachedUrls.includes(url));
  return cache.addAll(toCache);
}
