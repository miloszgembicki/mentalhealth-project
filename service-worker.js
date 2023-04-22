var cacheName = "classes-v1";
var cacheFiles = [
    "index.html",
    // "subjects.j",
    "public/character-image.jpg",
    "public/landscape.jpg",
    "public/mental.jpg"
];
self.addEventListener("install", function (e) {
    console.log("[Service Worker] Install");
    e.waitUntil(
        caches.open(cacheName).then(function (cache) {
            console.log("[Service Worker] Caching files");
            return cache.addAll(cacheFiles);
        })
    );
});
self.addEventListener("fetch", function (e) {
    e.respondWith(
        caches.match(e.request).then(function (cachedFile) {
            //return the file if it is not in the cache
            if (cachedFile) {
                console.log(
                    "[Service Worker] Resource fetched from the cache for: " +
                    e.request.url
                );
                return cachedFile;
            } else {
                return fetch(e.request).then(function (response) {
                    return caches.open(cacheName).then(function (cache) {
                        //add the new file to the cache
                        cache.put(e.request, response.clone());
                        console.log(
                            "[Service Worker] Resource fetched and saved in the cache for: " +
                            e.request.url
                        );
                        return response;
                    });
                });
            }
        })
    );
});