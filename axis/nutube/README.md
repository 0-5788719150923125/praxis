# nuTube

A local-first way to explore YouTube, built with [Godot](https://godotengine.org/) 4.6 for mobile. Everyone hates the algorithm - so nuTube tries to build a different one that runs entirely on your device.

## The idea

The recommendation algorithm lives on `localhost`, inside the app. There is no remote ranking service: nuTube keeps a small index on the device and ranks it with simple, inspectable local algorithms - keyword overlap today, embedding similarity searches and watch-history feedback later. You can read exactly why something was surfaced.

If it works for YouTube, the same shape extends to other video platforms: a central, on-device indexer where every source plugs in behind one generic item type, and a single local ranking brain decides what to show across all of them.

## Layout

- `scenes/main.tscn` - the main screen: a search bar over a feed.
- `scripts/main.gd` - wires the UI to the index, renders thumbnail cards, and shows a detail overlay (tap a card) with an explicit "Watch on YouTube" handoff.
- `scripts/youtube.gd` - `YouTubeSource`: resolves a watch URL into a feed item using YouTube's key-free endpoints (oEmbed for title/author, the thumbnail CDN for the image). No API key, no OAuth. Thumbnails are cached under `user://cache/`.
- `scripts/local_index.gd` - autoloaded as `LocalIndex`; the on-device store and recommender. Sources populate it via `upsert`; the index persists to `user://index.json`. This is the part to grow: a real source crawler and a better scorer.

## Caching

The first launch fetches metadata and thumbnails over the network; everything is then written to `user://` (the index as JSON, thumbnails as JPGs). Later launches render straight from disk and fetch nothing - `LocalIndex.needs_fetch()` gates the network so only genuinely-new videos are requested.

Network fetches use `HTTPClient` over an explicitly-resolved IPv4 address rather than the `HTTPRequest` node. On a host with no working IPv6 route, `HTTPRequest` stalls ~12s per connection waiting for the IPv6 connect to time out before falling back (it has no Happy-Eyeballs racing); resolving IPv4 ourselves - with a TLS common-name override so the certificate still validates against the hostname - drops a cold fetch from ~30s to under a second. See the header comment in `youtube.gd`.

## Status

Early scaffold. On launch it resolves three hardcoded videos (see `SEED_VIDEOS` in `main.gd`) and renders their thumbnails and titles; the search box re-ranks them locally. There is no crawler, similarity model, or in-app playback yet - tapping a card opens a detail view whose "Watch" button hands off to the system browser / YouTube app. Open `project.godot` in the Godot editor to poke at it.
