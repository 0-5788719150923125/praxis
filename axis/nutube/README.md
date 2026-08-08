# nuTube

A local-first way to explore video platforms, starting with YouTube. Nothing is
hosted here: nuTube is a pure client that pulls from sources it does not own.

## The idea

The recommendation algorithm lives on the device. There is no remote ranking
service - nuTube keeps a small index locally and ranks it with simple,
inspectable rules, so every card can tell you why it surfaced. Keyword overlap
today; watch-history feedback and better scoring later.

If it works for YouTube, the shape extends: every source plugs in behind one
generic `FeedItem`, and a single local ranker decides what to show across all of
them.

## Why Kotlin, and what happened to the Godot version

The prototype lived in Godot and is kept at [`godot/`](godot/) for reference. It
worked, and the two things people worry about with a game engine - idle battery
drain and UI lag - both turned out to be solvable. Godot can genuinely stop
redrawing when idle (`OS.low_processor_usage_mode` plus
`RenderingServer.viewport_set_update_mode(..., VIEWPORT_UPDATE_DISABLED)` on the
root viewport takes GPU draw calls to zero, measured).

What killed it was everything an aggregator needs that lives on the JVM:

- **Playback.** Godot's `VideoStream` has exactly one subclass,
  `VideoStreamTheora`. YouTube serves DASH with separate VP9/AV1/H.264 video and
  audio. ExoPlayer handles it, hardware-decoded, on a surface Android can hand
  straight to the display controller.
- **Extraction.** oEmbed gives a title and an author and nothing else - no
  search, no channel listings. Real extraction means running YouTube's player
  JavaScript to decipher stream URLs, and GDScript has no JS engine.
  NewPipeExtractor does it with Rhino.
- **Background indexing.** Android has had no background execution without
  `WorkManager` or a foreground service since API 26, and neither is reachable
  from GDScript. A crawler that runs while the app is closed has to be Kotlin.

Each of those alone forced JVM code into the project. Together they meant Godot
would have been a UI shell over a Kotlin app.

## Layout

Kotlin Multiplatform, split so that a second platform is an added target rather
than a rewrite.

- `core/` - the multiplatform module, `commonMain` only. `FeedItem`, `LocalIndex`
  (the on-device store and ranker), `VideoSource` (the contract every platform
  implements), `SourceRegistry` (which platforms are plugged in), and
  `PlaybackStreams`. Nothing here touches an Android or JVM API; it uses okio
  rather than `java.io` for exactly that reason. Adding `iosArm64()` or `jvm()`
  to `core/build.gradle.kts` is the whole change needed to run this logic
  elsewhere.
- `app/` - the Android application. The Compose UI, the ExoPlayer overlay, and
  `sources/youtube/`, which implements `VideoSource` over NewPipeExtractor.
- `godot/` - the original Godot 4.6 prototype, frozen.

The one thing that does not generalise: **NewPipeExtractor is a Java library**,
so the YouTube source can only ever live in a JVM target. An iOS build would
share the index, the ranker and the registry, but would need its own extraction
path and its own player. That is a real limit on how much KMP buys here, and it
is worth knowing before betting on it.

### Adding a platform

Implement `VideoSource` and register it in `NuTubeApp.onCreate`. Everything above
the interface - the index, the ranking, the feed, the player - is written against
`FeedItem` and never learns which platform an item came from. `SourceRegistry`
fans a search out across every registered source concurrently and drops the ones
that fail, so a rate-limited platform degrades instead of breaking the feed.

## Building

Needs JDK 17 and the Android SDK. `local.properties` points at the SDK and is
git-ignored, so it needs writing once per machine:

```sh
echo "sdk.dir=$HOME/Android/Sdk" > local.properties
./run.sh --apk        # build only; drops nutube.apk in this directory
./run.sh              # boot the emulator, install, launch, tail logs
./run.sh --help       # the rest
```

`nutube.apk` sits at the project root rather than under `app/build/` because
Syncthing excludes build directories; it is git-ignored but explicitly allowed
through in `axis/.stignore`.

No Android Studio required - the Gradle wrapper does everything.

## Status

Working: search fans out through the registry and folds results into the local
index; the index ranks and explains itself on every card; tapping a card plays
in-app in HD. Sharing or opening a YouTube link from any other app indexes it.

HD works by taking YouTube's separate video-only and audio-only tracks and
merging them with ExoPlayer's `MergingMediaSource` - the muxed stream it also
offers stops at 720p and is kept only as a fallback.

Not built yet: the background crawler (`WorkManager` is a dependency but unused),
persistence beyond a JSON file, background audio and PiP (manifest and
permissions are in place, the `MediaSessionService` is not), and any ranking
smarter than keyword overlap.
