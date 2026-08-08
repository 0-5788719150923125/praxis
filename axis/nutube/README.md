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

- `app/src/main/kotlin/.../data/` - `FeedItem`, the source-agnostic item type,
  and `LocalIndex`, the on-device store and ranker. This is the part to grow.
- `app/src/main/kotlin/.../source/` - `YouTubeSource` over NewPipeExtractor
  (search, resolve, stream URLs) and `NewPipeDownloader`, its OkHttp transport.
- `app/src/main/kotlin/.../ui/` - Compose feed, view model, and the ExoPlayer
  overlay.
- `godot/` - the original Godot 4.6 prototype, frozen.

## Building

Needs JDK 17 and the Android SDK. `local.properties` points at the SDK and is
git-ignored, so it needs writing once per machine:

```sh
echo "sdk.dir=$HOME/Android/Sdk" > local.properties
JAVA_HOME=/usr/lib/jvm/java-17-openjdk ./gradlew :app:assembleDebug
```

The APK lands in `app/build/outputs/apk/debug/`. `./gradlew installDebug` pushes
it to a connected device.

No Android Studio required - the Gradle wrapper does everything, and the Kotlin
LSP in an editor covers the rest.

## Status

Scaffold. Search hits YouTube and folds results into the local index; the index
ranks and explains itself; tapping a card plays in-app. Sharing or opening a
YouTube link from any other app indexes it.

Not built yet: the background crawler (`WorkManager` is wired as a dependency
but unused), persistence beyond a JSON file, background audio and PiP (the
manifest and permissions are in place, the `MediaSessionService` is not), and
any ranking smarter than keyword overlap.

Playback currently prefers YouTube's HLS/DASH manifest and falls back to the
best progressive stream, which YouTube caps at 720p. Proper adaptive playback
means feeding ExoPlayer the separate video and audio tracks.
