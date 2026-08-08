# nuTube

A local-first way to explore video platforms, starting with YouTube. Nothing is
hosted here: nuTube is a pure client that pulls from sources it does not own.

## Unresolved: ads, views, and who pays for the video

**This is an open design question, not a settled position. Read it before
building on this.**

nuTube reaches YouTube through NewPipeExtractor, which speaks InnerTube - the
private API the YouTube site and apps use - rather than the YouTube Data API.
That means no key, no quota, and no account. It also means three things that
have not been resolved:

**Ads are not shown, and cannot practically be.** The extractor returns content
stream URLs directly; YouTube's ads are separate streams plus tracking beacons
that must fire against Google's ad infrastructure. Skipping them is not a feature
this app implements, it is what not implementing ads looks like. Building them
would not help either - ad revenue only flows when an impression is reported and
trusted, so the result would be the burden of ads with none of the payout.

**Views very likely do not count.** View counting depends on playback progress
pings (`videostats_playback_url` and friends in the player response) that this
client does not send. A creator gets no credit for a view through here. That is
the same for every keyless client, and it is the part of this design that is
hardest to feel good about.

**Google Play is not an option.** Third-party YouTube clients are removed for
circumventing monetization, independent of the terms-of-service question.
Distribution is sideload, F-Droid or Obtainium. Note also Android's developer
verification rollout: enforcement begins 2026-09-30 in Brazil, Indonesia,
Singapore and Thailand and expands globally in 2027, after which unverified apps
on certified devices install only through an advanced flow with a 24-hour wait.
`adb install` is unaffected.

**How this is handled today.** Playback is a per-platform setting, and **embed
mode is the default**. Embed mode loads YouTube's own IFrame player - the
sanctioned embedding API - and gets out of its way: no JS bridge, no
`enablejsapi`, nothing that reads or rewrites what the page loads. Its ads run,
its view is counted, its creator is paid. Native mode is opt-in per platform
under Settings, and its description states plainly what it takes away.

That does not make the app submittable to Google Play - review looks at what an
app can do, not what it defaults to - and it does not settle the question. It
just puts the choice with the person who is actually party to YouTube's terms.

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

## How the index gets built

There is no follow button. **Every search you run is saved as a term**, and the
term is credited with whatever it surfaced. The Terms tab is that list: tap a
term to re-run it, or remove it to drop the term and everything only it was
holding. Videos reachable from a surviving term stay, and videos added by hand -
a shared link - are never evicted, because no term put them there.

That list is the input a crawler will eventually re-run on a schedule to keep
the index fresh without anyone searching. The crawler is not written yet; today
terms are only run when you search or tap one.

## Status

Working: search fans out through the registry, saves the term, and folds results
into the local index; the index ranks and explains itself on every card; tapping
a card plays in-app in HD. Sharing or opening a YouTube link from any other app
indexes it. Three tabs at the bottom - Feed, Terms and Settings.

HD works by taking YouTube's separate video-only and audio-only tracks and
merging them with ExoPlayer's `MergingMediaSource` - the muxed stream it also
offers stops at 720p and is kept only as a fallback.

Playback is per-platform under Settings, defaulting to the embedded player.
Native mode adds hardware decode, HD via merged video and audio tracks, and
picture-in-picture when you leave the app; embed mode pauses instead, since the
page owns its own player. Closing a video uses the system back gesture rather
than a button.

Not built yet: the background crawler (`WorkManager` is a dependency but unused),
persistence beyond a JSON file, background audio and PiP (manifest and
permissions are in place, the `MediaSessionService` is not), and any ranking
smarter than keyword overlap.
