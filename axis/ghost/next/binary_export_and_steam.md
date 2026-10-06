# ghost: binary export and Steam

Researched 2026-10-05; nothing built yet. The question: can ghost be sold (or given away) on Steam,
given that it downloads its own dependencies at runtime and some of them might forbid commercial
use? Answered with an audit of what ghost fetches and what a build would ship, plus web research -
sources at the end, every one checked on 2026-10-05. File references are `file:line` into
`axis/ghost/`. Not legal advice: the items marked **ASK** want a lawyer's or Valve's answer before
launch.

## The short answer

- **Nothing ghost downloads blocks a release.** Piper, the one that looked risky, is already
  handled: the voice allowlist holds only public-domain and CC BY voices, and no GPL Piper code is
  installed.
- **Fetching at runtime protects against terms triggered by DISTRIBUTING, not terms triggered by
  USING.** GPL and attribution duties attach to whoever conveys a copy; ghost conveys none, it has
  the user's machine fetch upstream's. Non-commercial and research-only licenses, YouTube's terms,
  anti-circumvention law and codec patents are about use, so fetching does nothing for them.
- **The real work is elsewhere:** the YouTube import, what a default export would pack, the Linux
  FFmpeg source, the Assistant, third-party notices - and the export itself, which ghost has never
  been through (there is no `export_presets.cfg`).

## Does price change anything?

Almost nothing here is triggered by charging. Price only matters to non-commercial licenses, and
only with no monetization at all.

| Restriction | Free, no monetization | Free + donations or premium |
|---|---|---|
| Copyleft and attribution (GPL eSpeak-NG/FFmpeg, CC BY LibriTTS, OFL fonts, BSD CMUdict, MIT Godot) | Same duties - giving copies away is still distribution | Same |
| Non-commercial voices (CC BY-NC-SA: ryan, hfc_*) | Arguably OK | Donations a gray zone; premium makes it commercial |
| Research-only (Blizzard 2013: lessac and all fine-tuned from it) | Still no - a consumer app is not research | No |
| Anthropic | Commercial Terms, unmodified binary, never touch credentials | Same |
| OpenAI (Codex on a ChatGPT login) | Its closest docs treat open-source apps apart from paid ones; a free build of MIT code sits nearer the open-source side | Premium is the paid-app side (an interest form) |
| Steam AI disclosure, guardrails, third-party account notice | Required | Required |
| Anti-circumvention (YouTube import) | Price-blind: DMCA §1201 bans offering a circumvention tool, sold or not | Same |

What monetizing adds is Steam's own rule that in-game transactions go through Steam Wallet:

- **Donations:** a "supporter pack" DLC. A Patreon or Ko-fi button inside the app risks running
  into the rule.
- **Premium features:** DLC or Steam microtransactions, at Valve's usual cut.
- **Premium = the AI modes** is the worst combination: Valve's FAQ expects you to provide and bill
  the AI service for players, and OpenAI would treat ghost as a paid app.

## Before shipping, ranked

1. **The YouTube import.** The one item with court rulings against tools like it (details below).
   ghost installs yt-dlp plus Deno to solve YouTube's challenges, and says so in its own panel text
   (`scripts/deps.gd:128`). Fetching instead of bundling does not help: §1201 and the German rule
   both reach PROVIDING the tool. Valve has not acted against this (VRChat ships yt-dlp), so the
   risk is rights holders, not review. Recommendation: leave URL import out of the Steam build and
   keep it in the source build. Keeping it is **ASK** (a lawyer).
2. **What a default export would pack.** Godot's export ignores `.gitignore`. `masks/` holds 27
   imported files (audio and waveforms from the clips stored there, YouTube downloads included),
   `feedback/` its screenshots, and `reference/arcbot.webp` is a photograph of unknown provenance
   used only by the README. Exclude all three, plus `build/` and `tests/`, in the preset.
3. **The Linux FFmpeg source.** Martin Riedl's Linux build is configured `--enable-nonfree`
   (DeckLink), which FFmpeg's license calls unredistributable - confirmed on the provisioned 9.0.2
   with `ffmpeg -version`. ghost is not the one redistributing it, but there is no reason to send
   customers there: BtbN's Linux GPL builds are a drop-in swap. Riedl's macOS build, Gyan's and
   BtbN's Windows builds are plain GPLv3.
4. **The Assistant.** It runs `claude --dangerously-skip-permissions` and
   `codex --dangerously-bypass-approvals-and-sandbox` against the checkout
   (`scripts/assistant_backends.gd:61`, `:68`). A Steam install has no checkout, and an agent with no
   permission checks on a customer's machine runs into Valve's ban on apps that "modify customer's
   computers in unexpected or harmful ways". Strip it from the build. (The tarot writer and painter
   are already locked down: `--tools ""` or ghost's own MCP tools, Codex read-only or
   workspace-write in a job folder.)
5. **Third-party notices** - see "What a build ships" below.

## Binary export: why a build would not run today

Not legal issues; the reasons an exported ghost would fail as the code stands.

- **No export preset** exists yet.
- **The Python hosts are found through `res://`.** `voice_host.gd:129`, `mask_editor.gd:5389` and
  `:5742`, and `page_capture.gd:153` hand Python `ProjectSettings.globalize_path("res://...")`.
  Godot's docs say that does not work in an exported project, and Python could not read inside a
  `.pck` anyway. The hosts' `.py` files and the `requirements.txt` files the Provisioner hashes and
  installs from must ship as loose files beside the executable (a Steam depot can carry them) and be
  found through `OS.get_executable_path().get_base_dir()`.
- **Non-resource files are left out of the `.pck`** unless the preset's include filter names them:
  `data/cmudict.dict`, `data/english.yml`, `storyboards/*.yml`, the hosts' files and every license
  file. Check `.json` too.
- **The video export and the audio bake relaunch the engine against the project folder** -
  `--path <project>`, plus `--script res://...` for the bake (`scripts/exporter.gd:550-555`,
  `scripts/mask_editor.gd:8036`) - and the render reads its resolution from an `override.cfg`
  written into the project root (`scripts/exporter.gd:886-890`). An exported build has no project
  folder: this becomes the binary relaunching itself, with the overrides passed some way other than
  a file in the install directory. Masking's reload check (`scripts/mask_editor.gd:6453`) runs
  `--editor`, which an export template does not have - dev-only, strip it with the Assistant.
- **`MASKS_DIR` is `res://masks`** (`scripts/mask_editor.gd:61`), read-only once exported. Masking's
  sessions and URL downloads belong in `user://`.
- **The Assistant finds the checkout as `res://../..`** (`scripts/assistant.gd:131`); strip it (above).
- **Programs that stay the machine's own** (`Deps.TOOLS`: setpriv, fallocate, xvfb-run) - open
  question below on whether a Steam-launched ghost can see them on Linux.
- Windows and macOS have never been run on real machines.

## Steam's rules

Verified on the Steamworks pages unless marked as inference.

- **Software is accepted** in listed categories, including "Audio/Video Production", "Animation &
  Modeling" and "Design & Illustration". Banned: advertising-based business models, blockchain/NFT.
- **AI disclosure (Content Survey).** Pre-Generated is AI content that ships; Live-Generated is
  "content created with the help of AI tools while the game is running", for which "you'll need to
  tell us what kind of guardrails you're putting on your AI". For ghost, Live-Generated covers the
  neural voices, tarot's text and images and the book/notebook illustrations (inference). Players
  can report live-generated content through the overlay; "Live-Generated AI Adult Only Sexual
  Content" is not allowed. AI coding tools used in development need no disclosure since Valve's
  January 2026 change ("Efficiency gains through the use of these tools is not the focus").
- **Store page:** "Requires 3rd-Party Account" and the AI-service notice ("Connects to 3rd-Party
  Service for AI Content Generation"). Disclose the first-run downloads and their sizes (from
  `scripts/deps.gd`): FFmpeg ~70 MB (~115 MB Windows), uv ~20 MB, Python ~35 MB, voice environment
  ~320 MB, download environment ~140 MB, page capture ~150 MB + Chromium ~170 MB, body/face
  tracking ~520 MB, Piper voices ~60 MB each.
- **Downloading programs after install:** no rule forbids it and no store field covers it. The
  nearest rule bans apps that "modify customer's computers in unexpected or harmful ways". So:
  expected (a consent prompt or at least the existing progress UI), disclosed, kept in ghost's own
  folders. Valve also expects uninstall to clean up what install created, and Steam has no Linux
  install scripts, so a "remove downloaded components" action in the Environment panel is worth
  having.
- **External AI services:** the Content Survey FAQ says that for an AI service with ongoing costs
  "you'll need to manage both access to that external service on behalf of your players and
  collecting payment from your player using a Steam-supported payment method." Bring-your-own-
  account tarot fits none of Valve's suggested models. AI Roguelite is live with "custom via API",
  so it can pass, but this is **ASK** (Steamworks support) before submitting. That tarot is grayed
  out without an agent, and every other mode works without one, helps.
- **In-game transactions use Steam Wallet** (the review checklist).
- **EULA:** the Steam Subscriber Agreement licenses products for personal, non-commercial use unless
  your terms say otherwise. Give users commercial rights to their exports in ghost's EULA.

## The AI providers

ghost never sees credentials; it runs the CLIs the user installed and logged into.

- **Anthropic: explicitly allowed.** The Agent SDK docs: "Unless previously approved, Anthropic does
  not allow third party developers to offer claude.ai login or rate limits for their products".
  The legal-and-compliance page separates the cases: running Claude Code in a product "requires
  agreeing to our Commercial Terms of Service", "The Claude Code binary must not be modified",
  developers "may not collect, store, or intermediate Claude.ai credentials or session tokens", and
  none of it prevents "an end user from signing in to the unmodified Claude Code binary with their
  own Claude subscription." ghost is that case. 2026 enforcement was against tools spoofing the
  client (OpenCode, OpenClaw). A planned change to how `claude -p` draws on a subscription was
  paused in June 2026 - it may still move.
- **OpenAI: unclear.** Its docs say "Use API key authentication for programmatic Codex CLI
  workflows", while `codex exec` reuses the saved login by default. Its Sign in with ChatGPT program
  serves "all open-source partners and selected private clients", and "a paid or remotely hosted
  app" fills in an interest form. Nothing addresses an app that spawns `codex exec`. Safe paths:
  the interest form, or telling Codex users to log in with an API key.
- **Amazon Bedrock:** the user's account and bills. Outputs are "Your Content" (AWS Service Terms
  50.2; Nova has AWS's IP indemnity). Stability's Bedrock terms give no revenue cap or attribution
  rule (the Community License's $1M cap covers self-hosted weights). Stability's policy bans nudity,
  so traditional Star and Lovers cards will likely be refused - a no-nudity line in the painter
  prompt also gives you a guardrail to describe to Valve.
- **Output ownership:** Anthropic assigns outputs to the user (consumer) or the customer owns them
  (commercial); OpenAI's user owns the output. With players on their own accounts, those rights are
  the player's, not yours.
- **Inherited obligations:** Anthropic's usage policy has consumer-facing interactive agents
  "disclose to users that they are interacting with AI"; OpenAI forbids presenting output as
  human-made. Tarot is plainly AI.

## The YouTube import in detail

- **Germany, final.** LG Hamburg (31 March 2023, 310 O 317/21) held Uberspace - merely the web host
  of youtube-dl's site - liable for aiding circumvention of YouTube's rolling cipher, treated as an
  effective technological measure. OLG Hamburg affirmed (21 November 2024, 5 U 54/23); the BGH
  declined review in October 2025. Steam sells in Germany.
- **United States, unsettled.** The RIAA's 2020 §1201 takedown of youtube-dl was reversed by GitHub.
  Yout v. RIAA was dismissed in 2022; the appeal was argued in February 2024 with no ruling by April
  2026. In 2026, claims over the same cipher survived motions to dismiss (Cordova v. Huneault; Sony
  v. Udio, where the cipher's status "requires a greater factual record"). No ruling yet names PO
  tokens or the JavaScript challenges yt-dlp-ejs solves.
- **Terms and stores.** YouTube's terms bar downloading "except ... as expressly authorized".
  Apple's guideline 5.2.3 bars it on the Mac App Store; Google made Microsoft's 2013 Windows Phone
  YouTube app drop downloads. No Steam removal found.
- **Options:** leave it out of the Steam build (recommended); or keep yt-dlp for other sites and
  block YouTube - though the challenge solver exists for YouTube; or keep it and take the risk with
  a lawyer's opinion.

## FFmpeg and video patents

- **Builds:** Gyan "essentials" is GPLv3; BtbN's `gpl` variants are GPLv3 (`--enable-gpl
  --enable-version3`); Riedl macOS GPLv3, Riedl Linux nonfree (above).
- **Running ffmpeg as a program is aggregation.** The FSF's FAQ: "pipes, sockets and command-line
  arguments are communication mechanisms normally used between two separate programs". ghost's GPL
  duties for FFmpeg are none; they stay with the builders.
- **Patents are not moved by downloading at runtime.** The exporter encodes H.264 with libx264 and
  AAC. Patent liability covers inducing infringement (35 U.S.C. §271(b)), and the app that drives the
  encoder is the natural target. Via LA's AVC license is $0 for 1-100,000 units a year, then $0.20,
  and $0.10 above 5 million. The last US AVC patents expire December 2026 to January 2028 (one
  outlier, US 9,356,620, in November 2030); AAC's last baseline patent expires in 2028, extensions
  in 2031. The HEVC pool moved to Access Advance in December 2025. Low risk and shrinking. Options:
  sign the free tier, encode with the OS encoders (`h264_mf` on Windows, `h264_videotoolbox` on
  macOS), or offer AV1/Opus.

## What ghost fetches, and under what license

| Fetched | License | Note |
|---|---|---|
| FFmpeg + ffprobe | GPLv3 (Riedl Linux: nonfree) | separate process |
| uv | MIT or Apache-2.0 | |
| CPython (via `uv python install`) | PSF and bundled permissive libraries | |
| onnxruntime, numpy, onnx | MIT, BSD-3-Clause, Apache-2.0 | |
| phonemizer, espeakng-loader (eSpeak-NG) | GPL-3.0 | imported IN-PROCESS by voice_host; fine while ghost is MIT (GPL-compatible) and never ships them; `phonemizer="ghost"` drops both |
| nltk (+ tagger and stopword data it fetches) | Apache-2.0 | |
| yt-dlp[default], Deno | Unlicense, MIT (mutagen inside is GPL-2.0+) | the problem is what it does, not its license |
| mediapipe + its two models | Apache-2.0 | model cards: "Any form of surveillance or identity recognition is explicitly out of scope" - guidance, not terms |
| opencv-contrib-python | Apache-2.0 | |
| playwright + Chromium | Apache-2.0, BSD-style | |
| Piper voices | per voice, below | |

The AI CLIs (claude, codex, aws) are the user's own installs, not fetched.

**Piper voices, read MODEL_CARD by MODEL_CARD** (repo `rhasspy/piper-voices`, tagged MIT; each
voice's card governs it):

| Voice | Dataset | License | Training |
|---|---|---|---|
| en_US-ljspeech-medium | LJ Speech | public domain | from scratch |
| en_US-libritts-high | LibriTTS (openslr 60) | CC BY 4.0 - credit Google LLC | from scratch |
| en_US-kristin-medium | LibriVox | public domain | from scratch |
| en_US-norman-medium | LibriVox | public domain | from scratch |
| en_US-john-medium | LibriVox | public domain | fine-tuned from kristin |

Excluded on purpose (`voice_host/backends/piper.py` header): lessac (Blizzard 2013, research only,
explicitly no "commercialization, sale or licencing of voice synthesis products") and every voice
fine-tuned from it (amy, joe, hfc_female, hfc_male, libritts_r); ryan and the hfc pair (CC
BY-NC-SA); kathleen (fine-tuned from ryan). Adding a voice means reading its card and its whole
fine-tuning chain.

## What a build ships, and the notices it owes

- **Godot** (MIT) and the libraries it bundles - Godot's "Complying with licenses" page lists the
  ones that need credit.
- **Fonts:** 11 OFL families - `fonts/hands/` (Caveat, Kalam, Patrick Hand) and `fonts/tarot/`
  (Bungee, Cinzel, Courier Prime, EB Garamond, IM Fell English, Limelight, Uncial Antiqua,
  UnifrakturMaguntia). OFL is fine in a commercial app; its text must travel with the fonts.
- **`data/cmudict.dict`:** CMU's BSD-style license - the notice reproduced in the documentation.
- **`data/libritts_speakers.json`:** derived from LibriTTS-P (LINE), CC BY 4.0 - credit.
- **`data/tarot/meanings.json`:** CC0 (Corpora) over an uncopyrighted source (McElroy). Nothing
  owed; credit anyway.
- **The libritts voice:** fetched, not shipped, but credit LibriTTS (Google LLC, CC BY 4.0) in the
  same place. Whether a model trained on it counts as "sharing" it is unsettled; a credit line costs
  nothing.
- **How:** Godot exports skip `.txt`/`.LICENSE` unless included, so the license files beside the
  fonts and data would silently drop out. Ship a `THIRD_PARTY_NOTICES.txt` beside the binary and an
  About panel. `docs.py` could generate it from `Deps` and the license files, as it renders the
  other docs off registries.

## Smaller items

- **The license comments disagree with the code.** `voice_host/requirements.txt:11-12` says
  phonemizer and espeakng-loader are not installed; lines 37-38 install them, and
  `voice_host/backends/piper.py:2084` uses eSpeak by default. `scripts/deps.gd:177` tells users no
  GPL Piper code is installed (true only of piper-tts). Fix before anyone audits from the comments.
- **The name.** Renamed 2026-10-05 to "Ghost Notes: A Spectral Experience": a Steam game is already
  called Ghost, and the word is unsearchable. Still run a trademark search on the new name before it
  goes on a store page. `application/config/name` stays "ghost" because Godot names the user:// folder
  after it; the export preset should set the product name (Windows `product_name`, the macOS bundle
  name) explicitly, and a shipped build probably wants its own custom user folder - with a migration
  from `godot/app_userdata/ghost`.
- **Everything runs locally** - face and body tracking included - and nothing reaches the
  developer. Worth keeping true: face-geometry data is what biometric-privacy laws such as
  Illinois's BIPA regulate.

## Open questions

- **Valve:** does bring-your-own-account tarot pass review? (**ASK** Steamworks support.)
- **Lawyer:** the YouTube import, if it stays.
- **OpenAI:** Codex on a ChatGPT login inside a paid app - the interest form.
- **Steam Linux Runtime:** can a Steam-launched ghost see the host's xvfb-run, setpriv and
  fallocate? A containerized runtime may hide host programs; renders depend on xvfb-run.
- **macOS:** signing and notarization for a Steam build - not researched.

## Sources

Checked 2026-10-05.

Steam
- https://partner.steamgames.com/doc/gettingstarted/contentsurvey
- https://partner.steamgames.com/doc/gettingstarted/onboarding
- https://partner.steamgames.com/doc/store/review_process
- https://partner.steamgames.com/doc/sdk/installscripts
- https://partner.steamgames.com/doc/sdk/uploading/distributing_opensource
- https://store.steampowered.com/news/group/4145017/view/3862463747997849618 (AI content, January 2024)
- https://store.steampowered.com/news/group/4145017/view/4547038620960934856 (required disclosures)
- https://www.pcgamer.com/software/ai/steam-updates-ai-disclosure-form-to-specify-that-its-focused-on-ai-generated-content-that-is-consumed-by-players-not-efficiency-tools-used-behind-the-scenes/
- https://store.steampowered.com/app/1889620 (AI Roguelite)
- https://store.steampowered.com/subscriber_agreement/

AI providers
- https://code.claude.com/docs/en/agent-sdk/overview
- https://code.claude.com/docs/en/legal-and-compliance
- https://support.claude.com/en/articles/15036540
- https://www.anthropic.com/legal/aup
- https://www.anthropic.com/legal/consumer-terms
- https://www.anthropic.com/legal/commercial-terms
- https://learn.chatgpt.com/docs/auth
- https://learn.chatgpt.com/docs/non-interactive-mode
- https://developers.openai.com/siwc/quickstart
- https://developers.openai.com/siwc/token-sharing-open-source
- https://openai.com/policies/terms-of-use/
- https://aws.amazon.com/service-terms/
- https://aws.amazon.com/legal/bedrock/third-party-models/
- https://aws.amazon.com/ai/responsible-ai/policy/

YouTube and yt-dlp
- https://freiheitsrechte.org/themen/starke-grundrechte-fuer-eine-lebendige-demokratie/uberspace-youtube-dl
- https://github.com/github/dmca/blob/master/2020/10/2020-10-23-RIAA.md
- https://github.blog/news-insights/policy-news-and-insights/standing-up-for-developers-youtube-dl-is-back/
- https://torrentfreak.com/yout-com-hopes-supreme-courts-cox-ruling-helps-its-case-riaa-disagrees/
- https://torrentfreak.com/ripping-clips-for-youtube-reaction-videos-can-violate-the-dmca-court-rules/
- https://shertremonte.com/2026/05/11/client-alert-for-ai-companies-how-you-get-your-training-data-matters-sdny-allows-dmca-claim-to-proceed-against-ai-music-generator-udio/
- https://www.youtube.com/static?template=terms
- https://developer.apple.com/app-store/review/guidelines/
- https://github.com/yt-dlp/yt-dlp/wiki/EJS

FFmpeg and patents
- https://ffmpeg.martin-riedl.de/download/linux/amd64/1789931100_9.0.2/versions.txt
- https://www.gyan.dev/ffmpeg/builds/
- https://github.com/BtbN/FFmpeg-Builds/blob/master/variants/defaults-gpl.sh
- https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md
- https://www.gnu.org/licenses/gpl-faq.html#MereAggregation
- https://www.via-la.com/licensing-programs/avc-h-264/
- https://meta.wikimedia.org/wiki/Have_the_patents_for_H.264_MPEG-4_AVC_expired_yet%3F
- https://accessadvance.com/2025/12/15/access-advance-and-via-licensing-alliance-announce-hevc-vvc-program-acquisition/
- https://www.law.cornell.edu/uscode/text/35/271

Models and data
- https://huggingface.co/rhasspy/piper-voices (each voice's MODEL_CARD)
- https://brycebeattie.com/files/tts/ (kristin, norman, john: "public domain")
- https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
- https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf
- https://www.openslr.org/60/
