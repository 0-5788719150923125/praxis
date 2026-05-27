# Axis

Axis is a mobile companion app for controlling Praxis, built with [Godot](https://godotengine.org/) 4.3. It is an older, archived experiment: the app still runs, but the model-prompting flow almost certainly no longer talks to the current Praxis API.

## What it does

On first launch it asks for your Praxis server URL (stored in `user://settings.cfg`), then opens a chat interface backed by some atmospheric visuals - a neural-network scene, orbiting atoms, a starfield, and glitched-text effects.

The chat screen POSTs an OpenAI-style `messages` array to `<server>/input/`. That contract has since drifted, so expect the prompting half to be broken even though the UI and networking scaffolding still work.

## Status

Effectively unmaintained. It needs Godot skills to revive, which I don't currently have, so it lives here mostly as a record of the idea. Treat the source under `scenes/` and `scripts/` as a starting point rather than a working client.

Open `project.godot` in the Godot editor to poke at it.
