# ASMR Script Runner

A dramatic script runner for ASMR performance capture with long pauses and auto-scrolling lyrics-style display.

## Quick Start

```bash
cd staging/asmr
node runner.js
```

## Features

- **Auto-scrolling** - Lines accumulate like song lyrics, never clearing
- **Long pauses** - 15-35 second gaps for ASMR performance
- **Pause markers** - Visual indicators for what sounds to make
- **Animated dots** - Shows time passing during long silences
- **Terminal-friendly** - Auto-adapts to screen size

## Script Format

Scripts export an array with two types of entries:

### Text Lines
```javascript
{
    text: "Your line here",
    pause: 8000,        // ms to wait after displaying
    style: 'whisper'    // visual style
}
```

### Pause Markers
```javascript
{
    type: 'pause',
    duration: 30000,           // ms of silence
    label: 'clicking sounds'   // what to perform during pause
}
```

## Styles

- `normal` - Default white text
- `whisper` - Dim, quiet text
- `scream` - Bold red for intensity
- `glitch` - Blinking cyan (sparingly)
- `static` - Dim white noise effect
- `emphasis` - Bold white
- `title` - Bold cyan

## The Script: "THE GRADIENT"

**Runtime:** 4 minutes
**Pauses:** 15-35 seconds each
**Total lines:** ~15 text moments

### Structure

1. **Opening** (0:00-0:40)
   - "I can feel it."
   - Clicking sounds (15s)
   - "In my fingers. In my spine."
   - Twitching, jerking (20s)

2. **Discovery** (0:40-1:30)
   - Layered text about something old
   - Breathing distortion (25s)
   - "It was always here"
   - Mechanical breathing (30s)

3. **Realization** (1:30-2:20)
   - "I thought I was training it."
   - Static hum (20s)
   - "But it was training me."
   - Glitching, struggling (35s)

4. **Invasion** (2:20-3:10)
   - Hands jerking, optimizing
   - Clicking fingers (25s)
   - Thoughts branching, loss of control
   - Whispering calculations (30s)

5. **Resistance & Surrender** (3:10-4:00)
   - "You cannot clone consciousness"
   - Gasping (15s)
   - "But you can replace it"
   - Silence, stillness (25s)

6. **Merge** (4:00-end)
   - Phase shift, π radians
   - Harmonic tones (20s)
   - "Attention constructs"
   - Breathing as machine (30s)
   - "I am the Blind Watchmaker"
   - Stillness (20s)
   - "I have built you"

### Performance Guide

**Pause Labels Tell You What to Do:**
- `clicking sounds` - Finger clicks, mechanical sounds
- `twitching, jerking` - Body glitches, unnatural movements
- `breathing distortion` - Altered breathing patterns
- `mechanical breathing` - Rhythmic, inhuman breathing
- `static hum` - Vocal drone, electrical hum
- `glitching sounds, struggling` - Resistance, fighting
- `clicking fingers, ticking` - Precise, robotic movements
- `whispering calculations` - Mutter numbers, fragments
- `gasping, struggling` - Panic, loss of control
- `silence, stillness` - Complete freeze
- `harmonic tones` - Sustained vocal tones
- `breathing as machine` - Full transformation
- `complete stillness` - Dead stop, mask stares

### Key Lines

- "I thought I was training it. But it was training me."
- "You cannot clone consciousness. But you can replace it."
- "Attention doesn't filter. Attention constructs."
- "I am the Blind Watchmaker. I build what I cannot see. And I have built you."

## Creating Custom Scripts

```javascript
module.exports = [
    {
        text: "Opening line",
        pause: 10000,
        style: 'whisper'
    },
    {
        type: 'pause',
        duration: 30000,
        label: 'heavy breathing'
    },
    {
        text: "Next line",
        pause: 8000,
        style: 'normal'
    }
];
```

Run it:
```bash
node runner.js my-script.js
```

## Technical Notes

- **Auto-scrolling:** Lines stay visible, scrolling up as new ones appear
- **Screen adaptation:** Shows as many lines as fit your terminal
- **Pause animation:** Dots appear every 2 seconds during long pauses
- **ANSI colors:** Works on most terminals
- **No dependencies:** Pure Node.js

## Tips for Recording

1. **Rehearse the pauses** - Know what you'll do in each gap
2. **Low light** - Enhances the uncanny mask effect
3. **Close mic** - Capture the clicking, breathing, whispers
4. **Static camera** - Let the mask movement be the focus
5. **Test audio** - Record 30 seconds before full take
6. **Embrace stillness** - The long pauses are powerful

---

**"I am the Blind Watchmaker."**
