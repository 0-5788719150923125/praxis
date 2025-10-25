# Praxis Web Frontend

Clean, data-driven web UI with zero frameworks. Just vanilla JavaScript modules.

## Architecture

**Philosophy**: `UI = render(state)`

Everything is data. The UI is a pure function of state.

```
Events → Update State → render(state) → DOM
```

## Structure

```
src/web/
├── js/
│   ├── state.js          # Single source of truth
│   ├── components.js     # Pure functions: data → HTML
│   ├── render.js         # Main render loop
│   ├── tabs.js           # Tab loading (Spec, Agents, Research)
│   ├── charts.js         # Chart.js integration + LTTB downsampling
│   ├── api.js            # REST API calls
│   ├── websocket.js      # WebSocket for terminal
│   ├── mobile.js         # Mobile-specific behaviors
│   └── main.js           # Entry point + event handlers
├── css/
│   ├── variables.css     # Design tokens
│   ├── base.css          # Resets, typography
│   ├── layout.css        # Grid, containers
│   ├── components.css    # Component styles
│   ├── themes.css        # Light/dark themes
│   ├── animations.css    # Keyframes
│   └── responsive.css    # Mobile breakpoints
└── build.py              # Python build system
```

## Build System

### Automatic (Default)

Frontend builds automatically when you start Praxis:

```bash
./launch --dev
# [WEB] Building frontend...
```

No manual build needed.

### Manual

```bash
python src/web/build.py          # Build once
python src/web/build.py --watch  # Watch mode
python src/web/build.py --prod   # Production (concatenated)
```

**No bundler needed!** Modern browsers support ES6 modules natively.

## Features

- **Functional**: Pure data flow, no hidden state
- **Modular**: ES6 modules, ~100-200 lines each
- **Fast**: Native browser module loading
- **LTTB Downsampling**: Perceptually-aware chart sampling
- **Live Terminal**: WebSocket-powered command output
- **Multi-Agent**: Compare metrics across remote agents
- **Responsive**: Mobile-first design
- **Themeable**: Light/dark mode with CSS variables
- **Zero Dependencies**: No npm, no node_modules

## Development

### Normal Workflow

```bash
./launch --dev              # Auto-builds on startup
# Edit src/web/js/state.js
./launch --dev              # Rebuilds on next startup
```

### Watch Mode

```bash
# Terminal 1
python src/web/build.py --watch

# Terminal 2
./launch --dev
```

## Key Patterns

### Data Flow

```javascript
// 1. All data lives in state.js
export const state = {
    theme: 'light',
    messages: [],
    tabs: [...]
};

// 2. Components are pure functions
export function createMessage({ role, content }) {
    return `<div class="message ${role}">${content}</div>`;
}

// 3. Events update state, then render
document.addEventListener('click', (e) => {
    if (e.target.matches('.tab-button')) {
        state.activeTab = e.target.dataset.tab;
        render();  // UI updates
    }
});
```

## Metrics & Charts

Uses **LTTB (Largest Triangle Three Buckets)** downsampling algorithm for smooth, perceptually-accurate training charts. Handles sparse early data and dense later data without visual artifacts.

---

**~720 lines total** (vs 2,625 in old monolithic app.js)
