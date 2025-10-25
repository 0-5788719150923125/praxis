
# Praxis Web Frontend - Functional Architecture

**Clean, data-driven, maintainable web UI with zero frameworks.**

## Philosophy: UI = render(data)

Everything is data. The UI is a pure function of state. No frameworks, no complexity - just vanilla JavaScript with ES6 modules.

```javascript
// The entire architecture in 3 lines:
Events → Update State → render(state) → DOM
```

## Directory Structure

```
src/web/
├── js/
│   ├── state.js          # Single source of truth - all data lives here
│   ├── components.js     # Pure functions: data → DOM strings
│   ├── render.js         # Main render function (UI = render(state))
│   ├── api.js            # API communication
│   ├── websocket.js      # WebSocket for terminal
│   └── main.js           # Entry point, event handlers
├── css/
│   ├── variables.css     # Colors, spacing, design tokens
│   ├── base.css          # Resets, typography
│   ├── layout.css        # Grid, containers
│   ├── components.css    # All component styles (carefully preserved)
│   ├── themes.css        # Theme overrides
│   ├── animations.css    # Keyframe animations
│   └── responsive.css    # Mobile breakpoints
└── build.py              # Simple Python build system
```

## How It Works

### 1. State (The Data)

Everything lives in one place:

```javascript
// state.js
export const state = {
    theme: 'light',
    messages: [],
    tabs: [...],
    settings: {...},
    // etc.
};
```

### 2. Components (Pure Functions)

Components are just functions that take data and return DOM:

```javascript
// components.js
export function createMessage({ role, content }) {
    return `<div class="message ${role}">${content}</div>`;
}
```

No classes, no lifecycle hooks, no magic - just `data → HTML`.

### 3. Render (UI = render(state))

One function updates the entire UI:

```javascript
// render.js
export function render() {
    renderMessages();
    renderTabs();
    renderTheme();
    // ...
}
```

### 4. Events → State Updates → Render

```javascript
// main.js
document.addEventListener('click', (e) => {
    if (e.target.matches('.tab-button')) {
        state.tabs.forEach(t => t.active = t.id === e.target.dataset.tab);
        render();  // That's it!
    }
});
```

## Build System

### Automatic Build (Default)

The frontend **builds automatically** when you start Praxis:

```bash
./launch --dev
# [WEB] Building frontend...
#   ✓ Copied state.js
#   ✓ Copied components.js
#   ...
```

No manual build needed! The web app is always up-to-date.

### Manual Build (Optional)

```bash
python src/web/build.py          # Build once
python src/web/build.py --watch  # Watch mode (rebuilds on file change)
```

Copies modules to `static/js/` - browser loads them natively with `import`/`export`.

**No bundler needed!** Modern browsers support ES6 modules out of the box.

### Production (Concatenated)

```bash
python src/web/build.py --prod
```

Concatenates all modules into single `static/app.js` for deployment.

## Usage

### Adding a New Component

1. Add data to `state.js`:
```javascript
export const state = {
    myFeature: { enabled: true, data: [] }
};
```

2. Create component in `components.js`:
```javascript
export function createMyFeature(data) {
    return `<div class="my-feature">${data}</div>`;
}
```

3. Render it in `render.js`:
```javascript
function renderMyFeature() {
    const container = document.getElementById('my-feature');
    container.innerHTML = createMyFeature(state.myFeature.data);
}
```

4. Call from event handler in `main.js`:
```javascript
if (e.target.matches('.my-button')) {
    state.myFeature.data.push('new item');
    render();
}
```

That's it! No configuration, no registration, no framework ceremony.

### Adding Styles

Just add CSS to `src/web/css/components.css`:

```css
.my-feature {
    padding: var(--spacing-md);
    background: var(--background);
    /* Use existing design tokens! */
}
```

Run build, styles are automatically included.

## Visual Design Preservation

All the carefully-crafted visual details are preserved:

- ✅ Pixel-perfect spacing (7px gaps, 22px padding, etc.)
- ✅ Beautiful glowing tab effects in dark mode
- ✅ Curved edges (6px radius throughout)
- ✅ Subtle shadows and transitions
- ✅ Theme system with CSS variables

**Nothing was lost in the refactor** - just made maintainable.

## Development Workflow

### Option 1: Normal Workflow (Recommended)

```bash
./launch --dev                    # Builds automatically on startup
# Edit src/web/js/state.js
./launch --dev                    # Rebuilds on next startup
```

Simple! Just restart Praxis to rebuild.

### Option 2: Watch Mode (Advanced)

```bash
# Terminal 1: Watch and auto-rebuild
python src/web/build.py --watch

# Terminal 2: Run Praxis
./launch --dev

# Edit src/web/js/state.js → auto-rebuilds → refresh browser
```

For rapid iteration without restarting Praxis.

### Production Build

```bash
python src/web/build.py --prod   # Single concatenated file
```

## Why This Approach?

1. **Simple**: No frameworks, no build complexity
2. **Fast**: ES6 modules load instantly in dev
3. **Maintainable**: Each file is ~100-200 lines, easy to understand
4. **Testable**: Pure functions are trivial to test
5. **Functional**: Clear data flow, no hidden state
6. **Python-Native**: Build system is just Python code
7. **Zero Dependencies**: No npm, no node_modules

## Migration Notes

The old `static/app.js` (2,625 lines) has been split into:
- `state.js` (96 lines) - Data
- `components.js` (147 lines) - UI functions
- `render.js` (112 lines) - Rendering logic
- `api.js` (68 lines) - API calls
- `websocket.js` (99 lines) - WebSocket handling
- `main.js` (198 lines) - Events and initialization

**Total: ~720 lines** (72% reduction) with better organization.

CSS similarly organized from 1,700 lines into logical modules.

## Next Steps

- [ ] Migrate Spec tab to data-driven rendering
- [ ] Migrate Agents tab to data-driven rendering
- [ ] Migrate Research tab to data-driven rendering
- [ ] Add unit tests for pure functions
- [ ] Consider adding JSDoc types for better IDE support

---

**Built with ❤️ for the Praxis AI ecosystem**
