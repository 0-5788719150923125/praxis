# Tesla Ball Logo - Escaped Tendril Anchoring Issue

## Current Problem
The escaped tendrils in `tesla-ball-logo.html` are NOT properly anchoring to fixed points in world space. They continue to rotate with the tetrahedron model instead of staying fixed.

## What Should Happen
1. A tendril grows from the center outward while the tetrahedron rotates
2. When the tendril reaches the tetrahedron boundary, it "escapes"
3. At the moment of escape, the tendril's tip position should be captured in ABSOLUTE WORLD COORDINATES
4. As the tetrahedron continues rotating, the escaped tendril should STRETCH between:
   - Base: A point on the tetrahedron surface (rotates with model)
   - Tip: The fixed world space anchor point (NEVER MOVES)
5. This creates an effect like electrical arcs anchored to invisible points in space

## What's Actually Happening
The escaped tendrils are still rotating with the tetrahedron. The anchor points are moving when they should be completely stationary in world space.

## Root Cause Analysis
The issue is likely in the coordinate transformation pipeline:
1. Tendrils are generated in LOCAL space (relative to tetrahedron center)
2. During growth, they rotate with the model via `rotate3D(localPos, globalRotX, globalRotY, globalRotZ)`
3. When escaping, we try to capture the world position, but something is wrong with how we:
   - Either capture the initial anchor point
   - Or how we use it during rendering

## Key Code Sections to Review

### Current Escape Logic (lines 346-385)
```javascript
// When tendril escapes, we store:
this.escapeWorldAnchor = {
    x: worldTip.x,
    y: worldTip.y,
    z: worldTip.z
};
```

### Current Drawing Logic (lines 436-463)
```javascript
if (this.state === 'escaped' && this.escapeWorldAnchor) {
    // Interpolate between rotating base and fixed anchor
    worldPos = this.escapeWorldAnchor; // For tip
}
```

## Debugging Steps Needed
1. **Verify the anchor is truly fixed**: Add console logging to check if `escapeWorldAnchor` values change over time
2. **Check rotation order**: Ensure we're not accidentally applying rotation AFTER storing the "world" position
3. **Verify coordinate spaces**: Make sure we're truly in world space when we think we are

## Potential Solution Approaches

### Approach 1: Store Screen Coordinates
Instead of storing 3D world coordinates, store the 2D screen position at escape time:
```javascript
// At escape time
this.escapeScreenX = screenX;
this.escapeScreenY = screenY;
```

### Approach 2: Reverse Transform
Store the LOCAL position and the rotation at escape time, then always transform using that frozen rotation:
```javascript
// At escape
this.escapeLocalPos = localTip;
this.escapeRotation = {x: globalRotX, y: globalRotY, z: globalRotZ};

// When drawing
worldPos = this.rotate3D(this.escapeLocalPos, this.escapeRotation.x, this.escapeRotation.y, this.escapeRotation.z);
```

### Approach 3: Debug Current Implementation
The current approach SHOULD work. Need to verify:
1. `globalRotX/Y/Z` are the actual current rotations
2. `rotate3D` is correctly transforming to world space
3. The stored anchor isn't being modified elsewhere

## Files
- `/home/crow/repos/praxis/staging/tesla-ball-logo.html` - Main animation file
- `/home/crow/repos/praxis/staging/serve.py` - Dev server (working correctly)

## Next Steps
1. Add debug logging to verify anchor positions remain constant
2. Check if the issue is in the capture or the rendering
3. Implement one of the solution approaches if current method can't be fixed