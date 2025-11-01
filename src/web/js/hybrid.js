/**
 * Praxis Web - Hybrid Split-Reality Theme Mode
 *
 * Creates a dynamic split-screen effect where dark and light themes
 * are divided by a plane aligned with one face of the rotating tetrahedron.
 *
 * "Prismatic attention rides the line between two worlds"
 */

let isActive = false;
let animationFrameId = null;
let hybridLayer = null;
let edgeBand = null;

/**
 * Start hybrid mode rendering
 */
export function startHybridMode() {
    if (isActive) return;

    isActive = true;

    // Add class to body to trigger full-width layout
    document.body.classList.add('hybrid-active');

    // Create the hybrid layer (light theme overlay)
    createHybridLayer();

    // Create edge band
    createEdgeBand();

    // Start the animation loop
    updateHybridClipping();
}

/**
 * Stop hybrid mode rendering
 */
export function stopHybridMode() {
    if (!isActive) return;

    isActive = false;

    // Remove hybrid-active class from body
    document.body.classList.remove('hybrid-active');

    // Stop content sync
    if (syncInterval) {
        clearInterval(syncInterval);
        syncInterval = null;
    }

    // Cancel animation frame
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    // Remove hybrid layer
    if (hybridLayer) {
        hybridLayer.remove();
        hybridLayer = null;
    }

    // Remove edge band
    if (edgeBand) {
        edgeBand.remove();
        edgeBand = null;
    }
}

/**
 * Create the hybrid layer (light theme rendering)
 */
function createHybridLayer() {
    // Remove existing layer if present
    if (hybridLayer) {
        hybridLayer.remove();
    }

    // Create wrapper div for light theme
    hybridLayer = document.createElement('div');
    hybridLayer.className = 'hybrid-overlay';
    hybridLayer.setAttribute('data-theme', 'light');

    // Clone the app-container (not entire body to avoid duplicate scripts)
    const appContainer = document.querySelector('.app-container');
    if (!appContainer) {
        console.error('[Hybrid] No .app-container found');
        return;
    }

    const appClone = appContainer.cloneNode(true);

    // REMOVE the entire header from clone but add spacer for alignment
    const headerClone = appClone.querySelector('.header');
    let headerHeight = 0;
    if (headerClone) {
        // Get height before removing
        const originalHeader = appContainer.querySelector('.header');
        if (originalHeader) {
            headerHeight = originalHeader.offsetHeight;
        }

        // Replace with transparent spacer to maintain layout
        const spacer = document.createElement('div');
        spacer.className = 'header-spacer-hybrid';
        spacer.style.height = headerHeight + 'px';
        spacer.style.width = '100%';
        spacer.style.flexShrink = '0';
        headerClone.replaceWith(spacer);
    }

    hybridLayer.appendChild(appClone);

    // Position overlay
    hybridLayer.style.position = 'fixed';
    hybridLayer.style.top = '0';
    hybridLayer.style.left = '0';
    hybridLayer.style.width = '100vw';
    hybridLayer.style.height = '100vh';
    hybridLayer.style.pointerEvents = 'none';
    hybridLayer.style.zIndex = '998';
    hybridLayer.style.overflow = 'hidden';

    // Background will be set via CSS variables - don't override here

    document.body.appendChild(hybridLayer);

    // Start content sync to keep dynamic content (charts, agents, etc.) updated
    startContentSync();
}

/**
 * Smart content sync - only update when necessary
 */
let syncInterval = null;

function startContentSync() {
    // Sync every 500ms (less aggressive than before)
    syncInterval = setInterval(() => {
        if (!isActive || !hybridLayer) {
            if (syncInterval) {
                clearInterval(syncInterval);
                syncInterval = null;
            }
            return;
        }

        syncHybridContent();
    }, 500);
}

function syncHybridContent() {
    if (!hybridLayer) return;

    const appContainer = document.querySelector('.app-container');
    if (!appContainer) return;

    // Clone current app content
    const appClone = appContainer.cloneNode(true);

    // Remove header from clone
    const headerClone = appClone.querySelector('.header');
    let headerHeight = 0;
    if (headerClone) {
        const originalHeader = appContainer.querySelector('.header');
        if (originalHeader) {
            headerHeight = originalHeader.offsetHeight;
        }

        // Replace with spacer
        const spacer = document.createElement('div');
        spacer.className = 'header-spacer-hybrid';
        spacer.style.height = headerHeight + 'px';
        spacer.style.width = '100%';
        spacer.style.flexShrink = '0';
        headerClone.replaceWith(spacer);
    }

    // Update hybrid layer content
    hybridLayer.innerHTML = '';
    hybridLayer.appendChild(appClone);
}

/**
 * Create narrow edge band element
 */
function createEdgeBand() {
    if (edgeBand) {
        edgeBand.remove();
    }

    edgeBand = document.createElement('div');
    edgeBand.className = 'edge-band';
    edgeBand.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        z-index: 1000;
    `;

    document.body.appendChild(edgeBand);
}

/**
 * Animation loop - update clipping based on tetrahedron rotation
 */
function updateHybridClipping() {
    if (!isActive) return;

    // Get prism geometry
    const geom = window.prismGeometry;
    if (!geom) {
        animationFrameId = requestAnimationFrame(updateHybridClipping);
        return;
    }

    // Project tetrahedron face to screen space
    const splittingFace = projectSplittingFace(geom);

    // Calculate clip paths for both layers
    const { lightClip } = calculateClipPaths(splittingFace);

    // Apply clipping to light overlay
    if (hybridLayer) {
        hybridLayer.style.clipPath = lightClip;
    }

    // Update edge band to follow the splitting line
    if (edgeBand && splittingFace.length >= 2) {
        updateEdgeBand(splittingFace);
    }

    // Continue animation
    animationFrameId = requestAnimationFrame(updateHybridClipping);
}

/**
 * Update edge band to follow splitting line with narrow projected band
 * Intensity centered on prism position
 */
function updateEdgeBand(triangle) {
    if (!edgeBand) return;

    const apex = triangle[0];
    const back = triangle[1];

    // Extend line to viewport boundaries
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const extendedLine = extendLineToViewport(apex, back, vw, vh);

    // Calculate perpendicular direction for band width
    const dx = extendedLine.end.x - extendedLine.start.x;
    const dy = extendedLine.end.y - extendedLine.start.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    const perpX = -dy / len;
    const perpY = dx / len;

    // Band width (pixels on each side of line) - narrow energy emission
    const bandWidth = 17.5; // 50% reduction - tight, focused edge glow

    // Create polygon for narrow band around the line
    const p1 = {
        x: extendedLine.start.x + perpX * bandWidth,
        y: extendedLine.start.y + perpY * bandWidth
    };
    const p2 = {
        x: extendedLine.end.x + perpX * bandWidth,
        y: extendedLine.end.y + perpY * bandWidth
    };
    const p3 = {
        x: extendedLine.end.x - perpX * bandWidth,
        y: extendedLine.end.y - perpY * bandWidth
    };
    const p4 = {
        x: extendedLine.start.x - perpX * bandWidth,
        y: extendedLine.start.y - perpY * bandWidth
    };

    // Create clip-path for the narrow band
    const bandClip = `polygon(${p1.x}px ${p1.y}px, ${p2.x}px ${p2.y}px, ${p3.x}px ${p3.y}px, ${p4.x}px ${p4.y}px)`;
    edgeBand.style.clipPath = bandClip;

    // Calculate prism center position for intensity correlation
    const geom = window.prismGeometry;
    if (geom) {
        const canvas = document.getElementById('prism-canvas');
        if (canvas) {
            const canvasRect = canvas.getBoundingClientRect();
            const canvasScreenX = canvasRect.left;
            const canvasScreenY = canvasRect.top;
            const canvasDisplaySize = 140;
            const scale = canvasRect.width / canvasDisplaySize;

            // Use turbulent center if available (dark mode), otherwise stable center
            const drawCenterX = geom.turbulentCenterX !== undefined ? geom.turbulentCenterX : geom.centerX;
            const drawCenterY = geom.turbulentCenterY !== undefined ? geom.turbulentCenterY : geom.centerY;

            // Convert to screen coordinates
            const prismScreenX = canvasScreenX + drawCenterX * scale;
            const prismScreenY = canvasScreenY + drawCenterY * scale;

            // Set CSS custom properties for radial gradient center
            edgeBand.style.setProperty('--prism-x', `${prismScreenX}px`);
            edgeBand.style.setProperty('--prism-y', `${prismScreenY}px`);
        }
    }
}

/**
 * Project the splitting face (apex-back-left) to 2D screen coordinates
 * Uses EXACT same projection math as prism.js for perfect alignment
 */
function projectSplittingFace(geom) {
    const { rotX, rotY, rotZ, centerX, centerY, maxRadius, vertices, turbulentCenterX, turbulentCenterY } = geom;

    // Get actual screen position of the prism canvas
    const canvas = document.getElementById('prism-canvas');
    if (!canvas) {
        return [
            { x: 100, y: 50 },
            { x: 200, y: 150 },
            { x: 50, y: 150 }
        ];
    }

    const canvasRect = canvas.getBoundingClientRect();
    const canvasScreenX = canvasRect.left;
    const canvasScreenY = canvasRect.top;

    // Canvas display size (140px as set in prism.js)
    const canvasDisplaySize = 140;
    const scale = canvasRect.width / canvasDisplaySize;

    // Use turbulent center if available (dark mode), otherwise stable center
    const drawCenterX = turbulentCenterX !== undefined ? turbulentCenterX : centerX;
    const drawCenterY = turbulentCenterY !== undefined ? turbulentCenterY : centerY;

    // Use a face that INCLUDES the apex: apex-back-left
    // This ensures the split is attached to the "point of attention"
    const face = [
        vertices.apex,  // The point - where attention focuses
        vertices.back,  // Base vertex 1
        vertices.left   // Base vertex 2
    ];

    // Project each vertex to screen space (EXACT same math as prism.js drawPyramidEdges)
    return face.map(v => {
        // Apply 3D rotation (exact same as prism.js)
        const rotated = rotate3D(v, rotX, rotY, rotZ);

        // Apply perspective projection (exact same as prism.js)
        const perspective = 2 / (2 - rotated.z * 0.3);

        // Convert to canvas coordinates using turbulent center (CRITICAL!)
        const canvasX = drawCenterX + rotated.x * maxRadius * perspective;
        const canvasY = drawCenterY + rotated.y * maxRadius * perspective * 0.8;

        // Convert to screen coordinates
        const screenX = canvasScreenX + canvasX * scale;
        const screenY = canvasScreenY + canvasY * scale;

        return { x: screenX, y: screenY };
    });
}

/**
 * Rotate a 3D point (replicating prism.js logic)
 */
function rotate3D(point, rotX, rotY, rotZ) {
    let { x, y, z } = point;

    // Rotate around X axis
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const y1 = y * cosX - z * sinX;
    const z1 = y * sinX + z * cosX;
    y = y1;
    z = z1;

    // Rotate around Y axis
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);
    const x1 = x * cosY + z * sinY;
    const z2 = -x * sinY + z * cosY;
    x = x1;
    z = z2;

    // Rotate around Z axis
    const cosZ = Math.cos(rotZ);
    const sinZ = Math.sin(rotZ);
    const x2 = x * cosZ - y * sinZ;
    const y2 = x * sinZ + y * cosZ;

    return { x: x2, y: y2, z: z2 };
}

/**
 * Calculate clip paths for dark and light layers
 *
 * Strategy: Split along apex-back edge (touches the "point of attention")
 */
function calculateClipPaths(triangle) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Use FIXED apex-back edge (touches the apex - the point of attention)
    const apex = triangle[0];  // The point
    const back = triangle[1];  // Back vertex
    const left = triangle[2];  // Left vertex (opposite side)

    // Extend apex-back edge infinitely
    const extendedLine = extendLineToViewport(apex, back, vw, vh);

    // Build polygons for each half
    const { leftPoly, rightPoly } = splitViewportByLine(extendedLine, vw, vh);

    // Determine which side 'left' vertex is on
    const dx = back.x - apex.x;
    const dy = back.y - apex.y;
    const cross = (left.x - apex.x) * dy - (left.y - apex.y) * dx;

    // Left vertex side gets light theme (day), opposite gets dark (night)
    const flip = cross > 0;
    const darkPoly = flip ? rightPoly : leftPoly;
    const lightPoly = flip ? leftPoly : rightPoly;

    // Convert to CSS clip-path format
    const darkClip = `polygon(${darkPoly.map(p => `${p.x}px ${p.y}px`).join(', ')})`;
    const lightClip = `polygon(${lightPoly.map(p => `${p.x}px ${p.y}px`).join(', ')})`;

    return { darkClip, lightClip };
}

/**
 * Extend a line segment to viewport boundaries (infinite extension)
 */
function extendLineToViewport(p1, p2, vw, vh) {
    // Direction vector
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;

    // Normalize direction
    const len = Math.sqrt(dx * dx + dy * dy);
    const dirX = dx / len;
    const dirY = dy / len;

    // Calculate diagonal distance (max possible distance across viewport)
    const maxDist = Math.sqrt(vw * vw + vh * vh);

    // Extend MUCH farther than viewport diagonal in both directions
    const extension = maxDist * 100; // 100x viewport diagonal for true infinity

    return {
        start: {
            x: p1.x - dirX * extension,
            y: p1.y - dirY * extension
        },
        end: {
            x: p1.x + dirX * extension,
            y: p1.y + dirY * extension
        }
    };
}

/**
 * Split viewport into two polygons along a line
 */
function splitViewportByLine(line, vw, vh) {
    // Viewport corners (clockwise from top-left)
    const corners = [
        { x: 0, y: 0, id: 0 },      // top-left [0]
        { x: vw, y: 0, id: 1 },     // top-right [1]
        { x: vw, y: vh, id: 2 },    // bottom-right [2]
        { x: 0, y: vh, id: 3 }      // bottom-left [3]
    ];

    // Viewport edges
    const edges = [
        { start: corners[0], end: corners[1], id: 0 }, // top
        { start: corners[1], end: corners[2], id: 1 }, // right
        { start: corners[2], end: corners[3], id: 2 }, // bottom
        { start: corners[3], end: corners[0], id: 3 }  // left
    ];

    // Find intersections with viewport edges
    const intersections = [];
    for (const edge of edges) {
        const int = lineIntersection(line.start, line.end, edge.start, edge.end);
        if (int) {
            intersections.push({ x: int.x, y: int.y, edgeId: edge.id });
        }
    }

    // Must have exactly 2 intersections for a proper split
    if (intersections.length !== 2) {
        // Fallback: vertical split
        return {
            leftPoly: [
                { x: 0, y: 0 },
                { x: vw/2, y: 0 },
                { x: vw/2, y: vh },
                { x: 0, y: vh }
            ],
            rightPoly: [
                { x: vw/2, y: 0 },
                { x: vw, y: 0 },
                { x: vw, y: vh },
                { x: vw/2, y: vh }
            ]
        };
    }

    // Sort intersections by edge ID
    intersections.sort((a, b) => a.edgeId - b.edgeId);
    const int1 = intersections[0];
    const int2 = intersections[1];

    // Build first polygon: int1 -> corners -> int2
    const poly1 = [int1];
    for (let i = int1.edgeId + 1; i <= int2.edgeId; i++) {
        poly1.push(corners[i]);
    }
    poly1.push(int2);

    // Build second polygon: int2 -> corners -> int1 (wrapping around)
    const poly2 = [int2];
    for (let i = int2.edgeId + 1; i <= int1.edgeId + 4; i++) {
        poly2.push(corners[i % 4]);
    }
    poly2.push(int1);

    return { leftPoly: poly1, rightPoly: poly2 };
}

/**
 * Calculate line-line intersection
 */
function lineIntersection(p1, p2, p3, p4) {
    const x1 = p1.x, y1 = p1.y;
    const x2 = p2.x, y2 = p2.y;
    const x3 = p3.x, y3 = p3.y;
    const x4 = p4.x, y4 = p4.y;
    
    const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(denom) < 0.0001) return null;
    
    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;
    
    if (u >= 0 && u <= 1) {
        return {
            x: x1 + t * (x2 - x1),
            y: y1 + t * (y2 - y1)
        };
    }
    
    return null;
}

