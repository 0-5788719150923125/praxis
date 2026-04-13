// Prism Ball Animation for Logo - Full Featured Version
//
// Shape-shifting based on theme:
// - DARK MODE: Tetrahedron (prism) - the original quantum echo
// - LIGHT MODE: 2x4 Board - "We build during the day"
//
// The 2x4 pattern references the Delta-8 architecture (2 experts × 4 iterations)
// and the construction metaphor from docs/exponentials.md:
//   "You know what carpenters call a 2×4? A board. You know what you build with? Boards."
//
// The spinning, flying 2x4 board is both hilarious and profound - it's a literal
// construction tool oscillating through 3D space, just like the network oscillates
// between expert perspectives to "build" understanding.
//
// Wait for canvas to be available (created by renderAppStructure)
(function initPrismWhenReady() {
    const canvas = document.getElementById('prism-canvas');
    if (!canvas) {
        // Canvas not ready yet, try again in a moment
        requestAnimationFrame(initPrismWhenReady);
        return;
    }

    const ctx = canvas.getContext('2d');

    // High-DPI canvas setup for zoom resistance
    // Always render at very high resolution to prevent pixelation
    const baseSize = 140; // Canvas render size (25% larger than container)
    const renderScale = 4; // Render at 4x resolution for crisp zooming
    const renderSize = baseSize * renderScale;

    // Set actual canvas size to high resolution
    canvas.width = renderSize;
    canvas.height = renderSize;

    // Set display size to match the larger canvas (will overflow container slightly)
    canvas.style.width = baseSize + 'px';
    canvas.style.height = baseSize + 'px';

    // Scale context to match high resolution
    ctx.scale(renderScale, renderScale);

    // Use baseSize for all drawing coordinates (the actual canvas size)
    const centerX = baseSize / 2;
    const centerY = baseSize / 2;
    const maxRadius = baseSize * 0.42;

    // 3D rotation angles - GLOBAL for proper pyramid rotation
    let globalRotX = 0;
    let globalRotY = 0;
    let globalRotZ = 0;
    let time = 0;

    // Expose geometry for hybrid mode rendering
    window.prismGeometry = {
        rotX: 0,
        rotY: 0,
        rotZ: 0,
        centerX: centerX,
        centerY: centerY,
        maxRadius: maxRadius,
        // Tetrahedron vertices (will be updated each frame)
        vertices: {
            apex: {x: 0, y: -0.5, z: 0},
            back: {x: 0, y: 0.5, z: -0.577},
            left: {x: -0.5, y: 0.5, z: 0.289},
            right: {x: 0.5, y: 0.5, z: 0.289}
        }
    };

    // Rotation velocities for full 3D rotation
    let rotVelX = 0.003 + Math.random() * 0.002;
    let rotVelY = 0.005 + Math.random() * 0.002;
    let rotVelZ = 0.002 + Math.random() * 0.001;

    // Motion behavior system
    const motionBehaviors = {
        SPINNING: 'spinning',      // Current default behavior
        FLYING: 'flying',          // Smooth forward motion with banking
        SOARING: 'soaring',        // Gentle gliding with lift
        SWOOPING: 'swooping',      // Dramatic diving and climbing
        PEACEFUL: 'peaceful',      // Slow, stable rotation
        HOVERING: 'hovering',      // Nearly still with micro-movements
        TUMBLING: 'tumbling'       // Chaotic tumbling motion
    };

    let currentBehavior = motionBehaviors.SPINNING;
    let behaviorStartTime = 0;
    let behaviorDuration = 5 + Math.random() * 5; // 5-10 seconds per behavior
    let nextBehavior = null;
    let transitionProgress = 0;
    let transitionDuration = 1.5; // 1.5 second transitions

    // Target velocities for smooth transitions
    let targetVelX = rotVelX;
    let targetVelY = rotVelY;
    let targetVelZ = rotVelZ;

    // Morphing parameters for gentle tetrahedron distortion
    let morphPhase = 0;
    let morphSpeed = 0.01;

    // Wind system for directional surges
    let windState = {
        direction: { x: 0, y: 0, z: 0 },
        strength: 0,
        gustTime: 0,
        gustDuration: 0,
        targetDirection: { x: 0, y: 0, z: 0 },
        currentDirection: { x: 0, y: 0, z: 0 }
    };

    // Turbulence system - dark mode only
    // RANDOM, INFREQUENT jolting - completely independent of hyperactivity
    // Like hitting air pockets - rare, unpredictable, jarring
    let turbulenceState = {
        active: false,
        intensity: 0,
        startTime: 0,
        duration: 0,
        nextCheck: 20 + Math.random() * 30, // First check at 20-50 seconds (rare!)
        // Rotational jolts (spinning off axis)
        joltRotX: 0,
        joltRotY: 0,
        joltRotZ: 0,
        // Translational jolts (spatial displacement)
        offsetX: 0,
        offsetY: 0,
        decay: 0.95,
        lastJoltTime: 0
    };

    // Propulsion state - separate from turbulence
    // Accumulates during hyperactivity, decays when calm
    let propulsionState = {
        velocityX: 0,  // Current velocity (accumulates during acceleration)
        velocityY: 0,
        positionX: 0,  // Displaced position from anchor
        positionY: 0,
        isActive: false,
        rampPhase: 0  // 0-1 during acceleration ramp
    };

    // Rubber band anchor system - pulls prism back to center
    // The further from center, the stronger the pull (Hooke's law)
    const anchorStrength = 0.12; // Spring constant (stronger for dark mode)

    // Update propulsion physics - like a car accelerating
    // Brief ramp up, then cruising at displaced position, fighting anchor
    function updatePropulsion(tendrils, lightMode, dt = 0.016) {
        /**
         * Propulsion system with acceleration physics.
         *
         * DARK MODE ONLY:
         * - Hyperactive mode (>30 tendrils) acts like engines firing
         * - Brief acceleration ramp (like car accelerating)
         * - Cruising: settled at displaced position, straining against anchor
         * - Anchor pulls back harder as distance increases (diminishing returns)
         * - When hyperactivity ends, prism decelerates and returns to center
         *
         * LIGHT MODE: Disabled
         */

        // Light mode: no propulsion
        if (lightMode) {
            // Decay any existing propulsion state
            propulsionState.velocityX *= 0.9;
            propulsionState.velocityY *= 0.9;
            propulsionState.positionX *= 0.9;
            propulsionState.positionY *= 0.9;
            propulsionState.isActive = false;
            return { x: 0, y: 0 };
        }

        const tendrilCount = tendrils.length;
        const isHyperactive = tendrilCount > 30;

        if (isHyperactive) {
            // HYPERACTIVE: Engines firing!
            if (!propulsionState.isActive) {
                // Just entered hyperactive mode - start acceleration ramp
                propulsionState.isActive = true;
                propulsionState.rampPhase = 0;
            }

            // Acceleration ramp (brief, like car accelerating)
            if (propulsionState.rampPhase < 1) {
                propulsionState.rampPhase = Math.min(1, propulsionState.rampPhase + 0.02); // Ramp over ~50 frames (~0.8 seconds)
            }

            // Calculate thrust direction from tendril aggregate
            let thrustDirX = 0;
            let thrustDirY = 0;
            let sampleCount = 0;

            const sampleStep = Math.max(1, Math.floor(tendrilCount / 20));
            for (let i = 0; i < tendrilCount; i += sampleStep) {
                const tendril = tendrils[i];
                if (tendril.opacity > 0.2) {
                    thrustDirX += tendril.baseDirection.x * tendril.opacity;
                    thrustDirY += tendril.baseDirection.y * tendril.opacity;
                    sampleCount++;
                }
            }

            if (sampleCount > 0) {
                thrustDirX /= sampleCount;
                thrustDirY /= sampleCount;

                const mag = Math.sqrt(thrustDirX * thrustDirX + thrustDirY * thrustDirY);
                if (mag > 0) {
                    thrustDirX /= mag;
                    thrustDirY /= mag;
                }
            }

            // Polynomial power growth (x^2.5 for smooth but strong acceleration)
            const hyperactivityFactor = Math.min(1, (tendrilCount - 30) / 70);
            const thrustPower = Math.pow(hyperactivityFactor, 2.5); // Polynomial growth

            // Smooth acceleration ramp (no jarring - pure stable stretching)
            // Power increases during ramp, then sustains
            const rampCurve = Math.sin(propulsionState.rampPhase * Math.PI / 2); // Smooth ease-in curve
            const maxAccel = 2.0; // Strong but stable
            const accel = thrustPower * maxAccel * rampCurve;

            // Update velocity SMOOTHLY (no jarring)
            propulsionState.velocityX += thrustDirX * accel;
            propulsionState.velocityY += thrustDirY * accel;

            // Strong damping for stability (prevents jarring oscillations)
            const damping = 0.94;
            propulsionState.velocityX *= damping;
            propulsionState.velocityY *= damping;

        } else {
            // NOT HYPERACTIVE: Deceleration phase
            if (propulsionState.isActive) {
                propulsionState.isActive = false;
                propulsionState.rampPhase = 0;
            }

            // Decelerate smoothly (like car coasting to stop)
            propulsionState.velocityX *= 0.93;
            propulsionState.velocityY *= 0.93;
        }

        // Update position based on velocity
        propulsionState.positionX += propulsionState.velocityX;
        propulsionState.positionY += propulsionState.velocityY;

        // Apply rubber band anchor force to position
        // This creates the tension - position wants to grow, anchor pulls back
        const anchorPull = 0.08; // Restoring force strength
        propulsionState.positionX -= propulsionState.positionX * anchorPull;
        propulsionState.positionY -= propulsionState.positionY * anchorPull;

        return {
            x: propulsionState.positionX,
            y: propulsionState.positionY
        };
    }

    function applyRubberBandAnchor(offsetX, offsetY, lightMode) {
        /**
         * Rubber band physics: pulls prism back to center.
         * The further from center, the stronger the restoring force (Hooke's law).
         *
         * F = -k * displacement
         *
         * DARK MODE: Stronger anchor to counteract exponential propulsion
         * - Creates visible tension against exponential tendril thrust
         * - Diminishing returns: can pull away but anchor fights back harder
         * - The ship strains against its chain but cannot escape
         *
         * LIGHT MODE: No anchor needed (no propulsion to fight)
         *
         * This creates the "chained to a point in space and time" feeling.
         */

        // Stronger anchor in dark mode to counteract exponential propulsion
        const k = lightMode ? anchorStrength : anchorStrength * 1.8;

        // Calculate restoring force (towards center)
        // Force increases linearly with distance (Hooke's law)
        const restoreX = -offsetX * k;
        const restoreY = -offsetY * k;

        return { x: restoreX, y: restoreY };
    }

    function updateWind(time) {
        // Random gusts every few seconds
        if (time - windState.gustTime > windState.gustDuration) {
            // New gust
            windState.gustTime = time;
            windState.gustDuration = 2 + Math.random() * 3; // 2-5 seconds

            // Random direction (favoring horizontal for dandelion effect)
            const angle = Math.random() * Math.PI * 2;
            windState.targetDirection = {
                x: Math.cos(angle),
                y: (Math.random() - 0.5) * 0.2, // less vertical movement
                z: Math.sin(angle)
            };
            windState.strength = 0.3 + Math.random() * 0.7;
        }

        // Smooth transition between wind directions
        const transitionSpeed = 0.05;
        windState.currentDirection.x += (windState.targetDirection.x - windState.currentDirection.x) * transitionSpeed;
        windState.currentDirection.y += (windState.targetDirection.y - windState.currentDirection.y) * transitionSpeed;
        windState.currentDirection.z += (windState.targetDirection.z - windState.currentDirection.z) * transitionSpeed;

        // Fade in/out wind strength
        const gustProgress = (time - windState.gustTime) / windState.gustDuration;
        const envelope = Math.sin(gustProgress * Math.PI); // smooth in/out

        return {
            x: windState.currentDirection.x * windState.strength * envelope,
            y: windState.currentDirection.y * windState.strength * envelope,
            z: windState.currentDirection.z * windState.strength * envelope,
            strength: windState.strength * envelope
        };
    }

    function updateTurbulence(time) {
        /**
         * Turbulence system - DARK MODE ONLY
         *
         * Simulates aircraft turbulence: periodic, random, infrequent jolts that
         * jostle the prism off its rotational axis. Creates waves of turbulence
         * that complement (don't replace) the existing rotation behaviors.
         *
         * Think: flying through rough air - sudden jolts, recovery, more jolts
         */

        // Only activate in dark mode
        if (isLightMode()) {
            turbulenceState.active = false;
            turbulenceState.joltRotX *= 0.8; // Quick decay if switching modes
            turbulenceState.joltRotY *= 0.8;
            turbulenceState.joltRotZ *= 0.8;
            turbulenceState.offsetX *= 0.8;
            turbulenceState.offsetY *= 0.8;
            return { rotX: 0, rotY: 0, rotZ: 0, offsetX: 0, offsetY: 0 };
        }

        // Check if it's time to consider turbulence (completely independent of surges/hyperactivity)
        if (!turbulenceState.active && time >= turbulenceState.nextCheck) {
            // Lower chance - turbulence is RARE (20% when checked)
            if (Math.random() < 0.2) {
                // TURBULENCE TRIGGERED!
                turbulenceState.active = true;
                turbulenceState.startTime = time;
                turbulenceState.duration = 3 + Math.random() * 5; // 3-8 seconds of sustained jolts
                turbulenceState.intensity = 0.6 + Math.random() * 0.4; // 0.6-1.0 intensity
                turbulenceState.lastJoltTime = time;
            }

            // Schedule next check (20-50 seconds from now - INFREQUENT)
            turbulenceState.nextCheck = time + 20 + Math.random() * 30;
        }

        // Apply turbulence jolts if active
        if (turbulenceState.active) {
            const elapsed = time - turbulenceState.startTime;

            // End turbulence after duration
            if (elapsed > turbulenceState.duration) {
                turbulenceState.active = false;
                turbulenceState.intensity = 0;
            } else {
                // Sustained turbulence: continuous random jolts of varying magnitude
                // Like flying through asteroid field - constant bumping
                const timeSinceLastJolt = time - turbulenceState.lastJoltTime;

                // Variable jolt frequency: sometimes rapid fire, sometimes sparse
                const joltInterval = 0.05 + Math.random() * 0.15; // 50-200ms between jolts

                if (timeSinceLastJolt > joltInterval) {
                    turbulenceState.lastJoltTime = time;

                    // Mix of small bumps (70%) and larger jolts (30%)
                    const isLargeJolt = Math.random() < 0.3;
                    const magnitude = isLargeJolt ?
                        turbulenceState.intensity * (1.5 + Math.random()) :  // Large jolt: 1.5-2.5x
                        turbulenceState.intensity * (0.3 + Math.random() * 0.4); // Small bump: 0.3-0.7x

                    // Rotational jolts (spinning off axis)
                    const joltRotMagnitude = magnitude * (0.015 + Math.random() * 0.02);
                    turbulenceState.joltRotX += (Math.random() - 0.5) * joltRotMagnitude;
                    turbulenceState.joltRotY += (Math.random() - 0.5) * joltRotMagnitude;
                    turbulenceState.joltRotZ += (Math.random() - 0.5) * joltRotMagnitude;

                    // Spatial displacement jolts (fuzzy anchor - actually moves position)
                    const joltPosMagnitude = magnitude * (10 + Math.random() * 15);
                    turbulenceState.offsetX += (Math.random() - 0.5) * joltPosMagnitude;
                    turbulenceState.offsetY += (Math.random() - 0.5) * joltPosMagnitude;
                }

                // Envelope: fade in/out turbulence intensity over duration
                const progress = elapsed / turbulenceState.duration;
                const envelope = Math.sin(progress * Math.PI); // Smooth in, smooth out

                // Apply decay to all jolts (they naturally settle)
                turbulenceState.joltRotX *= turbulenceState.decay;
                turbulenceState.joltRotY *= turbulenceState.decay;
                turbulenceState.joltRotZ *= turbulenceState.decay;
                turbulenceState.offsetX *= turbulenceState.decay;
                turbulenceState.offsetY *= turbulenceState.decay;

                return {
                    rotX: turbulenceState.joltRotX * envelope,
                    rotY: turbulenceState.joltRotY * envelope,
                    rotZ: turbulenceState.joltRotZ * envelope,
                    offsetX: turbulenceState.offsetX * envelope,
                    offsetY: turbulenceState.offsetY * envelope,
                    intensity: turbulenceState.intensity * envelope
                };
            }
        }

        // Decay jolts even when not active (smooth return to calm)
        turbulenceState.joltRotX *= 0.85;
        turbulenceState.joltRotY *= 0.85;
        turbulenceState.joltRotZ *= 0.85;
        turbulenceState.offsetX *= 0.85;
        turbulenceState.offsetY *= 0.85;

        return { rotX: 0, rotY: 0, rotZ: 0, offsetX: 0, offsetY: 0, intensity: 0 };
    }

    // Tendril class for each electric beam
    class Tendril {
        constructor(index) {
            this.index = index;
            this.reset();
        }

        reset() {
            // Origin at the true center of the shape (0, 0, 0)
            this.origin = {x: 0, y: 0, z: 0};

            // Check if we're in light mode for 2x4 board targets
            const lightMode = isLightMode();

            // 25% chance to target edges/vertices for better shape definition
            if (Math.random() < 0.25) {
                let targets;

                if (lightMode) {
                    // Target 2x4 board vertices and edge midpoints
                    // Using same dimensions as get2x4Edges
                    const w = 0.45, h = 1.575, d = 0.225;
                    targets = [
                        // 8 corners of the board
                        {x: -w/2, y: -h/2, z: d/2},   // front top left
                        {x: w/2, y: -h/2, z: d/2},    // front top right
                        {x: -w/2, y: h/2, z: d/2},    // front bottom left
                        {x: w/2, y: h/2, z: d/2},     // front bottom right
                        {x: -w/2, y: -h/2, z: -d/2},  // back top left
                        {x: w/2, y: -h/2, z: -d/2},   // back top right
                        {x: -w/2, y: h/2, z: -d/2},   // back bottom left
                        {x: w/2, y: h/2, z: -d/2},    // back bottom right
                        // Edge midpoints for better coverage
                        {x: 0, y: -h/2, z: d/2},      // front top edge center
                        {x: 0, y: h/2, z: d/2},       // front bottom edge center
                        {x: -w/2, y: 0, z: d/2},      // front left edge center
                        {x: w/2, y: 0, z: d/2}        // front right edge center
                    ];
                } else {
                    // Target tetrahedron vertices and edges (dark mode)
                    targets = [
                        // Base triangle corners (centered tetrahedron)
                        {x: 0, y: 0.5, z: -0.577},      // back vertex
                        {x: -0.5, y: 0.5, z: 0.289},     // left vertex
                        {x: 0.5, y: 0.5, z: 0.289},      // right vertex
                        // Apex
                        {x: 0, y: -0.5, z: 0},
                        // Edge midpoints for better edge coverage
                        {x: -0.25, y: 0.5, z: -0.144},   // back-left edge
                        {x: 0.25, y: 0.5, z: -0.144},    // back-right edge
                        {x: 0, y: 0.5, z: 0.289},        // left-right edge
                        {x: 0, y: 0, z: -0.289},         // apex edges
                        {x: -0.25, y: 0, z: 0.144},
                        {x: 0.25, y: 0, z: 0.144}
                    ];
                }

                const target = targets[Math.floor(Math.random() * targets.length)];

                // Add some noise to avoid perfect lines
                this.baseDirection = {
                    x: target.x + (Math.random() - 0.5) * 0.3,
                    y: target.y + (Math.random() - 0.5) * 0.3,
                    z: target.z + (Math.random() - 0.5) * 0.3
                };
            } else {
                // Generate random 3D direction for organic look
                const phi = Math.random() * Math.PI * 2; // Azimuth
                const theta = (Math.random() - 0.5) * Math.PI; // Full sphere coverage

                this.baseDirection = {
                    x: Math.cos(theta) * Math.cos(phi),
                    y: Math.sin(theta),
                    z: Math.cos(theta) * Math.sin(phi)
                };
            }

            // Normalize direction
            const mag = Math.sqrt(
                this.baseDirection.x ** 2 +
                this.baseDirection.y ** 2 +
                this.baseDirection.z ** 2
            );
            this.baseDirection.x /= mag;
            this.baseDirection.y /= mag;
            this.baseDirection.z /= mag;

            this.lifetime = 0;
            this.maxLifetime = 30 + Math.random() * 40;
            this.opacity = 0;
            this.growing = true;
            this.thickness = 0.5 + Math.random() * 1.5;
            this.waveSpeed = 0.02 + Math.random() * 0.02;
            this.waveAmount = 0.02 + Math.random() * 0.02;
            this.color = this.generateColor();
            this.phaseOffset = Math.random() * Math.PI * 2;
            this.lengthProgress = 0;
            this.state = 'growing';
            this.escapeWorldAnchor = null;
            this.willEscape = Math.random() < 0.3; // 30% chance of escaping
        }

        generateColor() {
            const colors = [
                { r: 34, g: 139, b: 34 },   // Forest green
                { r: 0, g: 255, b: 127 },    // Spring green
                { r: 50, g: 205, b: 50 },    // Lime green
                { r: 124, g: 252, b: 0 },    // Lawn green
                { r: 173, g: 255, b: 47 },   // Green yellow
                { r: 144, g: 238, b: 144 },  // Light green
                { r: 152, g: 251, b: 152 },  // Pale green
            ];
            return colors[Math.floor(Math.random() * colors.length)];
        }

        // Detect proximity to 2x4 board edges for enhanced glow (light mode)
        detect2x4EdgeProximity(point3D) {
            // 2x4 box dimensions - using same dimensions
            const w = 0.45, h = 1.575, d = 0.225;
            const bounds = {
                min: {x: -w/2, y: -h/2, z: -d/2},
                max: {x: w/2, y: h/2, z: d/2}
            };

            // Distance to each face of the box
            const distX = Math.min(
                Math.abs(point3D.x - bounds.min.x),
                Math.abs(point3D.x - bounds.max.x)
            );
            const distY = Math.min(
                Math.abs(point3D.y - bounds.min.y),
                Math.abs(point3D.y - bounds.max.y)
            );
            const distZ = Math.min(
                Math.abs(point3D.z - bounds.min.z),
                Math.abs(point3D.z - bounds.max.z)
            );

            // Minimum distance to any face
            const minDist = Math.min(distX, distY, distZ);

            return {
                distance: minDist,
                isNearEdge: minDist < 0.08,
                glowIntensity: Math.max(0, 1 - minDist / 0.08)
            };
        }

        // Detect proximity to pyramid edges for enhanced glow (dark mode)
        detectEdgeProximity(point3D) {
            // Calculate distance to pyramid boundaries
            const yNorm = point3D.y;
            const maxExtent = (yNorm + 1) / 2; // pyramid width at this height

            // Distance to nearest edge in x and z directions
            const xEdgeDist = maxExtent - Math.abs(point3D.x);
            const zEdgeDist = maxExtent - Math.abs(point3D.z);
            const minEdgeDist = Math.min(xEdgeDist, zEdgeDist);

            // Also check distance to apex edges
            const distToApex = Math.sqrt(point3D.x * point3D.x + point3D.z * point3D.z);
            const apexEdgeDist = Math.abs(distToApex - Math.abs(yNorm + 1) * 0.5);

            const finalDist = Math.min(minEdgeDist, apexEdgeDist);

            return {
                distance: finalDist,
                isNearEdge: finalDist < 0.15, // threshold for "near edge"
                glowIntensity: Math.max(0, 1 - finalDist / 0.15)
            };
        }

        // Rotate a 3D point
        rotate3D(point, rotX, rotY, rotZ) {
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

        unrotate3D(point, rotX, rotY, rotZ) {
            let { x, y, z } = point;

            // Reverse Z-axis rotation
            let cos = Math.cos(-rotZ);
            let sin = Math.sin(-rotZ);
            let tempX = x * cos - y * sin;
            let tempY = x * sin + y * cos;
            x = tempX;
            y = tempY;

            // Reverse Y-axis rotation
            cos = Math.cos(-rotY);
            sin = Math.sin(-rotY);
            tempX = x * cos + z * sin;
            let tempZ = -x * sin + z * cos;
            x = tempX;
            z = tempZ;

            // Reverse X-axis rotation
            cos = Math.cos(-rotX);
            sin = Math.sin(-rotX);
            tempY = y * cos - z * sin;
            tempZ = y * sin + z * cos;
            y = tempY;
            z = tempZ;

            return { x, y, z };
        }

        // Ray-box intersection for 2x4 board (light mode)
        find2x4Intersection(direction) {
            // Simple AABB (axis-aligned bounding box) ray intersection
            // 2x4 dimensions - long but not TOO unwieldy
            const w = 0.45, h = 1.575, d = 0.225;
            const bounds = {
                min: {x: -w/2, y: -h/2, z: -d/2},
                max: {x: w/2, y: h/2, z: d/2}
            };

            let maxT = 0.8; // Max tendril length

            // March along ray and check if we're inside the box
            for (let t = 0.02; t <= 1.0; t += 0.03) {
                const point = {
                    x: this.origin.x + direction.x * t,
                    y: this.origin.y + direction.y * t,
                    z: this.origin.z + direction.z * t
                };

                // Check if outside box bounds
                if (point.x < bounds.min.x || point.x > bounds.max.x ||
                    point.y < bounds.min.y || point.y > bounds.max.y ||
                    point.z < bounds.min.z || point.z > bounds.max.z) {
                    maxT = Math.min(maxT, t * 0.9);
                    break;
                }
            }

            return Math.min(maxT, 0.8);
        }

        // Ray-tetrahedron intersection in LOCAL space (dark mode)
        findPyramidIntersection(direction) {
            // Cache the result for this direction
            if (this.cachedIntersection &&
                this.cachedDirection === direction) {
                return this.cachedIntersection;
            }

            let maxT = 0.8; // Reduced max distance to keep tendrils more contained

            // More accurate intersection test with tighter bounds
            for (let t = 0.02; t <= 1.5; t += 0.03) { // Slightly finer steps
                const point = {
                    x: this.origin.x + direction.x * t,
                    y: this.origin.y + direction.y * t,
                    z: this.origin.z + direction.z * t
                };

                // Stricter tetrahedron bounds check
                if (point.y < -0.48 || point.y > 0.48) { // Slightly tighter bounds
                    maxT = Math.min(maxT, t * 0.9); // Scale down to keep inside
                    break;
                }

                // More accurate triangular base check
                const height = (point.y + 0.5); // 0 at apex, 1 at base
                if (height > 0.01) { // Small epsilon to avoid division issues
                    const scale = height * 0.95; // Slightly smaller scale for better containment
                    const scaledX = point.x / scale;
                    const scaledZ = point.z / scale;

                    // Tighter triangular boundary for better containment
                    if (scaledZ < -0.55 || scaledZ > 0.27 ||
                        Math.abs(scaledX) > 0.48) {
                        maxT = Math.min(maxT, t * 0.85); // Further reduce to stay inside
                        break;
                    }
                }
            }

            // Cache the result
            this.cachedDirection = direction;
            this.cachedIntersection = Math.min(maxT, 0.8);
            return this.cachedIntersection;
        }

        update() {
            this.lifetime++;

            // Grow the tendril quickly
            if (this.lengthProgress < 1) {
                this.lengthProgress = Math.min(1, this.lengthProgress + 0.1);
            }

            // Faster fade in/out for mass spawns
            if (this.growing) {
                this.opacity = Math.min(0.6, this.opacity + 0.05);
                if (this.lifetime > this.maxLifetime * 0.6) {
                    this.growing = false;
                }
            } else {
                this.opacity = Math.max(0, this.opacity - 0.03);
                if (this.opacity <= 0) {
                    this.reset();
                }
            }
        }

        draw(shadowPass = false, drawCenterX = centerX, drawCenterY = centerY) {
            // Find intersection with shape in LOCAL space
            // Use 2x4 box in light mode, tetrahedron in dark mode
            const lightMode = isLightMode();
            const maxLength = lightMode ?
                this.find2x4Intersection(this.baseDirection) :
                this.findPyramidIntersection(this.baseDirection);

            // AT ESCAPE: Capture anchor point
            if (this.state === 'growing' && this.lengthProgress >= 1 && this.willEscape) {
                this.state = 'escaped';
                const localTip = {
                    x: this.origin.x + this.baseDirection.x * maxLength,
                    y: this.origin.y + this.baseDirection.y * maxLength,
                    z: this.origin.z + this.baseDirection.z * maxLength
                };
                // The crucial step: transform to world space and store it
                this.escapeWorldAnchor = this.rotate3D(localTip, globalRotX, globalRotY, globalRotZ);
            }

            // Generate the tendril path
            const segments = [];
            const segmentCount = 15; // Good segment count for detail
            let maxEdgeGlow = 0; // Track maximum edge proximity for this tendril

            let pathDirection = this.baseDirection;
            let pathLength = maxLength * this.lengthProgress;

            if (this.state === 'escaped' && this.escapeWorldAnchor) {
                const localAnchor = this.unrotate3D(this.escapeWorldAnchor, globalRotX, globalRotY, globalRotZ);
                const dirToAnchor = {
                    x: localAnchor.x - this.origin.x,
                    y: localAnchor.y - this.origin.y,
                    z: localAnchor.z - this.origin.z
                };
                const mag = Math.sqrt(dirToAnchor.x ** 2 + dirToAnchor.y ** 2 + dirToAnchor.z ** 2);
                if (mag > 0) {
                    pathDirection = { x: dirToAnchor.x / mag, y: dirToAnchor.y / mag, z: dirToAnchor.z / mag };
                }
                pathLength = mag;
            }

            for (let i = 0; i <= segmentCount; i++) {
                const t = i / segmentCount;
                const adjustedT = t;

                // Add subtle wave motion
                const wave1 = Math.sin(this.lifetime * this.waveSpeed + t * 5 + this.phaseOffset) * this.waveAmount;
                const wave2 = Math.cos(this.lifetime * this.waveSpeed * 0.7 + t * 4) * this.waveAmount * 0.7;

                // Calculate position in LOCAL space (from origin at center)
                const localPos = {
                    x: this.origin.x + pathDirection.x * adjustedT * pathLength + wave1 * (1 - t * 0.5),
                    y: this.origin.y + pathDirection.y * adjustedT * pathLength + wave2 * (1 - t * 0.5),
                    z: this.origin.z + pathDirection.z * adjustedT * pathLength + wave1 * 0.3 * (1 - t * 0.5)
                };

                let edgeGlow = 0;
                if (this.state !== 'escaped') {
                    // Check edge proximity for enhanced effects
                    // Use correct detection based on light mode
                    const lightMode = isLightMode();
                    const edgeInfo = lightMode ?
                        this.detect2x4EdgeProximity(localPos) :
                        this.detectEdgeProximity(localPos);
                    maxEdgeGlow = Math.max(maxEdgeGlow, edgeInfo.glowIntensity);
                    edgeGlow = edgeInfo.glowIntensity;
                }

                // Apply GLOBAL pyramid rotation
                const worldPos = this.rotate3D(localPos, globalRotX, globalRotY, globalRotZ);

                // Project to 2D with perspective
                const perspective = 2 / (2 - worldPos.z * 0.3);
                const x2D = drawCenterX + worldPos.x * maxRadius * perspective;
                const y2D = drawCenterY + worldPos.y * maxRadius * perspective * 0.8;

                segments.push({
                    x: x2D,
                    y: y2D,
                    z: worldPos.z,
                    t: t,
                    edgeGlow: edgeGlow
                });
            }

            // Draw the tendril
            if (segments.length < 2) return;

            ctx.beginPath();
            ctx.moveTo(segments[0].x, segments[0].y);

            // Draw smooth curve
            for (let i = 1; i < segments.length - 1; i++) {
                const xc = (segments[i].x + segments[i + 1].x) / 2;
                const yc = (segments[i].y + segments[i + 1].y) / 2;
                ctx.quadraticCurveTo(segments[i].x, segments[i].y, xc, yc);
            }

            const last = segments[segments.length - 1];
            ctx.lineTo(last.x, last.y);

            if (shadowPass) {
                // Shadow rendering for light mode - very subtle dark glow
                ctx.strokeStyle = `rgba(0, 0, 0, ${this.opacity * 0.25})`; // Much more transparent
                ctx.lineWidth = (this.thickness * (1 - last.t * 0.3)) + 1.5; // Much thinner outline
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                // Softer, subtler blur
                ctx.shadowBlur = 4;
                ctx.shadowColor = `rgba(0, 0, 0, ${this.opacity * 0.15})`; // Very light shadow
                ctx.stroke();
                ctx.shadowBlur = 0;
            } else {
                // Normal rendering
                const gradient = ctx.createLinearGradient(
                    segments[0].x, segments[0].y,
                    last.x, last.y
                );

                gradient.addColorStop(0, `rgba(255, 255, 255, ${this.opacity * 0.9})`);
                gradient.addColorStop(0.3, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.opacity})`);
                gradient.addColorStop(0.8, `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.opacity * 0.7})`);
                gradient.addColorStop(1, `rgba(255, 255, 255, ${this.opacity * 0.5})`);

                ctx.strokeStyle = gradient;
                ctx.lineWidth = this.thickness * (1 - last.t * 0.3);
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';

                // Shadow effects for edge proximity
                const isSurge = tendrils.length > 30;
                if (isSurge && maxEdgeGlow > 0.5) {
                    // Only apply shadow during significant edge proximity
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = `rgba(${this.color.r}, ${this.color.g}, ${this.color.b}, ${this.opacity * 0.4})`;
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                } else {
                    ctx.stroke();
                }

                // Endpoint spot
                if (this.lengthProgress > 0.9 && this.thickness > 1 && this.opacity > 0.3) {
                    ctx.beginPath();
                    ctx.arc(last.x, last.y, 1.5, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity * 0.4})`;
                    ctx.fill();
                }
            }
        }
    }

    // Tendril management
    const minTendrils = 3;
    const normalTendrils = 12;
    const surgeTendrils = 50;
    const maxTendrils = 200; // Reduced for logo size
    const tendrils = [];

    // Initialize with a few tendrils
    for (let i = 0; i < minTendrils; i++) {
        tendrils.push(new Tendril(i));
    }

    let tendrilIndex = minTendrils;
    let lastSurgeTime = -1000;
    let frameCount = 0;

    // 2x4 Board edge definitions (for light mode)
    // "We build during the day" - a literal construction board
    function get2x4Edges(morphFactor = 0) {
        // Gentle morphing for the board (less dramatic than tetrahedron)
        const morph1 = Math.sin(morphPhase) * morphFactor;
        const morph2 = Math.cos(morphPhase * 1.3) * morphFactor;
        const morph3 = Math.sin(morphPhase * 0.7) * morphFactor;

        // 2x4 board dimensions - just right for unwieldiness
        // Real 2x4: 1.5" x 3.5" (finished dimensions)
        // Made longer and more board-like for visual impact
        const w = 0.45;   // Width (the "2" in 2x4)
        const h = 1.575;  // Height (the "4" in 2x4) - nice long board
        const d = 0.225;  // Depth (thickness)

        // Define 8 vertices of the rectangular board (centered at origin)
        const vertices = {
            // Front face (z = +d/2)
            ftl: {x: -w/2 + morph1 * 0.03, y: -h/2 + morph2 * 0.03, z: d/2},   // front top left
            ftr: {x: w/2 + morph2 * 0.03, y: -h/2 + morph3 * 0.03, z: d/2},    // front top right
            fbl: {x: -w/2 + morph3 * 0.03, y: h/2 + morph1 * 0.03, z: d/2},    // front bottom left
            fbr: {x: w/2 + morph1 * 0.03, y: h/2 + morph2 * 0.03, z: d/2},     // front bottom right

            // Back face (z = -d/2)
            btl: {x: -w/2 + morph2 * 0.03, y: -h/2 + morph1 * 0.03, z: -d/2},  // back top left
            btr: {x: w/2 + morph3 * 0.03, y: -h/2 + morph2 * 0.03, z: -d/2},   // back top right
            bbl: {x: -w/2 + morph1 * 0.03, y: h/2 + morph3 * 0.03, z: -d/2},   // back bottom left
            bbr: {x: w/2 + morph2 * 0.03, y: h/2 + morph1 * 0.03, z: -d/2}     // back bottom right
        };

        // Define 12 edges of the box (the board)
        return [
            // Front face edges (4)
            {start: vertices.ftl, end: vertices.ftr, type: 'top'},
            {start: vertices.ftr, end: vertices.fbr, type: 'side'},
            {start: vertices.fbr, end: vertices.fbl, type: 'bottom'},
            {start: vertices.fbl, end: vertices.ftl, type: 'side'},

            // Back face edges (4)
            {start: vertices.btl, end: vertices.btr, type: 'top'},
            {start: vertices.btr, end: vertices.bbr, type: 'side'},
            {start: vertices.bbr, end: vertices.bbl, type: 'bottom'},
            {start: vertices.bbl, end: vertices.btl, type: 'side'},

            // Connecting edges (4) - front to back
            {start: vertices.ftl, end: vertices.btl, type: 'depth'},
            {start: vertices.ftr, end: vertices.btr, type: 'depth'},
            {start: vertices.fbl, end: vertices.bbl, type: 'depth'},
            {start: vertices.fbr, end: vertices.bbr, type: 'depth'}
        ];
    }

    // Tetrahedron edge definitions with morphing (for dark mode)
    function getPyramidEdges(morphFactor = 0) {
        // Morphing distortion factors
        const morph1 = Math.sin(morphPhase) * morphFactor;
        const morph2 = Math.cos(morphPhase * 1.3) * morphFactor;
        const morph3 = Math.sin(morphPhase * 0.7) * morphFactor;

        // Define tetrahedron vertices (centered at 0,0,0 for proper rotation)
        const vertices = {
            // Base triangle vertices (at y = 0.5)
            b1: {x: 0 + morph1 * 0.1, y: 0.5, z: -0.577 + morph2 * 0.05},      // back vertex
            b2: {x: -0.5 + morph2 * 0.08, y: 0.5, z: 0.289 + morph3 * 0.05},   // left vertex
            b3: {x: 0.5 + morph3 * 0.08, y: 0.5, z: 0.289 + morph1 * 0.05},    // right vertex
            // Apex (at y = -0.5 for centering)
            apex: {x: 0 + morph2 * 0.05, y: -0.5 + morph1 * 0.1, z: 0 + morph3 * 0.05}
        };

        // Define edges (only 6 edges for tetrahedron)
        return [
            // Base edges (3)
            {start: vertices.b1, end: vertices.b2, type: 'base'},
            {start: vertices.b2, end: vertices.b3, type: 'base'},
            {start: vertices.b3, end: vertices.b1, type: 'base'},
            // Apex edges (3)
            {start: vertices.b1, end: vertices.apex, type: 'apex'},
            {start: vertices.b2, end: vertices.apex, type: 'apex'},
            {start: vertices.b3, end: vertices.apex, type: 'apex'}
        ];
    }

    // Optimized edge illumination calculation
    let edgeIlluminationCache = new Map();
    let lastCacheFrame = -1;

    function calculateEdgeIllumination(edge, tendrils, frameCount) {
        // Cache illumination values per frame
        if (frameCount !== lastCacheFrame) {
            edgeIlluminationCache.clear();
            lastCacheFrame = frameCount;
        }

        const cacheKey = `${edge.start.x},${edge.start.y},${edge.start.z}-${edge.end.x},${edge.end.y},${edge.end.z}`;
        if (edgeIlluminationCache.has(cacheKey)) {
            return edgeIlluminationCache.get(cacheKey);
        }

        let illumination = 0;
        const proximityThreshold = 0.4;
        let totalContribution = 0;

        // Sample tendrils for edge illumination
        const skipFactor = tendrils.length > 100 ? 3 : tendrils.length > 50 ? 2 : 1;
        const maxSamples = 30;
        let sampledCount = 0;

        for (let idx = 0; idx < tendrils.length && sampledCount < maxSamples; idx += skipFactor) {
            sampledCount++;
            const tendril = tendrils[idx];

            // Skip invisible tendrils
            if (tendril.opacity < 0.1) continue;

            // Check tendril proximity to edge
            const maxLength = tendril.cachedIntersection || 0.8;
            const endT = tendril.lengthProgress;

            // Check multiple points along tendril for edge proximity
            for (let sample = 0; sample <= 2; sample++) {
                const t = sample * 0.4 * endT;

                // Calculate position from tendril origin
                const origin = tendril.origin || {x: 0, y: 0, z: 0};
                const pos = {
                    x: origin.x + tendril.baseDirection.x * t * maxLength,
                    y: origin.y + tendril.baseDirection.y * t * maxLength,
                    z: origin.z + tendril.baseDirection.z * t * maxLength
                };

                // Quick distance check
                const dist = pointToLineDistanceSquared(pos, edge.start, edge.end);

                if (dist < proximityThreshold * proximityThreshold) {
                    const actualDist = Math.sqrt(dist);
                    const proximity = 1 - (actualDist / proximityThreshold);
                    totalContribution += proximity * tendril.opacity * 0.5;
                }
            }
        }

        illumination = Math.min(1, totalContribution);
        edgeIlluminationCache.set(cacheKey, illumination);
        return illumination;
    }

    // Optimized squared distance calculation (avoids sqrt)
    function pointToLineDistanceSquared(point, lineStart, lineEnd) {
        const line = {
            x: lineEnd.x - lineStart.x,
            y: lineEnd.y - lineStart.y,
            z: lineEnd.z - lineStart.z
        };

        const toPoint = {
            x: point.x - lineStart.x,
            y: point.y - lineStart.y,
            z: point.z - lineStart.z
        };

        const lineLengthSq = line.x * line.x + line.y * line.y + line.z * line.z;
        if (lineLengthSq === 0) return toPoint.x * toPoint.x + toPoint.y * toPoint.y + toPoint.z * toPoint.z;

        const t = Math.max(0, Math.min(1, (toPoint.x * line.x + toPoint.y * line.y + toPoint.z * line.z) / lineLengthSq));

        const closest = {
            x: lineStart.x + t * line.x - point.x,
            y: lineStart.y + t * line.y - point.y,
            z: lineStart.z + t * line.z - point.z
        };

        return closest.x * closest.x + closest.y * closest.y + closest.z * closest.z;
    }

    // Draw pyramid edges with illumination
    function drawPyramidEdges(edges, tendrils, frameCount, shadowPass = false, drawCenterX = centerX, drawCenterY = centerY) {
        ctx.save();

        for (let edge of edges) {
            // Calculate illumination for this edge
            const illumination = calculateEdgeIllumination(edge, tendrils, frameCount);

            // Only draw if there's some illumination
            if (illumination > 0.01) {
                // Rotate edge vertices
                const rotatedStart = tendrils[0].rotate3D(edge.start, globalRotX, globalRotY, globalRotZ);
                const rotatedEnd = tendrils[0].rotate3D(edge.end, globalRotX, globalRotY, globalRotZ);

                // Project to 2D
                const perspectiveStart = 2 / (2 - rotatedStart.z * 0.3);
                const perspectiveEnd = 2 / (2 - rotatedEnd.z * 0.3);

                const x1 = drawCenterX + rotatedStart.x * maxRadius * perspectiveStart;
                const y1 = drawCenterY + rotatedStart.y * maxRadius * perspectiveStart * 0.8;
                const x2 = drawCenterX + rotatedEnd.x * maxRadius * perspectiveEnd;
                const y2 = drawCenterY + rotatedEnd.y * maxRadius * perspectiveEnd * 0.8;

                // Draw edge with illumination-based opacity
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);

                if (shadowPass) {
                    // Shadow rendering for light mode - very subtle
                    ctx.strokeStyle = `rgba(0, 0, 0, ${illumination * 0.5})`; // Much lighter
                    ctx.lineWidth = 3 + illumination * 2; // Thinner lines
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = `rgba(0, 0, 0, ${illumination * 0.3})`; // Very subtle shadow
                    ctx.stroke();
                } else {
                    // Normal rendering
                    const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
                    const baseOpacity = illumination * 0.6;
                    const glowOpacity = illumination * 0.3;

                    // Edge type affects color
                    if (edge.type === 'apex') {
                        gradient.addColorStop(0, `rgba(50, 205, 50, ${baseOpacity})`);
                        gradient.addColorStop(0.5, `rgba(144, 238, 144, ${baseOpacity * 1.2})`);
                        gradient.addColorStop(1, `rgba(152, 251, 152, ${baseOpacity})`);
                    } else {
                        gradient.addColorStop(0, `rgba(34, 139, 34, ${baseOpacity})`);
                        gradient.addColorStop(0.5, `rgba(50, 205, 50, ${baseOpacity * 1.2})`);
                        gradient.addColorStop(1, `rgba(34, 139, 34, ${baseOpacity})`);
                    }

                    ctx.strokeStyle = gradient;
                    ctx.lineWidth = 1 + illumination * 1.5;

                    // Add glow effect
                    ctx.shadowBlur = 5 + illumination * 15;
                    ctx.shadowColor = `rgba(144, 238, 144, ${glowOpacity})`;

                    ctx.stroke();
                }
            }
        }

        ctx.restore();
    }

    // Check if we're in light mode
    function isLightMode() {
        return document.documentElement.getAttribute('data-theme') !== 'dark';
    }

    // Select next random behavior with weighted probabilities
    function selectNextBehavior() {
        const weights = {
            [motionBehaviors.SPINNING]: 15,
            [motionBehaviors.FLYING]: 20,
            [motionBehaviors.SOARING]: 20,
            [motionBehaviors.SWOOPING]: 10,
            [motionBehaviors.PEACEFUL]: 20,
            [motionBehaviors.HOVERING]: 10,
            [motionBehaviors.TUMBLING]: 5
        };

        // Don't repeat the same behavior immediately
        weights[currentBehavior] = 0;

        const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
        let random = Math.random() * totalWeight;

        for (const [behavior, weight] of Object.entries(weights)) {
            random -= weight;
            if (random <= 0) {
                return behavior;
            }
        }
        return motionBehaviors.PEACEFUL; // Fallback
    }

    // Calculate target velocities based on behavior
    function calculateBehaviorVelocities(behavior, elapsed) {
        const t = elapsed / 1000; // Convert to seconds

        switch (behavior) {
            case motionBehaviors.SPINNING:
                // Original spinning behavior
                return {
                    x: 0.003 + Math.random() * 0.002,
                    y: 0.005 + Math.random() * 0.002,
                    z: 0.002 + Math.random() * 0.001
                };

            case motionBehaviors.FLYING:
                // Smooth forward motion with banking turns
                const flyAngle = t * 0.3;
                return {
                    x: 0.002 + Math.sin(flyAngle) * 0.003,
                    y: 0.008, // Steady forward rotation
                    z: Math.cos(flyAngle * 0.7) * 0.002
                };

            case motionBehaviors.SOARING:
                // Gentle gliding with occasional lift
                const soarPhase = t * 0.2;
                return {
                    x: Math.sin(soarPhase) * 0.002,
                    y: 0.003 + Math.sin(soarPhase * 0.5) * 0.002,
                    z: Math.cos(soarPhase * 0.3) * 0.001
                };

            case motionBehaviors.SWOOPING:
                // Dramatic diving and climbing
                const swoopPhase = t * 0.5;
                return {
                    x: Math.sin(swoopPhase) * 0.01,
                    y: 0.005 + Math.cos(swoopPhase) * 0.008,
                    z: Math.sin(swoopPhase * 1.5) * 0.003
                };

            case motionBehaviors.PEACEFUL:
                // Very slow, stable rotation
                return {
                    x: 0.0005,
                    y: 0.001,
                    z: 0.0003
                };

            case motionBehaviors.HOVERING:
                // Nearly still with micro-movements
                const hoverPhase = t * 2;
                return {
                    x: Math.sin(hoverPhase) * 0.0003,
                    y: 0.0002 + Math.cos(hoverPhase * 0.7) * 0.0002,
                    z: Math.sin(hoverPhase * 1.3) * 0.0001
                };

            case motionBehaviors.TUMBLING:
                // Chaotic tumbling
                return {
                    x: (Math.random() - 0.5) * 0.02,
                    y: (Math.random() - 0.5) * 0.02,
                    z: (Math.random() - 0.5) * 0.015
                };

            default:
                return { x: rotVelX, y: rotVelY, z: rotVelZ };
        }
    }

    // Smooth interpolation function
    function smoothInterpolate(current, target, factor) {
        return current + (target - current) * factor;
    }

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);

        // Clear canvas
        ctx.clearRect(0, 0, baseSize, baseSize);

        // Update time
        time += 0.012;

        // Behavior system update
        const behaviorElapsed = time - behaviorStartTime;

        // Check if it's time to transition to a new behavior
        if (behaviorElapsed > behaviorDuration && !nextBehavior) {
            nextBehavior = selectNextBehavior();
            transitionProgress = 0;
        }

        // Handle behavior transitions
        if (nextBehavior) {
            transitionProgress += 0.012 / transitionDuration;

            if (transitionProgress >= 1) {
                // Complete the transition
                currentBehavior = nextBehavior;
                nextBehavior = null;
                behaviorStartTime = time;
                behaviorDuration = 4 + Math.random() * 8; // 4-12 seconds
                transitionProgress = 0;
            }
        }

        // Calculate target velocities based on current behavior
        const behaviorVels = calculateBehaviorVelocities(
            nextBehavior || currentBehavior,
            behaviorElapsed * 1000
        );

        // Smoothly interpolate to target velocities
        const smoothFactor = nextBehavior ? transitionProgress * 0.1 : 0.05;
        targetVelX = behaviorVels.x;
        targetVelY = behaviorVels.y;
        targetVelZ = behaviorVels.z;

        rotVelX = smoothInterpolate(rotVelX, targetVelX, smoothFactor);
        rotVelY = smoothInterpolate(rotVelY, targetVelY, smoothFactor);
        rotVelZ = smoothInterpolate(rotVelZ, targetVelZ, smoothFactor);

        // Check mode once for all physics calculations
        const lightMode = isLightMode();

        // Update turbulence (dark mode only)
        const turbulence = updateTurbulence(time);

        // Apply rotation velocities WITH turbulence jolts
        // Turbulence adds to rotation angles directly (not velocities)
        // This creates the "jostled off axis" feeling - sudden angular displacement
        globalRotX += rotVelX + turbulence.rotX;
        globalRotY += rotVelY + turbulence.rotY;
        globalRotZ += rotVelZ + turbulence.rotZ;

        // Update exposed geometry for hybrid mode (with morphing)
        const morph1 = Math.sin(morphPhase) * 0.15; // Same morphFactor as used in rendering
        const morph2 = Math.cos(morphPhase * 1.3) * 0.15;
        const morph3 = Math.sin(morphPhase * 0.7) * 0.15;

        window.prismGeometry.rotX = globalRotX;
        window.prismGeometry.rotY = globalRotY;
        window.prismGeometry.rotZ = globalRotZ;
        window.prismGeometry.vertices = {
            apex: {x: 0 + morph2 * 0.05, y: -0.5 + morph1 * 0.1, z: 0 + morph3 * 0.05},
            back: {x: 0 + morph1 * 0.1, y: 0.5, z: -0.577 + morph2 * 0.05},
            left: {x: -0.5 + morph2 * 0.08, y: 0.5, z: 0.289 + morph3 * 0.05},
            right: {x: 0.5 + morph3 * 0.08, y: 0.5, z: 0.289 + morph1 * 0.05}
        };

        // Update propulsion physics (dark mode only)
        // Car-like acceleration: brief ramp, cruise at offset, decelerate when calm
        const propulsion = updatePropulsion(tendrils, lightMode);

        // Combine all spatial offsets:
        // - Turbulence jolts (random bumps - dark mode only)
        // - Propulsion displacement (sustained pull-away during hyperactivity - dark mode only)
        const totalOffsetX = turbulence.offsetX + propulsion.x;
        const totalOffsetY = turbulence.offsetY + propulsion.y;

        // Fuzzy anchoring: Apply combined offsets to center point
        // The prism can drift away (propulsion + turbulence) but is pulled back (anchor)
        // Like a ship on a chain - it can move, but there's always tension pulling it back
        const turbulentCenterX = centerX + totalOffsetX;
        const turbulentCenterY = centerY + totalOffsetY;

        // Expose turbulent center for hybrid mode
        window.prismGeometry.turbulentCenterX = turbulentCenterX;
        window.prismGeometry.turbulentCenterY = turbulentCenterY;

        // Update morphing phase
        morphPhase += morphSpeed;

        // Calculate spawn intensity with MASSIVE surges
        const baseWave = Math.sin(time * 0.3) * 0.5 + 0.5;
        const mediumWave = Math.sin(time * 0.7 + Math.PI/4) * 0.3 + 0.3;
        const fastWave = Math.sin(time * 1.5 - Math.PI/3) * 0.2 + 0.2;

        // HUGE surge events - rare but massive
        let surgeMultiplier = 1;
        const surgeProbability = Math.sin(time * 0.15); // Slow wave for surge timing

        if (surgeProbability > 0.95 && time - lastSurgeTime > 5) {
            // MASSIVE surge event!
            surgeMultiplier = 10 + Math.random() * 15; // Scaled down for logo
            lastSurgeTime = time;
        } else if (surgeProbability > 0.7) {
            // Medium surge
            surgeMultiplier = 3 + Math.random() * 5;
        }

        const baseIntensity = (baseWave + mediumWave + fastWave) / 3;
        const spawnIntensity = Math.min(1, baseIntensity * surgeMultiplier);

        // Calculate target tendril count
        let targetCount;
        if (surgeMultiplier > 10) {
            // Large surge
            targetCount = Math.floor(40 + Math.random() * 60);
        } else if (surgeMultiplier > 3) {
            // Medium surge
            targetCount = Math.floor(20 + Math.random() * 20);
        } else {
            // Normal oscillation
            targetCount = Math.floor(minTendrils + (normalTendrils - minTendrils) * baseIntensity);
        }

        // Spawn tendrils during surges
        if (tendrils.length < targetCount) {
            const spawnRate = surgeMultiplier > 10 ? 5 : 2;
            for (let i = 0; i < spawnRate && tendrils.length < Math.min(targetCount, maxTendrils); i++) {
                tendrils.push(new Tendril(tendrilIndex++));
            }
        }

        // Sort tendrils by z-depth occasionally for performance
        if (frameCount % 3 === 0) {
            tendrils.sort((a, b) => {
                // Use cached z-values if available
                if (!a.sortZ || frameCount !== a.lastSortFrame) {
                    const aRotated = a.rotate3D(a.baseDirection, globalRotX, globalRotY, globalRotZ);
                    a.sortZ = aRotated.z;
                    a.lastSortFrame = frameCount;
                }
                if (!b.sortZ || frameCount !== b.lastSortFrame) {
                    const bRotated = b.rotate3D(b.baseDirection, globalRotX, globalRotY, globalRotZ);
                    b.sortZ = bRotated.z;
                    b.lastSortFrame = frameCount;
                }
                return a.sortZ - b.sortZ;
            });
        }

        // Update all tendrils first
        for (let i = 0; i < tendrils.length; i++) {
            tendrils[i].update();
        }

        // Get morphed shape edges (2x4 board in light mode, tetrahedron in dark mode)
        // Reuse lightMode variable already declared above
        const shapeEdges = lightMode ?
            get2x4Edges(0.15) :      // 2x4 board - "We build during the day"
            getPyramidEdges(0.15);   // Tetrahedron - for dark mode

        // SHADOW PASS: Draw shadows first if in light mode
        if (lightMode) {
            ctx.save();
            ctx.globalAlpha = 0.5; // Even more subtle overall transparency

            // Draw shadow tendrils (light mode uses stable center)
            for (let i = 0; i < tendrils.length; i++) {
                tendrils[i].draw(true, centerX, centerY); // shadowPass = true
            }

            // Draw shadow shape edges
            drawPyramidEdges(shapeEdges, tendrils, frameCount, true, centerX, centerY); // shadowPass = true

            ctx.restore();
        }

        // NORMAL PASS: Draw actual elements on top

        // Draw central core that responds to surges
        const coreSize = 5 + Math.sin(time * 2) * 1 + Math.min(10, surgeMultiplier);
        const coreGlow = 0.5 + Math.min(0.5, surgeMultiplier * 0.02);

        // Core shadow in light mode
        if (isLightMode()) {
            ctx.save();
            ctx.shadowBlur = 15;
            ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
            ctx.beginPath();
            ctx.arc(centerX, centerY, coreSize + 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fill();
            ctx.restore();
        }

        // Core uses turbulent center in dark mode, stable center in light mode
        const coreCenterX = lightMode ? centerX : turbulentCenterX;
        const coreCenterY = lightMode ? centerY : turbulentCenterY;

        const coreGradient = ctx.createRadialGradient(coreCenterX, coreCenterY, 0, coreCenterX, coreCenterY, coreSize * 1.5);
        coreGradient.addColorStop(0, `rgba(255, 255, 255, ${coreGlow})`);
        coreGradient.addColorStop(0.3, `rgba(200, 255, 200, ${coreGlow * 0.8})`);
        coreGradient.addColorStop(0.6, `rgba(50, 205, 50, ${coreGlow * 0.5})`);
        coreGradient.addColorStop(1, 'rgba(34, 139, 34, 0.05)');

        ctx.beginPath();
        ctx.arc(coreCenterX, coreCenterY, coreSize, 0, Math.PI * 2);
        ctx.fillStyle = coreGradient;
        ctx.fill();

        // Core glow during surges
        if (surgeMultiplier > 5) {
            ctx.shadowBlur = 20;
            ctx.shadowColor = `rgba(144, 238, 144, 0.5)`;
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        // Draw shape edges with proximity-based illumination
        // (2x4 board in light mode, tetrahedron in dark mode)
        // Use turbulent center in dark mode for jostling effect
        drawPyramidEdges(shapeEdges, tendrils, frameCount, false, coreCenterX, coreCenterY); // shadowPass = false
        frameCount++;

        // Draw all tendrils with turbulent center
        for (let i = 0; i < tendrils.length; i++) {
            tendrils[i].draw(false, coreCenterX, coreCenterY); // shadowPass = false
        }

        // Clean up dead tendrils - aggressive cleanup after surges
        if (surgeMultiplier <= 1 && tendrils.length > targetCount + 10) {
            // Fast cleanup after surge
            const removeCount = Math.min(10, tendrils.length - targetCount);
            for (let i = 0; i < removeCount; i++) {
                tendrils.shift();
            }
        } else if (tendrils.length > maxTendrils) {
            // Hard limit
            tendrils.splice(0, tendrils.length - maxTendrils);
        }
    }

    animate();
})();