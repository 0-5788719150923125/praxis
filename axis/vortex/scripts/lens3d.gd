extends RefCounted
class_name Lens3D

## Lens3D - a positionable perspective camera for the unified 3D path.
##
## The old [Mesh3D] projector was a fixed, centred lens (focal 3.2, eye on the +z
## axis); it could spin a solid in place but never *move the camera*. Lens3D is a
## real camera: an eye position looking at a target with a field of view, so a scene
## can push in, orbit, and frame in depth. Wide fov + a near eye is **forced
## perspective** - near geometry looms over far geometry, the dimensional read that
## a sheared 2D plane can only fake.
##
## It projects a world [Vector3] into centred *unit-fraction* screen space (the same
## space the rest of vortex draws in: origin = screen middle, y down); the caller
## multiplies by the pixel unit. Call [method prepare] once per frame before
## projecting (it caches the view basis), then [method project] / [method depth].

## Camera placement and lens.
var eye := Vector3(0, 0, 4)
var look := Vector3.ZERO
var up := Vector3.UP
## Vertical field of view in degrees. Small = long lens (flat), large = wide lens
## (strong forced perspective).
var fov := 55.0
## Anything closer than this along the view axis is treated as behind the camera.
var near := 0.05

# Cached view basis + focal, rebuilt by prepare().
var _r := Vector3.RIGHT
var _u := Vector3.UP
var _f := Vector3.FORWARD
var _focal := 2.0


## Rebuild the cached camera basis and focal length. Call once per frame after
## setting eye / look / fov, before any project()/depth() calls.
func prepare() -> void:
	var fwd := look - eye
	fwd = fwd.normalized() if fwd.length() > 1e-6 else Vector3(0, 0, -1)
	var right := fwd.cross(up)
	right = right.normalized() if right.length() > 1e-6 else Vector3(1, 0, 0)
	_r = right
	_u = right.cross(fwd).normalized()
	_f = fwd
	_focal = 1.0 / tan(deg_to_rad(fov) * 0.5)


## Camera-space depth of a world point: distance along the view axis. Greater than
## [member near] means in front of the camera. Used for the painter's sort.
func depth(p: Vector3) -> float:
	return (p - eye).dot(_f)


## Project a world point to centred unit-fraction screen coords. Returns a
## [Vector3]: (x, y, depth) where x/y are the screen offset (multiply by the pixel
## unit) and z is the camera-space depth. z <= near means the point is behind the
## camera and the caller should clip it.
func project(p: Vector3) -> Vector3:
	var v := p - eye
	var cz := v.dot(_f)
	var inv := _focal / maxf(near, cz)
	return Vector3(v.dot(_r) * inv, -v.dot(_u) * inv, cz)


## Place the eye on an orbit around [param center] at [param dist], with [param yaw]
## / [param pitch] in radians (pitch lifts the eye above the target). The lazy way
## to fly the camera; forced perspective comes from a small dist plus a wide fov.
func orbit(center: Vector3, dist: float, yaw: float, pitch: float) -> void:
	look = center
	var cp := cos(pitch)
	eye = center + Vector3(sin(yaw) * cp, sin(pitch), cos(yaw) * cp) * dist
