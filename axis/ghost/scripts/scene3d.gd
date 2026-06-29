extends GhostScene
class_name Scene3D

## Scene3D - the unified 3D rendering path (the convergence target).
##
## The project carries several render mechanisms forward (see GhostScene's
## render_kind); this is the one the rest should migrate onto. A Scene3D owns a
## perspective camera ([Lens3D]) and a world of two drawable kinds - solid bodies
## ([Mesh3D]) and flat quads ([Plane3D]) - and renders them depth-sorted under
## forced perspective. A subclass just populates the world and flies the camera:
##
##   build_params(rng)  - add_plane(...) / set up meshes; place the lens.
##   update(f, delta)    - move the lens, restate per-frame bodies, recolour planes.
##   _draw()             - begin_draw(); render_world().
##
## Bodies are restated each frame (their basis/glow change), so the usual shape is
## `bodies.clear()` then add_body(...) in update; planes are usually persistent and
## mutated in place. The whole world is sorted back-to-front by camera depth before
## drawing, so bodies and planes correctly occlude each other; each Mesh3D still
## sorts its own faces internally. The scene is still a Node2D drawn through the 2D
## [SceneView], so transitions, presence, and gentle shot drift all keep working -
## the lens is the *primary* camera, the 2D view adds only subtle framing on top.

## The camera the world is projected through.
var lens := Lens3D.new()
## Per-frame solid bodies: dicts of {mesh, basis, pos, scale, hue, sat, edge, alpha,
## glow, explode}. Usually rebuilt each frame via clear()/add_body.
var bodies: Array = []
## Persistent flat quads (see [Plane3D]); usually mutated in place each frame.
var planes: Array = []


func _init() -> void:
	render_kind = "scene3d"


## Drop all bodies and planes (call before repopulating a frame's world).
func clear_world() -> void:
	bodies.clear()
	planes.clear()


## Stage a solid body for this frame.
func add_body(mesh: Mesh3D, basis: Basis, pos: Vector3, scale: float, hue: float,
		sat := 0.5, edge := 0, alpha := 1.0, glow := 0.0, explode := 0.0) -> void:
	bodies.append({
		"mesh": mesh, "basis": basis, "pos": pos, "scale": scale, "hue": hue,
		"sat": sat, "edge": edge, "alpha": alpha, "glow": glow, "explode": explode})


## Add a flat quad to the world.
func add_plane(pl: Plane3D) -> void:
	planes.append(pl)


## Project, depth-sort the whole world back-to-front, and draw it through the lens.
## Call from _draw() after begin_draw(). Bodies sort by their centroid (pos); for
## well-separated bodies that reads correctly, and each mesh still resolves its own
## faces. Planes sort by their centre.
func render_world() -> void:
	lens.prepare()
	var u := unit()
	var items := []
	for b in bodies:
		items.append({"d": lens.depth(b.pos), "plane": null, "body": b})
	for pl in planes:
		items.append({"d": lens.depth(pl.center), "plane": pl, "body": null})
	items.sort_custom(func(a, c): return a.d > c.d)   # far first
	for it in items:
		if it.plane != null:
			it.plane.draw_through(self, lens, u)
		else:
			var b = it.body
			b.mesh.draw_through(self, lens, u, b.basis, b.pos, b.scale,
				b.hue, b.sat, b.edge, b.alpha, b.glow, b.explode)
