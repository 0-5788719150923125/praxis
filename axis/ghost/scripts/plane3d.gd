extends RefCounted
class_name Plane3D

## Plane3D - a flat quad genuinely placed in 3D space, projected through a [Lens3D].
##
## The "3D planes for geometry" primitive. A plane is a centre plus two half-extent
## edge vectors (so a corner is center ± u_axis ± v_axis) - orient it however you
## like in the world: lay them flat and stack for parallax, stand them up as bars,
## tilt them into depth. Unlike the old approach (a 2D polygon sheared by the view
## to *fake* depth), this is a real quad under a real camera, depth-sorted alongside
## the [Mesh3D] bodies. Carries a fill colour and an optional bright edge.

var center := Vector3.ZERO
var u_axis := Vector3.RIGHT     ## half-width edge vector
var v_axis := Vector3.UP        ## half-height edge vector
var color := Color.WHITE
var edge := Color(0, 0, 0, 0)   ## edge stroke; alpha 0 = no edge


func _init(c := Vector3.ZERO, uax := Vector3.RIGHT, vax := Vector3.UP, col := Color.WHITE) -> void:
	center = c
	u_axis = uax
	v_axis = vax
	color = col


## The four world-space corners, counter-clockwise.
func corners() -> Array:
	return [
		center - u_axis - v_axis,
		center + u_axis - v_axis,
		center + u_axis + v_axis,
		center - u_axis + v_axis,
	]


## Project the quad through the lens and draw it at pixel scale [param u_px]
## (centred space). Skips the quad if any corner falls behind the camera, which
## keeps the projection from flipping through infinity at grazing angles.
func draw_through(ci: CanvasItem, lens: Lens3D, u_px: float) -> void:
	var poly := PackedVector2Array()
	for c in corners():
		var pr := lens.project(c)
		if pr.z <= lens.near:
			return
		poly.append(Vector2(pr.x, pr.y) * u_px)
	ci.draw_colored_polygon(poly, color)
	if edge.a > 0.0:
		var e := poly.duplicate()
		e.append(poly[0])
		ci.draw_polyline(e, edge, 1.0, true)
