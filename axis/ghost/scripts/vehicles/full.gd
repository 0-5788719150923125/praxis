extends Vehicle
class_name FullVehicle

## FullVehicle - one scene, filling the frame. What ghost has always done.
##
## The IDENTITY vehicle, and it is worth having as a real object rather than as a
## `if vehicle == null` branch everywhere: it makes "the original show" a named,
## selectable thing in the registry instead of the absence of a choice, and it is what
## the [Director]'s four vetoes are measured against - every one of them returns its
## argument unchanged here, so a session on this vehicle behaves exactly as it did
## before vehicles existed.
##
## It draws nothing at all. Scenes are added straight to the stage SubViewport and
## composite themselves, as they always have.


func host_for(_incoming: GhostScene) -> Node:
	return stage
