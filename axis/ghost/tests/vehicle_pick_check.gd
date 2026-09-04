extends Node

## Smoke test for the Vehicle picker in the Generative panel: build the whole panel and
## read the OptionButton back. A parse check cannot see a wrong registry key or a
## Callable that captures the wrong thing, and this control is the only way most people
## will ever reach the setting.
##
## Run: tests/run_boot_probe.sh tests/vehicle_pick_check.gd 90
##
## It restores the setting it found, and FLUSHES that restore to disk rather than trusting
## the debounce - see the note at the restore itself.

func _ready() -> void:
	_run.call_deferred()


func _run() -> void:
	var ed := preload("res://scripts/generative_editor.gd").new()
	add_child(ed)
	await get_tree().process_frame
	var fails := 0
	# THE PICKER MUST AGREE WITH THE FILE, not merely with whatever the Director happens to
	# hold. Those came apart the moment Settings was added as an autoload listed AFTER
	# Director: Director read its remembered values before the file had been loaded, got
	# defaults, and the picker faithfully showed them. Nothing errored - a default is a
	# valid value - and the only visible symptom was a setting quietly reverting.
	var on_disk := ConfigFile.new()
	if on_disk.load(Settings.PATH) == OK:
		var want := String(on_disk.get_value("director", "vehicle", "full"))
		if Director.vehicle != want:
			print("vpick: FAILED - file says '%s' but Director holds '%s' (autoload order?)"
				% [want, Director.vehicle])
			fails += 1
	var opt: OptionButton = ed._vehicle_pick
	if opt == null:
		print("vpick: FAILED - the picker was never built")
		fails += 1
	else:
		var keys: Array = Vehicle.REGISTRY.keys()
		print("vpick: %d items, selected %d (%s)" % [
			opt.item_count, opt.selected, opt.get_item_text(maxi(0, opt.selected))])
		if opt.item_count != keys.size():
			print("vpick: FAILED - %d items for %d registered vehicles" % [
				opt.item_count, keys.size()])
			fails += 1
		if opt.get_item_text(maxi(0, opt.selected)) != String(Vehicle.LABELS.get(Director.vehicle, "")):
			print("vpick: FAILED - selection does not show the live setting '%s'" % Director.vehicle)
			fails += 1
		# and that choosing one actually reaches the Director
		var was := Director.vehicle
		var other := ""
		for k in keys:
			if String(k) != was:
				other = String(k)
				break
		opt.item_selected.emit(keys.find(other))
		if Director.vehicle != other:
			print("vpick: FAILED - selecting '%s' left Director.vehicle at '%s'" % [other, Director.vehicle])
			fails += 1
		else:
			print("vpick: selecting '%s' set Director.vehicle" % other)
		# LEAVE THE USER'S SETTING ALONE, and flush it rather than trusting the debounce.
		# set_vehicle only marks the config dirty; the write lands 400 ms later, and this
		# check quits well inside that - so the SELECTION above was what reached the disk
		# and the restore was not. It left `vehicle="comic"` in a config the user never
		# chose it in.
		Director.set_vehicle(was)
		Director._save_pacing()
	# THE FILM CONTROLS ARE BUILT AND SHOW THE STORED VALUE. They live in the same section
	# and are reached the same way, so they fail the same way: a control that never got
	# built is invisible rather than broken, and one whose initial value came from a
	# default instead of the file looks exactly like a setting that reverted. Only reads
	# here - the frequency dial is the user's, and a gate may not spend it.
	if ed._film_freq == null:
		print("vpick: FAILED - the film frequency slider was never built")
		fails += 1
	elif not is_equal_approx(float(ed._film_freq.value), Films.frequency()):
		print("vpick: FAILED - the film slider shows %.2f, the library says %.2f"
			% [ed._film_freq.value, Films.frequency()])
		fails += 1
	else:
		print("vpick: film frequency %.2f, %d clip(s) in the library"
			% [Films.frequency(), Films.clips().size()])
	if ed._film_list == null or ed._film_list.get_child_count() == 0:
		print("vpick: FAILED - the film list was never built (it shows a row either way)")
		fails += 1
	print("vpick: %s" % ("ALL OK" if fails == 0 else "%d FAILURE(S)" % fails))
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(fails)
