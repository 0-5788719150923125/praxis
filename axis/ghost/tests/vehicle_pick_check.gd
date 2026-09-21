extends Node

## Smoke test for the PICTURE SETTINGS in the Generative panel - the Vehicle picker, the film
## controls and the Look filters: build the whole panel and read the controls back. A parse
## check cannot see a wrong registry key or a Callable that captured the wrong thing, and
## these controls are the only way most people will ever reach the settings.
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
	# ONLY THE SETTINGS THIS VEHICLE CAN USE ARE SHOWN. Reported as "there are a number of
	# settings currently being displayed that ONLY work with the comic book vehicle". The
	# failure is not an error - a control that does nothing looks exactly like one that does -
	# so it is asserted as a property of the registry rather than of a hand-written list.
	if ed._vehicle_rows.is_empty():
		print("vpick: FAILED - no rows are tagged with a vehicle feature at all")
		fails += 1
	else:
		var cam := Director.vehicle
		for key in Vehicle.REGISTRY:
			var vk := String(key)
			# select() AND the signal: a real click does both, and the two are separate in
			# Godot - emitting alone leaves `selected` on the previous item.
			ed._vehicle_pick.select(Vehicle.REGISTRY.keys().find(vk))
			ed._vehicle_pick.item_selected.emit(Vehicle.REGISTRY.keys().find(vk))
			for tag in ed._vehicle_rows:
				var want: bool = Vehicle.uses(vk, String(tag))
				for row in ed._vehicle_rows[tag] as Array:
					if (row as Control).visible != want:
						print("vpick: FAILED - on '%s' the '%s' rows are %s, want %s"
							% [vk, tag, "shown" if (row as Control).visible else "hidden",
								"shown" if want else "hidden"])
						fails += 1
						break
			print("vpick: '%s' shows %s" % [vk, str(Vehicle.USES.get(vk, []))])
		# THE CONTROL: at least one tag must actually differ between two vehicles, or the
		# sweep above is satisfied by every row being visible everywhere.
		var differs := false
		for tag in ed._vehicle_rows:
			var seen := {}
			for key in Vehicle.REGISTRY:
				seen[Vehicle.uses(String(key), String(tag))] = true
			if seen.size() > 1:
				differs = true
		if not differs:
			print("vpick: FAILED - the control is wrong: no tagged group is hidden on any "
				+ "vehicle, so the sweep proves nothing")
			fails += 1
		Director.set_vehicle(cam)
		Director._save_pacing()
		ed._vehicle_pick.select(maxi(0, Vehicle.REGISTRY.keys().find(cam)))
		ed._sync_vehicle_rows()
	# THE LOOK FILTERS, and the same three questions: a row per registry entry, each showing
	# the live setting, and a tick that actually reaches the Director. The failure mode here is
	# the one filters.gd is written around - a checkbox that moves, saves, reloads and changes
	# no pixel - and it is invisible from inside a parse check or a shader compile.
	if ed._filter_rows.size() != Filters.REGISTRY.size():
		print("vpick: FAILED - %d filter rows for %d registered filters"
			% [ed._filter_rows.size(), Filters.REGISTRY.size()])
		fails += 1
	else:
		for key in Filters.REGISTRY:
			var k := String(key)
			var row: Dictionary = ed._filter_rows[k]
			var cb: CheckBox = row["box"]
			var sl: HSlider = row["slider"]
			var live := Director.filter_amount(k)
			if cb.button_pressed != (live > 0.0):
				print("vpick: FAILED - '%s' shows %s, the Director holds %.2f"
					% [k, "on" if cb.button_pressed else "off", live])
				fails += 1
			# A GREYED DIAL MUST BE GREYED, or an inert control looks live.
			if sl.editable != (live > 0.0):
				print("vpick: FAILED - '%s' dial is %s while the filter is %s"
					% [k, "live" if sl.editable else "greyed", "on" if live > 0.0 else "off"])
				fails += 1
		# One round trip through the FIRST filter, then put it back. Ticking it on with the
		# dial at zero must land on the registry's default rather than on silence - a box that
		# does nothing when ticked is the report this whole block exists for.
		var first := String(Filters.REGISTRY.keys()[0])
		var frow: Dictionary = ed._filter_rows[first]
		var fcb: CheckBox = frow["box"]
		var fsl: HSlider = frow["slider"]
		var was_amount := Director.filter_amount(first)
		fsl.set_value_no_signal(0.0)
		fcb.button_pressed = true
		if Director.filter_amount(first) <= 0.0:
			print("vpick: FAILED - ticking '%s' on left it at 0 (the dial was empty)" % first)
			fails += 1
		elif not is_equal_approx(Director.filter_amount(first), float(Filters.DEFAULTS[first])):
			print("vpick: FAILED - ticking '%s' applied %.2f, not its default %.2f"
				% [first, Director.filter_amount(first), float(Filters.DEFAULTS[first])])
			fails += 1
		else:
			print("vpick: ticking '%s' set the Director to its default %.2f"
				% [first, Director.filter_amount(first)])
		fcb.button_pressed = false
		if Director.filter_amount(first) != 0.0:
			print("vpick: FAILED - unticking '%s' left it at %.2f"
				% [first, Director.filter_amount(first)])
			fails += 1
		# LEAVE THE USER'S LOOK ALONE, flushed rather than left to the debounce - same reason
		# the vehicle restore above is flushed.
		Director.set_filter(first, was_amount)
		Director._save_pacing()
		Settings.flush()
	print("vpick: %s" % ("ALL OK" if fails == 0 else "%d FAILURE(S)" % fails))
	for _i in 3:
		await get_tree().process_frame
	get_tree().quit(fails)
