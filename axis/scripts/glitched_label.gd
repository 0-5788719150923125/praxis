# This code was essentially ported from here:
# https://github.com/thetarnav/glitched-writer
extends Label
class_name GlitchedLabel

signal glitch_completed

# Safe Unicode characters that create interesting visual effects
const GLITCH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?#$%^&*<>-_/\\[]{}|="
const COMPLEX_CHARS = "┌┐└┘├┤┬┴┼━┃┏┓┗┛┣┫┳┻╋┠┨┯┷┿┝┥┰┸╂┒┑┚┙┖┕┎┍┞┟┡┢┦┧┩┪┭┮┱┲┵┶┹┺┽┾╀╁╃╄╅╆╇╈╉╊"
const NOISE_CHARS = "※｜－×￣￤○◎●△▲▽▼☉☐☑☒☢☣☠☯☸⚠⚡⛔✓✕✗✘♠♡♢♣♤♥♦♧"
const ALL_GLITCH_CHARS = GLITCH_CHARS + COMPLEX_CHARS + NOISE_CHARS

var glitch_config = {
	"steps": Vector2(2, 5), # Reduced steps range
	"interval": Vector2(30, 80), # Significantly faster intervals
	"delay": Vector2(5, 15), # Reduced delays
	"change_chance": 0.7, # Keep this for visual interest
	"ghost_chance": 0.3, # Keep this for visual interest
	"max_ghosts": 0.3, # Keep this for visual interest
	"one_at_a_time": true,
	"extra_glitch_chars": 8 # How many extra characters to append to the end
}

var target_text: String = ""
var chars: Array = []
var is_writing: bool = false
var current_char_index: int = 0

class GlitchChar:
	var writer # Reference to main node
	var current: String # Current visible character
	var target: String # Target final character
	var ghosts: Array # Ghost characters [before, after]
	var steps_left: int # Steps before settling
	var revealed: bool = false # Whether this character has started revealing
	var is_extra: bool = false # Whether this is an extra character for glitching
	
	func _init(_writer, _target: String, _is_extra: bool = false) -> void:
		writer = _writer
		target = _target
		is_extra = _is_extra
		current = writer._get_random_glyph(true) if is_extra else ""
		ghosts = [[], []]
		steps_left = writer._get_random_from_range(writer.glitch_config.steps)
		revealed = is_extra # Extra characters start revealed
		
	func get_display_text() -> String:
		if not revealed:
			return ""
		return "".join(ghosts[0]) + current + "".join(ghosts[1])
		
	func is_finished() -> bool:
		if is_extra:
			return current == "" and ghosts[0].size() == 0 and ghosts[1].size() == 0
		return revealed and current == target and ghosts[0].size() == 0 and ghosts[1].size() == 0
		
	func step() -> void:
		if not revealed:
			current = writer._get_random_glyph(true)
			return
			
		if is_extra:
			# Extra characters should eventually disappear
			if randf() < 0.2: # Chance to start disappearing
				current = ""
			else:
				current = writer._get_random_glyph(true)
		elif steps_left > 0:
			if randf() < writer.glitch_config.change_chance:
				if steps_left < 3:
					current = writer._get_random_glyph(false)
				else:
					current = writer._get_random_glyph(true)
			
			if randf() < writer.glitch_config.ghost_chance:
				var max_ghosts = floor(writer.target_text.length() * writer.glitch_config.max_ghosts)
				var total_ghosts = ghosts[0].size() + ghosts[1].size()
				
				if total_ghosts < max_ghosts:
					var ghost = writer._get_random_glyph(true)
					if randf() < 0.5:
						ghosts[0].append(ghost)
					else:
						ghosts[1].append(ghost)
				else:
					if ghosts[0].size() > 0 and randf() < 0.5:
						ghosts[0].pop_back()
					elif ghosts[1].size() > 0:
						ghosts[1].pop_back()
			
			steps_left -= 1
		else:
			current = target
			if ghosts[0].size() > 0:
				ghosts[0].pop_back()
			elif ghosts[1].size() > 0:
				ghosts[1].pop_back()

var timer: Timer

func _ready():
	timer = Timer.new()
	add_child(timer)
	timer.timeout.connect(_on_timer_timeout)

func _get_random_from_range(range_value: Vector2) -> int:
	return randi_range(int(range_value.x), int(range_value.y))

func _get_random_glyph(use_complex: bool = false) -> String:
	if use_complex:
		return ALL_GLITCH_CHARS[randi() % ALL_GLITCH_CHARS.length()]
	else:
		return GLITCH_CHARS[randi() % GLITCH_CHARS.length()]

func _get_random_interval() -> float:
	return randf_range(glitch_config.interval.x, glitch_config.interval.y) / 1000.0

func write(string: String) -> void:
	target_text = string
	current_char_index = 0
	
	# Initialize characters - all hidden initially
	chars.clear()
	
	# Add some extra characters at the start for glitching
	var num_extra_start = glitch_config.extra_glitch_chars / 2
	for _i in range(num_extra_start):
		chars.append(GlitchChar.new(self, "", true))
	
	# Add actual text characters
	for i in range(string.length()):
		chars.append(GlitchChar.new(self, string[i]))
	
	# Add some extra characters at the end for glitching
	var num_extra_end = glitch_config.extra_glitch_chars - num_extra_start
	for _i in range(num_extra_end):
		chars.append(GlitchChar.new(self, "", true))
	
	is_writing = true
	timer.start(_get_random_interval())

func _on_timer_timeout() -> void:
	if not is_writing:
		timer.stop()
		return
		
	var all_finished = true
	var display_text = ""
	
	# Calculate the valid range for glitch effects
	var glitch_start = max(0, current_char_index - glitch_config.extra_glitch_chars)
	var glitch_end = min(chars.size(), current_char_index + glitch_config.extra_glitch_chars)
	
	# Reveal and update characters
	for i in range(chars.size()):
		var ch = chars[i]
		
		# Start revealing characters progressively, including some ahead
		if i <= current_char_index + 2 and not ch.revealed and not ch.is_extra:
			ch.revealed = true
		
		# Update characters within glitch range
		if ch.revealed or (i >= glitch_start and i <= glitch_end):
			if not ch.is_finished():
				all_finished = false
				ch.step()
		
		display_text += ch.get_display_text()
	
	# Update display
	text = display_text
	
	# Move to next character if current one is stable
	if current_char_index < chars.size() and chars[current_char_index].is_finished():
		current_char_index += 1
	
	if all_finished:
		is_writing = false
		timer.stop()
		text = target_text
		glitch_completed.emit()
	else:
		timer.start(_get_random_interval())
