extends Label
class_name GlitchedLabel

signal glitch_completed

# Safe Unicode characters that create interesting visual effects
const GLITCH_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?#$%^&*<>-_/\\[]{}|="
const COMPLEX_CHARS = "┌┐└┘├┤┬┴┼━┃┏┓┗┛┣┫┳┻╋┠┨┯┷┿┝┥┰┸╂┒┑┚┙┖┕┎┍┞┟┡┢┦┧┩┪┭┮┱┲┵┶┹┺┽┾╀╁╃╄╅╆╇╈╉╊"
const NOISE_CHARS = "※｜－×￣￤○◎●△▲▽▼☉☐☑☒☢☣☠☯☸⚠⚡⛔✓✕✗✘♠♡♢♣♤♥♦♧"
const ALL_GLITCH_CHARS = GLITCH_CHARS + COMPLEX_CHARS + NOISE_CHARS

var glitch_config = {
	"steps": Vector2(2, 5),        # Increased minimum steps for more visible effect
	"interval": Vector2(30, 80),  # Slightly slower intervals
	"delay": Vector2(5, 15),      # Short delays between characters
	"change_chance": 0.7,          # Increased chance to change characters
	"ghost_chance": 0.3,           # Increased ghost chance
	"max_ghosts": 0.3,            # Slightly more ghosts allowed
	"one_at_a_time": true         # Process characters sequentially
}

var target_text: String = ""
var chars: Array = []
var is_writing: bool = false
var current_char_index: int = 0

class GlitchChar:
	var writer              # Reference to main node
	var current: String     # Current visible character
	var target: String     # Target final character
	var ghosts: Array      # Ghost characters [before, after]
	var steps_left: int    # Steps before settling
	var revealed: bool = false  # Whether this character has started revealing
	
	func _init(_writer, _target: String) -> void:
		writer = _writer
		target = _target
		current = writer._get_random_glyph(true)  # Start with a complex character
		ghosts = [[], []]
		steps_left = writer._get_random_from_range(writer.glitch_config.steps)
		
	func get_display_text() -> String:
		if not revealed:
			return ""  # Hidden until it's this character's turn
		return "".join(ghosts[0]) + current + "".join(ghosts[1])
		
	func is_finished() -> bool:
		return revealed and current == target and ghosts[0].size() == 0 and ghosts[1].size() == 0
		
	func step() -> void:
		if not revealed:
			current = writer._get_random_glyph(true)
			return
			
		if steps_left > 0:
			if randf() < writer.glitch_config.change_chance:
				if steps_left < 3:  # As we get closer to finishing, use simpler chars
					current = writer._get_random_glyph(false)
				else:
					current = writer._get_random_glyph(true)
			
			if randf() < writer.glitch_config.ghost_chance:
				var max_ghosts = floor(writer.target_text.length() * writer.glitch_config.max_ghosts)
				var total_ghosts = ghosts[0].size() + ghosts[1].size()
				
				if total_ghosts < max_ghosts:
					var ghost = writer._get_random_glyph(true)  # Use complex chars for ghosts
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
		# Use the full character set including complex characters
		return ALL_GLITCH_CHARS[randi() % ALL_GLITCH_CHARS.length()]
	else:
		# Use only basic characters when getting close to the final state
		return GLITCH_CHARS[randi() % GLITCH_CHARS.length()]

func _get_random_interval() -> float:
	return randf_range(glitch_config.interval.x, glitch_config.interval.y) / 1000.0

func write(text: String) -> void:
	target_text = text
	current_char_index = 0
	
	# Initialize characters - all hidden initially
	chars.clear()
	for i in range(text.length()):
		chars.append(GlitchChar.new(self, text[i]))
	
	is_writing = true
	timer.start(_get_random_interval())

func _on_timer_timeout() -> void:
	if not is_writing:
		timer.stop()
		return
		
	var all_finished = true
	var display_text = ""
	
	# Reveal and update characters
	for i in range(chars.size()):
		var char = chars[i]
		
		# Start revealing characters progressively
		if i <= current_char_index and not char.revealed:
			char.revealed = true
		
		if char.revealed:
			if not char.is_finished():
				all_finished = false
				char.step()
		
		display_text += char.get_display_text()
	
	# Update display
	text = display_text
	
	# Move to next character if current one is stable
	if current_char_index < chars.size() and chars[current_char_index].is_finished():
		current_char_index += 1
	
	if all_finished:
		is_writing = false
		timer.stop()
		# Safety check - ensure final text matches target
		text = target_text  # Add this line
		glitch_completed.emit()
	else:
		timer.start(_get_random_interval())
