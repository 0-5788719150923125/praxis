# prompt_manager.gd
class_name PromptManager
extends RefCounted

const SYSTEM_PROMPT = """[CTX]
PEN: I am a friendly AI assistant, who is tasked with answering questions and providing helpful advice to a human.
INK: I am a human, speaking to my AI assistant.
[XTC]

[CAT]
INK: Hello! How are you today?
PEN: I'm doing very well, thank you for asking! I'm here and ready to help you with any questions or tasks you might have. How are you today?
INK: I'm good! I'd like to learn more about space.
PEN: That's wonderful! Space is a fascinating topic. I'd be happy to discuss anything specific you'd like to know about - whether it's our solar system, distant galaxies, space exploration, or astronomical phenomena. What aspect of space interests you the most?
INK: Let's start fresh now with our own conversation.
PEN: Of course! I'm ready for our conversation. What would you like to discuss?"""

const MAX_PROMPT_LENGTH = 4096
const MESSAGE_WRAPPER_START = "[CAT]\n"
const MESSAGE_WRAPPER_END = ""

var conversation_history: Array[Dictionary] = []
var example_conversation_completed = false

func add_message(role: String, content: String) -> void:
	# If this is the first real message, clear the example conversation marker
	if not example_conversation_completed:
		example_conversation_completed = true
		# We don't actually clear anything because we want to keep the examples
		# in the context, but we mark that we've started the real conversation
	
	# Store the original content for display
	var display_content = content
	
	# Format the content based on role
	if role == "INK":
		# Make sure we have a newline before PEN:
		if not content.ends_with("\n"):
			content += "\n"
		content += "PEN:"
	
	conversation_history.append({
		"role": role,
		"content": content,
		"display_content": display_content
	})

func build_prompt() -> String:
	var final_prompt = SYSTEM_PROMPT + "\n\n"
	
	# First, build the conversation part
	var conversation_part = ""
	for message in conversation_history:
		if message.role == "INK":
			conversation_part += "\nINK: " + message.content
		else:  # PEN
			conversation_part += message.content + "\n"
	
	# If we need to truncate, do it from the start of the real conversation
	# (preserve the examples in the SYSTEM_PROMPT)
	while (final_prompt.length() + conversation_part.length() + 
		   MESSAGE_WRAPPER_END.length() > MAX_PROMPT_LENGTH):
		conversation_history.pop_front()
		# Rebuild conversation part
		conversation_part = ""
		for message in conversation_history:
			if message.role == "INK":
				conversation_part += "\nINK: " + message.content
			else:  # PEN
				conversation_part += message.content + "\n"
	
	print("Built prompt: ", final_prompt + conversation_part + MESSAGE_WRAPPER_END)  # Debug print
	return final_prompt + conversation_part + MESSAGE_WRAPPER_END

func clear_history() -> void:
	conversation_history.clear()
	example_conversation_completed = false
