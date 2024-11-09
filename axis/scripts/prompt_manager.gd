class_name PromptManager
extends RefCounted

const SYSTEM_MESSAGE = """Assistant is an intelligent chatbot designed to help users with their questions and tasks.

Instructions:
- Provide helpful and accurate information
- Be friendly and engaging
- If unsure, admit uncertainty and suggest reliable sources"""

var conversation_history: Array[Dictionary] = []

func add_message(role: String, content: String) -> void:
	conversation_history.append({
		"role": role,
		"content": content
	})

func get_messages() -> Array:
	# Create a new array starting with the system message
	var messages = [{
		"role": "system",
		"content": SYSTEM_MESSAGE
	}]
	
	# Add all conversation messages
	messages.append_array(conversation_history)
	
	return messages

func clear_history() -> void:
	conversation_history.clear()
