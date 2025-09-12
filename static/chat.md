# Praxis Chat

This is a modern chat interface for interacting with AI models through the Praxis API server.

## Features

- Clean, responsive chat interface
- Message history with proper conversation context
- Settings to change the API endpoint
- Support for cross-origin requests with proper error handling
- Live-reload for development

## Configuration

The chat interface automatically connects to the current server's `/messages/` endpoint by default for conversation-style interactions. You can change this in the settings to connect to any compatible API endpoint.

## Cross-Origin (CORS) Support

The Praxis API server now includes CORS support, allowing connections from any origin. This means you can:

1. Run the main API server on one port (e.g., http://localhost:2100)
2. Run another API service on a different port (e.g., http://localhost:2101)
3. Use the chat interface from any of those servers to communicate with the other servers

## Troubleshooting

If you encounter connection issues when connecting to a different server:

1. Make sure the target server is running
2. Verify the server has CORS enabled (all Praxis servers now include this by default)
3. Check the URL format in settings (should include protocol, host, port and /messages/ path)
4. Check browser console for detailed error messages

## Development

To make modifications to the chat interface:

1. Edit the HTML, CSS, and JavaScript in the templates/index.html file
2. The server automatically reloads when changes are detected
3. The browser refreshes automatically thanks to the Socket.IO live-reload integration

## Technologies Used

- Flask and Flask-SocketIO for the backend
- Flask-CORS for cross-origin resource sharing
- Vanilla JavaScript for the frontend
- WebSockets for live reload during development