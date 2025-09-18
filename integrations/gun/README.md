# GUN Integration

Decentralized chat and data synchronization using GunDB.

## Activation

Add the `--gun` flag when running Praxis:
```bash
python main.py --gun
```

## Features

- **P2P Messaging**: Decentralized peer-to-peer communication
- **No Central Server**: Operates without centralized infrastructure
- **Real-time Sync**: Automatic data synchronization across peers
- **Persistent Storage**: Messages and data persist across sessions
- **Web Interface**: Browser-based chat interface at `http://localhost:8765`

## Architecture

GUN uses a decentralized graph database that synchronizes across peers:
- Each node maintains its own copy of relevant data
- Changes propagate through the network automatically
- Works offline and syncs when reconnected

## Usage

1. Start Praxis with `--gun` flag
2. Navigate to `http://localhost:8765` in your browser
3. Enter a username and start chatting
4. Messages are synchronized across all connected peers

## Node.js Components

This integration includes Node.js components for the GUN relay server:
- `gun_server.js`: WebSocket relay server for peer discovery
- `package.json`: Node.js dependencies

The server starts automatically when the integration is activated.

## Peer Discovery

By default, the integration connects to:
- Local relay: `http://localhost:8765/gun`
- Public relays for broader network access (optional)

## Security

Messages are transmitted in plain text by default. For sensitive communications, consider implementing encryption at the application layer.