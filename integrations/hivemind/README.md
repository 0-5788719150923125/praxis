# Hivemind Integration

This integration provides decentralized deep learning capabilities through the Hivemind library, enabling distributed training across a peer-to-peer network.

## Features

- **Decentralized Training**: Connect to the Hivemind swarm for distributed model training
- **P2P Networking**: Automatic peer discovery and connection management
- **Remote Experts**: Access and use experts hosted on other nodes in the network
- **Fault Tolerance**: Handles network failures and peer disconnections gracefully

## Installation

The Hivemind integration requires the `hivemind` package:

```bash
pip install hivemind>=1.1.0
```

## Usage

### Command Line

Enable Hivemind in your training with:

```bash
python main.py --hivemind
```

Optionally provide initial bootstrap peers:

```bash
python main.py --hivemind --initial-peers "/dns/peer1.example.com/tcp/31337/p2p/..." 
```

### Configuration

The integration is automatically loaded when the `--hivemind` flag is provided. It will:

1. Initialize a DHT (Distributed Hash Table) node
2. Connect to the Hivemind network using default bootstrap peers
3. Start discovering and registering experts
4. Enable distributed training across the network

### Architecture

The integration follows the standard Praxis integration pattern:

- **`spec.yaml`**: Defines dependencies, conditions, and capabilities
- **`main.py`**: Contains the core implementation including:
  - `HivemindOrchestrator`: Main orchestration class for Hivemind
  - `HivemindWrapper`: Wrapper for remote experts
  - `Integration`: Standard integration class following BaseIntegration
- **`__init__.py`**: Module exports

### How It Works

1. When `--hivemind` is enabled, the integration is loaded
2. The `on_decoder_init` hook injects a `HivemindOrchestrator` instance into decoders
3. The orchestrator:
   - Registers local experts to serve to the network
   - Discovers remote experts from other nodes
   - Manages routing between local and remote experts
4. Training proceeds with transparent distribution across the network

### Network Configuration

By default, the integration connects to public bootstrap peers from the Petals network. You can customize this by:

1. Providing your own bootstrap peers via `--initial-peers`
2. Modifying the `PUBLIC_INITIAL_PEERS` list in `main.py`
3. Using IPFS bootstrap peers (experimental)

### Limitations

When using Hivemind, be aware of these constraints:

1. All expert inputs must be Tensors (no None values)
2. Input shapes must be constant
3. All inputs/outputs must be part of the computation graph
4. Network latency affects training speed
5. Requires stable internet connection

### Troubleshooting

If you encounter connection issues:

1. Check your firewall allows P2P connections
2. Ensure ports 31337-31338 are accessible
3. Try different bootstrap peers
4. Check network connectivity with `ping bootstrap1.petals.dev`

For more information about Hivemind, see the [official documentation](https://github.com/learning-at-home/hivemind).