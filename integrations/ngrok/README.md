# ngrok Integration

HTTP tunneling for remote access to the Praxis dashboard.

## Activation

Add the `--ngrok` flag when running Praxis:
```bash
python main.py --ngrok
```

## Features

- **Remote Access**: Exposes your local dashboard to the internet via secure tunnel
- **Automatic Setup**: Tunnel URL is displayed in the console output
- **Collaboration**: Share your dashboard URL with team members
- **No Configuration**: Works out of the box with ngrok's free tier

## Requirements

The ngrok Python package is automatically installed when this integration is activated.

## Usage

When activated, you'll see output like:
```
[ngrok] Tunnel established at: https://abc123.ngrok.io
```

Share this URL to allow remote access to your training dashboard. The tunnel remains active for the duration of your training session.

## Security Note

The ngrok tunnel provides public access to your dashboard. While the dashboard is read-only for metrics viewing, be mindful when sharing URLs.