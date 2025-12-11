# Discord Integration

Chat bot integration that allows interaction with Praxis via Discord.

## Activation

Add the `--discord` flag when running Praxis:
```bash
python main.py --discord
```

## Features

- **@Mentions**: Bot responds when mentioned in any channel
- **Replies**: Bot responds to replies to its messages
- **DMs**: Bot responds to direct messages
- **Context Aware**: Uses recent channel history (~20 messages) for conversation context
- **Tool Calling**: Supports tool use via the generator's built-in capabilities

## Configuration

### Required

Set your Discord bot token via environment variable:
```bash
export DISCORD_TOKEN="your-bot-token"
```

Or use the `--discord-token` flag:
```bash
python main.py --discord --discord-token "your-bot-token"
```

### Optional

Override the bot's display name for prompt formatting:
```bash
python main.py --discord --discord-nickname "CustomName"
```

If not provided, the bot uses its Discord display name.

## Requirements

The discord.py package is automatically installed when this integration is activated.

## Usage

When activated, you'll see output like:
```
Starting Discord bot...
Discord bot 'Praxis' connected as Praxis#1234
Discord bot is online
```

The bot will then respond to:
1. Messages that @mention the bot
2. Replies to the bot's messages
3. Direct messages

## Response Behavior

- 33% chance to reply directly to the triggering message
- 67% chance to send a regular message in the channel
- Long responses are automatically split at Discord's 2000 character limit

## Security Note

The bot token provides full access to your Discord bot account. Keep it secure and never commit it to version control.
