"""Discord bot integration for Praxis."""

import asyncio
import os
import threading
from typing import Any, Optional

from praxis.integrations.base import BaseIntegration, IntegrationSpec


class DiscordBot:
    """Manages Discord bot connection and message handling."""

    def __init__(self, token: str, nickname: str, generator, tokenizer):
        self.token = token
        self.nickname = nickname
        self.generator = generator
        self.tokenizer = tokenizer
        self.client = None
        self._loop = None
        self._thread = None
        self.error = None
        self.ready = False  # Flag to track when bot is connected

    def start(self):
        """Start the Discord bot in a separate thread."""
        try:
            import discord
        except ImportError:
            print("discord.py not installed. Run: pip install discord.py")
            return False

        self._thread = threading.Thread(target=self._run_bot_async, daemon=True)
        self._thread.start()

        # Wait for connection (wait for on_ready event)
        import time

        max_wait = 30
        wait_time = 0
        while not self.ready and wait_time < max_wait:
            if self.error:
                print(f"Discord bot error: {self.error}")
                return False
            time.sleep(0.5)
            wait_time += 0.5

        if self.ready:
            return True
        else:
            print(f"Discord bot failed to connect within {max_wait}s timeout")
            if self.error:
                print(f"Last error: {self.error}")
            return False

    def _run_bot_async(self):
        """Run the Discord bot in an async event loop."""
        import discord

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._setup_bot())
        except Exception as e:
            self.error = str(e)
            print(f"Discord bot error: {e}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()

    async def _setup_bot(self):
        """Set up and run the Discord bot."""
        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.dm_messages = True

        self.client = discord.Client(intents=intents)

        # Store reference to self for the event handler
        bot_self = self

        @self.client.event
        async def on_ready():
            bot_self.ready = True
            # Use bot's display name if not provided via CLI
            if not bot_self.nickname:
                # display_name returns global_name if set, otherwise username
                bot_self.nickname = bot_self.client.user.display_name
            print(f"Discord bot '{bot_self.nickname}' connected as {bot_self.client.user}")

        @self.client.event
        async def on_message(message):
            await bot_self._handle_message(message)

        await self.client.start(self.token)

    async def _handle_message(self, message):
        """Handle incoming Discord messages."""
        import discord

        # Ignore messages from the bot itself
        if message.author == self.client.user:
            return

        # Check if we should respond
        should_respond = False

        # 1. Bot is @mentioned
        if self.client.user in message.mentions:
            should_respond = True

        # 2. Message is a reply to a bot message
        if message.reference and message.reference.resolved:
            if message.reference.resolved.author == self.client.user:
                should_respond = True

        # 3. Message is a DM
        if isinstance(message.channel, discord.DMChannel):
            should_respond = True

        if not should_respond:
            return

        # Show typing indicator while generating
        async with message.channel.typing():
            try:
                response = await self._generate_response(message)
                if response:
                    # Split long messages (Discord limit is 2000 chars)
                    # 33% chance to reply, 66% chance to send regular message
                    import random
                    use_reply = random.random() < 0.33

                    for i in range(0, len(response), 2000):
                        chunk = response[i : i + 2000]
                        if use_reply:
                            await message.reply(chunk)
                        else:
                            await message.channel.send(chunk)
            except Exception as e:
                print(f"Error generating response: {e}")
                await message.channel.send("Sorry, I encountered an error generating a response.")

    async def _generate_response(self, message) -> Optional[str]:
        """Generate a response using the Praxis generator."""
        from praxis.data.config import SYSTEM_PROMPT

        # Fetch recent channel history
        history = []
        try:
            async for msg in message.channel.history(limit=20):
                history.append(msg)
        except Exception as e:
            print(f"Error fetching history: {e}")
            history = [message]

        # Reverse to get chronological order
        history.reverse()

        # Start with system and developer prompts (matching training data format)
        formatted_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "developer", "content": "chat"},
        ]

        # Add conversation history
        for msg in history:
            if msg.author == self.client.user:
                # Bot's own messages are "assistant"
                formatted_messages.append(
                    {"role": "assistant", "content": msg.content}
                )
            else:
                # Other users' messages include username for context
                content = f"{msg.author.display_name}: {msg.content}"
                # Remove bot mentions from the content to clean up the prompt
                if self.client.user:
                    content = content.replace(f"<@{self.client.user.id}>", "").strip()
                formatted_messages.append({"role": "user", "content": content})

        # Need at least one user message beyond the prompts
        if len(formatted_messages) <= 2:
            return None

        # Run generation in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self._call_generator, formatted_messages
        )

        return response

    def _call_generator(self, messages: list) -> Optional[str]:
        """Call the generator synchronously (runs in thread pool)."""
        from praxis.api.utils import generate_from_messages

        try:
            result = generate_from_messages(
                messages=messages,
                generator=self.generator,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                truncate_to=2048,
            )
            return result
        except Exception as e:
            print(f"Generator error: {e}")
            return None

    def stop(self):
        """Stop the Discord bot."""
        if self.client and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.client.close(), self._loop
                )
                future.result(timeout=5.0)
            except Exception:
                pass

        if self._loop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass

        print("Discord bot disconnected")


# Global bot instance
_bot = None


class Integration(BaseIntegration):
    """Discord bot integration for Praxis."""

    def __init__(self, spec: IntegrationSpec):
        super().__init__(spec)
        self.bot = None

    def add_cli_args(self, parser) -> None:
        """Add Discord CLI arguments."""
        networking_group = None

        for group in parser._action_groups:
            if group.title == "networking":
                networking_group = group
                break

        if networking_group is None:
            networking_group = parser.add_argument_group("networking")

        networking_group.add_argument(
            "--discord",
            action="store_true",
            default=False,
            help="Enable Discord bot integration",
        )
        networking_group.add_argument(
            "--discord-token",
            type=str,
            default=None,
            help="Discord bot token (can also be set via DISCORD_TOKEN env var)",
        )
        networking_group.add_argument(
            "--discord-nickname",
            type=str,
            default=None,
            help="Bot's display name for prompt formatting (defaults to bot's Discord username)",
        )

    def on_api_server_start(self, host: str, port: int) -> None:
        """Start the Discord bot when API server starts.

        Args:
            host: API server host
            port: API server port
        """
        global _bot

        # Check if Discord is enabled
        try:
            from praxis.cli import get_cli_args

            args = get_cli_args()
            if not getattr(args, "discord", False):
                return
        except Exception:
            return

        if _bot is not None:
            print("Discord bot already running")
            return

        # Load from .env
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        # Get token from args or environment
        token = getattr(args, "discord_token", None) or os.getenv("DISCORD_TOKEN")
        if not token:
            print(
                "Discord error: No token provided. Set DISCORD_TOKEN env var or use --discord-token"
            )
            return

        # Get nickname (use bot's username as default)
        nickname = getattr(args, "discord_nickname", None)

        # Get generator and tokenizer from Flask app
        try:
            from praxis import api

            generator = api.app.config.get("api_server")
            if generator:
                generator = generator.generator
            else:
                generator = api.app.config.get("generator")

            tokenizer = api.app.config.get("tokenizer")

            if not generator:
                print("Discord error: Generator not available")
                return
        except Exception as e:
            print(f"Discord error: Could not access generator: {e}")
            return

        print(f"Starting Discord bot...")
        _bot = DiscordBot(token, nickname, generator, tokenizer)
        success = _bot.start()

        if success:
            print("Discord bot is online")
        else:
            print("Discord bot failed to start")
            _bot = None

    def cleanup(self) -> None:
        """Clean up Discord bot resources."""
        global _bot

        if _bot is not None:
            _bot.stop()
            _bot = None
