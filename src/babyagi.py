import logging
from collections import deque
from typing import Dict, Any, List, Optional, Type

import requests
from pydantic import BaseModel, Field
from steamship.invocable import PackageService, post, Config

from babyagi import solve_agi_problem
from response_cache import already_responded, record_response


class TelegramBuddyConfig(Config):
    """Config object containing required parameters to initialize a MyPackage instance."""

    bot_token: str = Field(description="The secret token for your Telegram bot")
    model_name: str = Field(
        description="The model name, either 'gpt-3.5-turbo' or 'gpt-4'",
        default="gpt-3.5-turbo",
    )

    max_tokens: int = Field(
        description="Maximum number of tokens that should be returned by the LLM for each call. "
        "This controls the length of the text responses.",
        default=256,
        ge=1,
        le=4000,
    )

    max_iterations: int = Field(
        description="Maximum number of steps AutoGPT should take before stopping. "
        "Must be between 0 and 10 (0 is interpreted to mean to run without limits).",
        default=3,
        ge=0,
        le=10,
    )


class LangChainTelegramChatbot(PackageService):
    """Deploy LangChain chatbots and connect them to Telegram."""

    config: TelegramBuddyConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return TelegramBuddyConfig

    def instance_init(self) -> None:
        """Connect the instance to Telegram."""
        # Unlink the previous instance
        requests.get(
            f"https://api.telegram.org/bot{self.config.bot_token}/deleteWebhook"
        )
        # Reset the bot
        requests.get(f"https://api.telegram.org/bot{self.config.bot_token}/getUpdates")
        # Connect the new instance
        requests.get(
            f"https://api.telegram.org/bot{self.config.bot_token}/setWebhook",
            params={
                "url": f"{self.context.invocable_url}respond",
                "allowed_updates": ["message"],
            },
        )

    @post("info")
    def info(self) -> dict:
        """Endpoint returning information about this bot."""
        resp = requests.get(
            f"https://api.telegram.org/bot{self.config.bot_token}/getMe"
        ).json()
        logging.info(f"/info: {resp}")
        return {"telegram": resp.get("result")}

    def _send_message(self, chat_id: str, message_text: str) -> None:
        requests.get(
            f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
            params={
                "chat_id": chat_id,
                "text": message_text,
            },
        )

    @post("respond", public=True)
    def respond(self, update_id: int, message: dict) -> str:
        """Telegram webhook contract."""
        message_text = message["text"]
        chat_id = message["chat"]["id"]
        message_id = message["message_id"]

        try:
            if message_text == "/stop":
                self._send_message(chat_id, "AutoGPT has been stopped.")
                return "ok"

            if already_responded(self.client, chat_id, message_id):
                logging.info(f"Skip message {chat_id} {message_id}")
                return "ok"

            record_response(self.client, chat_id, message_id)

            if message_text == "/start":
                self._send_message(
                    chat_id,
                    "Welcome to AutoGPT for Telegram! "
                    "I'm here to help you with your objectives. "
                    "Type an objective that you want me to solve.",
                )
                return "ok"

            self._send_message(
                chat_id, f"Hey! I'm going to solve the objective {message_text}"
            )

            for message in solve_agi_problem(
                client=self.client,
                objective=message_text,
                model_name=self.config.model_name,
                max_tokens=self.config.max_tokens,
                max_iterations=self.config.max_iterations,
            ):
                self._send_message(chat_id, message)

        except Exception as e:
            self._send_message(
                chat_id,
                f"I'm sorry something went wrong, "
                f"here's the exception I received: {e}",
            )

        return "ok"


service = LangChainTelegramChatbot()
