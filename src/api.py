"""Scaffolding to host your LangChain Chatbot on Steamship and connect it to Telegram."""
import logging
from typing import Type

import requests
from pydantic import Field
from steamship.invocable import PackageService, post, Config

from babyagi import solve_agi_problem
from response_cache import already_responded, record_response


class TelegramBuddyConfig(Config):
    """Config object containing required parameters to initialize a MyPackage instance."""

    bot_token: str = Field(description="The secret token for your Telegram bot")


class LangChainTelegramChatbot(PackageService):
    """Deploy LangChain chatbots and connect them to Telegram."""

    config: TelegramBuddyConfig

    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return TelegramBuddyConfig

    def instance_init(self) -> None:
        """Connect the instance to telegram."""
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
        resp = requests.get(f"https://api.telegram.org/bot{self.config.bot_token}/getMe").json()
        logging.info(f"/info: {resp}")
        return {"telegram": resp.get("result")}

    @post("respond", public=True)
    def respond(self, update_id: int, message: dict) -> str:
        """Telegram webhook contract."""
        message_text = message["text"]
        chat_id = message["chat"]["id"]
        message_id = message["message_id"]

        try:
            if already_responded(self.client, chat_id, message_id):
                logging.info(f"Skip message {chat_id} {message_id}")
                return "ok"

            record_response(self.client, chat_id, message_id)

            requests.get(
                f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
                params={
                    "chat_id": chat_id,
                    "text": f"Hey! I'm going to solve the objective {message_text}",
                },
            )

            for message in solve_agi_problem(self.client, message_text):
                requests.get(
                    f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
                    params={"chat_id": chat_id, "text": message},
                )
        except Exception as e:
            requests.get(
                f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
                params={"chat_id": chat_id, "text": f"I'm sorry something went wrong, "
                                                    f"here's the exception I received: {e}"},
            )

        return "ok"
