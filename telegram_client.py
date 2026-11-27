"""[B] Send messages to the Telegram channel.

The bot token comes from the environment (.env locally, server secrets in production).
It is never printed or logged: exception texts from `requests` contain the full URL with
the token, so only the exception TYPE is logged.
"""
import logging
import os
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv not installed
    pass

log = logging.getLogger(__name__)
API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_message(text, token=None, channel=None, retries=3, timeout=30, sleep=time.sleep) -> bool:
    """Post text to the channel. Returns True only if Telegram accepted it.

    Network errors, 429 and 5xx are retried; other 4xx errors (wrong token, wrong channel,
    bot is not an admin) cannot be fixed by retrying and return False at once.
    """
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    channel = channel or os.getenv("TELEGRAM_CHANNEL_ID")
    if not token or not channel:
        log.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHANNEL_ID is not set")
        return False

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL.format(token=token),
                                 data={"chat_id": channel, "text": text}, timeout=timeout)
            if resp.status_code == 200:
                return True
            log.error("Telegram answered %s", resp.status_code)
            if resp.status_code != 429 and 400 <= resp.status_code < 500:
                return False
        except requests.RequestException as exc:
            log.error("Telegram request failed: %s", type(exc).__name__)
        if attempt < retries:
            sleep(2 * attempt)
    return False
