"""Tests for Telegram document-size cap.

The public Telegram Bot API caps `getFile` at 20MB. A locally-hosted
`telegram-bot-api` server raises that ceiling to 2GB. We treat the presence
of `extra.base_url` as the explicit opt-in to the higher cap.
"""

import sys
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def test_max_doc_bytes_defaults_to_20mb_without_base_url():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))
    assert adapter._max_doc_bytes == 20 * 1024 * 1024


def test_max_doc_bytes_raised_to_2gb_when_base_url_set():
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"base_url": "http://localhost:8081/bot"},
        )
    )
    assert adapter._max_doc_bytes == 2 * 1024 * 1024 * 1024


def test_max_doc_bytes_empty_base_url_keeps_default():
    """An empty/falsy `base_url` should not flip the cap — only a real URL does."""
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="***", extra={"base_url": ""}),
    )
    assert adapter._max_doc_bytes == 20 * 1024 * 1024


def test_local_mode_send_document_passes_path_to_bot(tmp_path):
    media_path = tmp_path / "large.zip"
    media_path.write_bytes(b"x")
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "base_url": "http://localhost:8081/bot",
                "local_mode": True,
            },
        )
    )
    adapter._bot = SimpleNamespace(send_document=object())
    captured = {}

    async def fake_send(send_func, kwargs, metadata, reply_to_id, kind, reset_media=None):
        captured.update(kwargs)
        captured["reset_media"] = reset_media
        return SimpleNamespace(message_id=123)

    adapter._send_with_dm_topic_reply_anchor_retry = fake_send

    result = asyncio.run(adapter.send_document("273403055", str(media_path)))

    assert result.success is True
    assert captured["document"] == str(media_path)
    assert captured["reset_media"] is None
