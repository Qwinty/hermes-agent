from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as telegram_mod


def _make_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


@pytest.mark.asyncio
async def test_video_note_message_is_transcribed_from_mp4():
    runner = _make_runner()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = MessageEvent(
        text="",
        message_type=MessageType.VIDEO_NOTE,
        source=source,
        media_urls=["/tmp/video-note.mp4"],
        media_types=["video/mp4"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "video note words", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/video-note.mp4", None, "gateway")
    assert "video note words" in result


class _FilterToken:
    def __init__(self, *names):
        self.names = set(names)

    def __or__(self, other):
        return _FilterToken(*(self.names | other.names))


def test_telegram_media_filter_includes_video_note(monkeypatch):
    fake_filters = SimpleNamespace(
        PHOTO=_FilterToken("PHOTO"),
        VIDEO=_FilterToken("VIDEO"),
        VIDEO_NOTE=_FilterToken("VIDEO_NOTE"),
        AUDIO=_FilterToken("AUDIO"),
        VOICE=_FilterToken("VOICE"),
        Document=SimpleNamespace(ALL=_FilterToken("DOCUMENT")),
        Sticker=SimpleNamespace(ALL=_FilterToken("STICKER")),
    )
    monkeypatch.setattr(telegram_mod, "filters", fake_filters)

    combined = telegram_mod.TelegramAdapter._media_message_filter()

    assert "VIDEO_NOTE" in combined.names
