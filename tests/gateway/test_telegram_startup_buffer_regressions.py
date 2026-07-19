"""Late-finding regressions for Telegram startup media buffering."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


class _DummyTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="100",
        chat_type="dm",
        user_id="1",
        user_name="Tester",
    )


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(enabled=True, token="fake")
    adapter._pending_messages = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._media_downloads_in_progress_by_session = {}
    adapter.handle_message = AsyncMock()
    return adapter


def test_has_startup_media_pending_for_forwarded_text_batch():
    adapter = _make_adapter()
    source = _source()
    session_key = build_session_key(source)
    adapter._pending_text_batches[session_key] = MessageEvent(
        text="[Forwarded message | From: Alice]\n\nhello",
        message_type=MessageType.TEXT,
        source=source,
    )

    assert adapter.has_startup_media_pending(session_key) is True


def test_pop_startup_media_event_consumes_forwarded_text_batch():
    adapter = _make_adapter()
    source = _source()
    session_key = build_session_key(source)
    task = _DummyTask()
    adapter._pending_text_batches[session_key] = MessageEvent(
        text="[Forwarded message | From: Alice]\n\nhello",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_text_batch_tasks[session_key] = task

    event = adapter.pop_startup_media_event(session_key)

    assert event is not None
    assert event.text.startswith("[Forwarded message | From: Alice]")
    assert session_key not in adapter._pending_text_batches
    assert session_key not in adapter._pending_text_batch_tasks
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_video_download_is_tracked_as_startup_pending(monkeypatch):
    adapter = _make_adapter()
    source = _source()
    session_key = build_session_key(source)
    started = []
    finished = []

    def track_start(event):
        key = adapter._event_session_key(event)
        started.append(key)
        return key

    def track_done(key):
        finished.append(key)

    monkeypatch.setattr(adapter, "_track_media_download_start", track_start)
    monkeypatch.setattr(adapter, "_track_media_download_done", track_done)
    monkeypatch.setattr(
        adapter,
        "_telegram_media_size_allowed",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes",
        lambda *_args, **_kwargs: "/tmp/video.mp4",
    )

    video = MagicMock()
    file_obj = AsyncMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"video"))
    file_obj.file_path = "clip.mp4"
    video.get_file = AsyncMock(return_value=file_obj)
    video.file_size = 128

    msg = MagicMock()
    msg.message_id = 1
    msg.text = ""
    msg.caption = None
    msg.photo = None
    msg.video = video
    msg.audio = None
    msg.voice = None
    msg.sticker = None
    msg.document = None
    msg.media_group_id = None
    msg.chat = SimpleNamespace(id=100, type="private", title=None, full_name="Tester")
    msg.from_user = SimpleNamespace(id=1, full_name="Tester")
    msg.message_thread_id = None

    update = SimpleNamespace(message=msg, update_id=11)
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda _msg: True)
    monkeypatch.setattr(adapter, "_should_process_message", lambda _msg: True)
    monkeypatch.setattr(adapter, "_media_message_type", lambda _msg: MessageType.VIDEO)
    monkeypatch.setattr(
        adapter,
        "_build_message_event",
        lambda _msg, msg_type, update_id=None: MessageEvent(
            text="",
            message_type=msg_type,
            source=source,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_apply_telegram_group_observe_attribution",
        lambda event: event,
    )

    await adapter._handle_media_message(update, MagicMock())

    assert started == [session_key]
    assert finished == [session_key]
