"""Regression tests for gateway long-running status notifications."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


class NotifyCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent: list[dict] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        message_id = f"msg-{len(self.sent) + 1}"
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "message_id": message_id,
                "metadata": metadata,
                "monotonic": time.monotonic(),
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id, "type": "dm"}


class QueuedFollowupAgent:
    """First turn finishes quickly; queued follow-up stays alive briefly.

    The first _run_agent invocation recurses into the queued follow-up before
    its own "Still working..." timer has fired. Without cancelling the outer
    timer, both the outer and inner notifiers send status bubbles for the same
    Telegram topic a few milliseconds apart.
    """

    calls = 0

    def __init__(self, **kwargs):
        self.tools = []
        self.model = kwargs.get("model", "fake-model")
        self.session_id = kwargs.get("session_id")
        self.is_interrupted = False
        self.tool_progress_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.interim_assistant_callback = None
        self.status_callback = None
        self.reasoning_config = None
        self.service_tier = None
        self.request_overrides = {}

    def interrupt(self, message=None) -> None:
        self.is_interrupted = True

    def get_activity_summary(self) -> dict:
        return {
            "last_activity_ts": time.time(),
            "last_activity_desc": "running fake tool",
            "seconds_since_activity": 0.0,
            "current_tool": "execute_code",
            "api_call_count": 5,
            "max_iterations": 200,
            "budget_used": 5,
        }

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls += 1
        if type(self).calls == 1:
            time.sleep(0.01)
            return {
                "final_response": "first done",
                "messages": [{"role": "assistant", "content": "first done"}],
                "api_calls": 1,
            }

        # Long enough for exactly one inner notifier tick at 50ms, but short
        # enough that a second inner tick should not fire before cleanup.
        time.sleep(0.085)
        return {
            "final_response": "second done",
            "messages": [{"role": "assistant", "content": "second done"}],
            "api_calls": 5,
        }


def _make_runner(adapter: NotifyCaptureAdapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._long_running_notify_tokens = {}
    runner._session_run_generation = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_native_image_paths_by_session = {}
    runner._pending_skills_reload_notes = {}
    runner._queued_events = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False, emit=None)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=SimpleNamespace(
            enabled=False,
            transport="off",
            edit_interval=0.1,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0.0,
        ),
    )
    return runner


def _install_fakes(monkeypatch):
    gateway_run = importlib.import_module("gateway.run")

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", QueuedFollowupAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    monkeypatch.setattr(gateway_run, "_reload_runtime_env_preserving_config_authority", lambda: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"display": {"interim_assistant_messages": False, "streaming": False}},
    )
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_resolve_session_agent_runtime",
        lambda self, **kwargs: ("fake-model", {"api_key": "fake", "provider": "test"}),
    )
    monkeypatch.setattr(
        gateway_run.GatewayRunner,
        "_resolve_turn_agent_config",
        lambda self, msg, model, runtime: {"model": model, "runtime": runtime},
    )

    import tools.approval as approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda *a, **k: None)
    monkeypatch.setattr(approval, "unregister_gateway_notify", lambda *a, **k: None)
    monkeypatch.setattr(approval, "set_current_session_key", lambda key: None)
    monkeypatch.setattr(approval, "reset_current_session_key", lambda token: None)

    return gateway_run


@pytest.mark.asyncio
async def test_queued_followup_cancels_outer_still_working_notifier(monkeypatch):
    adapter = NotifyCaptureAdapter()
    runner = _make_runner(adapter)
    _install_fakes(monkeypatch)

    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "off")
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.05")
    monkeypatch.setenv("HERMES_AGENT_TIMEOUT", "0")
    QueuedFollowupAgent.calls = 0

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="273403055",
        chat_type="dm",
        user_id="maxim",
        thread_id="356070",
    )
    session_key = "agent:main:telegram:dm:273403055:356070"
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="first",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
        event_message_id="root-1",
    )

    assert result["final_response"] == "second done"
    first_done_at = next(m["monotonic"] for m in adapter.sent if m["content"] == "first done")
    stale_outer = [
        m
        for m in adapter.sent
        if "Still working" in m["content"]
        and m["monotonic"] > first_done_at
        and m["metadata"].get("telegram_reply_to_message_id") == "root-1"
    ]
    assert stale_outer == [], adapter.sent

    queued_statuses = [
        m
        for m in adapter.sent
        if "Still working" in m["content"]
        and m["monotonic"] > first_done_at
        and m["metadata"].get("telegram_reply_to_message_id") == "queued-1"
    ]
    assert queued_statuses, adapter.sent
    assert any("running: execute_code" in m["content"] for m in queued_statuses)
