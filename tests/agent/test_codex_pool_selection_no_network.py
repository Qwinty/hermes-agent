import asyncio
import threading
import time

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    STATUS_EXHAUSTED,
    CredentialPool,
    PooledCredential,
    _CodexUsageStatus,
)


@pytest.mark.asyncio
async def test_codex_select_offloads_live_usage_probe_from_running_event_loop(
    monkeypatch,
):
    entry = PooledCredential(
        provider="openai-codex",
        id="cred-one",
        label="stale exhaustion",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual",
        access_token="opaque-test-token",
        last_status=STATUS_EXHAUSTED,
        last_status_at=time.time(),
        last_error_code=429,
        last_error_reason="usage_limit_reached",
        last_error_reset_at=time.time() + 3600,
    )
    pool = CredentialPool("openai-codex", [entry])
    started = threading.Event()
    release = threading.Event()

    def fake_usage_status(_candidate):
        started.set()
        release.wait(timeout=1.0)
        return _CodexUsageStatus(available=True, allowed=True)

    monkeypatch.setattr(
        "agent.credential_pool._fetch_codex_entry_usage_status",
        fake_usage_status,
    )

    timer = threading.Timer(0.25, release.set)
    timer.start()
    before = time.monotonic()
    try:
        selected = pool.select()
        elapsed = time.monotonic() - before
        assert selected is None
        assert elapsed < 0.1
        assert await asyncio.to_thread(started.wait, 1.0) is True
        await pool._codex_reconcile_future
    finally:
        release.set()
        timer.cancel()

    assert pool.select().id == "cred-one"
