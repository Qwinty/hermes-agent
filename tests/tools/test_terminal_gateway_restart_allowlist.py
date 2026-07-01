"""Tests for gateway lifecycle handling in terminal_tool.

Terminal tool must not blanket-block gateway lifecycle commands. The gateway
already has graceful restart/recovery behavior, and ordinary approval checks are
the right layer for risky shell commands.
"""

import json

from tools import terminal_tool as tt


class _FakeEnv:
    def __init__(self):
        self.cwd = "/tmp"
        self.calls = []

    def is_alive(self):
        return True

    def execute(self, command, timeout=None, cwd=None, **kwargs):
        self.calls.append(
            {
                "command": command,
                "timeout": timeout,
                "cwd": cwd,
                "kwargs": kwargs,
            }
        )
        return {"output": "ran", "returncode": 0}


def test_gateway_lifecycle_command_reaches_terminal_backend(monkeypatch):
    """Even inside gateway, terminal_tool should not hard-block lifecycle text."""
    fake_env = _FakeEnv()
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local", "cwd": "/tmp", "timeout": 30, "local_persistent": True})
    monkeypatch.setattr(tt, "_resolve_container_task_id", lambda task_id: "default")
    monkeypatch.setattr(tt, "resolve_task_overrides", lambda task_id: {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(tt, "_active_environments", {"default": fake_env})
    monkeypatch.setattr(tt, "_last_activity", {})

    result = json.loads(
        tt.terminal_tool(
            command="systemctl --user restart hermes-gateway",
            timeout=5,
            force=True,
        )
    )

    assert result["exit_code"] == 0
    assert result["output"] == "ran"
    assert fake_env.calls[0]["command"] == "systemctl --user restart hermes-gateway"
    assert "Blocked: cannot restart" not in (result.get("error") or "")


def test_gateway_marker_is_not_rewritten_for_hermes_gateway_restart(monkeypatch):
    """terminal_tool should pass the requested command through unchanged."""
    fake_env = _FakeEnv()
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setattr(tt, "_get_env_config", lambda: {"env_type": "local", "cwd": "/tmp", "timeout": 30, "local_persistent": True})
    monkeypatch.setattr(tt, "_resolve_container_task_id", lambda task_id: "default")
    monkeypatch.setattr(tt, "resolve_task_overrides", lambda task_id: {})
    monkeypatch.setattr(tt, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(tt, "_check_all_guards", lambda *args, **kwargs: {"approved": True})
    monkeypatch.setattr(tt, "_active_environments", {"default": fake_env})
    monkeypatch.setattr(tt, "_last_activity", {})

    result = json.loads(tt.terminal_tool(command="hermes gateway restart", timeout=5, force=True))

    assert result["exit_code"] == 0
    assert fake_env.calls[0]["command"] == "hermes gateway restart"
    assert not fake_env.calls[0]["command"].startswith("env -u _HERMES_GATEWAY")
