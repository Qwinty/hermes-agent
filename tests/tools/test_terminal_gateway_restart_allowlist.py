"""Tests for the local gateway-restart allowlist in terminal_tool.

The upstream hard-block (commit 245b95b09) rejects all gateway lifecycle
commands when ``_HERMES_GATEWAY=1``. Our local override exempts
``hermes gateway restart`` because it uses SIGUSR1 graceful drain (not
SIGTERM), so child processes survive.
"""

from tools.terminal_tool import (
    _gateway_restart_command_for_child,
    _is_safe_gateway_restart_only,
)


class TestIsSafeGatewayRestartOnly:
    def test_plain_restart_allowed(self):
        assert _is_safe_gateway_restart_only("hermes gateway restart")

    def test_blocks_restart_flags_inside_gateway(self):
        assert not _is_safe_gateway_restart_only("hermes gateway restart --system")
        assert not _is_safe_gateway_restart_only("hermes gateway restart --all")
        assert not _is_safe_gateway_restart_only("hermes gateway restart --foo=bar")

    def test_blocks_shell_substitution_in_flag_value(self):
        assert not _is_safe_gateway_restart_only("hermes gateway restart --x=$(id)")
        assert not _is_safe_gateway_restart_only("hermes gateway restart --x=`id`")
        assert not _is_safe_gateway_restart_only(
            "hermes gateway restart --x=$(pkill${IFS}-f${IFS}hermes-gateway)"
        )
        assert not _is_safe_gateway_restart_only("hermes gateway restart --x=<(id)")
        assert not _is_safe_gateway_restart_only("hermes gateway restart --x=>/tmp/pwn")

    def test_blocks_newline_separated_tokens(self):
        assert not _is_safe_gateway_restart_only("hermes\ngateway\nrestart")
        assert not _is_safe_gateway_restart_only("hermes gateway restart\n--all")

    def test_blocks_uppercase_command(self):
        assert not _is_safe_gateway_restart_only("HERMES GATEWAY RESTART")

    def test_restart_double_spaces_allowed(self):
        assert _is_safe_gateway_restart_only("hermes  gateway  restart")

    def test_systemctl_restart_blocked(self):
        assert not _is_safe_gateway_restart_only(
            "systemctl --user restart hermes-gateway"
        )

    def test_pkill_blocked(self):
        assert not _is_safe_gateway_restart_only("pkill -f hermes.*gateway")

    def test_hermes_gateway_stop_blocked(self):
        assert not _is_safe_gateway_restart_only("hermes gateway stop")

    def test_hermes_gateway_start_blocked(self):
        assert not _is_safe_gateway_restart_only("hermes gateway start")

    def test_restart_combined_with_systemctl_blocked(self):
        assert not _is_safe_gateway_restart_only(
            "hermes gateway restart && systemctl restart hermes-gateway"
        )

    def test_restart_combined_with_pkill_blocked(self):
        assert not _is_safe_gateway_restart_only(
            "pkill -f hermes-gateway; hermes gateway restart"
        )

    def test_env_wrapper_blocked(self):
        assert not _is_safe_gateway_restart_only(
            "env -u _HERMES_GATEWAY hermes gateway restart"
        )

    def test_pipe_blocked(self):
        assert not _is_safe_gateway_restart_only("hermes gateway restart | cat")

    def test_non_lifecycle_command_returns_false(self):
        assert not _is_safe_gateway_restart_only("ls -la")

    def test_empty_command_returns_false(self):
        assert not _is_safe_gateway_restart_only("")

    def test_child_command_unsets_gateway_marker(self):
        assert _gateway_restart_command_for_child("hermes gateway restart") == (
            "env -u _HERMES_GATEWAY hermes gateway restart"
        )
