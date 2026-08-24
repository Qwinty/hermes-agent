"""Tests for empty model fallback — when provider is configured but model is missing."""

from unittest.mock import patch


class TestGetDefaultModelForProvider:
    """Unit tests for hermes_cli.models.get_default_model_for_provider."""

    def test_known_provider_returns_first_model(self):
        from hermes_cli.models import get_default_model_for_provider
        result = get_default_model_for_provider("openai-codex")
        # Should return first model from _PROVIDER_MODELS["openai-codex"]
        assert result
        assert isinstance(result, str)





    def test_catalog_label_overrides_constant(self):
        """A ``"default": true`` label in the cached catalog manifest wins over
        the in-repo constant, so maintainers can rotate the silent default
        without shipping a release."""
        from unittest.mock import patch

        from hermes_cli import models as models_mod

        with patch(
            "hermes_cli.model_catalog.get_default_model_from_cache",
            return_value="qwen/qwen3.8-max",
        ):
            assert (
                models_mod.get_preferred_silent_default_model("nous")
                == "qwen/qwen3.8-max"
            )
            # nous catalog carries qwen3.8-max, so the full resolver follows.
            assert (
                models_mod.get_default_model_for_provider("nous")
                == "qwen/qwen3.8-max"
            )






class TestGatewayEmptyModelFallback:
    """Test that _resolve_session_agent_runtime fills in empty model from provider catalog."""

    def test_empty_model_filled_from_provider(self):
        """When config has no model but provider is openai-codex, use first codex model."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        # Mock _resolve_gateway_model to return empty string
        # Mock _resolve_runtime_agent_kwargs to return openai-codex provider
        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Model should have been filled in from provider catalog
        assert model, "Model should not be empty when provider is known"
        assert isinstance(model, str)
        assert kwargs["provider"] == "openai-codex"

    def test_nonempty_model_not_overridden(self):
        """When config has a model set, don't override it."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "test-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        assert model == "gpt-5.4", "Explicit model should not be overridden"

    def test_empty_model_no_provider_stays_empty(self):
        """When both model and provider are empty, model stays empty."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}

        with patch("gateway.run._resolve_gateway_model", return_value=""), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "",
                 "api_key": "test-key",
                 "base_url": "https://example.com",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime()

        # Can't fill in a default without knowing the provider
        assert model == ""
    def test_telegram_guest_mode_model_overrides_runtime(self):
        """Telegram guest-mode calls can use a dedicated model/provider."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        user_config = {
            "telegram": {
                "guest_mode_model": {
                    "provider": "custom:CommandCode",
                    "model": "deepseek-v4-pro",
                },
            },
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.5"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "codex-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "custom",
                 "api_key": "dummy",
                 "base_url": "http://127.0.0.1:8099/v1",
                 "api_mode": "chat_completions",
             }):
            model, kwargs = runner._resolve_session_agent_runtime(
                user_config=user_config,
                guest_mode_invocation=True,
            )

        assert model == "deepseek-v4-pro"
        assert kwargs["provider"] == "custom"
        assert kwargs["base_url"] == "http://127.0.0.1:8099/v1"
        assert kwargs["api_mode"] == "chat_completions"

    def test_telegram_guest_mode_reasoning_effort_overrides_global(self):
        """Telegram guest-mode calls can use a dedicated reasoning effort."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_reasoning_overrides = {}
        user_config = {
            "agent": {"reasoning_effort": "high"},
            "telegram": {
                "guest_mode_model": {
                    "provider": "openai-codex",
                    "model": "gpt-5.5",
                    "reasoning_effort": "low",
                },
            },
        }

        reasoning = runner._resolve_session_reasoning_config(
            user_config=user_config,
            guest_mode_invocation=True,
        )

        assert reasoning == {"enabled": True, "effort": "low"}

    def test_telegram_guest_mode_reasoning_effort_does_not_override_normal_calls(self):
        """The dedicated guest reasoning level must not affect regular sessions."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_reasoning_overrides = {}
        user_config = {
            "agent": {"reasoning_effort": "high"},
            "telegram": {"guest_mode_reasoning_effort": "low"},
        }

        with patch("gateway.run._load_gateway_runtime_config", return_value=user_config):
            reasoning = runner._resolve_session_reasoning_config(
                user_config=user_config,
                guest_mode_invocation=False,
            )

        assert reasoning == {"enabled": True, "effort": "high"}

    def test_telegram_guest_mode_model_does_not_override_normal_calls(self):
        """The dedicated guest model must not affect regular Telegram sessions."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        user_config = {
            "telegram": {
                "guest_mode_model": {
                    "provider": "custom:CommandCode",
                    "model": "deepseek-v4-pro",
                },
            },
        }

        with patch("gateway.run._resolve_gateway_model", return_value="gpt-5.5"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "openai-codex",
                 "api_key": "codex-key",
                 "base_url": "https://chatgpt.com/backend-api/codex",
                 "api_mode": "codex_responses",
             }):
            model, kwargs = runner._resolve_session_agent_runtime(
                user_config=user_config,
                guest_mode_invocation=False,
            )

        assert model == "gpt-5.5"
        assert kwargs["provider"] == "openai-codex"
        assert kwargs["api_mode"] == "codex_responses"



class TestResolveGatewayModel:
    """Test _resolve_gateway_model reads model from config correctly."""

    def test_returns_default_key(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"default": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_model_key_fallback(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {"model": "gpt-5.4"}}) == "gpt-5.4"

    def test_returns_empty_when_missing(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": {}}) == ""

    def test_returns_empty_when_no_model_section(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({}) == ""

    def test_string_model_config(self):
        from gateway.run import _resolve_gateway_model
        assert _resolve_gateway_model({"model": "my-model"}) == "my-model"
