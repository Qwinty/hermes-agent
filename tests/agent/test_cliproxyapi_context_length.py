"""Regression tests for CLIProxyAPI context-window detection."""

from __future__ import annotations

from unittest.mock import patch


def test_loopback_provider_plugin_does_not_make_cpa_url_known_provider():
    """A user plugin on 127.0.0.1 must not classify every loopback proxy.

    Maxim runs CommandCode on 127.0.0.1:8099 and CLIProxyAPI on
    127.0.0.1:8317.  If the CommandCode provider profile contributes the bare
    127.0.0.1 host to the global URL→provider map, CPA is misclassified as
    CommandCode; Hermes then skips CPA's /v1/models metadata and falls through
    to the generic OpenAI gpt-5.5 fallback (1.05M) instead of CPA's Codex
    metadata (272K).
    """
    from agent import model_metadata as mm

    assert mm._infer_provider_from_url("http://127.0.0.1:8317/v1") is None
    assert not mm._is_known_provider_base_url("http://127.0.0.1:8317/v1")


def test_cliproxyapi_gpt55_uses_endpoint_context_length_before_family_fallback():
    from agent.model_metadata import get_model_context_length

    base_url = "http://127.0.0.1:8317/v1"
    with patch(
        "agent.model_metadata.fetch_endpoint_model_metadata",
        return_value={"gpt-5.5": {"context_length": 272_000}},
    ), patch("agent.model_metadata.get_cached_context_length", return_value=None):
        assert (
            get_model_context_length(
                "gpt-5.5",
                base_url=base_url,
                api_key="test-key",
                provider="cliproxyapi",
            )
            == 272_000
        )


def test_private_provider_profile_host_not_auto_registered(monkeypatch):
    from agent import model_metadata as mm

    assert mm._is_ambiguous_local_provider_host("127.0.0.1")
    assert mm._is_ambiguous_local_provider_host("localhost")
    assert mm._is_ambiguous_local_provider_host("10.0.0.5")
    assert not mm._is_ambiguous_local_provider_host("api.example.com")
