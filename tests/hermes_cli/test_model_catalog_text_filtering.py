"""Regression tests for text/chat model filtering in picker catalogs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.models import (
    _filter_model_catalog_entries,
    _xai_curated_models,
    is_non_text_model_id,
)
from providers.base import ProviderProfile


def test_string_fallback_flags_non_text_model_ids():
    assert is_non_text_model_id("grok-imagine-image-quality")
    assert is_non_text_model_id("gpt-image-2-medium")
    assert is_non_text_model_id("xiaomi-tts-v1")
    assert is_non_text_model_id("text-embedding-3-large")
    assert not is_non_text_model_id("grok-4.3")
    assert not is_non_text_model_id("mimo-v2.5-pro")


def test_rich_catalog_filter_uses_modalities_and_endpoint_metadata():
    entries = [
        {"id": "grok-4.3", "input_modalities": ["text"], "output_modalities": ["text"]},
        {"id": "vision-chat", "modalities": {"input": ["text", "image"], "output": ["text"]}},
        {"id": "grok-imagine-image-quality", "modalities": {"input": ["text"], "output": ["image"]}},
        {"id": "xiaomi-voice", "modalities": {"input": ["text"], "output": ["audio"]}},
        {"id": "embedder", "type": "embedding"},
        {"id": "reranker", "capabilities": {"type": "rerank"}},
        {"id": "image-only-endpoint", "supported_endpoints": ["/images/generations"]},
    ]

    assert _filter_model_catalog_entries(entries) == ["grok-4.3", "vision-chat"]


def test_provider_profile_fetch_models_applies_rich_catalog_filter(monkeypatch):
    payload = {
        "data": [
            {"id": "gpt-5.5", "input_modalities": ["text"], "output_modalities": ["text"]},
            {"id": "grok-imagine-image-quality", "input_modalities": ["text"], "output_modalities": ["image"]},
            {"id": "xiaomi-tts", "type": "tts"},
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            import json

            return json.dumps(payload).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: FakeResponse())

    profile = ProviderProfile(name="test", base_url="http://example.test/v1")
    assert profile.fetch_models(api_key="sk-test") == ["gpt-5.5"]


def test_xai_curated_models_uses_agentic_models_dev_filter():
    mock_data = {
        "xai": {
            "models": {
                "grok-4.3": {"tool_call": True},
                "grok-imagine-image-quality": {
                    "tool_call": False,
                    "modalities": {"input": ["text"], "output": ["image"]},
                },
                "grok-voice-tts": {"tool_call": False},
                "grok-live-2": {"tool_call": True},
            }
        }
    }

    with patch("agent.models_dev._load_disk_cache", return_value=mock_data):
        models = _xai_curated_models()

    assert "grok-4.3" in models
    assert "grok-imagine-image-quality" not in models
    assert "grok-voice-tts" not in models
    assert "grok-live-2" not in models


def test_list_authenticated_providers_final_safety_net_filters_static_and_custom_ids(monkeypatch):
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch(
            "hermes_cli.models.cached_provider_model_ids",
            return_value=["grok-4.3", "grok-imagine-image-quality", "xai-tts-v1"],
        ),
    ):
        rows = list_authenticated_providers(
            current_provider="",
            user_providers={
                "my-cpa": {
                    "name": "My CPA",
                    "base_url": "http://127.0.0.1:8317/v1",
                    "models": ["gpt-5.5", "gpt-image-2-medium", "xiaomi-tts-v1"],
                    "discover_models": False,
                }
            },
        )

    all_models = {row["slug"]: row.get("models", []) for row in rows}
    assert "grok-imagine-image-quality" not in sum(all_models.values(), [])
    assert "xai-tts-v1" not in sum(all_models.values(), [])
    assert all_models["my-cpa"] == ["gpt-5.5"]
