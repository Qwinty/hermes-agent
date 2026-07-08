from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_constants import parse_reasoning_effort
from providers import get_provider_profile


def _kwargs_for(model: str, effort: str | None):
    profile = get_provider_profile("cliproxyapi")
    assert profile is not None
    return ChatCompletionsTransport().build_kwargs(
        model,
        [{"role": "user", "content": "hi"}],
        tools=None,
        provider_profile=profile,
        reasoning_config=parse_reasoning_effort(effort) if effort else None,
        timeout=30,
        max_tokens=None,
        ephemeral_max_output_tokens=None,
        max_tokens_param_fn=lambda n: {"max_tokens": n},
        request_overrides=None,
        session_id="test-session",
        base_url="http://127.0.0.1:8317/v1",
        supports_reasoning=False,
    )


def test_cliproxyapi_gpt_chat_completions_emits_top_level_reasoning_effort():
    kwargs = _kwargs_for("gpt-5.5", "high")

    assert kwargs["reasoning_effort"] == "high"
    assert "extra_body" not in kwargs


def test_cliproxyapi_preserves_xhigh_for_gpt_chat_completions():
    kwargs = _kwargs_for("gpt-5.5", "xhigh")

    assert kwargs["reasoning_effort"] == "xhigh"


def test_cliproxyapi_clamps_max_to_xhigh_for_gpt_chat_completions():
    kwargs = _kwargs_for("gpt-5.5", "max")

    assert kwargs["reasoning_effort"] == "xhigh"


def test_cliproxyapi_claude_family_slug_emits_reasoning_effort():
    kwargs = _kwargs_for("neko-sonnet-5", "xhigh")

    assert kwargs["reasoning_effort"] == "xhigh"
    assert "extra_body" not in kwargs


def test_cliproxyapi_claude_family_slug_preserves_max_reasoning_effort():
    kwargs = _kwargs_for("neko-opus-4.8", "max")

    assert kwargs["reasoning_effort"] == "max"


def test_cliproxyapi_claude_family_slug_reasoning_none_disables_thinking():
    kwargs = _kwargs_for("neko-fable-5", "none")

    assert kwargs["reasoning_effort"] == "none"


def test_cliproxyapi_claude_family_matching_is_not_tied_to_neko_prefix():
    kwargs = _kwargs_for("sonnet-4.6", "high")

    assert kwargs["reasoning_effort"] == "high"


def test_cliproxyapi_claude_family_matching_handles_provider_prefixed_alias():
    kwargs = _kwargs_for("some-provider/opus-4-8", "max")

    assert kwargs["reasoning_effort"] == "max"


def test_cliproxyapi_neko_prefix_alone_does_not_emit_reasoning_effort():
    kwargs = _kwargs_for("neko-random-model", "high")

    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_cliproxyapi_does_not_emit_reasoning_effort_for_gemini_slug():
    kwargs = _kwargs_for("gemini-3.1-pro-high", "high")

    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_cliproxyapi_reasoning_none_omits_reasoning_effort():
    kwargs = _kwargs_for("gpt-5.5", "none")

    assert "reasoning_effort" not in kwargs


def test_cliproxyapi_grok_45_emits_top_level_reasoning_effort():
    kwargs = _kwargs_for("grok-4.5", "medium")

    assert kwargs["reasoning_effort"] == "medium"
    assert kwargs["extra_body"]["prompt_cache_key"] == "test-session"
    assert kwargs["extra_headers"]["x-grok-conv-id"] == "test-session"
    assert kwargs["extra_headers"]["X-Session-ID"] == "test-session"


def test_cliproxyapi_grok_45_clamps_xhigh_and_max_to_high():
    for effort in ("xhigh", "max"):
        kwargs = _kwargs_for("grok-4.5", effort)
        assert kwargs["reasoning_effort"] == "high", effort


def test_cliproxyapi_grok_45_clamps_minimal_to_low():
    kwargs = _kwargs_for("grok-4.5", "minimal")
    assert kwargs["reasoning_effort"] == "low"


def test_cliproxyapi_grok_45_reasoning_none_omits_effort_but_keeps_cache_key():
    # Grok 4.5 cannot disable reasoning; omit the dial → upstream default high.
    kwargs = _kwargs_for("grok-4.5", "none")

    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["prompt_cache_key"] == "test-session"
    assert kwargs["extra_headers"]["x-grok-conv-id"] == "test-session"


def test_cliproxyapi_grok_43_also_emits_reasoning_effort():
    kwargs = _kwargs_for("grok-4.3", "low")
    assert kwargs["reasoning_effort"] == "low"


def test_cliproxyapi_prefixed_grok_slug_emits_reasoning_effort():
    kwargs = _kwargs_for("x-ai/grok-4.5", "high")
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_headers"]["x-grok-conv-id"] == "test-session"


def test_cliproxyapi_composer_grok_keeps_cache_key_without_effort_dial():
    # Composer models do not expose reasoning_effort; still sticky-cache them.
    kwargs = _kwargs_for("grok-composer-2.5-fast", "high")

    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["prompt_cache_key"] == "test-session"
    assert kwargs["extra_headers"]["x-grok-conv-id"] == "test-session"
