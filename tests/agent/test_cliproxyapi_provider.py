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


def test_cliproxyapi_does_not_emit_reasoning_effort_for_gemini_slug():
    kwargs = _kwargs_for("gemini-3.1-pro-high", "high")

    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs


def test_cliproxyapi_reasoning_none_omits_reasoning_effort():
    kwargs = _kwargs_for("gpt-5.5", "none")

    assert "reasoning_effort" not in kwargs
