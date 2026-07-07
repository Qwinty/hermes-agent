"""CLIProxyAPI provider profile.

CLIProxyAPI is an OpenAI-compatible local proxy that fronts several CLI/OAuth
providers.  Its GPT/Codex-style and Claude/Neko chat-completions routes accept
top-level ``reasoning_effort``; CPA then translates that into the upstream
provider's native thinking shape. Generic ``custom`` providers do not emit that
field, so a named profile is needed when a configured provider points at CPA.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


_GPT_REASONING_MODEL_PREFIXES = (
    "gpt-",
    "chatgpt-",
    "codex-",
    "o1",
    "o3",
    "o4",
    "openai/",
)

_CLAUDE_REASONING_MODEL_FAMILY_TOKENS = {
    "claude",
    "sonnet",
    "opus",
    "fable",
    "haiku",
    "mythos",
}


def _is_gpt_reasoning_model(model: str | None) -> bool:
    model_l = (model or "").strip().lower()
    return bool(model_l) and model_l.startswith(_GPT_REASONING_MODEL_PREFIXES)


def _model_segments(model: str | None) -> set[str]:
    """Return normalized identifier segments for provider/family matching."""
    normalized = (model or "").strip().lower()
    if not normalized:
        return set()
    for sep in "/:._-()[]":
        normalized = normalized.replace(sep, " ")
    return {part for part in normalized.split() if part}


def _is_claude_reasoning_model(model: str | None) -> bool:
    # CPA's model metadata is the strongest signal when present: NekoCode Claude
    # aliases are advertised as owned_by=anthropic, type=claude, reasoning=true.
    # The transport does not have live /v1/models metadata in this hook today, so
    # keep a conservative family-token fallback that is independent of prefixes
    # like ``neko-`` and catches future aliases such as ``some/opener-sonnet-5``.
    segments = _model_segments(model)
    return bool(segments & _CLAUDE_REASONING_MODEL_FAMILY_TOKENS)


def _supports_top_level_reasoning_effort(model: str | None) -> bool:
    return _is_gpt_reasoning_model(model) or _is_claude_reasoning_model(model)


class CLIProxyAPIProfile(ProviderProfile):
    """Local CLIProxyAPI OpenAI-compatible proxy."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not _supports_top_level_reasoning_effort(model):
            return {}, {}

        if not isinstance(reasoning_config, dict):
            return {}, {}

        if reasoning_config.get("enabled") is False:
            if _is_claude_reasoning_model(model):
                return {}, {"reasoning_effort": "none"}
            return {}, {}

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        effort = {"minimal": "low"}.get(effort, effort)

        if _is_gpt_reasoning_model(model) and effort == "max":
            # CPA's GPT/Codex catalog exposes xhigh as the highest chat-completions
            # effort today; Claude/Neko models keep max because Anthropic documents it.
            effort = "xhigh"

        if effort in {"low", "medium", "high", "xhigh", "max"}:
            return {}, {"reasoning_effort": effort}

        return {}, {}


cliproxyapi = CLIProxyAPIProfile(
    name="cliproxyapi",
    aliases=("cli-proxy-api", "cli_proxy_api", "cliproxy", "cpa"),
    display_name="CLIProxyAPI",
    description="Local OpenAI-compatible proxy for CLI subscription providers",
    env_vars=(),
    base_url="",
)

register_provider(cliproxyapi)
