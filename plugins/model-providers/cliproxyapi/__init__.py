"""CLIProxyAPI provider profile.

CLIProxyAPI is an OpenAI-compatible local proxy that fronts several CLI/OAuth
providers.  Its GPT/Codex-style, Claude/Neko, and Grok chat-completions routes
accept top-level ``reasoning_effort``; CPA then translates that into the
upstream provider's native thinking shape. Generic ``custom`` providers do not
emit that field, so a named profile is needed when a configured provider points
at CPA.
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

# Grok models that accept a reasoning-effort dial through CPA/xAI.
# Keep in sync with agent.model_metadata._GROK_EFFORT_CAPABLE_PREFIXES.
# Official Grok 4.5 efforts: low | medium | high (default high).
# CPA catalog for grok-4.5/4.3 also advertises none, but xAI docs say
# reasoning cannot be disabled for 4.5 — we therefore omit the field on
# "none"/disabled rather than sending none.
_GROK_EFFORT_CAPABLE_PREFIXES = (
    "grok-3-mini",
    "grok-4.20-multi-agent",
    "grok-4.3",
    "grok-4.5",
)

_GROK_STANDARD_EFFORTS = frozenset({"low", "medium", "high"})
_GROK_MULTI_AGENT_EFFORTS = frozenset({"low", "medium", "high", "xhigh"})


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


def _strip_model_slug(model: str | None) -> str:
    """Bare model id with aggregator/provider prefixes removed."""
    name = (model or "").strip().lower()
    if not name:
        return ""
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name


def _is_grok_model(model: str | None) -> bool:
    """Any Grok chat slug (used for prompt-cache sticky routing)."""
    return _strip_model_slug(model).startswith("grok-")


def _is_grok_reasoning_model(model: str | None) -> bool:
    name = _strip_model_slug(model)
    if not name:
        return False
    return any(name.startswith(prefix) for prefix in _GROK_EFFORT_CAPABLE_PREFIXES)


def _is_grok_multi_agent(model: str | None) -> bool:
    return _strip_model_slug(model).startswith("grok-4.20-multi-agent")


def _supports_top_level_reasoning_effort(model: str | None) -> bool:
    return (
        _is_gpt_reasoning_model(model)
        or _is_claude_reasoning_model(model)
        or _is_grok_reasoning_model(model)
    )


def _clamp_grok_effort(model: str | None, effort: str) -> str | None:
    """Map Hermes efforts onto the Grok dial CPA/xAI accept.

    Grok 4.5 / 4.3 / 3-mini: low | medium | high
    Grok 4.20 multi-agent: low | medium | high | xhigh
    """
    effort = (effort or "").strip().lower()
    effort = {"minimal": "low"}.get(effort, effort)

    if _is_grok_multi_agent(model):
        effort = {"max": "xhigh"}.get(effort, effort)
        return effort if effort in _GROK_MULTI_AGENT_EFFORTS else None

    # Standard Grok effort models (incl. grok-4.5): no xhigh/max/none dial.
    effort = {"xhigh": "high", "max": "high"}.get(effort, effort)
    return effort if effort in _GROK_STANDARD_EFFORTS else None


class CLIProxyAPIProfile(ProviderProfile):
    """Local CLIProxyAPI OpenAI-compatible proxy."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Sticky routing for Grok prompt caching through CPA.
        # CPA's xAI executor reads prompt_cache_key from the original client
        # payload and sets both upstream body prompt_cache_key and the
        # x-grok-conv-id header. X-Session-ID also stabilizes CPA session
        # affinity. See xAI docs: maximizing cache hits.
        sid = str(session_id or "").strip()
        if sid and _is_grok_model(model):
            top_level["extra_headers"] = {
                "X-Session-ID": sid,
                "x-grok-conv-id": sid,
            }
            extra_body["prompt_cache_key"] = sid

        if not _supports_top_level_reasoning_effort(model):
            return extra_body, top_level

        if not isinstance(reasoning_config, dict):
            return extra_body, top_level

        if reasoning_config.get("enabled") is False:
            if _is_claude_reasoning_model(model):
                top_level["reasoning_effort"] = "none"
            # GPT/Codex: omit (safer than forcing none).
            # Grok 4.5: reasoning cannot be disabled; omit → upstream default high.
            return extra_body, top_level

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        effort = {"minimal": "low"}.get(effort, effort)

        if _is_grok_reasoning_model(model):
            mapped = _clamp_grok_effort(model, effort)
            if mapped:
                top_level["reasoning_effort"] = mapped
            return extra_body, top_level

        if _is_gpt_reasoning_model(model) and effort == "max":
            # CPA's GPT/Codex catalog exposes xhigh as the highest chat-completions
            # effort today; Claude/Neko models keep max because Anthropic documents it.
            effort = "xhigh"

        if effort in {"low", "medium", "high", "xhigh", "max"}:
            top_level["reasoning_effort"] = effort

        return extra_body, top_level


cliproxyapi = CLIProxyAPIProfile(
    name="cliproxyapi",
    aliases=("cli-proxy-api", "cli_proxy_api", "cliproxy", "cpa"),
    display_name="CLIProxyAPI",
    description="Local OpenAI-compatible proxy for CLI subscription providers",
    env_vars=(),
    base_url="",
)

register_provider(cliproxyapi)
