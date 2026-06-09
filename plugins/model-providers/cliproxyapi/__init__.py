"""CLIProxyAPI provider profile.

CLIProxyAPI is an OpenAI-compatible local proxy that fronts several CLI/OAuth
providers.  Its GPT/Codex-style chat-completions routes accept top-level
``reasoning_effort``; generic ``custom`` providers do not emit that field, so a
named profile is needed when a configured provider points at CPA.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


_REASONING_MODEL_PREFIXES = (
    "gpt-",
    "chatgpt-",
    "codex-",
    "o1",
    "o3",
    "o4",
    "openai/",
)


def _supports_top_level_reasoning_effort(model: str | None) -> bool:
    model_l = (model or "").strip().lower()
    return bool(model_l) and model_l.startswith(_REASONING_MODEL_PREFIXES)


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
            return {}, {}

        effort = str(reasoning_config.get("effort") or "").strip().lower()
        effort = {
            "minimal": "low",
            "xhigh": "high",
            "max": "high",
        }.get(effort, effort)

        if effort in {"low", "medium", "high"}:
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
