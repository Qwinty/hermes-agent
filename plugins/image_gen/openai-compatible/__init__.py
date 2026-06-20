"""OpenAI-compatible image generation backend.

This provider targets OpenAI-compatible image endpoints exposed by local or
remote gateways, including CLIProxyAPI (CPA):

- POST /v1/images/generations for text-to-image
- POST /v1/images/edits for image-to-image / editing

It intentionally does not reuse the chat model provider. Image generation has a
separate provider registry and should be configured independently under
``image_gen.openai-compatible`` so gateways can choose different auth, base URL,
model, size and response-format semantics.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)

_PROVIDER = "openai-compatible"
DEFAULT_BASE_URL = "http://127.0.0.1:8317/v1"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_RESPONSE_FORMAT = ""
_VIRTUAL_GPT_IMAGE_2_QUALITIES = {
    "gpt-image-2-low": "low",
    "gpt-image-2-medium": "medium",
    "gpt-image-2-high": "high",
}

_SIZE_MAP = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_image_gen_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _cfg_section() -> Dict[str, Any]:
    cfg = _load_image_gen_config()
    section = cfg.get("openai-compatible") if isinstance(cfg.get("openai-compatible"), dict) else {}
    # Accept a shorter alias too; useful for local configs while the provider id
    # stays explicit and upstream-friendly.
    alias = cfg.get("openai_compatible") if isinstance(cfg.get("openai_compatible"), dict) else {}
    merged: Dict[str, Any] = {}
    if isinstance(alias, dict):
        merged.update(alias)
    if isinstance(section, dict):
        merged.update(section)
    return merged


def _resolve_env_value(value: Any) -> str:
    """Resolve simple env indirections without logging secrets.

    Supported forms:
    - ``env:NAME``
    - ``$NAME``
    - ``${NAME}``
    Other values are returned as strings unchanged.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("env:"):
        return os.environ.get(text[4:].strip(), "").strip()
    if text.startswith("${") and text.endswith("}"):
        return os.environ.get(text[2:-1].strip(), "").strip()
    if text.startswith("$") and len(text) > 1:
        return os.environ.get(text[1:].strip(), "").strip()
    return text


def _resolve_base_url() -> str:
    cfg = _cfg_section()
    value = (
        cfg.get("base_url")
        or os.environ.get("OPENAI_COMPATIBLE_IMAGE_BASE_URL")
        or os.environ.get("CPA_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return str(value).strip().rstrip("/")


def _resolve_api_key() -> str:
    cfg = _cfg_section()
    value = (
        cfg.get("api_key")
        or os.environ.get("OPENAI_COMPATIBLE_IMAGE_API_KEY")
        or os.environ.get("CPA_API_KEY")
        or os.environ.get("CLIPROXY_API_KEY")
    )
    key = _resolve_env_value(value)
    if key:
        return key

    # CPA often stores the service key in its YAML config rather than a process
    # env var. Keep this as a convenience fallback for localhost deployments.
    config_path = str(cfg.get("config_path") or os.environ.get("CPA_CONFIG") or "/opt/CLIProxyAPI/config.yaml")
    try:
        import re

        text = Path(config_path).read_text(encoding="utf-8")
        m = re.search(r"(?m)^api-keys:\s*\n\s*-\s*['\"]?([^'\"\n#]+)", text)
        if m:
            return m.group(1).strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read OpenAI-compatible image API key from %s: %s", config_path, exc)
    return ""


def _resolve_model() -> str:
    cfg = _cfg_section()
    top = _load_image_gen_config().get("model")
    value = os.environ.get("OPENAI_COMPATIBLE_IMAGE_MODEL") or cfg.get("model") or top or DEFAULT_MODEL
    return str(value).strip() or DEFAULT_MODEL


def _resolve_quality() -> str:
    cfg = _cfg_section()
    value = os.environ.get("OPENAI_COMPATIBLE_IMAGE_QUALITY") or cfg.get("quality") or ""
    return str(value).strip()


def _resolve_response_format() -> str:
    cfg = _cfg_section()
    value = cfg.get("response_format") or os.environ.get("OPENAI_COMPATIBLE_IMAGE_RESPONSE_FORMAT") or DEFAULT_RESPONSE_FORMAT
    return str(value).strip() or DEFAULT_RESPONSE_FORMAT


def _resolve_output_format() -> str:
    cfg = _cfg_section()
    value = cfg.get("output_format") or os.environ.get("OPENAI_COMPATIBLE_IMAGE_OUTPUT_FORMAT") or ""
    return str(value).strip()


def _resolve_max_reference_images() -> int:
    cfg = _cfg_section()
    raw = cfg.get("max_reference_images", 16)
    try:
        return max(1, int(raw))
    except Exception:  # noqa: BLE001
        return 16


def _resolve_size(aspect_ratio: str) -> str:
    aspect = resolve_aspect_ratio(aspect_ratio)
    cfg = _cfg_section()
    configured = cfg.get("size")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    sizes = cfg.get("sizes")
    if isinstance(sizes, dict):
        candidate = sizes.get(aspect)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return _SIZE_MAP.get(aspect, _SIZE_MAP["square"])


def _model_catalog_entry(model_id: str) -> Dict[str, Any]:
    return {
        "id": model_id,
        "display": model_id,
        "speed": "varies",
        "strengths": "OpenAI-compatible /v1/images gateway; supports text-to-image and image edits when upstream does",
        "price": "gateway-dependent",
    }


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------


def _local_file_tuple(path_ref: str) -> Tuple[str, Tuple[str, io.BytesIO, str]]:
    path = Path(path_ref).expanduser()
    data = path.read_bytes()
    bio = io.BytesIO(data)
    bio.name = path.name or "image.png"
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return bio.name, (bio.name, bio, mime)


def _is_remote_ref(ref: str) -> bool:
    lower = ref.strip().lower()
    return lower.startswith(("http://", "https://", "data:"))


def _file_to_data_url(path_ref: str) -> str:
    path = Path(path_ref).expanduser()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _image_ref_for_json(ref: str) -> str:
    ref = ref.strip()
    if _is_remote_ref(ref):
        return ref
    return _file_to_data_url(ref)


def _find_media_item(obj: Any) -> Optional[Tuple[str, str]]:
    if isinstance(obj, dict):
        for key in ("b64_json", "url"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return key, val.strip()
        for key in ("image_url", "video_url"):
            val = obj.get(key)
            if isinstance(val, dict) and isinstance(val.get("url"), str) and val.get("url", "").strip():
                return "url", val["url"].strip()
            if isinstance(val, str) and val.strip():
                return "url", val.strip()
        for val in obj.values():
            found = _find_media_item(val)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = _find_media_item(val)
            if found:
                return found
    return None


def _save_response_image(result: Dict[str, Any], *, model_id: str) -> str:
    media = None
    data = result.get("data")
    if isinstance(data, list) and data:
        media = _find_media_item(data[0])
    if media is None:
        media = _find_media_item(result)
    if media is None:
        raise ValueError("response contained neither b64_json nor image URL")

    kind, value = media
    prefix = "openai_compatible_" + "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_id)
    if kind == "b64_json":
        return str(save_b64_image(value, prefix=prefix))
    return str(save_url_image(value, prefix=prefix))


class OpenAICompatibleImageGenProvider(ImageGenProvider):
    """OpenAI-compatible ``/v1/images/*`` backend."""

    @property
    def name(self) -> str:
        return _PROVIDER

    @property
    def display_name(self) -> str:
        return "OpenAI-compatible Images"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        cfg = _cfg_section()
        models = cfg.get("models")
        out: List[Dict[str, Any]] = []
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str) and item.strip():
                    out.append(_model_catalog_entry(item.strip()))
                elif isinstance(item, dict) and isinstance(item.get("id"), str):
                    entry = _model_catalog_entry(item["id"].strip())
                    entry.update({k: v for k, v in item.items() if v is not None})
                    out.append(entry)
        if out:
            return out
        configured = _resolve_model()
        base_model = "gpt-image-2" if configured in _VIRTUAL_GPT_IMAGE_2_QUALITIES else configured
        ids = [base_model]
        if base_model == "gpt-image-2":
            ids = ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]
        return [_model_catalog_entry(mid) for mid in ids]

    def default_model(self) -> Optional[str]:
        return _resolve_model()

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI-compatible Images",
            "badge": "proxy",
            "tag": "OpenAI-compatible /v1/images/generations + /v1/images/edits; works with local CPA/CLIProxyAPI",
            "env_vars": [
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
                    "prompt": "Image API base URL (for CPA: http://127.0.0.1:8317/v1)",
                },
                {
                    "key": "OPENAI_COMPATIBLE_IMAGE_API_KEY",
                    "prompt": "Image API key / bearer token",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"], "max_reference_images": _resolve_max_reference_images()}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        model_id = str(kwargs.get("model") or _resolve_model()).strip() or DEFAULT_MODEL
        provider_name = _PROVIDER
        virtual_quality = _VIRTUAL_GPT_IMAGE_2_QUALITIES.get(model_id)
        api_model_id = "gpt-image-2" if virtual_quality else model_id

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=provider_name,
                model=model_id,
                aspect_ratio=aspect,
            )

        api_key = _resolve_api_key()
        if not api_key:
            return error_response(
                error=(
                    "No API key configured for OpenAI-compatible image generation. "
                    "Set image_gen.openai-compatible.api_key, OPENAI_COMPATIBLE_IMAGE_API_KEY, "
                    "CPA_API_KEY, or CLIPROXY_API_KEY."
                ),
                error_type="auth_required",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        refs: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            refs.append(image_url.strip())
        for ref in normalize_reference_images(reference_image_urls) or []:
            refs.append(ref)
        max_refs = _resolve_max_reference_images()
        refs = refs[:max_refs]
        is_edit = bool(refs)
        modality = "image" if is_edit else "text"

        size = _resolve_size(aspect)
        response_format = _resolve_response_format()
        quality = virtual_quality or _resolve_quality()
        output_format = _resolve_output_format()

        fields: Dict[str, Any] = {
            "model": api_model_id,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }
        if response_format:
            fields["response_format"] = response_format
        if quality:
            fields["quality"] = quality
        if output_format:
            fields["output_format"] = output_format

        cfg = _cfg_section()
        for key in (
            "background",
            "input_fidelity",
            "moderation",
            "aspect_ratio",
            "resolution",
        ):
            value = cfg.get(key)
            if isinstance(value, str) and value.strip():
                fields[key] = value.strip()
        for key in ("output_compression", "partial_images"):
            value = cfg.get(key)
            if isinstance(value, int):
                fields[key] = value

        base_url = _resolve_base_url()
        endpoint = f"{base_url}/images/edits" if is_edit else f"{base_url}/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": "Hermes-OpenAI-Compatible-ImageGen/1.0"}

        try:
            if is_edit:
                all_local = all(not _is_remote_ref(ref) for ref in refs)
                if all_local:
                    files_payload: List[Tuple[str, Tuple[str, io.BytesIO, str]]] = []
                    field_name = "image" if len(refs) == 1 else "image[]"
                    for ref in refs:
                        _name, file_tuple = _local_file_tuple(ref)
                        files_payload.append((field_name, file_tuple))
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        data={k: str(v) for k, v in fields.items()},
                        files=files_payload,
                        timeout=300,
                    )
                else:
                    payload = dict(fields)
                    payload["images"] = [{"image_url": _image_ref_for_json(ref)} for ref in refs]
                    headers["Content-Type"] = "application/json"
                    response = requests.post(endpoint, headers=headers, json=payload, timeout=300)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(endpoint, headers=headers, json=fields, timeout=300)
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                body = resp.json() if resp is not None else {}
                err = body.get("error", {}).get("message") or body.get("error") or resp.text[:500]
            except Exception:
                err = resp.text[:500] if resp is not None else str(exc)
            return error_response(
                error=f"OpenAI-compatible image request failed ({status}): {err}",
                error_type="api_error",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="OpenAI-compatible image request timed out (300s)",
                error_type="timeout",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"OpenAI-compatible image request failed: {exc}",
                error_type="provider_exception",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"OpenAI-compatible image endpoint returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            image_ref = _save_response_image(result, model_id=model_id)
        except Exception as exc:  # noqa: BLE001
            return error_response(
                error=f"Could not extract/save image output: {exc}",
                error_type="empty_response",
                provider=provider_name,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=provider_name,
            modality=modality,
            extra={"size": size, "endpoint": "/images/edits" if is_edit else "/images/generations"},
        )


def register(ctx: Any) -> None:
    """Register this provider with the image generation registry."""
    ctx.register_image_gen_provider(OpenAICompatibleImageGenProvider())
