from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import yaml


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGNgAAAAAgABXfstdgAAAABJRU5ErkJggg=="


def _load_provider_module():
    path = Path(__file__).resolve().parents[2] / "plugins" / "image_gen" / "openai-compatible" / "__init__.py"
    spec = importlib.util.spec_from_file_location("openai_compatible_image_provider_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_cfg(home: Path, cfg: dict):
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


class _Response:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError("boom", response=self)


def test_generation_posts_json_to_openai_compatible_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CLIPROXY_API_KEY", "cpa_test")
    _write_cfg(tmp_path, {
        "image_gen": {
            "provider": "openai-compatible",
            "model": "gpt-image-2-medium",
            "openai-compatible": {"base_url": "http://127.0.0.1:8317/v1"},
        }
    })
    mod = _load_provider_module()
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Response({"data": [{"b64_json": PNG_B64}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    out = mod.OpenAICompatibleImageGenProvider().generate("draw a cat", aspect_ratio="square")

    assert out["success"] is True
    assert out["provider"] == "openai-compatible"
    assert out["modality"] == "text"
    assert Path(out["image"]).exists()
    assert captured["url"] == "http://127.0.0.1:8317/v1/images/generations"
    assert captured["kwargs"]["json"]["model"] == "gpt-image-2"
    assert captured["kwargs"]["json"]["quality"] == "medium"
    assert captured["kwargs"]["json"]["size"] == "1024x1024"
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer cpa_test"


def test_local_image_edit_posts_multipart_to_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CLIPROXY_API_KEY", "cpa_test")
    source = tmp_path / "source.png"
    source.write_bytes(base64.b64decode(PNG_B64))
    _write_cfg(tmp_path, {
        "image_gen": {
            "provider": "openai-compatible",
            "model": "grok-imagine-image-quality",
            "openai-compatible": {"base_url": "http://127.0.0.1:8317/v1"},
        }
    })
    mod = _load_provider_module()
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Response({"data": [{"b64_json": PNG_B64}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    out = mod.OpenAICompatibleImageGenProvider().generate(
        "add a tiny cat in his hands",
        aspect_ratio="square",
        image_url=str(source),
    )

    assert out["success"] is True
    assert out["modality"] == "image"
    assert captured["url"] == "http://127.0.0.1:8317/v1/images/edits"
    assert captured["kwargs"]["data"]["model"] == "grok-imagine-image-quality"
    assert captured["kwargs"]["data"]["prompt"] == "add a tiny cat in his hands"
    assert "files" in captured["kwargs"]
    assert captured["kwargs"]["files"][0][0] == "image"
    filename, fileobj, mime = captured["kwargs"]["files"][0][1]
    assert filename == "source.png"
    assert mime == "image/png"
    assert fileobj.read().startswith(b"\x89PNG")


def test_remote_image_edit_posts_json_image_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CLIPROXY_API_KEY", "cpa_test")
    _write_cfg(tmp_path, {
        "image_gen": {
            "provider": "openai-compatible",
            "model": "grok-imagine-image",
            "openai-compatible": {"base_url": "http://127.0.0.1:8317/v1"},
        }
    })
    mod = _load_provider_module()
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _Response({"data": [{"b64_json": PNG_B64}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)

    out = mod.OpenAICompatibleImageGenProvider().generate(
        "use this as a style reference",
        image_url="https://example.com/a.png",
        reference_image_urls=["https://example.com/b.png"],
    )

    assert out["success"] is True
    assert captured["url"].endswith("/images/edits")
    payload = captured["kwargs"]["json"]
    assert payload["images"] == [
        {"image_url": "https://example.com/a.png"},
        {"image_url": "https://example.com/b.png"},
    ]
    assert captured["kwargs"]["headers"]["Content-Type"] == "application/json"


def test_capabilities_and_model_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CLIPROXY_API_KEY", "cpa_test")
    _write_cfg(tmp_path, {
        "image_gen": {
            "provider": "openai-compatible",
            "model": "gpt-image-2-medium",
            "openai-compatible": {"max_reference_images": 7},
        }
    })
    mod = _load_provider_module()
    provider = mod.OpenAICompatibleImageGenProvider()

    assert provider.is_available() is True
    assert provider.capabilities() == {"modalities": ["text", "image"], "max_reference_images": 7}
    assert [m["id"] for m in provider.list_models()] == [
        "gpt-image-2-low",
        "gpt-image-2-medium",
        "gpt-image-2-high",
    ]
