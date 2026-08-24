from unittest.mock import MagicMock, patch


def test_transcribe_deepgram_accepts_null_provider_config(tmp_path, monkeypatch):
    from tools import transcription_tools

    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test")
    monkeypatch.setattr(
        transcription_tools,
        "_load_stt_config",
        lambda: {"provider": "deepgram", "deepgram": None},
    )

    response = MagicMock(status_code=200)
    response.json.return_value = {
        "results": {
            "channels": [
                {"alternatives": [{"transcript": "hello world"}]}
            ]
        }
    }

    with patch("requests.post", return_value=response):
        result = transcription_tools._transcribe_deepgram(
            str(audio_path),
            "nova-3",
        )

    assert result["success"] is True
    assert result["transcript"] == "hello world"
