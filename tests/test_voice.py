"""Voice tests are limited: TTS just verifies the function runs without raising
(actual audio is not asserted). STT is skipped unless a fixture .wav exists."""
import os
import pytest
from app.voice import speak, transcribe

def test_speak_to_file_does_not_raise(tmp_path):
    out = tmp_path / "test.wav"
    speak("hello world", to_file=str(out))
    # If TTS engine is unavailable, speak() returns None and logs a warning.
    # We don't assert the file exists — just that speak() did not raise.

@pytest.mark.skipif(not os.path.exists("tests/fixtures/sample.wav"),
                    reason="no audio fixture")
def test_transcribe_returns_text():
    text = transcribe("tests/fixtures/sample.wav")
    assert isinstance(text, str)
