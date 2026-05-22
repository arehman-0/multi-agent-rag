"""STT via faster-whisper, TTS via pyttsx3. Both local, no API keys."""
import logging
from pathlib import Path
import pyttsx3
from faster_whisper import WhisperModel

log = logging.getLogger(__name__)
_whisper: WhisperModel | None = None


def _get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def transcribe(audio_path: str) -> str:
    try:
        model = _get_whisper()
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as exc:
        log.warning("transcribe failed: %s", exc)
        return ""


def speak(text: str, to_file: str | None = None) -> str | None:
    try:
        engine = pyttsx3.init()
        if to_file:
            engine.save_to_file(text, to_file)
            engine.runAndWait()
            return to_file
        engine.say(text)
        engine.runAndWait()
        return None
    except Exception as exc:
        log.warning("TTS failed: %s", exc)
        return None
