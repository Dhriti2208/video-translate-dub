from pathlib import Path
import whisper


_model = None


def load_model(model_name: str = "base"):
    """Load Whisper once and reuse it for subsequent audio files."""
    global _model

    if _model is None:
        _model = whisper.load_model(model_name)

    return _model


def transcribe_audio(audio_path: str, model_name: str = "base") -> dict:
    """Transcribe an audio file and return text plus segment timestamps."""
    path = Path(audio_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = load_model(model_name)

    result = model.transcribe(
        str(path),
        fp16=False,
        verbose=False,
    )

    segments = [
        {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": segment["text"].strip(),
        }
        for segment in result.get("segments", [])
    ]

    return {
        "language": result.get("language"),
        "text": result.get("text", "").strip(),
        "segments": segments,
    }
