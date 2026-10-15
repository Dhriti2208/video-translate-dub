
from src.asr.whisper_asr import transcribe_audio
from src.translation.translator import translate_text


def process_audio(audio_path: str) -> dict:
    """Transcribe audio and translate the recognized speech."""
    transcription = transcribe_audio(audio_path)
    english_text = transcription["text"]

    result = {
        "source_audio": audio_path,
        "detected_language": transcription["language"],
        "english_transcript": english_text,
        "segments": transcription["segments"],
        "hindi_translation": "",
        "marathi_translation": "",
    }

    # This translation module currently accepts English as input.
    if transcription["language"] == "en" and english_text:
        result["hindi_translation"] = translate_text(
            english_text, "hindi"
        )
        result["marathi_translation"] = translate_text(
            english_text, "marathi"
        )

    return result


if __name__ == "__main__":
    import json

    output = process_audio("data/input/speech_test.wav")
    print(json.dumps(output, ensure_ascii=False, indent=2))
