import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract mono, 16 kHz WAV audio from a video using FFmpeg."""
    video = Path(video_path).resolve()
    output = Path(output_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed: output WAV is empty.")

    return str(output)
