# AI Video Translation and Dubbing System

## Target Languages
- English
- Hindi
- Marathi

## Input
Any video containing a speaking person.

## Output
- Translated Video
- Translated Audio
- Original Transcript
- Translated Text

## Pipeline

Video
↓
Audio Extraction
↓
Speech Recognition
↓
Transcript + Timestamps
↓
Translation
↓
Speaker-Preserving Voice Generation
↓
Duration Alignment
↓
Lip Synchronization
↓
Final Video / Audio / Text

## Main Research Contribution

Timestamp-aware duration alignment between the original speech
and translated generated speech.

## Models

ASR:
faster-whisper

Translation:
IndicTrans2

TTS:
To be benchmarked

Lip Sync:
Wav2Lip / MuseTalk

Video Processing:
FFmpeg