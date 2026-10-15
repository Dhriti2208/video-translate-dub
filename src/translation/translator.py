
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model = None
_processor = None


def load_translation_model():
    """Load the translation model once and reuse it."""
    global _tokenizer, _model, _processor

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        ).to(_device)
        _processor = IndicProcessor(inference=True)

    return _tokenizer, _model, _processor


def translate_text(text: str, target_language: str) -> str:
    """Translate English text into Hindi or Marathi."""
    language_codes = {
        "hindi": "hin_Deva",
        "marathi": "mar_Deva",
        "english": "eng_Latn",
    }

    language = target_language.strip().lower()

    if language not in language_codes:
        raise ValueError(
            "Supported languages: English, Hindi, Marathi"
        )

    if not text or not text.strip():
        return ""

    if language == "english":
        return text.strip()

    tokenizer, model, processor = load_translation_model()
    target_code = language_codes[language]

    prepared = processor.preprocess_batch(
        [text.strip()],
        src_lang="eng_Latn",
        tgt_lang=target_code,
    )

    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(_device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            num_beams=3,
            max_new_tokens=128,
        )

    decoded = tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
    )

    return processor.postprocess_batch(
        decoded,
        lang=target_code,
    )[0]

