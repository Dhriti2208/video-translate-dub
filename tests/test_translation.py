
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)
print("Loading translation model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device)

processor = IndicProcessor(inference=True)

def translate(text, target_language):
    source_language = "eng_Latn"

    prepared = processor.preprocess_batch(
        [text],
        src_lang=source_language,
        tgt_lang=target_language
    )

    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            num_beams=3,
            max_new_tokens=128
        )

    decoded = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )

    return processor.postprocess_batch(
        decoded,
        lang=target_language
    )[0]

sentence = "Hello, how are you today?"

print("\nEnglish:", sentence)
print("Hindi:", translate(sentence, "hin_Deva"))
print("Marathi:", translate(sentence, "mar_Deva"))
