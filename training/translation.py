"""Translate a user prompt from any language into the system's default language (English).

Backends:
  local (default)  Helsinki-NLP/opus-mt-mul-en (Hugging Face Transformers, Apache-2.0, ~310 MB):
                   121 source languages -> English, runs on the training VM itself, no cloud
                   service or key. Downloaded once into the Hugging Face cache.
  aws              Amazon Translate with automatic language detection; needs AWS credentials
                   with translate:TranslateText and comprehend:DetectDominantLanguage.

Prompts that are already English (plain ASCII text that names a supported cancer type) are
used as they are, without translation.
"""
import functools
import os

DEFAULT_LANGUAGE = "en"
LOCAL_MODEL = "Helsinki-NLP/opus-mt-mul-en"


class TranslationError(RuntimeError):
    pass


# ---------------------------------------------------------------- local (Hugging Face)
@functools.lru_cache(maxsize=1)
def _local_model():
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as e:
        raise TranslationError(f"local translation needs transformers + sentencepiece: {e}") from e
    try:
        tokenizer = MarianTokenizer.from_pretrained(LOCAL_MODEL)
        model = MarianMTModel.from_pretrained(LOCAL_MODEL).eval()
    except Exception as e:  # no network on first use, disk full, ...
        raise TranslationError(f"could not load the translation model {LOCAL_MODEL}: {e}") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device), device


def translate_local(text):
    import torch

    tokenizer, model, device = _local_model()
    batch = tokenizer([text], return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model.generate(**batch, num_beams=4, max_new_tokens=128)
    return tokenizer.decode(out[0], skip_special_tokens=True), "auto"


# ---------------------------------------------------------------- Amazon Translate (optional)
def translate_aws(text, region=None, client=None):
    from botocore.exceptions import BotoCoreError, ClientError

    if client is None:
        import boto3

        region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("translate", region_name=region)
    source = "auto"
    for _ in range(2):
        try:
            r = client.translate_text(Text=text, SourceLanguageCode=source, TargetLanguageCode=DEFAULT_LANGUAGE)
            return r["TranslatedText"], r["SourceLanguageCode"]
        except ClientError as e:
            err = e.response.get("Error", {})
            code, message = err.get("Code", ""), err.get("Message", "")
            detected = e.response.get("DetectedLanguageCode") or err.get("DetectedLanguageCode")
            if code == "DetectedLanguageLowConfidenceException" and detected and source == "auto":
                source = detected  # short prompts: accept the low-confidence guess and retry once
                if source == DEFAULT_LANGUAGE:
                    return text, DEFAULT_LANGUAGE
                continue
            if "same" in message.lower() and "language" in message.lower():
                return text, DEFAULT_LANGUAGE
            raise TranslationError(f"Amazon Translate failed ({code}): {message}") from e
        except BotoCoreError as e:
            raise TranslationError(f"Amazon Translate is not reachable: {e}") from e
    raise TranslationError("Amazon Translate could not detect the prompt's language")


BACKENDS = {"local": translate_local, "aws": translate_aws}


def to_default_language(text, backend="local"):
    """Return (text in English, source language code or 'auto')."""
    if backend not in BACKENDS:
        raise TranslationError(f"unknown translation backend '{backend}' (choose from {', '.join(BACKENDS)})")
    return BACKENDS[backend](text)
