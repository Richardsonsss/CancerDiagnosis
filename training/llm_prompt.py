"""Understand a user prompt in any language with a local instruction-tuned language model.

Default model: Qwen/Qwen3-4B-Instruct-2507 (Apache-2.0, ~8 GB, downloaded once into the Hugging
Face cache; runs on the training VM's GPU, or slowly on its CPU). The model is shown the registered
cancer types and must answer with one of their ids or "none", plus an English rendering of the
request. Its answer is validated against the registry, so it can only pick a supported task or
reject the prompt - it never invents a dataset. Decoding is greedy, so answers are repeatable.
"""
import functools
import json
import os
import re

DEFAULT_LLM = os.getenv("PROMPT_LLM", "Qwen/Qwen3-4B-Instruct-2507")


class LLMUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _load(model_id):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise LLMUnavailable(f"transformers is not installed: {e}") from e
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        cuda = torch.cuda.is_available()
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if cuda else torch.float32)
        return tok, model.to("cuda" if cuda else "cpu").eval()
    except Exception as e:  # no network on first use, not enough memory, ...
        raise LLMUnavailable(f"could not load {model_id}: {type(e).__name__}: {str(e)[:200]}") from e


def _system_prompt(tasks):
    lines = [f'- "{k}": {t["name"]} - input: {t["modality"]}' for k, t in tasks.items()]
    return (
        "You route requests for training a medical image classifier to one of these supported tasks:\n"
        + "\n".join(lines) + "\n"
        "Rules: choose the single task that matches the cancer type the user wants to recognise. "
        "For skin cancer, choose \"skin\" only if the user mentions dermoscopy / dermatoscope images; "
        "otherwise choose \"skin_phone\". If the requested cancer type is not in the list, answer \"none\". "
        "The request may be written in any language.\n"
        'Reply with JSON only: {"task": "<task id or none>", "english": "<the request translated to English>"}')


def llm_understand(prompt, tasks, model_id=DEFAULT_LLM):
    """Return (task id or None, English rendering of the prompt)."""
    import torch

    tok, model = _load(model_id)
    messages = [{"role": "system", "content": _system_prompt(tasks)}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", answer, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    task = str(data.get("task", "")).strip().lower()
    english = str(data.get("english", "")).strip() or prompt
    return (task if task in tasks else None), english
