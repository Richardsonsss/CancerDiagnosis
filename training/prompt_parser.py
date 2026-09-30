"""Turn a free-text user prompt in any language into a registered cancer task.

Methods (--prompt_model):
  llm (default)  a local instruction-tuned language model (llm_prompt.py, Qwen3-4B-Instruct-2507)
                 reads the prompt in any language and picks a registered task or rejects it; if the
                 model cannot be loaded, the translate method is used instead
  translate      the prompt is translated into English (translation.py: local opus-mt-mul-en, or
                 Amazon Translate) and matched against the task keywords in registry.TASKS
  keywords       English prompts only: keyword matching, no model is loaded
The result is always a task from the registry, or a PromptError listing the supported types.

    >>> understand_prompt("Quiero reconocer el cáncer de piel")["task_id"]
    'skin_phone'
"""
import re

from registry import TASKS
from translation import DEFAULT_LANGUAGE, to_default_language


class PromptError(ValueError):
    pass


def supported():
    return "; ".join(f"{t['name']} ({k})" for k, t in TASKS.items())


def match_task(english_text):
    """Task id whose keywords best match an English text (longest matched keywords win)."""
    text = english_text.lower()
    scores = {k: sum(len(kw) for kw in t["keywords"] if re.search(rf"\b{re.escape(kw)}", text))
              for k, t in TASKS.items()}
    best = max(scores.values())
    hits = [k for k, s in scores.items() if s == best and s > 0]
    if len(hits) > 1:  # tie: a task marked as the default for its topic wins (e.g. skin_phone over skin)
        top = max(TASKS[k].get("priority", 0) for k in hits)
        hits = [k for k in hits if TASKS[k].get("priority", 0) == top]
    if not hits:
        raise PromptError(f"cannot tell which cancer type '{english_text}' refers to. Supported: {supported()}")
    if len(hits) > 1:
        raise PromptError(f"'{english_text}' matches several cancer types {hits}; please name one. "
                          f"Supported: {supported()}")
    return hits[0]


def _translate_and_match(prompt, translator):
    if prompt.isascii():
        try:  # already English: no translation needed
            return {"task_id": match_task(prompt), "prompt": prompt, "english": prompt,
                    "language": DEFAULT_LANGUAGE, "method": "keywords"}
        except PromptError:
            pass
    english, language = translator(prompt)
    return {"task_id": match_task(english), "prompt": prompt, "english": english, "language": language,
            "method": "translate"}


def understand_prompt(prompt, method="llm", translator=to_default_language, llm=None):
    """{'task_id', 'prompt', 'english', 'language', 'method'} for a prompt in any language."""
    if not prompt or not prompt.strip():
        raise PromptError(f"the prompt is empty. Supported: {supported()}")
    if method == "keywords":
        return {"task_id": match_task(prompt), "prompt": prompt, "english": prompt,
                "language": DEFAULT_LANGUAGE, "method": "keywords"}
    if method == "llm":
        from llm_prompt import LLMUnavailable, llm_understand

        try:
            task_id, english = (llm or llm_understand)(prompt, TASKS)
        except LLMUnavailable as e:
            print(f"  language model unavailable ({e}); using translation + keywords instead")
            return _translate_and_match(prompt, translator)
        if task_id is None:
            raise PromptError(f"'{prompt}' (\"{english}\") is not a supported cancer type. Supported: {supported()}")
        return {"task_id": task_id, "prompt": prompt, "english": english, "language": "auto", "method": "llm"}
    return _translate_and_match(prompt, translator)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:] or ["recognise skin cancer", "detect brain tumours in MRI", "stomach cancer"]:
        try:
            r = understand_prompt(p)
            print(f"{p!r} [{r['method']}] -> {r['english']!r} -> {r['task_id']}")
        except PromptError as e:
            print(f"{p!r} -> rejected: {e}")
