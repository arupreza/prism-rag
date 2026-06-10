"""
Cluster summarizer — in-process Qwen via transformers, no HTTP server.
Loads model lazily on first call. Same public interface: summarize(child_texts, domain).
"""
from __future__ import annotations
import os
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv(
    "LLM_LOCAL_PATH",
    "checkpoints/source_model/qwen_2_5",
)
MAX_NEW_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_TOKENS", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "immigration": (
        "You are summarizing legal/immigration policy chunks. "
        "Produce a factual, dense summary that preserves: statutes, visa categories, "
        "eligibility criteria, deadlines, and jurisdiction. No opinions."
    ),
    "trading": (
        "You are summarizing trading/finance chunks. "
        "Preserve: strategy names, indicators, parameters, risk rules, market conditions, "
        "and quantitative results. Be precise with numbers."
    ),
    "ai": (
        "You are summarizing AI/ML technical content including code. "
        "Preserve: algorithm names, key APIs, function signatures, hyperparameters. "
        "Keep critical code snippets in fenced ```python blocks. "
        "Summarize prose; do NOT paraphrase code."
    ),
}

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    print(f"[summarizer] loading Qwen from {MODEL_PATH} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    _model.eval()
    print(f"[summarizer] loaded ({_model.dtype}, {DEVICE})")


@torch.inference_mode()
def summarize(child_texts: Sequence[str], domain: str) -> str:
    _load()
    system = PROMPTS.get(domain, PROMPTS["ai"])
    joined = "\n\n---\n\n".join(child_texts)
    if len(joined) > 12000:
        joined = joined[:12000]
    user = (
        f"Summarize the following {len(child_texts)} chunks into one cohesive "
        f"summary:\n\n{joined}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(DEVICE)
    out = _model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.2,
        pad_token_id=_tokenizer.eos_token_id,
    )
    gen = out[0][inputs.input_ids.shape[1]:]
    text = _tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text
