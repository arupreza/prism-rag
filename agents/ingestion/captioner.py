"""
VLM captioner — Qwen/Qwen2.5-VL-3B-Instruct, in-process. Lazy load on first call.
Turns each extracted figure into a searchable text explanation.

Public API:
    caption_image(image_path, domain, pdf_caption="") -> str
    caption_doc(loaded_doc, domain) -> int   # mutates image segments in place
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

if TYPE_CHECKING:
    from .loader import LoadedDoc

# Resolve from repo root so it works regardless of CWD (captioner.py is at
# agents/ingestion/, so parents[2] == repo root — same trick config.py uses).
_REPO_ROOT     = Path(__file__).resolve().parents[2]
_DEFAULT_VLM   = _REPO_ROOT / "checkpoints/awq_models/qwen_vision_awq_w4a16"
MODEL_ID       = os.getenv("VLM_MODEL", str(_DEFAULT_VLM))
MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))
DEVICE_MAP     = os.getenv("VLM_DEVICE_MAP", "auto")

PROMPTS = {
    "ai": (
        "Describe this figure from an ML/AI document. State what it shows "
        "(architecture, plot axes and trend, or table content) in 1-3 dense "
        "sentences. Include any numbers, axis labels, or legend terms visible. "
        "No preamble, no markdown."
    ),
    "trading": (
        "Describe this finance/trading figure: chart type, axes, indicators, and "
        "the trend or key values shown. Be quantitative. 1-3 sentences. No preamble."
    ),
    "immigration": (
        "Describe this figure/diagram from a legal or immigration policy document: "
        "what it depicts and any labels, categories, or values. 1-3 sentences. No preamble."
    ),
}

_model = None
_proc = None


def _load() -> None:
    global _model, _proc
    if _model is not None:
        return
    print(f"[captioner] loading AWQ VLM from {MODEL_ID} ...")
    # AWQ W4A16: kernels run in fp16. Do NOT force bf16 — let transformers read
    # the dtype from the checkpoint's quantization_config. Requires `autoawq`
    # (+ autoawq-kernels) installed; transformers auto-detects awq from config.json.
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=DEVICE_MAP,
        local_files_only=True,     # never reach for the HF hub
        trust_remote_code=True,
    )
    _proc = AutoProcessor.from_pretrained(
        MODEL_ID, local_files_only=True, trust_remote_code=True
    )
    _model.eval()
    print(f"[captioner] loaded ({_model.dtype})")


@torch.inference_mode()
def caption_image(image_path: str, domain: str, pdf_caption: str = "") -> str:
    _load()
    instr = PROMPTS.get(domain, PROMPTS["ai"])
    if pdf_caption:
        instr += (
            f'\nThe document\'s own caption reads: "{pdf_caption}". '
            "Use it for context but describe what the image actually shows."
        )
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": instr}],
    }]
    text = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _proc(text=[text], images=[img], padding=True, return_tensors="pt").to(_model.device)
    out = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return _proc.decode(trimmed, skip_special_tokens=True).strip()


def caption_doc(doc: "LoadedDoc", domain: str) -> int:
    """Fill each image segment's text with '<pdf caption>\\n<VLM caption>'. Returns count."""
    n = 0
    for seg in doc.segments:
        if seg.kind != "image" or not seg.image_path:
            continue
        try:
            vlm = caption_image(seg.image_path, domain, pdf_caption=seg.text)
        except Exception as e:  # one bad image must not kill the whole doc
            print(f"[captioner] WARN failed on {seg.image_path}: {e}")
            continue
        seg.text = f"{seg.text}\n{vlm}".strip() if seg.text else vlm
        n += 1
    return n