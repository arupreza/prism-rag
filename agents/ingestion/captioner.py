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
_DEFAULT_VLM   = _REPO_ROOT / "checkpoints/source_model/vision_model"  # fp16, NOT awq
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

# Used for scanned / image-only pages (no text layer) -> verbatim OCR, not a caption.
OCR_PROMPT = (
    "Transcribe ALL text in this image verbatim, preserving the reading order and "
    "paragraph breaks. Output only the transcribed text — no commentary, no markdown "
    "fences, no headings you invent. If the image has no readable text, output a single "
    "short line describing it."
)
OCR_MAX_NEW_TOKENS = int(os.getenv("VLM_OCR_MAX_NEW_TOKENS", "1536"))  # a dense page >> 256

_model = None
_proc = None


def _load() -> None:
    global _model, _proc
    if _model is not None:
        return
    print(f"[captioner] loading VLM from {MODEL_ID} ...")
    # Local fp16/bf16 vision checkpoint (NOT quantized) -> no autoawq/Triton path.
    # torch_dtype="auto" reads the model's native dtype from its config.
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map=DEVICE_MAP,
        local_files_only=True,     # weights are local -> never hit the HF hub
        trust_remote_code=True,
    )
    _proc = AutoProcessor.from_pretrained(
        MODEL_ID, local_files_only=True, trust_remote_code=True
    )
    _model.eval()
    print(f"[captioner] loaded ({_model.dtype})")


@torch.inference_mode()
def caption_image(image_path: str, domain: str, pdf_caption: str = "",
                  transcribe: bool = False) -> str:
    _load()
    if transcribe:
        instr = OCR_PROMPT
        max_new = OCR_MAX_NEW_TOKENS
    else:
        instr = PROMPTS.get(domain, PROMPTS["ai"])
        if pdf_caption:
            instr += (
                f'\nThe document\'s own caption reads: "{pdf_caption}". '
                "Use it for context but describe what the image actually shows."
            )
        max_new = MAX_NEW_TOKENS
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": instr}],
    }]
    text = _proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _proc(text=[text], images=[img], padding=True, return_tensors="pt").to(_model.device)
    out = _model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return _proc.decode(trimmed, skip_special_tokens=True).strip()


def caption_doc(doc: "LoadedDoc", domain: str) -> tuple[int, int]:
    """Resolve every image segment in place.

    - Normal figure -> caption appended to text (kind stays 'image').
    - Scanned page (seg.is_scan) -> verbatim transcription; promoted to kind='text'
      so it flows through normal text chunking, while keeping image_path for provenance.

    Returns (n_captioned_figures, n_transcribed_pages).
    """
    n_cap = n_ocr = 0
    for seg in doc.segments:
        if seg.kind != "image" or not seg.image_path:
            continue
        scan = getattr(seg, "is_scan", False)
        try:
            out = caption_image(seg.image_path, domain,
                                pdf_caption="" if scan else seg.text,
                                transcribe=scan)
        except Exception as e:  # one bad image must not kill the whole doc
            print(f"[captioner] WARN failed on {seg.image_path}: {e}")
            continue
        if scan:
            if out:
                seg.text = out
                seg.kind = "text"     # transcription is real document text
                n_ocr += 1
        else:
            seg.text = f"{seg.text}\n{out}".strip() if seg.text else out
            n_cap += 1
    return n_cap, n_ocr