# training/awq_quantize_vision_worker.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import json
import random
import shutil
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from awq import AutoAWQForCausalLM
import awq.models.qwen2_5_vl as _qwen_vl_mod
from peft import PeftModel
from PIL import Image
import io

# ----------------------------- config -----------------------------
BASE_MODEL = "checkpoints/source_model/vision_model"
ADAPTER    = None
MERGED     = "checkpoints/awq_models/qwen_vision_merged_fp16"
CALIB      = "checkpoints/clallibration_data/vision/llava_instruct_150k"
CALIB_HF_ID = "lmms-lab/LLaVA-OneVision-Data"
CALIB_HF_SUBSET = "CLEVR-Math(MathV360K)"
OUT        = "checkpoints/awq_models/qwen_vision_awq_w4a16"

NUM_SAMPLES = 128
MAX_SEQ_LEN = 2048
KEEP_MERGED = False
random.seed(0)

SYS = "You are a helpful vision-language assistant. Describe images precisely."

# ----------------------------- patch AutoAWQ for transformers >=4.52 -----------------------------
# transformers >=4.52 moved Qwen2.5-VL decoder layers under `model.language_model.layers`.
# AutoAWQ (deprecated) still looks for `model.layers`. Patch the registered class
# regardless of its exact exported name.
def _get_model_layers(model):
    inner = model.model
    if hasattr(inner, "layers"):
        return inner.layers
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model.layers
    raise AttributeError(f"Cannot locate decoder layers on {type(inner)}")

def _move_embed(model, device):
    inner = model.model
    lm = inner.language_model if hasattr(inner, "language_model") else inner
    lm.embed_tokens = lm.embed_tokens.to(device)
    if hasattr(inner, "visual"):
        inner.visual = inner.visual.to(device)

_patched = False
for _name in dir(_qwen_vl_mod):
    _cls = getattr(_qwen_vl_mod, _name)
    if isinstance(_cls, type) and hasattr(_cls, "get_model_layers"):
        _cls.get_model_layers = staticmethod(_get_model_layers)
        if hasattr(_cls, "move_embed"):
            _cls.move_embed = staticmethod(_move_embed)
        print(f"[patch] patched {_name} for transformers >=4.52")
        _patched = True

if not _patched:
    raise RuntimeError("Could not find Qwen2.5-VL AWQ class to patch")

# ----------------------------- stage 1: merge LoRA -> FP16 (optional) -----------------------------
SOURCE = BASE_MODEL
if ADAPTER is not None:
    if not os.path.exists(os.path.join(MERGED, "config.json")):
        print(f"[merge] loading base on CPU: {BASE_MODEL}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(ADAPTER, trust_remote_code=True)

        print(f"[merge] attaching adapter: {ADAPTER}")
        model = PeftModel.from_pretrained(base, ADAPTER, device_map="cpu")
        model = model.merge_and_unload()

        os.makedirs(MERGED, exist_ok=True)
        model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(MERGED)
        print(f"[merge] saved FP16 -> {MERGED}")

        del model, base, processor
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[merge] skipped — {MERGED} already exists")
    SOURCE = MERGED
else:
    print("[merge] no adapter — quantizing base model directly")

# ----------------------------- stage 2: load model for AWQ -----------------------------
model = AutoAWQForCausalLM.from_pretrained(
    SOURCE,
    safetensors=True,
    low_cpu_mem_usage=True,
    use_cache=False,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(SOURCE, trust_remote_code=True)
tokenizer = processor.tokenizer

# ----------------------------- load calibration dataset (auto-download fallback) -----------------------------
parquet_files = sorted(glob.glob(f"{CALIB}/data/*.parquet"))
jsonl_files   = sorted(glob.glob(f"{CALIB}/data/*.jsonl"))

if parquet_files:
    print(f"[calib] loading {len(parquet_files)} parquet file(s) from {CALIB}")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
elif jsonl_files:
    print(f"[calib] loading {len(jsonl_files)} jsonl file(s) from {CALIB}")
    ds = load_dataset("json", data_files=jsonl_files, split="train")
else:
    try:
        print(f"[calib] trying load_from_disk: {CALIB}")
        ds = load_from_disk(CALIB)
        if isinstance(ds, DatasetDict):
            ds = ds["train" if "train" in ds else list(ds.keys())[0]]
    except Exception:
        print(f"[calib] local cache empty — downloading {CALIB_HF_ID}:{CALIB_HF_SUBSET}")
        os.makedirs(CALIB, exist_ok=True)
        ds = load_dataset(
            CALIB_HF_ID,
            CALIB_HF_SUBSET,
            split="train",
            cache_dir=CALIB,
        )
        save_path = os.path.join(CALIB, "arrow")
        ds.save_to_disk(save_path)
        print(f"[calib] cached dataset -> {save_path}")

print(f"[calib] columns: {ds.column_names}")

# ----------------------------- normalize schema -----------------------------
def norm_vl(ex):
    conv = ex.get("conversations")
    if conv and len(conv) >= 2:
        q = conv[0].get("value", "").replace("<image>", "").strip()
        a = conv[1].get("value", "").strip()
    else:
        q = ex.get("question") or ex.get("prompt") or "Describe this image."
        a = ex.get("answer") or ex.get("response") or ""
    return q, a

rows = [norm_vl(ds[i]) for i in range(len(ds))]
rows = [r for r in rows if r[0]]
if not rows:
    raise RuntimeError(f"No usable rows. Columns: {ds.column_names}. Fix norm_vl().")

rows = random.sample(rows, min(NUM_SAMPLES, len(rows)))
print(f"[calib] samples: {len(rows)}")

# ----------------------------- build text-only calibration -----------------------------
# AutoAWQ's calibration forward pass expects causal-LM signature (input_ids only).
# Feeding pixel_values would break init_quant. The vision tower stays FP16 via
# modules_to_not_convert, so calibrating the LM on text-only inputs is acceptable.
def build_text(q, a):
    msgs = [
        {"role": "system",    "content": SYS},
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a or "An image."},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)

texts = [build_text(q, a) for q, a in rows]
print(f"[calib] text samples: {len(texts)}")

# ----------------------------- AWQ quantize -----------------------------
quant_config = {
    "zero_point":   True,
    "q_group_size": 128,
    "w_bit":        4,
    "version":      "GEMM",
    "modules_to_not_convert": ["lm_head", "visual", "merger", "vision_tower"],
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=texts,
    max_calib_seq_len=MAX_SEQ_LEN,
    max_calib_samples=NUM_SAMPLES,
)

# ----------------------------- save -----------------------------
model.save_quantized(OUT)
processor.save_pretrained(OUT)

cfg_path = os.path.join(OUT, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["quantization_config"] = {
    "quant_method": "awq",
    "bits":         quant_config["w_bit"],
    "group_size":   quant_config["q_group_size"],
    "zero_point":   quant_config["zero_point"],
    "version":      quant_config["version"].lower(),
    "modules_to_not_convert": quant_config["modules_to_not_convert"],
}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Saved AWQ checkpoint -> {OUT}")

# ----------------------------- cleanup -----------------------------
if ADAPTER is not None and not KEEP_MERGED:
    shutil.rmtree(MERGED, ignore_errors=True)
    print(f"[cleanup] removed FP16 dir: {MERGED}")