# training/awq_quantize_trader.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import json
import random
import shutil
import torch
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from peft import PeftModel

# ----------------------------- config -----------------------------
BASE = "/home/lisa/Arupreza/PRISM-RAG"

BASE_MODEL = f"{BASE}/checkpoints/source_model/qwen_2_5"
ADAPTER    = f"{BASE}/checkpoints/awq_models/qwen_trader_sft_lora"
MERGED     = f"{BASE}/checkpoints/awq_models/qwen_trader_sft_merged_fp16"
DATA_DIR   = f"{BASE}/checkpoints/clallibration_data/trader"
DS_SUJET   = f"{DATA_DIR}/SujetFinance"
DS_ALPACA  = f"{DATA_DIR}/finance_alpaca"
OUT        = f"{BASE}/checkpoints/awq_models/qwen_trader_awq_w4a16"

NUM_SAMPLES = 128
MAX_SEQ_LEN = 2048
KEEP_MERGED = False
random.seed(0)

# system prompt matches SFT training script for calibration distribution match
SYS = (
    "You are an expert financial analyst and trading mentor. "
    "Answer questions about markets, trading strategies, technical analysis, "
    "risk management, and financial concepts with precision. "
    "Be concise, structured, and accurate. If uncertain, say so."
)

# ----------------------------- stage 1: merge LoRA -> FP16 (CPU) -----------------------------
# NOTE: your SFT script already produced a merged fp16 dir. We skip merge if it exists.
if not os.path.exists(os.path.join(MERGED, "config.json")):
    print(f"[merge] loading base on CPU: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)

    if len(tokenizer) != base.get_input_embeddings().weight.size(0):
        print(f"[merge] resizing embeddings: {base.get_input_embeddings().weight.size(0)} -> {len(tokenizer)}")
        base.resize_token_embeddings(len(tokenizer))

    print(f"[merge] attaching adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(base, ADAPTER, device_map="cpu")
    model = model.merge_and_unload()

    os.makedirs(MERGED, exist_ok=True)
    model.save_pretrained(MERGED, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(MERGED)
    print(f"[merge] saved FP16 -> {MERGED}")

    del model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
else:
    print(f"[merge] skipped — {MERGED} already exists")

# ----------------------------- stage 2: AWQ quantize (GPU) -----------------------------
model = AutoAWQForCausalLM.from_pretrained(
    MERGED,
    safetensors=True,
    low_cpu_mem_usage=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(MERGED, trust_remote_code=True)

# ----------------------------- load calibration dataset -----------------------------
def load_any(path):
    parquet_files = sorted(glob.glob(f"{path}/data/*.parquet")) + sorted(glob.glob(f"{path}/*.parquet"))
    jsonl_files   = sorted(glob.glob(f"{path}/data/*.jsonl"))   + sorted(glob.glob(f"{path}/*.jsonl"))
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, split="train")
    if jsonl_files:
        return load_dataset("json", data_files=jsonl_files, split="train")
    try:
        return load_dataset(path, split="train")
    except Exception:
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train" if "train" in ds else list(ds.keys())[0]]
        return ds

print("[calib] loading SujetFinance + finance_alpaca")
sujet  = load_any(DS_SUJET)
alpaca = load_any(DS_ALPACA)
print(f"[calib] sujet  cols: {sujet.column_names}")
print(f"[calib] alpaca cols: {alpaca.column_names}")

# ----------------------------- normalize schema -----------------------------
def norm_sujet(ex):
    user = (ex.get("user_prompt") or ex.get("inputs") or ex.get("instruction")
            or ex.get("question") or "")
    ans  = (ex.get("answer") or ex.get("outputs") or ex.get("output")
            or ex.get("response") or "")
    return user, ans

def norm_alpaca(ex):
    instr = ex.get("instruction", "")
    inp   = ex.get("input", "") or ""
    user  = f"{instr}\n\n{inp}".strip() if inp else instr
    return user, ex.get("output", "")

rows  = [norm_sujet(sujet[i])   for i in range(len(sujet))]
rows += [norm_alpaca(alpaca[i]) for i in range(len(alpaca))]
rows  = [r for r in rows if r[0] and r[1] and len(r[1]) > 10]

if not rows:
    raise RuntimeError("No usable rows after normalization. Check column names.")

rows = random.sample(rows, min(NUM_SAMPLES, len(rows)))

def build(user, assistant):
    msgs = [
        {"role": "system",    "content": SYS},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)

texts = [build(*r) for r in rows]
print(f"[calib] samples: {len(texts)}")

# ----------------------------- AWQ quantize -----------------------------
quant_config = {
    "zero_point":   True,
    "q_group_size": 128,
    "w_bit":        4,
    "version":      "GEMM",
    "modules_to_not_convert": ["lm_head"],
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
tokenizer.save_pretrained(OUT)

cfg_path = os.path.join(OUT, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["quantization_config"] = {
    "quant_method": "awq",
    "bits":         quant_config["w_bit"],
    "group_size":   quant_config["q_group_size"],
    "zero_point":   quant_config["zero_point"],
    "version":      quant_config["version"].lower(),
}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

print(f"Saved AWQ checkpoint -> {OUT}")

# ----------------------------- cleanup -----------------------------
if not KEEP_MERGED:
    shutil.rmtree(MERGED, ignore_errors=True)
    print(f"[cleanup] removed intermediate FP16 dir: {MERGED}")