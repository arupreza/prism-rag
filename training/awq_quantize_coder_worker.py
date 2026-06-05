# training/awq_quantize_coder_worker.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import glob
import json
import random
import shutil
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from peft import PeftModel

# ----------------------------- config -----------------------------
BASE_MODEL = "checkpoints/source_model/qwen_coder"
ADAPTER    = "checkpoints/awq_models/qwen_coder_grpo_lora/checkpoint-200"
MERGED     = "checkpoints/awq_models/qwen_coder_merged_fp16"
CALIB      = "checkpoints/clallibration_data/coder/verifiable-coding-problems-python"
OUT        = "checkpoints/awq_models/qwen_coder_awq_w4a16"

NUM_SAMPLES = 128
MAX_SEQ_LEN = 2048
KEEP_MERGED = False
random.seed(0)

# system prompt matches GRPO training script for calibration distribution match
SYS = (
    "You are an expert Python programmer. Solve the problem below.\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\nExplain your approach briefly.\n</reasoning>\n"
    "<code>\n```python\n# your complete solution here\n```\n</code>"
)

# ----------------------------- stage 1: merge LoRA -> FP16 (CPU) -----------------------------
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
parquet_files = sorted(glob.glob(f"{CALIB}/data/*.parquet"))
jsonl_files   = sorted(glob.glob(f"{CALIB}/data/*.jsonl"))

if parquet_files:
    print(f"[calib] loading {len(parquet_files)} parquet file(s)")
    coder = load_dataset("parquet", data_files=parquet_files, split="train")
elif jsonl_files:
    print(f"[calib] loading {len(jsonl_files)} jsonl file(s)")
    coder = load_dataset("json", data_files=jsonl_files, split="train")
else:
    # fallback: try the HF auto-detect, then save_to_disk format
    try:
        coder = load_dataset(CALIB, split="train")
    except Exception:
        coder = load_from_disk(CALIB)
        if isinstance(coder, DatasetDict):
            coder = coder["train" if "train" in coder else list(coder.keys())[0]]

print(f"[calib] dataset columns: {coder.column_names}")
print(f"[calib] first row sample: { {k: str(v)[:120] for k, v in list(coder[0].items())[:6]} }")

# ----------------------------- normalize schema -----------------------------
# verifiable-coding-problems-python (from training script) carries:
#   - problem text in one of: prompt / problem / question
#   - tests + function_name inside verification_info (dict or JSON string)
# we only need (problem, solution_proxy) for AWQ calibration — solution can be empty
def norm_coder(ex):
    prob = ex.get("problem_statement")
    sol  = ex.get("gold_standard_solution") or ""
    return prob, sol

rows = [norm_coder(coder[i]) for i in range(len(coder))]
rows = [r for r in rows if r[0]]

if not rows:
    raise RuntimeError(
        f"No usable rows after normalization. Columns were: {coder.column_names}. "
        f"Update norm_coder() with the correct prompt column name."
    )

rows = random.sample(rows, min(NUM_SAMPLES, len(rows)))

def build(problem, solution):
    msgs = [
        {"role": "system",    "content": SYS},
        {"role": "user",      "content": problem},
        {"role": "assistant", "content": solution or "# solution"},
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