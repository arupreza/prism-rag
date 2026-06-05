"""
GRPO fine-tuning for Qwen Coder — plain TRL + PEFT + bnb 4-bit.
"""
import os
import re
import sys
import json
import tempfile
import subprocess
from typing import List, Dict, Any

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import GRPOConfig, GRPOTrainer

os.environ["WANDB_PROJECT"] = "PRISM-RAG"
os.environ["WANDB_ENTITY"]  = "arupreza-soonchunhyang-university"

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE          = "/home/lisa/Arupreza/PRISM-RAG"
MODEL_PATH    = f"{BASE}/checkpoints/source_model/qwen_coder"
DATA_DIR      = f"{BASE}/checkpoints/clallibration_data/coder"
SAVE_DIR      = f"{BASE}/checkpoints/awq_models"

DS_VERIFIABLE = f"{DATA_DIR}/verifiable-coding-problems-python"
DS_LEETCODE   = f"{DATA_DIR}/LeetCodeDataset"

LORA_DIR        = f"{SAVE_DIR}/qwen_coder_grpo_lora"
MERGED_FP16_DIR = f"{SAVE_DIR}/qwen_coder_grpo_merged_fp16"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def is_bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# ─────────────────────────────────────────────────────────────────
# 1. MODEL — bnb 4-bit + LoRA (no Unsloth)
# ─────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 2048
LORA_RANK   = 32

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading model (bnb 4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16 if is_bf16_supported() else torch.float16,
)
model.config.use_cache = False

print("Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────
# 2. SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are an expert Python programmer. Solve the problem below.\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\n"
    "Explain your approach briefly.\n"
    "</reasoning>\n"
    "<code>\n"
    "```python\n"
    "# your complete solution here\n"
    "```\n"
    "</code>"
)

# ─────────────────────────────────────────────────────────────────
# 3. DATASET LOADING + NORMALIZATION
# ─────────────────────────────────────────────────────────────────
def normalize_verifiable(example: Dict[str, Any]) -> Dict[str, Any]:
    problem = (
        example.get("prompt")
        or example.get("problem")
        or example.get("question", "")
    )
    vinfo = example.get("verification_info", {})
    if isinstance(vinfo, str):
        try: vinfo = json.loads(vinfo)
        except Exception: vinfo = {}
    test_cases = vinfo.get("test_cases", []) or vinfo.get("tests", [])
    test_code = "\n".join(str(t) for t in test_cases) if isinstance(test_cases, list) else str(test_cases)
    return {
        "problem":     problem,
        "test_code":   test_code,
        "entry_point": vinfo.get("function_name", ""),
    }

def normalize_leetcode(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "problem":     example.get("prompt") or example.get("problem", ""),
        "test_code":   example.get("test", ""),
        "entry_point": example.get("entry_point", ""),
    }

def to_chat_format(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": example["problem"]},
        ],
        "test_code":   example["test_code"],
        "entry_point": example["entry_point"],
    }

print("Loading datasets...")
ds_verif = load_dataset(DS_VERIFIABLE, split="train")
ds_leet  = load_dataset(DS_LEETCODE,   split="train")

ds_verif = ds_verif.map(normalize_verifiable, remove_columns=ds_verif.column_names)
ds_leet  = ds_leet.map(normalize_leetcode,    remove_columns=ds_leet.column_names)

ds_verif = ds_verif.filter(lambda x: bool(x["test_code"]) and bool(x["problem"]))
ds_leet  = ds_leet.filter(lambda x: bool(x["test_code"]) and bool(x["problem"]))

train_ds = concatenate_datasets([ds_verif, ds_leet]).shuffle(seed=42)
train_ds = train_ds.map(to_chat_format)
print(f"Total training examples: {len(train_ds)}")

# ─────────────────────────────────────────────────────────────────
# 4. CODE EXECUTION HELPER
# ─────────────────────────────────────────────────────────────────
EXEC_TIMEOUT = 8

def run_python(code: str) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                        delete=False, dir="/tmp") as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True,
            timeout=EXEC_TIMEOUT,
        )
        return {
            "success": result.returncode == 0,
            "stderr":  result.stderr[-500:],
            "stdout":  result.stdout[-200:],
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stderr": "TIMEOUT", "stdout": ""}
    except Exception as e:
        return {"success": False, "stderr": str(e), "stdout": ""}
    finally:
        try: os.remove(path)
        except Exception: pass

def extract_python_code(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.S)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.S)
    if m: return m.group(1).strip()
    return ""

# ─────────────────────────────────────────────────────────────────
# 5. REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def reward_format(prompts, completions, **kwargs) -> List[float]:
    pat = re.compile(
        r"<reasoning>.*?</reasoning>\s*<code>\s*```python\s*\n.*?```\s*</code>",
        re.S,
    )
    return [0.5 if pat.search(c[0]["content"]) else 0.0 for c in completions]

def reward_syntax(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        code = extract_python_code(c[0]["content"])
        if not code:
            rewards.append(0.0)
            continue
        try:
            compile(code, "<candidate>", "exec")
            rewards.append(0.25)
        except SyntaxError:
            rewards.append(0.0)
    return rewards

def reward_correctness(prompts, completions, test_code, entry_point, **kwargs) -> List[float]:
    rewards = []
    for c, tests in zip(completions, test_code):
        code = extract_python_code(c[0]["content"])
        if not code or not tests:
            rewards.append(0.0)
            continue
        full_program = code + "\n\n" + tests + "\n"
        if "check(" not in full_program and "def check(" in tests:
            full_program += "\ncheck(Solution())\n"
        res = run_python(full_program)
        rewards.append(2.0 if res["success"] else 0.0)
    return rewards

def reward_length(prompts, completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        n = len(tokenizer.encode(c[0]["content"]))
        rewards.append(0.1 if 100 <= n < 800 else 0.0)
    return rewards

# ─────────────────────────────────────────────────────────────────
# 6. GRPO CONFIG
# ─────────────────────────────────────────────────────────────────
cfg = GRPOConfig(
    output_dir=LORA_DIR,

    # Optimizer
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

    # Batching
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # GRPO
    num_generations=8,
    max_prompt_length=768,
    max_completion_length=1024,
    beta=0.04,
    epsilon=0.2,
    temperature=0.9,

    # Schedule
    num_train_epochs=1,
    max_steps=1000,
    save_steps=100,
    logging_steps=5,

    # Precision
    bf16=is_bf16_supported(),
    fp16=not is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # No vLLM — use HF generate (stable)
    use_vllm=False,

    # Logging
    report_to="wandb",
    run_name="qwen-coder-grpo",
)

# ─────────────────────────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_format,       # +0.5
        reward_syntax,       # +0.25
        reward_correctness,  # +2.0
        reward_length,       # +0.1
    ],
    args=cfg,
    train_dataset=train_ds,
)

print(f"Starting GRPO training on {len(train_ds)} examples...")
trainer.train()

# ─────────────────────────────────────────────────────────────────
# 8. SAVE LoRA adapters
# ─────────────────────────────────────────────────────────────────
print("Saving LoRA adapters...")
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

# ─────────────────────────────────────────────────────────────────
# 9. MERGE LoRA → fp16 (feed this to AWQ next)
# ─────────────────────────────────────────────────────────────────
print("Merging LoRA into base (fp16)...")
del model
torch.cuda.empty_cache()

base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
merged = PeftModel.from_pretrained(base, LORA_DIR).merge_and_unload()
merged.save_pretrained(MERGED_FP16_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_FP16_DIR)

print("✓ Done.")
print(f"  LoRA adapters:     {LORA_DIR}")
print(f"  Merged fp16:       {MERGED_FP16_DIR}")
print(f"  Next: AWQ quantize {MERGED_FP16_DIR}")