"""
SFT fine-tuning for Qwen — trader knowledge model.
Stack: TRL SFTTrainer + PEFT LoRA + bnb 4-bit.
"""
import os
import json
from typing import Dict, Any, List

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

os.environ["WANDB_PROJECT"] = "PRISM-RAG-Trader"

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
BASE       = "/home/lisa/Arupreza/PRISM-RAG"
MODEL_PATH = f"{BASE}/checkpoints/source_model/qwen_2_5"   # change if you use a different base
DATA_DIR   = f"{BASE}/checkpoints/clallibration_data/trader"
SAVE_DIR   = f"{BASE}/checkpoints/awq_models"

DS_SUJET   = f"{DATA_DIR}/SujetFinance"
DS_ALPACA  = f"{DATA_DIR}/finance_alpaca"

LORA_DIR        = f"{SAVE_DIR}/qwen_trader_sft_lora"
MERGED_FP16_DIR = f"{SAVE_DIR}/qwen_trader_sft_merged_fp16"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def is_bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# ─────────────────────────────────────────────────────────────────
# 1. MODEL — bnb 4-bit + LoRA
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
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # right padding for SFT (different from GRPO)

print("Loading model (bnb 4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
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
    "You are an expert financial analyst and trading mentor. "
    "Answer questions about markets, trading strategies, technical analysis, "
    "risk management, and financial concepts with precision. "
    "Be concise, structured, and accurate. If uncertain, say so."
)

# ─────────────────────────────────────────────────────────────────
# 3. DATASET LOADING + NORMALIZATION
# ─────────────────────────────────────────────────────────────────
# Both datasets are instruction-style. Normalize to (instruction, input, output)
# then convert to chat format and let the tokenizer apply the chat template.

def normalize_sujet(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sujet-Finance-Instruct-177k typically has fields like:
    'system_prompt', 'user_prompt', 'answer'  (or 'inputs'/'outputs' variants)
    """
    user = (
        example.get("user_prompt")
        or example.get("inputs")
        or example.get("instruction")
        or example.get("question", "")
    )
    answer = (
        example.get("answer")
        or example.get("outputs")
        or example.get("output")
        or example.get("response", "")
    )
    sys_p = example.get("system_prompt") or SYSTEM
    return {"system": sys_p, "user": user, "assistant": answer}

def normalize_alpaca(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    finance-alpaca has: 'instruction', 'input', 'output'
    """
    instr = example.get("instruction", "")
    inp   = example.get("input", "") or ""
    user  = f"{instr}\n\n{inp}".strip() if inp else instr
    out   = example.get("output", "")
    return {"system": SYSTEM, "user": user, "assistant": out}

def to_chat(example: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system",    "content": example["system"]},
        {"role": "user",      "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

print("Loading datasets...")
ds_sujet  = load_dataset(DS_SUJET,  split="train")
ds_alpaca = load_dataset(DS_ALPACA, split="train")

print(f"Sujet columns:  {ds_sujet.column_names}")
print(f"Alpaca columns: {ds_alpaca.column_names}")

ds_sujet  = ds_sujet.map(normalize_sujet,   remove_columns=ds_sujet.column_names)
ds_alpaca = ds_alpaca.map(normalize_alpaca, remove_columns=ds_alpaca.column_names)

# Filter empty/garbage rows
def valid(x):
    return bool(x["user"]) and bool(x["assistant"]) and len(x["assistant"]) > 10

ds_sujet  = ds_sujet.filter(valid)
ds_alpaca = ds_alpaca.filter(valid)

train_ds = concatenate_datasets([ds_sujet, ds_alpaca]).shuffle(seed=42)
train_ds = train_ds.map(to_chat, remove_columns=["system","user","assistant"])

# Drop rows that exceed MAX_SEQ_LEN after tokenization (cheap pre-filter)
def fits(x):
    return len(tokenizer.encode(x["text"])) <= MAX_SEQ_LEN

train_ds = train_ds.filter(fits)
print(f"Total training examples: {len(train_ds)}")

# ─────────────────────────────────────────────────────────────────
# 4. SFT CONFIG
# ─────────────────────────────────────────────────────────────────
cfg = SFTConfig(
    output_dir=LORA_DIR,

    # Optimizer
    learning_rate=2e-4,             # higher than GRPO; standard for LoRA SFT
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",

    # Batching
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16

    # SFT-specific
    max_length=MAX_SEQ_LEN,
    packing=False,                  # set True if you want to pack short seqs (faster)
    dataset_text_field="text",

    # Schedule
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=3,
    logging_steps=10,

    # Precision
    bf16=is_bf16_supported(),
    fp16=not is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Logging
    report_to="wandb",
    run_name="qwen-trader-sft",
)

# ─────────────────────────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=train_ds,
)

print(f"Starting SFT training on {len(train_ds)} examples...")
trainer.train()

# ─────────────────────────────────────────────────────────────────
# 6. SAVE LoRA adapters
# ─────────────────────────────────────────────────────────────────
print("Saving LoRA adapters...")
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

# ─────────────────────────────────────────────────────────────────
# 7. MERGE LoRA → fp16 (for AWQ next)
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
print(f"  LoRA adapters: {LORA_DIR}")
print(f"  Merged fp16:   {MERGED_FP16_DIR}")
print(f"  Next: AWQ quantize {MERGED_FP16_DIR}")