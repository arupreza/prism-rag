# finetune_models/awq_quantize_law_worker.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # set before torch import

import glob
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# ----------------------------- config -----------------------------
SRC   = "checkpoints/source_model/law_llm"
CALIB = "checkpoints/clallibration_data/legal"
OUT   = "checkpoints/awq_models/law_llm_awq_w4a16"

NUM_SAMPLES   = 128
MAX_SEQ_LEN   = 1024           # calibration length (not serving length)
REFUSAL_FRAC  = 0.30           # share of "not in context" samples
random.seed(0)

# ----------------------------- model -----------------------------
# autoawq handles device placement internally during quantization
# low_cpu_mem_usage keeps RAM tame on big models
model = AutoAWQForCausalLM.from_pretrained(
    SRC,
    safetensors=True,
    low_cpu_mem_usage=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

# ----------------------------- datasets -----------------------------
# cuad-qa: script loaders are dead in datasets v4 -> use HF auto-parquet branch
cuad = load_dataset(
    "theatticusproject/cuad-qa",
    revision="refs/convert/parquet",
    split="train",
)

# LegalQAEval: raw jsonl (val + test), carries the unanswerable cases
lqa = load_dataset(
    "json",
    data_files=glob.glob(f"{CALIB}/LegalQAEval/*.jsonl"),
    split="train",
)

# ----------------- normalize both schemas -> (context, question, answer|None) -----------------
REFUSAL = "The answer is not contained in the provided context."

def norm_cuad(ex):
    a = ex["answers"]["text"]               # dict of lists
    return ex["context"], ex["question"], (a[0] if a else None)

def norm_lqa(ex):
    a = ex["answers"]                       # list of {text,start,end}
    return ex["text"], ex["question"], (a[0]["text"] if a else None)

rows  = [norm_cuad(cuad[i]) for i in range(len(cuad))]
rows += [norm_lqa(lqa[i])  for i in range(len(lqa))]

# ----------------- balance answerable vs refusal BEFORE truncating -----------------
answerable   = [r for r in rows if r[2]]
unanswerable = [r for r in rows if not r[2]]

n_ref = min(int(REFUSAL_FRAC * NUM_SAMPLES), len(unanswerable))
n_ans = NUM_SAMPLES - n_ref
rows = random.sample(answerable, min(n_ans, len(answerable))) + \
        random.sample(unanswerable, n_ref)
random.shuffle(rows)

ctx_pool = [r[0][:1500] for r in rows if r[0]]   # distractor chunks for retrieval noise

# ----------------- build RAG-shaped calibration (Mistral template, answer included) -----------------
SYS = ("You are a legal assistant. Answer the question using ONLY the context. "
        "Cite the relevant clause. If the answer is not in the context, say so.")

def build(ctx, q, ans):
    chunks = [ctx[:1500]] + random.sample(ctx_pool, k=min(2, len(ctx_pool)))
    random.shuffle(chunks)
    retrieved = "\n\n---\n\n".join(chunks)
    # Mistral has no separate system role -> fold SYS into the user turn
    user = f"{SYS}\n\nContext:\n{retrieved}\n\nQuestion: {q}"
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ans if ans else REFUSAL},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)

texts = [build(*r) for r in rows]
# autoawq takes a list[str] directly as calib_data — no pre-tokenization needed

print(f"calib samples: {len(texts)} | refusals: {sum(1 for r in rows if not r[2])}")

# ----------------------------- AWQ quantize -----------------------------
# W4A16 equivalent in autoawq: w_bit=4 weights, activations stay fp16 (default)
# GEMM version is faster than GEMV on most modern GPUs (Ampere+)
quant_config = {
    "zero_point":   True,
    "q_group_size": 128,
    "w_bit":        4,
    "version":      "GEMM",
    # autoawq ignores lm_head and embed_tokens by default — no need to specify
    "modules_to_not_convert": ["lm_head"],
}

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=texts,                      # list[str] of pre-templated samples
    max_calib_seq_len=MAX_SEQ_LEN,
    max_calib_samples=NUM_SAMPLES,
)

# ----------------------------- save -----------------------------
model.save_quantized(OUT)
# ensure tokenizer + chat template ship with the checkpoint (vLLM needs them)
tokenizer.save_pretrained(OUT)
print(f"Saved AWQ checkpoint -> {OUT}")