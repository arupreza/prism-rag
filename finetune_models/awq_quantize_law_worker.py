# finetune_models/awq_quantize_law_worker.py
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # set before torch import

import glob
import random
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# ----------------------------- config -----------------------------
SRC   = "checkpoints/source_model/law_llm"
CALIB = "checkpoints/clallibration_data/legal"
OUT   = "checkpoints/awq_models/law_llm_awq_w4a16"

NUM_SAMPLES   = 128
MAX_SEQ_LEN   = 512           # calibration length (not serving length)
REFUSAL_FRAC  = 0.30           # share of "not in context" samples
random.seed(0)

# ----------------------------- model -----------------------------
# CPU load: llm-compressor onloads layers to GPU one at a time (memory-safe)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(SRC)

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

# oneshot needs a HF Dataset, not a list
ds = Dataset.from_list([
    dict(tokenizer(t, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False))
    for t in texts
])

print(f"calib samples: {len(ds)} | refusals: {sum(1 for r in rows if not r[2])}")

# ----------------------------- AWQ recipe -----------------------------
# scheme lives ON AWQModifier in current llm-compressor (no separate QuantizationModifier)
recipe = [
    AWQModifier(
        ignore=["lm_head"],
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

# ----------------------------- quantize + save -----------------------------
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQ_LEN,
    num_calibration_samples=NUM_SAMPLES,
    output_dir=OUT,
)

# ensure tokenizer + chat template ship with the checkpoint (vLLM needs them)
tokenizer.save_pretrained(OUT)
print(f"Saved AWQ checkpoint -> {OUT}")