"""Phase 3 cluster summarizer — in-process Qwen2.5-7B via transformers.

No HTTP, no vLLM, no port 8000. Loads the model once into this process and
generates locally. Same public interface as before:

    ClusterSummarizer().summarize(texts) -> (title, summary)

so agents/tree_builder/build.py needs no changes.

Map-reduce: a cluster can hold hundreds of leaves. Member texts are packed into
char-budgeted blocks, each block summarized, then the block summaries summarized
again — keeps every generation under the context window instead of truncating.

Runtime note: this generates sequentially (no continuous batching like vLLM),
so it is slower per call. It is an offline batch job — acceptable. If the
summarizer shares the GPU with BGE-M3, set EMBED_DEVICE=cpu for the build.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agents.ingestion.config import SUMMARIZER_MODEL

CHAR_BUDGET = 18_000          # ~5k tokens of input per generation
MAX_NEW_TOKENS = 512

_SYS = (
    "You compress documents into one dense, factual summary. "
    "Preserve named entities, numbers, and concrete claims. Invent nothing. "
    "Write neutral prose, no preamble."
)


class ClusterSummarizer:
    def __init__(self, model_name: str = SUMMARIZER_MODEL, char_budget: int = CHAR_BUDGET):
        self.char_budget = char_budget
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        self.model.eval()

    # ── low-level generation ──────────────────────────────────────────────
    @torch.inference_mode()
    def _call(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        messages = [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy — deterministic summaries
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]   # strip the prompt tokens
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ── map-reduce packing ────────────────────────────────────────────────
    def _blocks(self, texts: list[str]):
        block: list[str] = []
        size = 0
        for t in texts:
            t = t[: self.char_budget]
            if block and size + len(t) > self.char_budget:
                yield block
                block, size = [], 0
            block.append(t)
            size += len(t)
        if block:
            yield block

    def _summarize_block(self, texts: list[str]) -> str:
        joined = "\n\n---\n\n".join(texts)
        return self._call(
            f"Summarize the following documents into one cohesive paragraph.\n\n"
            f"{joined}\n\nSummary:"
        )

    # ── public ────────────────────────────────────────────────────────────
    def summarize(self, texts: list[str]) -> tuple[str, str]:
        blocks = list(self._blocks(texts))
        if len(blocks) == 1:
            summary = self._summarize_block(blocks[0])
        else:
            partials = [self._summarize_block(b) for b in blocks]
            summary = self._summarize_block(partials)        # reduce step

        title = self._call(
            f"Give a 3-8 word topic title for this summary. "
            f"Output the title only, no quotes.\n\n{summary}",
            max_new_tokens=32,
        )
        return title.strip().strip('"').strip(), summary