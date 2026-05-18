"""
PRISM-RAG — Domain Dataset Downloader
=======================================
Downloads all available data from verified HuggingFace sources.

Verified data availability:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Domain   │ Source                                  │ Total Rows  │ Size     │
├──────────┼─────────────────────────────────────────┼─────────────┼──────────┤
│ politics │ vblagoje/cc_news                        │ 708,241     │ 1.12 GB  │
│ politics │ Eugleo/us-congressional-speeches        │ 17,400,000  │ ~50 GB   │
│ finance  │ ashraq/financial-news-articles          │ 306,242     │ ~500 MB  │
│ ai_tech  │ CShorten/ML-ArXiv-Papers                │ 118,000     │ ~200 MB  │
│ medical  │ ccdv/pubmed-summarization               │ 119,924     │ ~1 GB    │
│ medical  │ ccdv/arxiv-summarization                │ 203,037     │ ~2 GB    │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    uv run python data/download.py                      # all domains
    uv run python data/download.py --domains politics   # one domain
    uv run python data/download.py --limit 50000        # cap all sources
    uv run python data/download.py --no-limit           # download everything
"""

import os
import json
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN not found in .env\n"
        'Add:  HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"'
    )

login(token=HF_TOKEN, add_to_git_credential=False)
print("[✓] HuggingFace login successful")

BASE_DIR = Path("/home/lisa/Arupreza/PRISM-RAG/data")

# ── Dataset sources (all verified, public, Parquet-formatted) ─────────────────
SOURCES = [

    # ── POLITICS ──────────────────────────────────────────────────────────────
    {
        "domain":     "politics",
        "name":       "cc_news",
        "hf_name":    "vblagoje/cc_news",
        "hf_config":  None,
        "split":      "train",
        "stream":     False,
        "total_rows": 708_241,
        "default_limit": None,          # download ALL 708K
        "desc":       "708K English news articles Jan 2017 - Dec 2019",
        "normalizer": lambda r: {
            "id":       r.get("url", ""),
            "title":    r.get("title", ""),
            "text":     r.get("text", ""),
            "source":   "cc_news",
            "metadata": {
                "date":        r.get("date", ""),
                "domain":      r.get("domain", ""),
                "description": r.get("description", ""),
                "url":         r.get("url", ""),
            },
        },
    },

    {
        "domain":     "politics",
        "name":       "congressional_speeches",
        "hf_name":    "Eugleo/us-congressional-speeches",
        "hf_config":  None,
        "split":      "train",
        "stream":     True,
        "total_rows": 17_400_000,
        "default_limit": 700_000,       # cap — 17M is too large for prototype
        "desc":       "17.4M US Congressional speeches 1873-2024",
        "normalizer": lambda r: {
            "id":       str(r.get("speech_id", "")),
            "title":    f"Speech — {r.get('speaker', 'Unknown')} — {str(r.get('date', ''))[:10]}",
            "text":     r.get("text", ""),
            "source":   "congressional_speeches",
            "metadata": {
                "speaker": r.get("speaker", ""),
                "chamber": r.get("chamber", ""),
                "date":    str(r.get("date", ""))[:10],
                "state":   r.get("state", ""),
                "gender":  r.get("gender", ""),
            },
        },
    },

    # ── FINANCE ───────────────────────────────────────────────────────────────
    {
        "domain":     "finance",
        "name":       "financial_news",
        "hf_name":    "ashraq/financial-news-articles",
        "hf_config":  None,
        "split":      "train",
        "stream":     False,
        "total_rows": 306_242,
        "default_limit": None,          # download ALL 306K
        "desc":       "306K financial news articles from Reuters/CNBC/WSJ (2018)",
        "normalizer": lambda r: {
            "id":       r.get("url", ""),
            "title":    r.get("title", ""),
            "text":     r.get("text", ""),
            "source":   "financial_news",
            "metadata": {
                "url": r.get("url", ""),
            },
        },
    },

    # ── AI TECHNOLOGY ─────────────────────────────────────────────────────────
    {
        "domain":     "ai_tech",
        "name":       "ml_arxiv_papers",
        "hf_name":    "CShorten/ML-ArXiv-Papers",
        "hf_config":  None,
        "split":      "train",
        "stream":     False,
        "total_rows": 118_000,
        "default_limit": None,          # download ALL 118K
        "desc":       "118K ML & AI ArXiv papers — titles + abstracts",
        "normalizer": lambda r: {
            "id":       str(r.get("Unnamed: 0", r.get("id", ""))),
            "title":    r.get("title", ""),
            "text":     r.get("title", "") + "\n\n" + r.get("abstract", ""),
            "source":   "ml_arxiv_papers",
            "metadata": {
                "authors":    r.get("authors", ""),
                "categories": r.get("categories", ""),
                "doi":        r.get("doi", ""),
            },
        },
    },

    # ── MEDICAL ───────────────────────────────────────────────────────────────
    {
        "domain":     "medical",
        "name":       "pubmed_papers",
        "hf_name":    "ccdv/pubmed-summarization",
        "hf_config":  "document",
        "split":      "train",
        "stream":     True,
        "total_rows": 119_924,
        "default_limit": None,          # download ALL ~120K
        "desc":       "119,924 PubMed full text papers (Cohan et al. 2018, Parquet)",
        "normalizer": lambda r: {
            "id":       str(abs(hash(r.get("article", "")[:100]))),
            "title":    (r.get("abstract", "") or "PubMed Paper")[:150],
            "text":     r.get("article", ""),
            "source":   "pubmed_papers",
            "metadata": {
                "abstract": r.get("abstract", ""),
            },
        },
    },

    {
        "domain":     "medical",
        "name":       "arxiv_papers",
        "hf_name":    "ccdv/arxiv-summarization",
        "hf_config":  "document",
        "split":      "train",
        "stream":     True,
        "total_rows": 203_037,
        "default_limit": None,          # download ALL ~203K
        "desc":       "203,037 ArXiv full text papers (Cohan et al. 2018, Parquet)",
        "normalizer": lambda r: {
            "id":       str(abs(hash(r.get("article", "")[:100]))),
            "title":    (r.get("abstract", "") or "ArXiv Paper")[:150],
            "text":     r.get("article", ""),
            "source":   "arxiv_papers",
            "metadata": {
                "abstract": r.get("abstract", ""),
            },
        },
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_header(text: str):
    bar = "=" * 65
    print(f"\n{bar}\n  {text}\n{bar}")


def save_jsonl(dataset, path: Path, limit, stream, normalizer) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            try:
                record = normalizer(row)
                if not record.get("text", "").strip():
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                continue

            if limit and count >= limit:
                break

            if count % 10_000 == 0 and count > 0:
                print(f"    {count:>10,} docs saved...", end="\r")

    print()
    return count


def download_source(src: dict, limit_override=None) -> dict:
    limit  = limit_override if limit_override is not None else src["default_limit"]
    domain = src["domain"]
    name   = src["name"]

    print(f"\n  Source : {src['hf_name']}")
    print(f"  Desc   : {src['desc']}")
    print(f"  Total  : {src['total_rows']:,} rows available")
    print(f"  Limit  : {'ALL' if not limit else f'{limit:,}'}")

    t0 = time.time()

    kwargs = dict(
        split=src["split"],
        token=HF_TOKEN,
        streaming=src["stream"],
    )
    try:
        if src["hf_config"]:
            ds = load_dataset(src["hf_name"], src["hf_config"], **kwargs)
        else:
            ds = load_dataset(src["hf_name"], **kwargs)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return {"domain": domain, "name": name, "count": 0, "errors": [str(e)]}

    path  = BASE_DIR / domain / f"{name}.jsonl"
    count = save_jsonl(ds, path, limit, src["stream"], src["normalizer"])

    size = path.stat().st_size
    size_str = (
        f"{size/1e9:.2f} GB" if size > 1e9 else
        f"{size/1e6:.1f} MB" if size > 1e6 else
        f"{size/1e3:.1f} KB"
    )
    pct = f"{(count/src['total_rows']*100):.1f}%" if src["total_rows"] else "?"

    print(f"  Saved  : {count:,} docs  ({pct} of total)  {size_str}")
    print(f"  Path   : {path}")
    print(f"  Time   : {time.time()-t0:.1f}s")

    return {"domain": domain, "name": name, "count": count,
            "size": size_str, "pct": pct, "errors": []}


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_stats: list):
    print_header("DOWNLOAD SUMMARY")

    domains = {}
    for s in all_stats:
        domains.setdefault(s["domain"], []).append(s)

    for domain, sources in domains.items():
        total = sum(s["count"] for s in sources)
        print(f"\n  {domain.upper()} — {total:,} total docs")
        for s in sources:
            flag = "⚠" if s["errors"] else "✓"
            print(f"    {flag} {s['name']:<30} {s['count']:>10,} docs")

    print(f"\n  {'File':<55} {'Lines':>10}")
    print(f"  {'-'*55} {'-'*10}")
    for f in sorted(BASE_DIR.rglob("*.jsonl")):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        rel   = str(f.relative_to(BASE_DIR))
        print(f"  {rel:<55} {lines:>10,}")

    grand_total = sum(s["count"] for s in all_stats)
    print(f"\n  Grand total : {grand_total:,} documents")
    print(f"  Base dir    : {BASE_DIR}")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PRISM-RAG Dataset Downloader")
    parser.add_argument(
        "--domains", nargs="+",
        choices=["politics", "finance", "ai_tech", "medical"],
        default=["politics", "finance", "ai_tech", "medical"],
        help="Domains to download"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Override limit for ALL sources (for quick test)"
    )
    parser.add_argument(
        "--no-limit", action="store_true",
        help="Download EVERYTHING — ignores default caps (WARNING: very large)"
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print_header("PRISM-RAG — Dataset Downloader")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  Domains  : {', '.join(args.domains)}")

    print(f"\n  Available data:")
    print(f"  {'Domain':<10} {'Source':<35} {'Available':>12}  {'Will download':>14}")
    print(f"  {'-'*10} {'-'*35} {'-'*12}  {'-'*14}")
    for src in SOURCES:
        if src["domain"] not in args.domains:
            continue
        limit = 0 if args.no_limit else (args.limit or src["default_limit"])
        will  = "ALL" if not limit else f"{limit:,}"
        print(f"  {src['domain']:<10} {src['name']:<35} {src['total_rows']:>12,}  {will:>14}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for d in args.domains:
        (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

    all_stats = []
    for src in SOURCES:
        if src["domain"] not in args.domains:
            continue
        print_header(f"{src['domain'].upper()} — {src['name']}")
        limit_override = (
            0 if args.no_limit else
            args.limit if args.limit is not None else
            None
        )
        stats = download_source(src, limit_override=limit_override)
        all_stats.append(stats)

    manifest = {
        "base_dir":      str(BASE_DIR),
        "hf_token_used": HF_TOKEN[:8] + "...",
        "domains":       args.domains,
        "stats":         all_stats,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(BASE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print_summary(all_stats)
    print_header("Done")


if __name__ == "__main__":
    main()