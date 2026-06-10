"""Phase 4 — query CLI.

Calls tree_search.TreeSearcher directly (no HTTP) so retrieval logic can be
debugged without the FastAPI layer.

Usage (from repo root):
    python scripts/05_query_cli.py "What did Congress say about voter ID laws?"
    python scripts/05_query_cli.py "mRNA vaccine R&D financial impact" \
        --domain finance --mode collapsed --k 5

Flags:
    --mode {top_down,collapsed}   default top_down
    --domain DOMAIN               default None (cross-domain routing via tree)
    --source SOURCE               default None (any source in the domain)
    --k INT                       default 5  (returned leaves)
    --beam INT                    default 6  (top_down beam width)
    --fanout INT                  default 50 (collapsed flat-ANN candidates)
    --alpha FLOAT                 default 0.30 (collapsed ancestor-boost weight)
    --json                        print full JSON instead of human view
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.retrieval.tree_search import (        # noqa: E402
    DEFAULT_ALPHA, DEFAULT_BEAM, DEFAULT_FANOUT, DEFAULT_K, TreeSearcher,
)


def _truncate(s: str, n: int = 240) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_human(result) -> None:
    print(f"\nquery : {result.query}")
    print(f"mode  : {result.mode}   domain={result.domain}  source={result.source}  k={result.k}")
    if result.extras:
        print(f"extras: {result.extras}")

    if result.mode == "top_down":
        print("\ntraversal path:")
        for i, level in enumerate(result.path):
            tag = "roots" if i == 0 else f"step {i}"
            print(f"  [{tag}] {len(level)} nodes")
            for h in level:
                title = h.title or "(no title)"
                print(f"    L{h.level}  sim={h.sim:+.4f}  {h.source or '-':<24} {title}")
                print(f"            {_truncate(h.summary, 200)}")
    else:
        print("\ninternal-node hits used for re-rank:")
        for h in (result.path[0] if result.path else []):
            title = h.title or "(no title)"
            print(f"    L{h.level}  sim={h.sim:+.4f}  {h.source or '-':<24} {title}")

    print(f"\ntop-{result.k} leaves:")
    if not result.leaves:
        print("  (none — check DB state and domain/source filters)")
        return
    for rank, h in enumerate(result.leaves, 1):
        print(f"\n  #{rank}  sim={h.sim:+.4f}  {h.domain}/{h.source}  chunk={h.chunk_id}")
        wrapped = textwrap.fill(_truncate(h.summary, 600), width=100,
                                initial_indent="    ", subsequent_indent="    ")
        print(wrapped)


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM-RAG tree-guided retrieval CLI")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["top_down", "collapsed"], default="top_down")
    ap.add_argument("--domain", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    ap.add_argument("--fanout", type=int, default=DEFAULT_FANOUT)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--json", action="store_true", help="emit full JSON")
    args = ap.parse_args()

    searcher = TreeSearcher()
    result = searcher.retrieve(
        args.query,
        mode=args.mode,
        domain=args.domain,
        source=args.source,
        k=args.k,
        beam=args.beam,
        fanout=args.fanout,
        alpha=args.alpha,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()