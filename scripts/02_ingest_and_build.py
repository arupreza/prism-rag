"""
End-to-end ingest: sources (PDFs + code) -> figures captioned by VLM ->
parents + children -> RAPTOR over children -> Postgres.

Pipeline per file:
    load_any        extract text/code/image segments (images saved to disk)
    caption_doc     Qwen2.5-VL writes an explanation into each image segment
    chunk_segments  segments -> parent/child chunks (image caption is searchable)
    embed           BGE-M3 over all children
    insert_*        write parents + children (image_path carried for figures)
Then RAPTOR summary tree is built over the children of the whole domain.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from agents.ingestion.loader import iter_domain_sources, load_any
from agents.ingestion.captioner import caption_doc
from agents.ingestion.chunker import chunk_segments
from agents.ingestion.encoder import embed
from agents.ingestion.db import (
    conn, upsert_document, insert_parent, insert_children, insert_summary, mean_pool,
)
from agents.tree_builder.build import build_tree
from agents.tree_builder.summarizer import summarize


def ingest_domain(root: Path, domain: str) -> None:
    print(f"\n=== {domain.upper()} ===")
    leaf_ids: list[int] = []     # CHILD ids -> feed RAPTOR
    leaf_emb: list[list[float]] = []
    leaf_txt: list[str] = []     # use child text for clustering signal

    with conn() as c:
        for src_path in iter_domain_sources(root, domain):
            print(f"[{domain}] {src_path.name}")
            doc = load_any(src_path, domain)  # type: ignore[arg-type]

            # VLM resolves figures (caption) and scanned pages (verbatim OCR).
            n_cap, n_ocr = caption_doc(doc, domain)
            if n_cap or n_ocr:
                print(f"  captioned {n_cap} figure(s), transcribed {n_ocr} scanned page(s)")

            parents = chunk_segments(doc.segments)
            if not parents:
                continue

            doc_id = upsert_document(
                c, domain=domain, source_path=str(src_path.resolve()),
                title=doc.title, n_pages=doc.n_pages, sha256=doc.sha256,
            )

            # Embed all children of all parents in one batch per file
            all_children = []
            parent_to_slice = []  # (start, end) index slices into all_children
            for p in parents:
                start = len(all_children)
                all_children.extend(p.children)
                parent_to_slice.append((start, len(all_children)))
            child_texts = [ck.content for ck in all_children]
            child_vecs = embed(child_texts, batch_size=32)

            for p, (s, e) in zip(parents, parent_to_slice):
                ch_embs = child_vecs[s:e]
                ch_objs = all_children[s:e]
                parent_emb = mean_pool(ch_embs)
                pid = insert_parent(
                    c, document_id=doc_id, domain=domain,
                    parent=p, mean_emb=parent_emb,
                )
                cids = insert_children(
                    c, document_id=doc_id, domain=domain,
                    parent_id=pid, children=ch_objs, embeddings=ch_embs,
                )
                leaf_ids.extend(cids)
                leaf_emb.extend(ch_embs)
                leaf_txt.extend([ck.content for ck in ch_objs])
            c.commit()
            print(f"  parents={len(parents)} children={len(all_children)}")

        if not leaf_ids:
            print(f"[{domain}] no leaves -> skip tree")
            return

        print(f"[{domain}] building RAPTOR over {len(leaf_ids)} children...")
        parents_nodes = build_tree(
            leaf_ids=leaf_ids,
            leaf_embeddings=leaf_emb,
            leaf_contents=leaf_txt,
            domain=domain,
            summarizer=summarize,
            embed_fn=embed,
        )
        placeholder_to_real: dict[int, int] = {}
        for p in parents_nodes:
            real_children = [placeholder_to_real.get(cid, cid) for cid in p.children]
            real_id = insert_summary(
                c, domain=domain, level=p.level,
                cluster_id=p.cluster_id or 0,
                children_ids=real_children,
                content=p.content, embedding=p.embedding,
            )
            placeholder_to_real[p.id] = real_id
        c.commit()
        print(f"[{domain}] summary nodes: {len(parents_nodes)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("PDF_DB"))
    ap.add_argument("--domains", nargs="+",
                    default=["immigration", "trading", "ai"],
                    choices=["immigration", "trading", "ai"])
    args = ap.parse_args()
    for d in args.domains:
        ingest_domain(args.root, d)


if __name__ == "__main__":
    main()