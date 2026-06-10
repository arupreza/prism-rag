"""
Paragraph-aware chunker with parent/child split.

Strategy:
    1. Split each Segment into paragraphs (blank-line split, merge short orphans).
    2. paragraph <= MAX_PARENT_TOKENS  -> ONE parent row, ONE child row (identical content).
    3. paragraph >  MAX_PARENT_TOKENS  -> ONE parent row + N child shards (token windows).
    4. Code segments: each function/class block is a parent. Long bodies -> shard children.
    5. Image segments: the caption/explanation is the parent+child content (single shard);
       the parent/child carry image_path so retrieval can display the figure.

Returns parents list, where each parent carries its own children list.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Literal

import tiktoken
from .loader import Segment

_ENC = tiktoken.get_encoding("cl100k_base")

# Tunables
MAX_PARENT_TOKENS = 1200     # if paragraph is bigger, shard it
CHILD_TOKENS      = 350      # embedding-optimal window
CHILD_OVERLAP     = 60
MIN_PARA_TOKENS   = 40       # merge tiny paragraphs into next one

CODE_BOUNDARY = re.compile(
    r"(?m)^(?=\s*(?:def |async def |class |fn |pub fn |function |#include|namespace ))"
)
PARA_SPLIT = re.compile(r"\n\s*\n+")   # blank-line paragraph boundary

ContentType = Literal["text", "code", "image"]


@dataclass
class ChildChunk:
    content: str
    token_count: int
    page_start: int
    page_end: int
    content_type: ContentType
    language: str | None
    image_path: str | None = None


@dataclass
class ParentChunk:
    content: str
    token_count: int
    page_start: int
    page_end: int
    content_type: ContentType
    language: str | None
    image_path: str | None = None
    children: list[ChildChunk] = field(default_factory=list)


def _tok_len(s: str) -> int:
    return len(_ENC.encode(s, disallowed_special=()))


# ---------- paragraph splitting (text) ----------
def _paragraphs(text: str) -> list[str]:
    raw = [p.strip() for p in PARA_SPLIT.split(text) if p.strip()]
    if not raw:
        return []
    merged: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for p in raw:
        tl = _tok_len(p)
        if tl < MIN_PARA_TOKENS:
            buf.append(p)
            buf_tok += tl
            continue
        if buf:
            merged.append("\n\n".join(buf + [p]))
            buf, buf_tok = [], 0
        else:
            merged.append(p)
    if buf:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + "\n\n".join(buf)
        else:
            merged.append("\n\n".join(buf))
    return merged


def _shard_children(text: str, parent: ParentChunk) -> list[ChildChunk]:
    """Token-window shard of an oversized parent into embedding-sized children."""
    toks = _ENC.encode(text, disallowed_special=())
    out: list[ChildChunk] = []
    step = CHILD_TOKENS - CHILD_OVERLAP
    for start in range(0, len(toks), step):
        win = toks[start : start + CHILD_TOKENS]
        if not win:
            break
        out.append(
            ChildChunk(
                content=_ENC.decode(win),
                token_count=len(win),
                page_start=parent.page_start,
                page_end=parent.page_end,
                content_type=parent.content_type,
                language=parent.language,
                image_path=parent.image_path,
            )
        )
        if start + CHILD_TOKENS >= len(toks):
            break
    return out


def _text_segment_to_parents(seg: Segment) -> list[ParentChunk]:
    parents: list[ParentChunk] = []
    for para in _paragraphs(seg.text):
        tl = _tok_len(para)
        parent = ParentChunk(
            content=para, token_count=tl,
            page_start=seg.page, page_end=seg.page,
            content_type="text", language=None,
            image_path=seg.image_path,        # set for transcribed scans; None otherwise
        )
        if tl <= MAX_PARENT_TOKENS:
            parent.children = [
                ChildChunk(content=para, token_count=tl,
                           page_start=seg.page, page_end=seg.page,
                           content_type="text", language=None,
                           image_path=seg.image_path)
            ]
        else:
            parent.children = _shard_children(para, parent)
        parents.append(parent)
    return parents


# ---------- code splitting ----------
def _code_segment_to_parents(seg: Segment) -> list[ParentChunk]:
    parts = [p for p in CODE_BOUNDARY.split(seg.text) if p.strip()]
    if not parts:
        parts = [seg.text]

    parents: list[ParentChunk] = []
    for block in parts:
        tl = _tok_len(block)
        parent = ParentChunk(
            content=block.rstrip(), token_count=tl,
            page_start=seg.page, page_end=seg.page,
            content_type="code", language=seg.language,
        )
        if tl <= MAX_PARENT_TOKENS:
            parent.children = [
                ChildChunk(content=block.rstrip(), token_count=tl,
                           page_start=seg.page, page_end=seg.page,
                           content_type="code", language=seg.language)
            ]
        else:
            parent.children = _shard_children(block, parent)
        parents.append(parent)
    return parents


# ---------- image (caption) ----------
def _image_segment_to_parents(seg: Segment) -> list[ParentChunk]:
    cap = seg.text.strip()
    if not cap:                          # no caption -> nothing searchable; drop
        return []
    tl = _tok_len(cap)                   # captions are short -> single shard
    parent = ParentChunk(
        content=cap, token_count=tl,
        page_start=seg.page, page_end=seg.page,
        content_type="image", language=None,
        image_path=seg.image_path,
    )
    parent.children = [
        ChildChunk(content=cap, token_count=tl,
                   page_start=seg.page, page_end=seg.page,
                   content_type="image", language=None,
                   image_path=seg.image_path)
    ]
    return [parent]


# ---------- public API ----------
def chunk_segments(segments: list[Segment]) -> list[ParentChunk]:
    out: list[ParentChunk] = []
    for seg in segments:
        if seg.kind == "code":
            out.extend(_code_segment_to_parents(seg))
        elif seg.kind == "image":
            out.extend(_image_segment_to_parents(seg))
        else:
            out.extend(_text_segment_to_parents(seg))
    return out