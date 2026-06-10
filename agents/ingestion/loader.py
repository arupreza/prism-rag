"""
PDF loader for PRISM-RAG v2.
- PyMuPDF for text + layout.
- For 'ai' domain: detects code blocks via font (monospace) + indentation heuristics.
- Returns List[Page] with per-page (text_segments, code_segments).
"""
from __future__ import annotations
import hashlib, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import fitz  # PyMuPDF

Domain = Literal["immigration", "trading", "ai"]

MONO_FONT_HINTS = ("Mono", "Courier", "Consolas", "Menlo", "Code", "Fira")
CODE_LINE_HINT = re.compile(
    r"^\s*(def |class |import |from |#include|using namespace|fn |let |const |async |await |return |if |for |while |//|#)"
)


@dataclass
class Segment:
    text: str
    page: int
    kind: Literal["text", "code"] = "text"
    language: str | None = None


@dataclass
class LoadedDoc:
    path: Path
    domain: Domain
    title: str
    n_pages: int
    sha256: str
    segments: list[Segment] = field(default_factory=list)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _is_code_span(span: dict) -> bool:
    font = span.get("font", "")
    return any(h in font for h in MONO_FONT_HINTS)


def _detect_language(code: str) -> str | None:
    s = code[:500]
    if re.search(r"\bdef\s+\w+\(|\bimport\s+\w+|^from\s+\w+\s+import", s, re.M):
        return "python"
    if re.search(r"#include\s*<|::|std::", s):
        return "cpp"
    if re.search(r"\bfn\s+\w+\(|let\s+mut\b|impl\s+\w+", s):
        return "rust"
    if re.search(r"\bfunction\s+\w+\(|=>\s*{|const\s+\w+\s*=", s):
        return "javascript"
    return None


def _extract_page(page: fitz.Page, page_no: int, code_aware: bool) -> list[Segment]:
    if not code_aware:
        txt = page.get_text("text").strip()
        return [Segment(text=txt, page=page_no, kind="text")] if txt else []

    # block-level extraction with font info
    blocks = page.get_text("dict")["blocks"]
    segs: list[Segment] = []
    buf_text: list[str] = []

    def flush_text():
        if buf_text:
            joined = "\n".join(buf_text).strip()
            if joined:
                segs.append(Segment(text=joined, page=page_no, kind="text"))
            buf_text.clear()

    for blk in blocks:
        if blk.get("type") != 0:  # not text block
            continue
        lines = blk.get("lines", [])
        # decide if block is predominantly monospace -> code
        mono_chars = total_chars = 0
        block_text_lines = []
        for ln in lines:
            line_str = "".join(sp["text"] for sp in ln["spans"])
            block_text_lines.append(line_str)
            for sp in ln["spans"]:
                n = len(sp["text"])
                total_chars += n
                if _is_code_span(sp):
                    mono_chars += n
        block_text = "\n".join(block_text_lines).rstrip()
        if not block_text.strip():
            continue

        mono_ratio = mono_chars / max(total_chars, 1)
        code_hint = bool(CODE_LINE_HINT.search(block_text))
        is_code = mono_ratio > 0.6 or (mono_ratio > 0.3 and code_hint)

        if is_code:
            flush_text()
            segs.append(
                Segment(
                    text=block_text,
                    page=page_no,
                    kind="code",
                    language=_detect_language(block_text),
                )
            )
        else:
            buf_text.append(block_text)
    flush_text()
    return segs


def load_pdf(path: Path, domain: Domain) -> LoadedDoc:
    doc = fitz.open(path)
    title = (doc.metadata or {}).get("title") or path.stem
    segs: list[Segment] = []
    code_aware = domain == "ai"
    for i, page in enumerate(doc):
        segs.extend(_extract_page(page, page_no=i + 1, code_aware=code_aware))
    return LoadedDoc(
        path=path,
        domain=domain,
        title=title,
        n_pages=doc.page_count,
        sha256=_sha256(path),
        segments=segs,
    )


def iter_domain_pdfs(root: Path, domain: Domain) -> Iterator[Path]:
    dom_dir = root / domain
    if not dom_dir.exists():
        return
    yield from sorted(dom_dir.rglob("*.pdf"))

# ---------- code/source file support ----------

CODE_EXTENSIONS = {
    ".py":   "python",
    ".pyi":  "python",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".cxx":  "cpp",
    ".h":    "cpp",
    ".hpp":  "cpp",
    ".rs":   "rust",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".jsx":  "javascript",
    ".tsx":  "typescript",
}

SKIP_DIRS = {
    "__pycache__", ".git", ".github", "node_modules", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "output", "testing", "tests",
}


def load_source_file(path: Path, domain: Domain) -> LoadedDoc:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return LoadedDoc(path=path, domain=domain, title=path.stem,
                         n_pages=0, sha256=_sha256(path), segments=[])
    language = CODE_EXTENSIONS.get(path.suffix.lower())
    seg = Segment(text=text, page=1, kind="code", language=language)
    return LoadedDoc(path=path, domain=domain, title=path.stem,
                     n_pages=1, sha256=_sha256(path), segments=[seg])


def _should_skip(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)


def iter_domain_sources(root: Path, domain: Domain) -> Iterator[Path]:
    dom_dir = root / domain
    if not dom_dir.exists():
        return
    files: list[Path] = []
    for p in dom_dir.rglob("*"):
        if not p.is_file() or _should_skip(p):
            continue
        ext = p.suffix.lower()
        if ext == ".pdf":
            files.append(p)
        elif domain == "ai" and ext in CODE_EXTENSIONS:
            files.append(p)
    yield from sorted(files)


def load_any(path: Path, domain: Domain) -> LoadedDoc:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path, domain)
    return load_source_file(path, domain)
