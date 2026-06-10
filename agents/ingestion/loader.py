"""
PDF + source-file loader for PRISM-RAG v2.
- PyMuPDF for text + layout.
- For 'ai' domain: detects code blocks via font (monospace) + indentation heuristics.
- Images/figures (all domains): saved to disk, emitted as kind="image" segments.
  The caption (PDF caption + later a VLM caption) becomes the searchable text.
- Returns LoadedDoc with a flat list[Segment].
"""
from __future__ import annotations
import hashlib, os, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import fitz  # PyMuPDF

Domain = Literal["immigration", "trading", "ai"]

MONO_FONT_HINTS = ("Mono", "Courier", "Consolas", "Menlo", "Code", "Fira")
CODE_LINE_HINT = re.compile(
    r"^\s*(def |class |import |from |#include|using namespace|fn |let |const |async |await |return |if |for |while |//|#)"
)

# ── Image handling (all domains) ─────────────────────────────────────────────
IMAGE_DIR      = Path(os.getenv("PRISM_IMAGE_DIR", "./data/_images"))
MIN_IMAGE_AREA = int(os.getenv("PRISM_MIN_IMAGE_AREA", str(64 * 64)))  # px²; skip logos/rules
SCAN_TEXT_MAX  = int(os.getenv("PRISM_SCAN_TEXT_MAX", "100"))          # <this many chars + big image => scanned page
SCAN_AREA_FRAC = float(os.getenv("PRISM_SCAN_AREA_FRAC", "0.5"))       # image must cover >this frac of page
CAPTION_HINT   = re.compile(r"^\s*(fig(?:ure)?\.?|table|chart|scheme|plot|diagram)\b", re.I)


@dataclass
class Segment:
    text: str
    page: int
    kind: Literal["text", "code", "image"] = "text"
    language: str | None = None
    image_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    is_scan: bool = False          # page-image with ~no text layer -> transcribe, not caption


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


# ── image extraction (runs for every domain) ────────────────────────────────
def _save_image(data: bytes, ext: str, sha: str, page_no: int, idx: int) -> Path:
    dst = IMAGE_DIR / sha
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / f"p{page_no:04d}_img{idx:02d}.{ext}"
    out.write_bytes(data)
    return out


def _pair_caption(img_bbox, text_blocks) -> str:
    """Nearest text block starting below the image; prefer 'Figure/Table' leads."""
    _, _, _, iy1 = img_bbox
    best, best_gap = "", 1e9
    for (bx, by, bx1, by1), txt in text_blocks:
        if by < iy1 - 2:                  # caption must begin below the image
            continue
        gap = by - iy1
        if gap < best_gap and (CAPTION_HINT.search(txt) or gap < 40):
            best, best_gap = txt.strip(), gap
    return best


def _extract_images(page: fitz.Page, page_no: int, doc: fitz.Document, sha: str) -> list[Segment]:
    info = page.get_text("dict")
    text_blocks = [
        (b["bbox"], "".join(sp["text"] for ln in b.get("lines", []) for sp in ln["spans"]))
        for b in info["blocks"] if b.get("type") == 0
    ]
    segs: list[Segment] = []
    idx = 0
    for b in info["blocks"]:
        if b.get("type") != 1:            # 1 = image block
            continue
        if b.get("width", 0) * b.get("height", 0) < MIN_IMAGE_AREA:
            continue
        data = b.get("image")
        ext = b.get("ext", "png")
        if not data:
            # Fallback: dict block carried no inline bytes (xobject) -> pull via xref.
            xref = b.get("number") or b.get("xref")
            if not xref:
                continue
            try:
                ei = doc.extract_image(int(xref))
                data, ext = ei["image"], ei["ext"]
            except Exception:
                continue
        idx += 1
        path = _save_image(data, ext, sha, page_no, idx)
        segs.append(
            Segment(
                text=_pair_caption(b["bbox"], text_blocks),  # PDF caption; VLM fills the rest
                page=page_no,
                kind="image",
                image_path=str(path),
                bbox=tuple(round(float(x), 1) for x in b["bbox"]),
            )
        )
    return segs


def load_pdf(path: Path, domain: Domain) -> LoadedDoc:
    doc = fitz.open(path)
    sha = _sha256(path)
    title = (doc.metadata or {}).get("title") or path.stem
    segs: list[Segment] = []
    code_aware = domain == "ai"
    for i, page in enumerate(doc):
        page_no = i + 1
        text_segs = _extract_page(page, page_no=page_no, code_aware=code_aware)
        img_segs = _extract_images(page, page_no=page_no, doc=doc, sha=sha)

        # Scanned/image-only page: little or no text layer + a page-covering image.
        # Tag those images so the VLM transcribes (OCR) them instead of captioning.
        page_text_len = sum(len(s.text.strip()) for s in text_segs)
        if img_segs and page_text_len < SCAN_TEXT_MAX:
            page_area = abs(page.rect.width * page.rect.height) or 1.0
            for s in img_segs:
                if s.bbox:
                    x0, y0, x1, y1 = s.bbox
                    if abs((x1 - x0) * (y1 - y0)) > SCAN_AREA_FRAC * page_area:
                        s.is_scan = True

        segs.extend(text_segs)
        segs.extend(img_segs)
    return LoadedDoc(
        path=path,
        domain=domain,
        title=title,
        n_pages=doc.page_count,
        sha256=sha,
        segments=segs,
    )


def iter_domain_pdfs(root: Path, domain: Domain) -> Iterator[Path]:
    dom_dir = root / domain
    if not dom_dir.exists():
        return
    yield from sorted(dom_dir.rglob("*.pdf"))


# ---------- code/source file support ----------
CODE_EXTENSIONS = {
    ".py": "python", ".pyi": "python",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
    ".rs": "rust",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
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