"""Generate the PRISM-RAG animated data-flow GIF.

Act 1: multimodal ingestion. Act 2: RAPTOR tree build.
Act 3: THREE worked examples (immigration / trading / code), each showing
which agent handles which stage and which AWQ worker answers.
"""
from PIL import Image, ImageDraw, ImageFont
import math

W, H = 1000, 720
BG = (13, 17, 23)
BOX_DONE = (26, 127, 55)
BOX_TODO = (154, 103, 0)
BOX_STORE = (9, 105, 218)
BOX_DIM = (40, 46, 54)
EDGE = (110, 118, 128)
TXT = (240, 246, 252)
DOT_INGEST = (63, 185, 80)
DOT_TREE = (88, 166, 255)
DOMAIN_COL = {"law": (255, 99, 132), "trading": (255, 196, 0), "code": (170, 130, 255)}

F  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
FT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
FQ = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)

boxes = {
    # offline (top)
    "src":    (95, 118, 130, 60, "PDFs + JSONL", "3 domains", BOX_DONE),
    "loader": (265, 118, 150, 60, "Loader", "PyMuPDF: text+code+figs", BOX_DONE),
    "vlm":    (265, 214, 150, 52, "VLM Captioner", "Qwen2.5-VL", BOX_DONE),
    "chunk":  (462, 118, 186, 60, "Parent-Child Chunker", "parents<=1200 / kids 350", BOX_DONE),
    "emb":    (645, 118, 130, 60, "BGE-M3", "1024-d embed", BOX_DONE),
    "db":     (858, 214, 188, 92, "PostgreSQL + pgvector", "parents - children - tree\nHNSW - trgm - tsv", BOX_STORE),
    "tree":   (645, 244, 150, 60, "RAPTOR Tree", "UMAP + BIC-GMM\n+ LLM summaries", BOX_DONE),
    # online (bottom)
    "query":  (82, 520, 116, 56, "User Query", "", BOX_STORE),
    "gw":     (235, 520, 140, 64, "Gateway (R3)", "domain router:\ncosine vs tree roots", BOX_TODO),
    "walk":   (408, 520, 148, 64, "Tree Walk (R0)", "beam=6 descend\ndomain subtree", BOX_TODO),
    "fuse":   (578, 520, 150, 64, "Hybrid RRF (R1)", "dense + lexical", BOX_TODO),
    "dedup":  (578, 408, 150, 52, "Dedup -> Parent", "full paragraph out", BOX_STORE),
    # three domain workers (which agent answers)
    "w_law":  (806, 452, 150, 50, "Law Worker", "AWQ - refusal-trained", BOX_TODO),
    "w_trd":  (806, 520, 150, 50, "Trader Worker", "AWQ - SFT/QLoRA", BOX_TODO),
    "w_cod":  (806, 588, 150, 50, "Coder Worker", "AWQ - GRPO", BOX_TODO),
    "ans":    (908, 668, 124, 48, "Cited Answer", "", BOX_STORE),
}

def pt(k, dx=0, dy=0):
    cx, cy, *_ = boxes[k]
    return (cx + dx, cy + dy)

paths = {
    "ingest": [pt("src", 65), pt("loader", -75), pt("loader", 75), pt("chunk", -93),
               pt("chunk", 93), pt("emb", -65), pt("emb", 65), pt("db", -94, -14)],
    "figs":   [pt("loader", 0, 30), pt("vlm", 0, -26), pt("vlm", 75), (380, 214), pt("chunk", 0, 30)],
    "tree_in":  [pt("db", -94, 24), pt("tree", 75, 0)],
    "tree_out": [pt("tree", 75, 16), pt("db", -94, 38)],
    "q1": [pt("query", 58), pt("gw", -70)],
    "q2": [pt("gw", 70), pt("walk", -74)],
    "q3": [pt("walk", 74), pt("fuse", -75)],
    "db_walk": [pt("db", 0, 46), (858, 348), (470, 348), (470, 470), pt("walk", 30, -32)],
    "q4": [pt("fuse", 0, -32), pt("dedup", 0, 26)],
    # dedup -> worker fan: three parallel elbows, no shared trunk
    "w_law": [pt("dedup", 75, -12), (704, 396), (704, 452), pt("w_law", -75)],
    "w_trd": [pt("dedup", 75,   0), (714, 408), (714, 520), pt("w_trd", -75)],
    "w_cod": [pt("dedup", 75,  12), (724, 420), (724, 588), pt("w_cod", -75)],
    # worker -> answer: staggered verticals into the answer box top
    "a_law": [pt("w_law", 75), (894, 452), (894, 630), pt("ans", -28, -24)],
    "a_trd": [pt("w_trd", 75), (908, 520), (908, 630), pt("ans",   0, -24)],
    "a_cod": [pt("w_cod", 75), (922, 588), (922, 630), pt("ans",  28, -24)],
}

def poly_len(p): return sum(math.dist(p[i], p[i+1]) for i in range(len(p)-1))
def along(p, t):
    d = t * poly_len(p)
    for i in range(len(p)-1):
        seg = math.dist(p[i], p[i+1])
        if d <= seg:
            r = d/seg if seg else 0
            return (p[i][0]+(p[i+1][0]-p[i][0])*r, p[i][1]+(p[i+1][1]-p[i][1])*r)
        d -= seg
    return p[-1]

def draw_box(d, k, glow=0.0, dim=False, glow_col=None):
    cx, cy, w, h, title, sub, col = boxes[k]
    if dim: col = BOX_DIM
    x0, y0, x1, y1 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    if glow > 0:
        gc = glow_col or (255, 255, 255)
        d.rounded_rectangle([x0-4, y0-4, x1+4, y1+4], radius=11, outline=gc, width=3)
    d.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=col, outline=(255,255,255) if not dim else (90,98,106), width=1)
    tcol = TXT if not dim else (130, 138, 146)
    d.text((cx, cy - (10 if sub else 0)), title, font=FB, fill=tcol, anchor="mm")
    if sub:
        for j, line in enumerate(sub.split("\n")):
            d.text((cx, cy + 8 + j*12), line, font=FS, fill=(215,222,230) if not dim else (110,118,126), anchor="mm")

def draw_edge(d, p, active=False, col=None):
    c = col if (active and col) else ((200,210,220) if active else EDGE)
    d.line(p, fill=c, width=3 if active else 2)
    (x0,y0),(x1,y1) = p[-2], p[-1]
    a = math.atan2(y1-y0, x1-x0); L=9
    d.polygon([(x1,y1),(x1-L*math.cos(a-0.45),y1-L*math.sin(a-0.45)),
               (x1-L*math.cos(a+0.45),y1-L*math.sin(a+0.45))], fill=c)

def dot(d, xy, col, r=7):
    x,y = xy
    d.ellipse([x-r-3,y-r-3,x+r+3,y+r+3], fill=col)
    d.ellipse([x-r,y-r,x+r,y+r], fill=col, outline=(255,255,255), width=2)


def label(d, xy, text, col):
    x, y = xy
    w = d.textlength(text, font=FS)
    d.rounded_rectangle([x - w/2 - 6, y - 10, x + w/2 + 6, y + 10],
                        radius=6, fill=(22, 28, 36), outline=col, width=1)
    d.text((x, y), text, font=FS, fill=col, anchor="mm")

# ── Act 3 worked examples ────────────────────────────────────────────────────
# stages: 0 query 1 gateway 2 walk 3 fuse 4 dedup 5 worker 6 answer
EXAMPLES = [
    dict(
        dom="law", worker="w_law", ansedge="a_law",
        query='"What are the F-2-7-7 visa renewal requirements?"',
        stages=[
            "User asks an exact-identifier legal question",
            "Gateway router: query embedding vs tree roots -> domain = IMMIGRATION (law subtree only)",
            "Tree Walk agent: beam=6 descends the immigration tree; cluster summaries act as topic filter",
            "Hybrid RRF: dense misses rare code 'F-2-7-7' -> LEXICAL (trgm) hit rescues it; ranks fused",
            "Dedup -> Parent: matched child shards mapped to FULL regulation clauses (parents)",
            "LAW worker (AWQ): answers with [chunk_id] citations - REFUSES if answer not in context",
            "Cited answer returned: clauses + tree path + worker_used = law",
        ],
    ),
    dict(
        dom="trading", worker="w_trd", ansedge="a_trd",
        query='"How did AAPL react after the Q3 earnings call?"',
        stages=[
            "User asks a finance question involving a ticker + a price chart",
            "Gateway router: cosine-argmax -> domain = TRADING (trading subtree only)",
            "Tree Walk agent: descends trading tree to earnings-news clusters",
            "Hybrid RRF: lexical matches 'AAPL'; dense matches 'earnings reaction'; RRF fuses both",
            "Dedup -> Parent: returns full news paragraphs + a VLM-captioned PRICE CHART (image_path)",
            "TRADER worker (AWQ, SFT on Sujet-Finance-177k): quantitative answer from chunks + chart caption",
            "Cited answer: numbers + chart reference + worker_used = trader",
        ],
    ),
    dict(
        dom="code", worker="w_cod", ansedge="a_cod",
        query='"Show Python code for soft GMM cluster assignment"',
        stages=[
            "User asks a coding question against the AI-papers corpus",
            "Gateway router: cosine-argmax -> domain = AI; code intent -> CODER worker selected",
            "Tree Walk agent: descends AI tree to clustering-methods clusters (code blocks are chunks too)",
            "Hybrid RRF: dense finds GMM method paragraphs; lexical pins 'GaussianMixture' identifiers",
            "Dedup -> Parent: full code blocks returned (each function/class = one parent)",
            "CODER worker (AWQ, GRPO: format+compile+unit-test rewards): emits <reasoning> then <code>",
            "Cited answer: runnable code + sources + worker_used = coder",
        ],
    ),
]
WORKER_KEYS = ["w_law", "w_trd", "w_cod"]
Q_SEGS = ["q1", "q2", "q3", "q4"]          # + worker edge + answer edge per example

ACT_FRAMES = [44, 32]            # act1 ingest, act2 tree
EX_FRAMES = 84                   # per example in act 3
N = sum(ACT_FRAMES) + 3 * EX_FRAMES

frames = []
for f in range(N):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # which act / example
    if f < ACT_FRAMES[0]:
        act, t, ex = 0, f/(ACT_FRAMES[0]-1), None
    elif f < ACT_FRAMES[0]+ACT_FRAMES[1]:
        act, t, ex = 1, (f-ACT_FRAMES[0])/(ACT_FRAMES[1]-1), None
    else:
        g = f - ACT_FRAMES[0] - ACT_FRAMES[1]
        ei, rem = divmod(g, EX_FRAMES)
        act, t, ex = 2, rem/(EX_FRAMES-1), EXAMPLES[ei]

    # ── headers ──
    d.text((20, 16), "PRISM-RAG  -  end-to-end data flow", font=FT, fill=TXT)
    if act == 0:
        d.text((20, 42), "ACT 1  -  multimodal ingestion: PDF -> chunks -> embeddings -> DB", font=F, fill=(155,225,255))
    elif act == 1:
        d.text((20, 42), "ACT 2  -  RAPTOR tree build: cluster -> summarize -> re-embed", font=F, fill=(155,225,255))
    else:
        ei = EXAMPLES.index(ex)
        dc = DOMAIN_COL[ex["dom"]]
        d.text((20, 42), f"ACT 3  -  QUERY EXAMPLE {ei+1}/3  -  domain: {ex['dom'].upper()}", font=F, fill=dc)
        d.text((20, 64), ex["query"], font=FQ, fill=TXT)

    # ── stage progress for act 3 ──
    stage = None
    if act == 2:
        SW = [1, 1, 3, 2, 1, 1, 1]           # stage weights: DB->tree-walk slowest
        TOT = sum(SW)
        acc, stage, su = 0.0, 6, 1.0
        for si, w in enumerate(SW):
            if t * TOT < acc + w:
                stage = si
                su = (t * TOT - acc) / w      # 0..1 progress within this stage
                break
            acc += w
        # stage caption banner (bottom)
        d.rounded_rectangle([14, H-58, W-14, H-22], radius=8, fill=(22,28,36), outline=DOMAIN_COL[ex["dom"]], width=2)
        d.text((28, H-40), f"[{stage+1}/7]  " + ex["stages"][stage], font=F, fill=TXT, anchor="lm")
        # stage tick marks
        for s in range(7):
            col = DOMAIN_COL[ex["dom"]] if s <= stage else (60,68,76)
            d.ellipse([28+s*16, H-70, 38+s*16, H-60], fill=col)
    else:
        d.text((20, H-40), "green = built   amber = R-phase planned   blue = data", font=FS, fill=(139,148,158))

    # ── edges ──
    for name, p in paths.items():
        if name in ("w_law","w_trd","w_cod","a_law","a_trd","a_cod"):
            if act == 2:
                mine = (name == ex["worker"]) or (name == ex["ansedge"])
                draw_edge(d, p, active=mine and stage >= 4, col=DOMAIN_COL[ex["dom"]] if mine else None)
            else:
                draw_edge(d, p, active=False)
        elif name in ("ingest","figs"):
            draw_edge(d, p, active=(act==0))
        elif name.startswith("tree"):
            draw_edge(d, p, active=(act==1))
        elif name == "db_walk":
            draw_edge(d, p, active=(act==2 and stage in (2,3)), col=DOT_TREE)
        else:  # q1..q4
            draw_edge(d, p, active=(act==2))

    # ── boxes ──
    if act == 2:
        dc = DOMAIN_COL[ex["dom"]]
        stage_box = [["query"],["gw"],["walk"],["fuse"],["dedup"],[ex["worker"]],["ans"]][stage]
        for k in boxes:
            if k in WORKER_KEYS:
                if k == ex["worker"]:
                    draw_box(d, k, glow=1.0 if stage>=5 else 0.0, glow_col=dc)
                else:
                    draw_box(d, k, dim=True)          # non-selected workers dimmed
            else:
                draw_box(d, k, glow=1.0 if k in stage_box else 0.0, glow_col=dc)
    else:
        glow_map = {}
        if act == 0:
            seq = ["src","loader","vlm","chunk","emb","db"]
            glow_map[seq[min(int(t*len(seq)), len(seq)-1)]] = 1.0
        else:
            glow_map["db" if t < 0.33 or t > 0.85 else "tree"] = 1.0
        for k in boxes:
            draw_box(d, k, glow=glow_map.get(k, 0.0))

    # ── packets ──
    if act == 0:
        dot(d, along(paths["ingest"], t), DOT_INGEST)
        dot(d, along(paths["figs"], (t*1.4) % 1.0), (180,120,255), r=5)
    elif act == 1:
        if t < 0.5: dot(d, along(paths["tree_in"], t/0.5), DOT_TREE)
        else:       dot(d, along(paths["tree_out"], (t-0.5)/0.5), DOT_TREE)
        cx, cy, *_ = boxes["tree"]
        r = 38 + 6*math.sin(t*math.tau*2)
        d.ellipse([cx-r, cy-r, cx+r, cy+r], outline=DOT_TREE, width=2)
    else:
        dc = DOMAIN_COL[ex["dom"]]
        # packet synced to stage clock: stage 0 dwells at query box,
        # stage s>=1 traverses edge s-1 over that stage's window
        edges = ["q1", "q2", "q3", "q4", ex["worker"], ex["ansedge"]]
        if stage == 0:
            dot(d, pt("query"), dc)
        else:
            dot(d, along(paths[edges[stage - 1]], min(max(su, 0.0), 1.0)), dc)
        # DB -> Tree Walk -> RRF data stream, made explicit:
        if stage == 2:
            # stream of blue packets: DB sends child shards + summaries to the walk
            u = min(max(su, 0.0), 1.0)
            for off in (0.0, 0.18, 0.36):
                uu = u - off
                if 0.0 <= uu <= 1.0:
                    dot(d, along(paths["db_walk"], uu), DOT_TREE, r=5)
            label(d, (470, 332), "DB sends: child shards + tree summaries", DOT_TREE)
            label(d, (408, 462), "walk keeps top-beam clusters -> candidate leaves", DOT_TREE)
        elif stage == 3:
            # candidate set flows walk -> RRF alongside the query
            u = min(max(su, 0.0), 1.0)
            for off in (0.0, 0.22):
                uu = u - off
                if 0.0 <= uu <= 1.0:
                    dot(d, along(paths["q3"], uu), DOT_TREE, r=5)
            label(d, (493, 458), "candidate leaf set ->", DOT_TREE)
            label(d, (578, 580), "RRF fuses dense + lexical ranks of candidates", DOT_TREE)
        elif stage == 4:
            label(d, (607, 462), "top-k children -> their FULL parents", (180, 220, 255))

    frames.append(img)

frames[0].save("/mnt/user-data/outputs/prism_dataflow.gif",
               save_all=True, append_images=frames[1:],
               duration=150, loop=0, optimize=True)
print("frames:", N, " loop:", N*150/1000, "s")
