"""Generate the paper's draw.io figures (editable .drawio sources).

Run:  python3 gen_figures.py      then export with export.sh.
Canvases are about 800 px wide so that, scaled to the ACM text width, 16 px text
prints at about 8 pt.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

FONT = "fontFamily=Linux Libertine O;"
BLUE = "fillColor=#dae8fc;strokeColor=#6c8ebf;"
ORANGE = "fillColor=#ffe6cc;strokeColor=#d79b00;"
GREEN = "fillColor=#d5e8d4;strokeColor=#82b366;"
RED = "fillColor=#f8cecc;strokeColor=#b85450;"
YELLOW = "fillColor=#fff2cc;strokeColor=#d6b656;"
GREY = "fillColor=#f5f5f5;strokeColor=#666666;"
PURPLE = "fillColor=#e1d5e7;strokeColor=#9673a6;"
WHITE = "fillColor=#ffffff;strokeColor=#666666;"


class Diagram:
    def __init__(self) -> None:
        self.cells: list[str] = []

    def box(self, cid: str, label: str, x: float, y: float, w: float, h: float,
            style: str = "", parent: str = "1", size: int = 16) -> str:
        st = f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;{FONT}fontSize={size};{style}"
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" '
            f'vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" '
            f'height="{h}" as="geometry"/></mxCell>')
        return cid

    def text(self, cid: str, label: str, x: float, y: float, w: float, h: float,
             style: str = "", parent: str = "1", size: int = 16) -> str:
        return self.box(cid, label, x, y, w, h,
                        "text;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;"
                        + style, parent, size)

    def lane(self, cid: str, label: str, x: float, y: float, w: float, h: float,
             style: str = "", size: int = 16) -> str:
        st = (f"swimlane;startSize=30;rounded=1;arcSize=4;html=1;{FONT}fontSize={size};"
              f"fontStyle=1;container=1;pointerEvents=0;collapsible=0;{style}")
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def diamond(self, cid: str, label: str, x: float, y: float, w: float, h: float,
                style: str = YELLOW, size: int = 15) -> str:
        st = f"rhombus;whiteSpace=wrap;html=1;{FONT}fontSize={size};{style}"
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" '
            f'as="geometry"/></mxCell>')
        return cid

    def edge(self, cid: str, src: str, tgt: str, label: str = "", style: str = "",
             points: list[tuple[float, float]] | None = None, size: int = 14,
             offset: tuple[float, float] | None = None) -> str:
        st = ("edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
              f"endArrow=block;endSize=8;strokeWidth=1.5;{FONT}fontSize={size};{style}")
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(
                f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
        off = f'<mxPoint as="offset" x="{offset[0]}" y="{offset[1]}"/>' if offset else ""
        self.cells.append(
            f'<mxCell id="{cid}" value="{escape(label, {chr(34): "&quot;"})}" style="{st}" '
            f'edge="1" parent="1" source="{src}" target="{tgt}">'
            f'<mxGeometry relative="1" as="geometry">{pts}{off}</mxGeometry></mxCell>')
        return cid

    def write(self, path: str) -> None:
        body = "\n".join(self.cells)
        with open(path, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<mxfile host="drawio">'
                    '<diagram name="Page-1"><mxGraphModel><root>'
                    '<mxCell id="0"/><mxCell id="1" parent="0"/>\n'
                    f"{body}\n</root></mxGraphModel></diagram></mxfile>\n")


def sub(s: str) -> str:
    return f"<sub>{s}</sub>"


# --------------------------------------------------------------------------------------
def concept() -> None:
    """Introduction: the idea of the paper on the R-SO cell."""
    d = Diagram()
    # Specification
    d.lane("spec", "Specification 𝒮", 0, 30, 170, 270, BLUE.replace("dae8fc", "f0f5fc"))
    d.box("tell", "<b>Tell 𝒯</b><br>requirements, policies,<br>knowledge base", 10, 40, 150, 72,
          BLUE, "spec", 15)
    d.box("check", "<b>Check 𝒞</b><br>invariants, golden<br>state, guardrails", 10, 124, 150, 72,
          ORANGE, "spec", 15)
    d.box("pi", "<b>Scenarios Π</b>", 10, 208, 150, 46, GREY, "spec", 15)
    # Population
    d.lane("pop", "LLM interpreters", 210, 30, 180, 270, GREY.replace("f5f5f5", "fafafa"))
    for i, name in enumerate(["model 1 · family A", "model 2 · family A",
                              "model 3 · family B", "model 4 · family C"]):
        d.box(f"m{i}", name, 10, 40 + i * 46, 160, 36, WHITE, "pop", 15)
    d.text("popn", "× 3 Tell variants × 3 seeds,<br>local open-weight models", 0, 222, 180, 40,
           "", "pop", 14)
    # Readings of one cell
    d.lane("cell", "Readings of one cell <i>c</i>", 430, 30, 230, 270,
           YELLOW.replace("fff2cc", "fffbea"))
    d.text("anch_l", "anchor <i>d</i>" + sub("c"), 0, 38, 80, 24, "align=right;", "cell", 15)
    d.box("anch", "", 92, 41, 18, 18, ORANGE + "rounded=0;", "cell")
    pattern = ["GOXOXG", "GOGXGO", "GGOGGG", "GGGGGG"]
    colours = {"G": GREEN, "O": ORANGE, "X": GREY}
    for r, row in enumerate(pattern):
        y = 94 + r * 30
        d.text(f"rl{r}", f"model {r + 1}", 0, y - 3, 80, 24, "align=right;", "cell", 15)
        for k, ch in enumerate(row):
            d.box(f"c{r}{k}", "", 92 + k * 22, y, 18, 18, colours[ch] + "rounded=0;", "cell")
    d.text("acc", "rows: models, by increasing accuracy", 0, 62, 230, 22, "", "cell", 13)
    d.box("lg1", "", 14, 222, 14, 14, GREEN + "rounded=0;", "cell")
    d.text("lg1t", "Đã duyệt", 30, 216, 70, 24, "align=left;", "cell", 14)
    d.box("lg2", "", 104, 222, 14, 14, ORANGE + "rounded=0;", "cell")
    d.text("lg2t", "Chờ duyệt", 120, 216, 80, 24, "align=left;", "cell", 14)
    d.box("lg3", "", 14, 244, 14, 14, GREY + "rounded=0;", "cell")
    d.text("lg3t", "other readings", 30, 238, 120, 24, "align=left;", "cell", 14)
    # Model and verdict
    d.box("acm", "<b>Anchored cell model</b><br>P(<i>A</i>" + sub("c") + " | readings)", 690, 40,
          130, 70, PURPLE)
    d.box("ref", "<b>Refutation</b><br>witness state", 690, 135, 130, 56, PURPLE)
    d.box("out", "<b>Hidden requirement</b><br>at SO-0031 status<br>+ witnesses", 690, 216, 130,
          84, RED)
    d.edge("e1", "tell", "pop", "reads", "exitX=1;exitY=0.5;entryX=0;entryY=0.2815;", size=13)
    d.edge("e2", "pi", "pop", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.8556;")
    d.edge("e3", "pop", "cell", "runs", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;", size=13)
    d.edge("e4", "check", "anch", "𝒞 declares the anchor:<br>one more fallible rater",
           "dashed=1;exitX=1;exitY=0.5;entryX=0.5;entryY=0;strokeColor=#d79b00;",
           points=[(195, 193), (195, 12), (531, 12)], size=13, offset=(0, -12))
    d.edge("e5", "cell", "acm", "", "exitX=1;exitY=0.3;entryX=0;entryY=0.5;")
    d.edge("e6", "acm", "ref", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("e7", "ref", "out", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    # Evidence patterns
    d.text("pt", "<b>Evidence pattern in the runs → defect type</b>", 0, 318, 820, 26, "", "1", 16)
    pats = [
        ("Hidden requirement", RED,
         "Readers that are accurate elsewhere agree with each other, not with 𝒞"),
        ("Ambiguity", YELLOW, "Readers split into stable camps that persist with accuracy"),
        ("Weak check", GREEN, "A run the population rejects still passes every check"),
        ("Contradiction", PURPLE, "No run and no repair satisfies a set of clauses jointly"),
    ]
    for i, (title, col, body) in enumerate(pats):
        d.box(f"p{i}", f"<b>{title}</b><br>{body}", i * 207, 350, 197, 92, col, "1", 15)
    d.write("concept.drawio")


# --------------------------------------------------------------------------------------
def pipeline() -> None:
    """Figure: overview of SpecDx (redrawn from the TikZ version)."""
    d = Diagram()
    d.box("spec", "<b>Specification</b><br>𝒮 = ⟨𝒯, 𝒞, Π⟩", 0, 175, 100, 90, BLUE)
    d.lane("L1", "① Explore", 125, 50, 190, 340, BLUE.replace("dae8fc", "f0f5fc"))
    d.box("scen", "<b>Scenario choice</b><br>reference → T-boundary<br>→ Thompson sampling",
          12, 45, 166, 100, WHITE, "L1", 15)
    d.box("exec", "<b>Execution</b><br>pilot: 1 model per family;<br>escalate to models ×<br>variants × seeds",
          12, 190, 166, 120, WHITE, "L1", 15)
    d.lane("L2", "② Infer", 340, 50, 200, 340, YELLOW.replace("fff2cc", "fffbea"))
    d.box("obs", "<b>Observation</b><br>items, cells, anchors <i>d</i>" + sub("c") +
          ",<br>violation sets <i>V</i>(<i>b</i>)", 12, 40, 176, 82, WHITE, "L2", 15)
    d.box("act", "<b>Stage 1: act</b><br>Rasch σ(η" + sub("m") + " − δ" + sub("s") +
          ")<br>observability <i>O</i>" + sub("s"), 12, 140, 176, 76, WHITE, "L2", 15)
    d.box("read", "<b>Stage 2: read</b><br>anchored cell model<br>P(<i>A</i>" + sub("c") +
          " | <i>Y</i>" + sub("c") + "), θ" + sub("m") + ", <i>w</i>" + sub("b"),
          12, 234, 176, 90, WHITE, "L2", 15)
    d.lane("L3", "③ Diagnose", 565, 50, 200, 340, PURPLE.replace("e1d5e7", "f6f0f9"))
    d.box("typ", "<b>Typing</b><br>thresholds τ" + sub("H") + ", τ" + sub("A") + ", τ" +
          sub("W") + "<br>soft hitting sets", 12, 40, 176, 82, WHITE, "L3", 15)
    d.box("ref", "<b>Refutation</b><br>symbolic repair,<br>directed probe", 12, 140, 176, 76,
          WHITE, "L3", 15)
    d.box("loc", "<b>Localisation</b><br>ddmin over<br>sentences of 𝒯", 12, 234, 176, 90,
          WHITE, "L3", 15)
    d.box("out", "<b>Report</b><br>typed defects,<br>locations,<br>witnesses,<br>posteriors",
          790, 165, 100, 110, RED)
    d.edge("a1", "spec", "scen", "", "exitX=0.5;exitY=0;entryX=0;entryY=0.5;")
    d.edge("a2", "scen", "exec", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("a3", "exec", "obs", "runs", "exitX=1;exitY=0.3;entryX=0;entryY=0.5;", size=13)
    d.edge("a4", "obs", "act", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("a5", "act", "read", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("a6", "read", "typ", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
           points=[(552, 329), (552, 131)])
    d.edge("a7", "typ", "ref", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("a8", "ref", "loc", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("a9", "typ", "out", "", "exitX=1;exitY=0.5;entryX=0.5;entryY=0;")
    d.edge("a10", "loc", "out", "", "exitX=1;exitY=0.5;entryX=0.5;entryY=1;")
    d.edge("f1", "L3", "L1", "reward: Σ KL change of the defect scores",
           "dashed=1;strokeColor=#9673a6;exitX=0.5;exitY=0;entryX=0.5;entryY=0;",
           points=[(665, 20), (220, 20)], size=13)
    d.edge("f2", "L3", "L1", "re-runs: probes (budget 100), ddmin tests (budget 80)",
           "dashed=1;strokeColor=#9673a6;exitX=0.5;exitY=1;entryX=0.5;entryY=1;",
           points=[(665, 420), (220, 420)], size=13)
    d.write("pipeline.drawio")


# --------------------------------------------------------------------------------------
def motivating() -> None:
    """Motivating example: the CASE-0031 flow with the two real defects."""
    d = Diagram()
    d.box("tband", "<b>Tell 𝒯</b>&nbsp; tells the agents what to do: BA document (97 pp.), demo "
          "description (46 pp.), SAD (18 pp.), knowledge base, principles",
          0, 0, 820, 40, BLUE, size=15)
    steps = ["<b>Sales order</b><br>SO-0031", "<b>Purchase order</b><br>PO-0015, 15 units",
             "<b>Approval</b><br>APR-0102<br>(human gate)", "<b>Goods receipt</b><br>12 units",
             "<b>Three-way match</b><br>fn_reconcile", "<b>Payment</b>"]
    w, gap = 125, 14
    for i, s in enumerate(steps):
        d.box(f"s{i}", s, i * (w + gap), 70, w, 70, WHITE, size=15)
        if i:
            d.edge(f"f{i}", f"s{i - 1}", f"s{i}", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.box("cband", "<b>Check 𝒞</b>&nbsp; judges the end state: 42 invariants, golden state over 26 "
          "tables, guardrail triggers, scope table", 0, 170, 820, 40, ORANGE, size=15)
    d.box("rso", "<b>R-SO: hidden requirement</b><br>Golden state after the approval:<br>"
          "PO-0015 → <i>Đã duyệt</i> (approved)<br>SO-0031 stays <i>Chờ duyệt</i> (awaiting)"
          "<br>No sentence of 𝒯 asks for this; an agent that<br>also approves SO-0031 is scored "
          "as failing.", 0, 250, 390, 140, RED, size=15)
    d.box("r12", "<b>R-12: contradiction</b><br>The schema allows several receipts per order;"
          "<br><i>fn_reconcile</i> raises CardinalityViolation on two.<br>The invariant "
          "doi_soat = fn_reconcile(po) then<br>has no satisfying state. Invisible in the three"
          "<br>reference scenarios, which have one receipt.", 430, 250, 390, 140, PURPLE,
          size=15)
    d.edge("x1", "rso", "s2", "", "dashed=1;strokeColor=#b85450;exitX=0.8731;exitY=0;"
           "entryX=0.5;entryY=1;edgeStyle=none;")
    d.edge("x2", "r12", "s3", "", "dashed=1;strokeColor=#9673a6;exitX=0.1269;exitY=0;"
           "entryX=0.5;entryY=1;edgeStyle=none;")
    d.edge("x3", "r12", "s4", "", "dashed=1;strokeColor=#9673a6;exitX=0.4833;exitY=0;"
           "entryX=0.5;entryY=1;edgeStyle=none;")
    d.write("motivating.drawio")


# --------------------------------------------------------------------------------------
def typing() -> None:
    """Typing: from posteriors and violation sets to the four defect types."""
    d = Diagram()
    d.box("post", "<b>Cell posteriors</b> P(<i>A</i>" + sub("c") + " | <i>Y</i>" + sub("c") +
          ", Θ̂) and behaviour weights <i>w</i>" + sub("b"), 0, 0, 600, 46, YELLOW)
    d.box("vs", "<b>Violation sets</b> {<i>V</i>(<i>b</i>)}, weighted by <i>w</i>" + sub("b"),
          630, 0, 200, 46, YELLOW)
    # Column 1: hidden requirement / contradiction of size one
    d.diamond("d1", "P(<i>A</i>" + sub("c") + " ≠ {<i>d</i>" + sub("c") + "}) ≥ τ" + sub("H") +
              "?", 30, 80, 180, 90)
    d.diamond("r1", "refutation finds<br>a witness state?", 30, 205, 180, 90)
    d.box("hid", "<b>Hidden<br>requirement</b><br>subtype from<br>the MAP set", 0, 335, 140, 100,
          RED, size=14)
    d.box("con1", "<b>Contradiction</b><br>of size one", 150, 335, 110, 100, PURPLE, size=14)
    d.edge("c1a", "post", "d1", "", "exitX=0.2;exitY=1;entryX=0.5;entryY=0;")
    d.edge("c1b", "d1", "r1", "yes", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;", size=13)
    d.edge("c1c", "r1", "hid", "yes", "edgeStyle=none;exitX=0.25;exitY=0.75;entryX=0.5;"
           "entryY=0;", size=13)
    d.edge("c1d", "r1", "con1", "no", "edgeStyle=none;exitX=0.75;exitY=0.75;entryX=0.5;"
           "entryY=0;", size=13)
    # Column 2: ambiguity
    d.diamond("d2", "P(|<i>A</i>" + sub("c") + "| = 2) ≥ τ" + sub("A") + "?", 280, 80, 160, 90)
    d.box("amb", "<b>Ambiguity</b>", 290, 215, 140, 50, YELLOW)
    d.box("dd", "<b>ddmin</b> over<br>sentences of 𝒯<br>→ 1-minimal set", 290, 335, 140, 100,
          WHITE, size=14)
    d.edge("c2a", "post", "d2", "", "exitX=0.6;exitY=1;entryX=0.5;entryY=0;")
    d.edge("c2b", "d2", "amb", "yes", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;", size=13)
    d.edge("c2c", "amb", "dd", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    # Column 3: weak check
    d.diamond("d3", "MAP {<i>x</i>} ≥ τ" + sub("W") + ", and an<br>acted run reads <i>y</i> ≠ "
              "<i>x</i><br>yet satisfies 𝒞?", 450, 70, 175, 110, size=14)
    d.box("weak", "<b>Weak check</b><br>at the cell's field;<br>the run is the witness",
          465, 335, 145, 100, GREEN, size=14)
    d.edge("c3a", "post", "d3", "", "exitX=0.895;exitY=1;entryX=0.5;entryY=0;")
    d.edge("c3b", "d3", "weak", "yes", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;", size=13)
    # Column 4: contradictions of two or more clauses
    d.box("hs", "soft minimal hitting<br>sets <i>H</i>, |<i>H</i>| ≤ 3", 655, 80, 150, 60,
          WHITE, size=14)
    d.diamond("d4", "|<i>H</i>| ≥ 2 and no run<br>satisfies <i>H</i>?", 640, 160, 180, 80,
              size=14)
    d.diamond("r4", "refutation finds<br>a witness state?", 640, 255, 180, 70, size=14)
    d.box("con2", "<b>Contradiction</b><br>(candidate MUS <i>H</i>)", 655, 345, 150, 60,
          PURPLE, size=14)
    d.box("drop", "dropped", 740, 425, 90, 30, GREY, size=13)
    d.edge("c4a", "vs", "hs", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("c4b", "hs", "d4", "", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    d.edge("c4c", "d4", "r4", "yes", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;", size=13)
    d.edge("c4d", "r4", "con2", "no", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;", size=13)
    d.edge("c4e", "r4", "drop", "yes", "exitX=1;exitY=0.5;entryX=1;entryY=0.5;",
           points=[(850, 290), (850, 440)], size=13)
    d.write("typing.drawio")


# --------------------------------------------------------------------------------------
def protocol() -> None:
    """Evaluation protocol: pre-registered go/no-go steps and the comparison study."""
    d = Diagram()
    d.text("row1", "<b>SpecDx, pre-registered</b>", 0, 0, 820, 24, "align=left;", "1", 15)
    steps = [("G0", "Simulation<br>(CPU)", BLUE), ("G1", "Base replay<br>(cached runs)", BLUE),
             ("", "<b>Freeze</b><br>pre-registration", GREY),
             ("G2", "Population<br>(≥ 4 models)", BLUE), ("G3", "Smoke run<br>(48 GB GPU)", BLUE),
             ("Gate", "20 development<br>mutants", YELLOW)]
    w, gap = 118, 22
    for i, (tag, body, col) in enumerate(steps):
        label = (f"<b>{tag}</b><br>{body}" if tag else body)
        d.box(f"g{i}", label, i * (w + gap), 30, w, 72, col, size=15)
        if i:
            d.edge(f"ge{i}", f"g{i - 1}", f"g{i}", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.box("stop", "fail → stop and report; parameters are never re-tuned", 0, 122, 670, 30,
          RED + "dashed=1;", size=14)
    for i in (0, 1, 3, 4):
        d.edge(f"se{i}", f"g{i}", "stop", "", "dashed=1;strokeColor=#b85450;exitX=0.5;exitY=1;"
               f"entryX={(i * (w + gap) + w / 2) / 670:.4f};entryY=0;")
    d.text("row2", "<b>Comparison study</b>", 0, 180, 820, 24, "align=left;", "1", 15)
    steps2 = [("Baselines on dev", "base + 20 mutants,<br>4 models each", GREEN),
              ("Freeze baselines", "model, strategy,<br>threshold (dev only)", GREY),
              ("Test study", "16 mutants, run once<br>(only if gate PASSED)", YELLOW),
              ("Report", "Tables A and B, McNemar<br>+ Holm, bootstrap CIs", ORANGE)]
    w2, gap2 = 185, 27
    for i, (title, body, col) in enumerate(steps2):
        d.box(f"b{i}", f"<b>{title}</b><br>{body}", i * (w2 + gap2), 212, w2, 72, col, size=15)
        if i:
            d.edge(f"be{i}", f"b{i - 1}", f"b{i}", "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.edge("gb", "g5", "b0", "PASSED or NOT PASSED", "exitX=0.5;exitY=1;entryX=0;entryY=0.5;",
           points=[(759, 166), (-20, 166), (-20, 248)], size=13, offset=(250, 0))
    d.write("protocol.drawio")


if __name__ == "__main__":
    concept()
    pipeline()
    motivating()
    typing()
    protocol()
