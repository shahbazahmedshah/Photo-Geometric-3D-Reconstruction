"""Generates docs/pipeline.drawio (and a quick PNG preview for layout checks)."""
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# palette: (fill, stroke)
C = {
    "scene": ("#E3F1E6", "#3A9D5D"),
    "fwd": ("#ECEFF4", "#5B6B82"),
    "fwdin": ("#FFFFFF", "#8A97AB"),
    "sv": ("#EAF3FA", "#2A6F97"),
    "svb": ("#FFFFFF", "#2A6F97"),
    "mv": ("#FDF1E6", "#C76A1A"),
    "mvb": ("#FFFFFF", "#C76A1A"),
    "fuse": ("#F1EAF8", "#7D5BA6"),
    "eval": ("#E6F4F1", "#1F7A6D"),
    "out": ("#F4F4F4", "#777777"),
}

# id: (x, y, w, h, title, detail, colour, kind)
NODES = [
    ("scene", 30, 300, 210, 250, "Analytic Scene",
     "z = f(x, y)<br>p = ∂z/∂x, q = ∂z/∂y<br>n = (−p, −q, 1)/‖·‖<br>albedo ρ(x, y)<br><br>"
     "<i>sphere · gaussian<br>peaks · ripple</i>", "scene", "box"),

    ("fwd", 280, 150, 240, 560, "Forward Models Φ", "", "fwd", "group"),
    ("lamb", 295, 195, 210, 80, "Lambertian shading", "I<sub>k</sub> = ρ max(0, n·s<sub>k</sub>)", "fwdin", "box"),
    ("defoc", 295, 290, 210, 80, "Thin-lens defocus", "σ(Z) ∝ (Av/2)|1/F − 1/v − 1/Z|", "fwdin", "box"),
    ("pin", 295, 385, 210, 80, "Pinhole projection", "x ~ K [R | t] X  ·  ray casting", "fwdin", "box"),
    ("warp", 295, 480, 210, 80, "Exact motion warp", "I<sub>2</sub>(x + w(x)) = I<sub>1</sub>(x)", "fwdin", "box"),
    ("noise", 295, 600, 210, 90, "Sensor model", "+ N(0, σ²) · 8-bit quantisation<br><i>applied to every image</i>", "fwdin", "dashed"),

    ("lane1", 560, 40, 510, 370, "Single viewpoint  ·  Course 3", "", "sv", "lane"),
    ("ps", 590, 90, 200, 90, "Photometric Stereo", "g = (SᵀS)⁻¹SᵀI<br>ρ = ‖g‖,  n = g/ρ", "svb", "box"),
    ("integ", 840, 90, 210, 90, "Normal Integration", "Poisson: ∇²z = p<sub>x</sub> + q<sub>y</sub><br>Frankot–Chellappa (FFT)", "svb", "box"),
    ("sfs", 590, 200, 200, 90, "Shape from Shading", "min Σ(E − R(p,q))²<br>+ λ(‖∇p‖² + ‖∇q‖²)", "svb", "box"),
    ("dfd", 590, 305, 200, 90, "Depth from Defocus", "σ₁² − σ₂² = 2(g₁ − g₂)/∇²g<br>→ Z by thin-lens lookup", "svb", "box"),

    ("lane2", 560, 440, 510, 420, "Multiple viewpoints  ·  Course 4", "", "mv", "lane"),
    ("calib", 590, 495, 200, 90, "Camera Calibration", "DLT  Ap = 0 → RQ → LM<br>K, R, t", "mvb", "box"),
    ("stereo", 840, 495, 210, 90, "Rectified Stereo", "SSD block match<br>Z = f b / d", "mvb", "box"),
    ("epi", 590, 605, 200, 90, "Uncalibrated Stereo", "x₂ᵀ F x₁ = 0  (8-point)<br>E = KᵀFK → R, t", "mvb", "box"),
    ("tri", 840, 605, 210, 90, "Triangulation", "linear DLT<br>→ sparse metric 3D points", "mvb", "box"),
    ("flow", 590, 715, 200, 90, "Optical Flow", "I<sub>x</sub>u + I<sub>y</sub>v + I<sub>t</sub> = 0<br>Lucas–Kanade · Horn–Schunck", "mvb", "box"),
    ("sfm", 840, 715, 210, 90, "Structure from Motion", "W = M S  (rank 3)<br>metric upgrade QQᵀ", "mvb", "box"),

    ("fuse", 1130, 345, 240, 175, "Photo-Geometric Fusion",
     "min ‖A<sub>n</sub>z − b<sub>n</sub>‖² + λ²‖A<sub>p</sub>z − ẑ‖²<br><br>"
     "normals → high frequencies<br>points → scale + low frequencies<br>λ by hold-out validation", "fuse", "box"),

    ("eval", 1430, 240, 240, 240, "Evaluation",
     "angular error · depth RMS<br>reprojection · EPE<br>pose / shape error<br><br>"
     "Monte Carlo ≥ 100 trials<br>noise · lights · baseline sweeps", "eval", "box"),
    ("out", 1430, 580, 240, 120, "Outputs",
     "fig01 – fig15 · *.json<br>summary.md (O1 – O8)<br>report · slides", "out", "box"),
]

# (id, src, dst, label, exit(x,y), entry(x,y), waypoints, style-extra)
EDGES = [
    ("e1", "lamb", "ps", "{I<sub>k</sub>}, S", (1, 0.3), (0, 0.5), [(545, 219), (545, 135)], ""),
    ("e2", "lamb", "sfs", "E, s", (1, 0.75), (0, 0.5), [], ""),
    ("e3", "defoc", "dfd", "g₁, g₂", (1, 0.6), (0, round(33 / 90, 4)), [], ""),
    ("e4", "pin", "lane2", "views, x ↔ X", (1, 0.4), (0, 0.11), [(530, 417), (530, 486)], ""),
    ("e5", "warp", "flow", "I₁, I₂", (1, 0.55), (0, 0.5), [(545, 524), (545, 760)], ""),
    ("e6", "ps", "integ", "n, ρ", (1, 0.5), (0, 0.5), [], ""),
    ("e7", "sfs", "integ", "p, q", (1, 0.5), (0.5, 1), [(945, 245)], ""),
    ("e8", "calib", "stereo", "K", (1, 0.45), (0, 0.45), [], ""),
    ("e9", "calib", "tri", "P₁, P₂", (1, 0.8), (0, 0.25), [(815, 567), (815, 627)], ""),
    ("e10", "epi", "tri", "R, t", (1, 0.6), (0, 0.6), [], ""),
    ("e11", "flow", "sfm", "tracks → W", (1, 0.5), (0, 0.5), [], ""),
    ("e12", "integ", "fuse", "normal rows A<sub>n</sub>", (1, 0.5), (0.5, 0), [(1250, 135)], ""),
    ("e13", "tri", "fuse", "sparse ẑ", (1, 0.5), (0.5, 1), [(1250, 650)], ""),
    ("e14", "fuse", "eval", "fused z", (1, 0.25), (0, 0.6), [], ""),
    ("e15", "lane1", "eval", "per-method estimates", (1, 0.05), (0, 0.15), [(1395, 58.5), (1395, 276)], ""),
    ("e16", "lane2", "eval", "per-method estimates", (1, 0.75), (0, 0.9), [(1395, 755), (1395, 456)], ""),
    ("e17", "eval", "out", "", (0.5, 1), (0.5, 0), [], ""),
    ("e18", "scene", "eval", "exact ground truth  z, n, K, R, t, w", (0.5, 0), (0.5, 0),
     [(135, 15), (1550, 15)], "dashed=1;dashPattern=6 4;strokeColor=#3A9D5D;fontColor=#3A9D5D;"),
    ("e19", "scene", "fwd", "", (1, 0.5), (0, 0.5), [], "strokeWidth=2.5;"),
]


def label(title, detail, kind):
    if kind == "lane":
        return f"<b>{title}</b>"
    if kind == "group":
        return f"<b style='font-size:13px'>{title}</b>"
    t = f"<b style='font-size:12px'>{title}</b>"
    if detail:
        t += f"<br><span style='font-size:10px'>{detail}</span>"
    return t


def node_style(kind, col):
    fill, stroke = C[col]
    base = f"html=1;whiteSpace=wrap;fontFamily=Helvetica;fillColor={fill};strokeColor={stroke};fontColor=#1F2933;"
    if kind == "lane":
        return ("swimlane;horizontal=1;startSize=30;rounded=1;arcSize=4;fontSize=13;"
                f"swimlaneFillColor={fill};" + base + f"fillColor={stroke};fontColor=#FFFFFF;")
    if kind == "group":
        return "rounded=1;arcSize=4;verticalAlign=top;spacingTop=8;strokeWidth=1.5;" + base
    if kind == "dashed":
        return "rounded=1;arcSize=8;dashed=1;verticalAlign=middle;" + base
    return "rounded=1;arcSize=8;verticalAlign=middle;strokeWidth=1.5;" + base


def build_xml():
    pos = {n[0]: n for n in NODES}
    parent = {}
    for nid, x, y, w, h, *_ in NODES:
        for lane in ("lane1", "lane2"):
            lx, ly, lw, lh = pos[lane][1:5]
            if nid not in ("lane1", "lane2") and lx < x < lx + lw and ly < y < ly + lh:
                parent[nid] = lane
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for nid, x, y, w, h, title, detail, col, kind in NODES:
        p = parent.get(nid, "1")
        gx, gy = (x - pos[p][1], y - pos[p][2]) if p != "1" else (x, y)
        v = html.escape(label(title, detail, kind), quote=True)
        cells.append(f'<mxCell id="{nid}" value="{v}" style="{node_style(kind, col)}" vertex="1" parent="{p}">'
                     f'<mxGeometry x="{gx}" y="{gy}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for eid, s, d, lab, ex, en, wps, extra in EDGES:
        st = ("edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeColor=#3C4858;"
              "fontSize=10;fontFamily=Helvetica;labelBackgroundColor=#FFFFFF;"
              f"exitX={ex[0]};exitY={ex[1]};exitDx=0;exitDy=0;entryX={en[0]};entryY={en[1]};entryDx=0;entryDy=0;"
              + extra)
        pts = "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in wps)
        arr = f'<Array as="points">{pts}</Array>' if wps else ""
        cells.append(f'<mxCell id="{eid}" value="{html.escape(lab, quote=True)}" style="{st}" edge="1" parent="1" '
                     f'source="{s}" target="{d}"><mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
    title = ('<mxCell id="title" value="&lt;b style=&#x27;font-size:16px&#x27;&gt;Photo-Geometric 3D Shape Recovery'
             '&lt;/b&gt;&lt;br&gt;&lt;span style=&#x27;font-size:11px&#x27;&gt;data flow of the reference solution'
             '&lt;/span&gt;" style="text;html=1;align=left;verticalAlign=top;fontFamily=Helvetica;fontColor=#1F2933;" '
             'vertex="1" parent="1"><mxGeometry x="30" y="40" width="240" height="60" as="geometry"/></mxCell>')
    cells.append(title)
    body = "".join(cells)
    return ('<mxfile host="app.diagrams.net"><diagram id="pipeline" name="Pipeline">'
            '<mxGraphModel dx="1700" dy="900" grid="1" gridSize="5" guides="1" tooltips="1" connect="1" arrows="1" '
            'fold="1" page="1" pageScale="1" pageWidth="1720" pageHeight="900" math="0" shadow="0">'
            f'<root>{body}</root></mxGraphModel></diagram></mxfile>')


def anchor(n, f):
    _, x, y, w, h, *_ = n
    return x + f[0] * w, y + f[1] * h


def preview(path):
    """Rough matplotlib rendering of the same geometry (for layout checks only)."""
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    pos = {n[0]: n for n in NODES}
    fig, ax = plt.subplots(figsize=(17.2, 9))
    strip = lambda s: re.sub(r"<[^>]+>", "", s.replace("<br>", "\n"))  # noqa: E731
    for nid, x, y, w, h, title, detail, col, kind in NODES:
        fill, stroke = C[col]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", fc=fill, ec=stroke,
                                    lw=1.4, ls="--" if kind == "dashed" else "-"))
        if kind == "lane":
            ax.add_patch(FancyBboxPatch((x, y), w, 30, boxstyle="round,pad=0,rounding_size=6", fc=stroke, ec=stroke))
            ax.text(x + w / 2, y + 15, title, ha="center", va="center", color="w", weight="bold", fontsize=9)
        elif kind == "group":
            ax.text(x + w / 2, y + 18, title, ha="center", va="center", weight="bold", fontsize=9)
        else:
            ax.text(x + w / 2, y + h / 2, title + "\n" + strip(detail), ha="center", va="center", fontsize=6.5)
    for eid, s, d, lab, ex, en, wps, extra in EDGES:
        p = [anchor(pos[s], ex)] + list(wps) + [anchor(pos[d], en)]
        # orthogonal completion like draw.io
        full = [p[0]]
        for q in p[1:]:
            a = full[-1]
            if a[0] != q[0] and a[1] != q[1]:
                full.append((q[0], a[1]))
            full.append(q)
        xs, ys = zip(*full)
        ax.plot(xs, ys, color="#3A9D5D" if "dashed" in extra else "#3C4858", lw=1,
                ls="--" if "dashed" in extra else "-")
        ax.annotate("", xy=full[-1], xytext=full[-2], arrowprops=dict(arrowstyle="-|>", color="#3C4858", lw=1))
        if lab:
            mid = full[len(full) // 2 - 1] if len(full) > 2 else full[0]
            mid2 = full[len(full) // 2]
            ax.text((mid[0] + mid2[0]) / 2, (mid[1] + mid2[1]) / 2, strip(lab), fontsize=6.5, ha="center",
                    va="center", bbox=dict(fc="w", ec="none", pad=0.5))
    ax.set_xlim(0, 1720)
    ax.set_ylim(900, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    out = os.path.join(HERE, "pipeline.drawio")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_xml())
    print("wrote", out)
    if "--preview" in sys.argv:
        preview(os.path.join(HERE, "pipeline_preview.png"))
