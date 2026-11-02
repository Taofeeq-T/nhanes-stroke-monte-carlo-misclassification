"""
Workflow figure (manuscript Figure 6): analytical pipeline for the revised analysis.

Writes artifacts/fig_pipeline_workflow.png (300 dpi), .pdf and .svg.
Usage: python scripts/fig_pipeline_workflow.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
artifacts_dir.mkdir(exist_ok=True)
W, H = 1430, 1720
plt.rcParams["font.family"] = "DejaVu Sans"
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

PAL = {
    "blue":   ("#E4EEF9", "#7FA4D8", "#1F4E8C"),
    "gray":   ("#EFEEEA", "#B3AD9F", "#4A4A4A"),
    "purple": ("#ECEAFB", "#A79FE3", "#4B3A9E"),
    "green":  ("#E1F2E9", "#7BBE9B", "#1E6B45"),
    "red":    ("#F9ECE7", "#D7998A", "#8A3324"),
    "orange": ("#FCEBD8", "#DE9F5E", "#7A4710"),
    "lgreen": ("#E9F3DC", "#9CC47B", "#3E6B1A"),
    "pink":   ("#FBE6E8", "#DB8A90", "#8E1B23"),
}
ARROW = "#68708A"
NOTE = "#6B7385"
LC, RC, CC = 530, 1117, 824          # column centres
BW, CW = 496, 508                     # box widths (side columns, centre column)


def box(cx, top, h, color, title, lines, w=BW):
    fill, edge, text = PAL[color]
    ax.add_patch(FancyBboxPatch((cx - w / 2, top), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=fill, ec=edge, lw=1.8))
    n = 1 + len(lines)
    y0 = top + h / 2 - (n - 1) * 11.5 - 2
    ax.text(cx, y0, title, ha="center", va="center", fontsize=16.5, color=text)
    for k, ln in enumerate(lines):
        ax.text(cx, y0 + 27 + k * 22.5, ln, ha="center", va="center", fontsize=12.8, color=text)
    return {"cx": cx, "top": top, "bot": top + h, "l": cx - w / 2, "r": cx + w / 2, "mid": top + h / 2}


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 color=ARROW, lw=1.8, shrinkA=0, shrinkB=0))


def dashed(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], ls=(0, (4, 4)), color="#C3C6CF", lw=1.6)


def note(b, lines):
    y = b["mid"] - (len(lines) - 1) * 11.5
    for k, ln in enumerate(lines):
        ax.text(255, y + k * 23, ln, ha="right", va="center", fontsize=13.2, color=NOTE)
    dashed(262, b["mid"], b["l"], b["mid"])


ax.text(CC, 50, "Analytical pipeline", ha="center", va="center", fontsize=20, color="#222222")

b1 = box(CC, 88, 116, "blue", "Data extraction",
         ["NHANES 2003–2023, 9 cycles", "N = 52,507 with a valid stroke response"], w=600)
b2 = box(CC, 246, 116, "blue", "Cohort restriction",
         ["Aged ≥20 years, MEC-examined, non-pregnant;",
          "positive phlebotomy weight in 2021–2023",
          "N = 47,696 (1,962 reported stroke, 4.1%)"], w=600)
arrow(CC, b1["bot"], CC, b2["top"])
b3 = box(CC, 404, 92, "gray", "Stratified 80/20 split", ["Stratified on stroke status, seed = 42"], w=CW)
arrow(CC, b2["bot"], CC, b3["top"])

b4l = box(LC, 558, 116, "purple", "80% training set", ["N = 38,156", "1,570 reported stroke (4.1%)"])
b4r = box(RC, 558, 116, "purple", "20% holdout set", ["N = 9,540, 392 reported stroke (4.1%)", "Never used in fitting"])
arrow(CC - 85, b3["bot"], LC, b4l["top"])
arrow(CC + 85, b3["bot"], RC, b4r["top"])

b5 = box(LC, 717, 128, "green", "Missingness assessment",
         ["Training set only", "Tiered framework (A / B / C / D)", "No variable met the Tier C threshold"])
arrow(LC, b4l["bot"], LC, b5["top"])
note(b5, ["Tier D excluded:", "sleep, triglycerides"])

b6 = box(LC, 889, 160, "red", "Repeated stratified CV",
         ["5-fold × 5 repeats = 25 fits", "HyperImpute fit on each fold's training rows,",
          "applied to its validation rows (no refitting)", "AutoGluon best_quality"])
arrow(LC, b5["bot"], LC, b6["top"])
note(b6, ["Optimal threshold", "(Youden's J from", "OOF predictions)"])

b7l = box(LC, 1093, 138, "green", "HyperImpute (final)",
          ["Fitted on full training set, 20 variables", "Column-wise model selection",
           "Models stored for holdout"])
b7r = box(RC, 1093, 138, "green", "Holdout imputation",
          ["Stored training models applied", "No refitting on holdout rows"])
arrow(LC, b6["bot"], LC, b7l["top"])
dashed(RC, b4r["bot"], RC, b7r["top"] - 2)
arrow(b7l["r"], b7l["mid"], b7r["l"], b7r["mid"])
note(b7l, ["Learners: logistic reg.,", "random forest, CatBoost"])

b8l = box(LC, 1275, 92, "red", "Final model retrain", ["Full training set, extended time"])
b8r = box(RC, 1275, 92, "orange", "Holdout evaluation", ["AUROC, AUPRC, F1, calibration"])
arrow(LC, b7l["bot"], LC, b8l["top"])
arrow(RC, b7r["bot"], RC, b8r["top"])
arrow(b8l["r"], b8l["mid"], b8r["l"], b8r["mid"])

b9 = box(RC, 1411, 92, "lgreen", "SHAP interpretation", ["Beeswarm, dependence, bar plots"])
arrow(RC, b8r["bot"], RC, b9["top"])

b10 = box(RC, 1547, 138, "pink", "Probabilistic bias analysis",
          ["1,000 Monte Carlo iterations", "Latent stroke outcome reconstructed",
           "Se ~ U(0.36, 0.98), Sp ~ U(0.96, 0.996)"])
arrow(RC, b9["bot"], RC, b10["top"])
b10l = box(LC, 1547, 138, "gray", "Sensitivity analyses",
           ["Maximum likelihood reconstruction", "of latent risk (no truncation)",
            "q(s) cross-fitted within the holdout"])
arrow(b10["l"], b10["mid"], b10l["r"], b10l["mid"])

for ext in ("png", "pdf", "svg"):
    fig.savefig(artifacts_dir / f"fig_pipeline_workflow.{ext}", dpi=300)
print("Saved: fig_pipeline_workflow.png / .pdf / .svg")