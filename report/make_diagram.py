"""Generates the system architecture diagram used in Chapter 2 of the report."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

BOX_STYLE = dict(boxstyle="round,pad=0.35,rounding_size=0.12", linewidth=1.3)

def box(x, y, w, h, text, fc, ec="#1f2937", fontsize=9.5, weight="bold", tc="#111827"):
    p = FancyBboxPatch((x, y), w, h, fc=fc, ec=ec, **BOX_STYLE)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=tc, linespacing=1.4)
    return (x, y, w, h)

def arrow(b1, b2, side1="right", side2="left", text=None):
    def anchor(b, side):
        x, y, w, h = b
        return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
                "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]
    p1, p2 = anchor(b1, side1), anchor(b2, side2)
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                         color="#374151", linewidth=1.3)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.18, text, ha="center", fontsize=7.8, color="#374151", style="italic")

# ── Row 1: clients ──────────────────────────────────────────────
b_web = box(0.4, 5.6, 3.0, 1.0, "React + Vite Web App\n(Dashboard, Detector,\nRun Experiment, Charts)", "#dbeafe")
b_cli = box(6.6, 5.6, 3.0, 1.0, "CLI / standalone scripts\n(gpt2/sta1.py,\ngpt2/synthid.py)", "#dbeafe")

# ── Row 2: API ──────────────────────────────────────────────────
b_api = box(3.2, 4.0, 3.6, 1.0, "FastAPI backend  (api/server.py)\n/detect  /experiments/run  /results  /config", "#fef3c7")

arrow(b_web, b_api, "bottom", "top")
arrow(b_cli, b_api, "bottom", "top")

# ── Row 3: core src/ modules ─────────────────────────────────────
b_model = box(0.3, 2.3, 2.1, 1.1, "model_loader.py\nHuggingFace\nAutoModelForCausalLM", "#dcfce7")
b_wm = box(2.7, 2.3, 2.3, 1.1, "watermarking/\nSTA1Watermark\nSynthIDWatermark", "#dcfce7")
b_det = box(5.3, 2.3, 2.3, 1.1, "detection/\nSTA1Detector\nSynthIDDetector", "#dcfce7")
b_eval = box(7.9, 2.3, 1.9, 1.1, "evaluation/\nWatermarkEvaluator", "#dcfce7")

arrow(b_api, b_wm, "bottom", "top")
arrow(b_api, b_det, "bottom", "top")
arrow(b_model, b_wm, "right", "left")
arrow(b_wm, b_det, "right", "left", "generated\ntoken ids")
arrow(b_det, b_eval, "right", "left")

# ── Row 4: config + storage ──────────────────────────────────────
b_cfg = box(0.3, 0.5, 2.6, 1.0, "config.yaml\n(gamma, keys, thresholds,\ngeneration params)", "#fee2e2")
b_res = box(6.4, 0.5, 3.2, 1.0, "results/*.json, *.png\n(metrics, plots, comparison CSV)", "#fee2e2")

arrow(b_cfg, b_wm, "top", "bottom")
arrow(b_eval, b_res, "bottom", "top")

ax.set_title("Figure 2.1 — System Architecture", fontsize=12, weight="bold", pad=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "architecture.png"), dpi=170)
print("saved architecture.png")