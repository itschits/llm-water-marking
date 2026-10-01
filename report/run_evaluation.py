"""
Evaluation harness for the Semester 3 project report.

Runs three causal LMs (GPT-2, Llama 3.2 1B, Qwen2.5 1.5B) through both
watermarking algorithms (STA-1, SynthID-Text) over a fixed set of prompts,
scores every generation with both detectors, and uses the project's own
WatermarkEvaluator to compute TPR/FPR/AUC-ROC and produce the figures
embedded in the report.

Uses only modules already present in src/ — no algorithm logic is
reimplemented here.

Run:
    python report/run_evaluation.py                 # all models
    python report/run_evaluation.py GPT-2 Qwen2.5-1.5B   # subset (others reused from disk)
    python report/run_evaluation.py --reuse         # no generation; recompute metrics + figures
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.model_loader import ModelLoader  # noqa: E402
from src.watermarking.sta1_watermark import STA1Watermark  # noqa: E402
from src.watermarking.synthid_watermark import SynthIDWatermark  # noqa: E402
from src.detection.sta1_detector import STA1Detector  # noqa: E402
from src.detection.synthid_detector import SynthIDDetector  # noqa: E402
from src.evaluation.evaluator import WatermarkEvaluator  # noqa: E402

OUT_DIR = os.path.join(ROOT, "report", "results")
FIG_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
THRESHOLD = 4.0
SEED = 0

# (display name, HuggingFace id, parameter count label)
MODELS = [
    ("GPT-2", "gpt2", "124M"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "1.24B"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "1.54B"),
]

PROMPTS = [
    "Artificial intelligence is transforming the world because",
    "The history of the Roman Empire shows that",
    "Climate change is one of the greatest challenges of our time, and",
    "In recent years, the global stock market has experienced",
    "The human brain is a remarkably complex organ that",
    "Renewable energy sources such as solar and wind power",
    "The novel begins with a description of a small town where",
    "Quantum computing promises to revolutionize computation by",
    "During the COVID-19 pandemic, healthcare systems around the world",
    "The French Revolution was a period of radical social change because",
    "Space exploration has always fascinated humanity because",
    "The theory of evolution explains how species",
    "Modern smartphones have changed the way people communicate by",
    "Global supply chains were severely disrupted when",
    "The Great Barrier Reef is home to thousands of species, and",
    "Machine learning algorithms can be used to predict",
    "The rise of social media has had a profound impact on",
    "Ancient Egyptian civilization is known for its",
    "Electric vehicles are becoming increasingly popular due to",
    "The internet has fundamentally changed how businesses operate by",
    "Deep-sea exploration reveals ecosystems that",
    "The invention of the printing press allowed",
    "Cybersecurity has become a top priority for organisations since",
    "The global economy in the twenty-first century depends heavily on",
    "Urban planning in growing cities must consider",
    "The discovery of penicillin changed medicine because",
    "Autonomous vehicles rely on sensors and software to",
    "The stock of a small technology company rose sharply after",
    "Traditional agriculture is being reshaped by new technologies that",
    "The history of philosophy begins with questions about",
]

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def slug(name: str) -> str:
    return name.replace(".", "_").replace("-", "_").lower()


def mean(xs):
    return sum(xs) / len(xs)


def run_model(display: str, hub_id: str) -> dict:
    """Generate and score all prompts for one model; returns its per-prompt records."""
    torch.manual_seed(SEED)
    loader = ModelLoader(hub_id, device=DEVICE)
    model, tokenizer = loader.load()
    vocab_size = loader.get_vocab_size()
    print(f"\n=== {display} ({hub_id}) — vocab={vocab_size}, device={DEVICE} ===")

    sta1_wm = STA1Watermark(vocab_size, seed=42, gamma=0.5, delta=2.0)
    sid_wm = SynthIDWatermark(vocab_size, keys=[2024, 42, 1234, 5678], depth=4, n_candidates=5)
    sta1_det = STA1Detector(sta1_wm, z_threshold=THRESHOLD)
    sid_det = SynthIDDetector(sid_wm, z_threshold=THRESHOLD)

    def generate_plain(prompt: str) -> str:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=0,
                top_p=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    def timed(fn, *args, **kw):
        t0 = time.time()
        out = fn(*args, **kw)
        return out, time.time() - t0

    records = []
    t_start = time.time()
    for i, prompt in enumerate(PROMPTS, 1):
        plain_text, _ = timed(generate_plain, prompt)
        sta1_text, sta1_s = timed(sta1_wm.generate, model, tokenizer, prompt,
                                  max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        sid_text, sid_s = timed(sid_wm.generate, model, tokenizer, prompt,
                                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)

        plain_ids = tokenizer.encode(plain_text, add_special_tokens=False)
        sta1_ids = tokenizer.encode(sta1_text, add_special_tokens=False)
        sid_ids = tokenizer.encode(sid_text, add_special_tokens=False)

        rec = {
            "prompt": prompt,
            "plain_text": plain_text,
            "sta1_text": sta1_text,
            "synthid_text": sid_text,
            "sta1_tokens": len(sta1_ids),
            "synthid_tokens": len(sid_ids),
            "sta1_gen_seconds": sta1_s,
            "synthid_gen_seconds": sid_s,
            "sta1_on_plain": sta1_det.score(plain_ids),
            "sta1_on_sta1": sta1_det.score(sta1_ids),
            "sta1_on_synthid": sta1_det.score(sid_ids),      # cross-check
            "synthid_on_plain": sid_det.score(plain_ids),
            "synthid_on_synthid": sid_det.score(sid_ids),
            "synthid_on_sta1": sid_det.score(sta1_ids),      # cross-check
        }
        records.append(rec)
        print(f"[{i:2d}/{len(PROMPTS)}] sta1 {sta1_s:5.1f}s  synthid {sid_s:5.1f}s  "
              f"STA1 z={rec['sta1_on_sta1']:.2f}  SynthID z={rec['synthid_on_synthid']:.2f}  "
              f"(baseline STA1={rec['sta1_on_plain']:.2f}, SynthID={rec['synthid_on_plain']:.2f})",
              flush=True)

    print(f"{display}: total time {time.time()-t_start:.1f}s")

    with open(os.path.join(OUT_DIR, f"raw_results_{slug(display)}.json"), "w", encoding="utf-8") as fh:
        json.dump({"model": display, "hub_id": hub_id, "vocab_size": vocab_size,
                   "device": DEVICE, "records": records}, fh, indent=2)

    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return records


# ── Generate (or reuse) per-model results ────────────────────────────────
selected = set(sys.argv[1:]) or {m[0] for m in MODELS}
print(f"Prompts: {len(PROMPTS)}  |  max_new_tokens={MAX_NEW_TOKENS}  |  temperature={TEMPERATURE}  "
      f"|  device={DEVICE}  |  running: {sorted(selected)}")

raw = {}
for display, hub_id, _ in MODELS:
    path = os.path.join(OUT_DIR, f"raw_results_{slug(display)}.json")
    if display in selected:
        run_model(display, hub_id)
    with open(path, encoding="utf-8") as fh:
        raw[display] = json.load(fh)

# ── Aggregate metrics via the project's own evaluator ────────────────────
evaluator = WatermarkEvaluator(output_dir=FIG_DIR)
summary = {
    "n_samples": len(PROMPTS),
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": TEMPERATURE,
    "threshold": THRESHOLD,
    "device": DEVICE,
    "models": {},
}
results_for_eval = {}

for display, hub_id, params in MODELS:
    recs = raw[display]["records"]
    per_algo = {}
    results_for_eval[display] = {}
    for key, algo_name, wm_key, plain_key in [
        ("sta1", "STA-1", "sta1_on_sta1", "sta1_on_plain"),
        ("synthid", "SynthID-Text", "synthid_on_synthid", "synthid_on_plain"),
    ]:
        wm = [r[wm_key] for r in recs]
        un = [r[plain_key] for r in recs]
        metrics = evaluator.compute_detection_metrics(wm, un, threshold=THRESHOLD)
        per_algo[key] = {
            "metrics": metrics,
            "mean_z_watermarked": mean(wm),
            "min_z_watermarked": min(wm),
            "mean_z_unwatermarked": mean(un),
            "max_z_unwatermarked": max(un),
            "mean_gen_seconds": mean([r[f"{key}_gen_seconds"] for r in recs]),
            "mean_tokens": mean([r[f"{key}_tokens"] for r in recs]),
        }
        results_for_eval[display][algo_name] = {
            "metrics": metrics, "watermarked_scores": wm, "unwatermarked_scores": un,
        }
        evaluator.plot_score_distributions(wm, un, algo_name, display, threshold=THRESHOLD)
        plt.close("all")

    summary["models"][display] = {
        "hub_id": hub_id,
        "params": params,
        "vocab_size": raw[display]["vocab_size"],
        **per_algo,
        "cross_check": {
            "sta1_detector_on_synthid_text_mean_z": mean([r["sta1_on_synthid"] for r in recs]),
            "synthid_detector_on_sta1_text_mean_z": mean([r["synthid_on_sta1"] for r in recs]),
        },
    }

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))

# ── Figures ──────────────────────────────────────────────────────────────
evaluator.plot_roc_curves(results_for_eval)
plt.close("all")

# Mean watermarked vs. plain z-score per model, one panel per algorithm
names = [m[0] for m in MODELS]
x = np.arange(len(names))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (key, title) in zip(axes, [("sta1", "STA-1"), ("synthid", "SynthID-Text")]):
    wm = [summary["models"][n][key]["mean_z_watermarked"] for n in names]
    un = [summary["models"][n][key]["mean_z_unwatermarked"] for n in names]
    b1 = ax.bar(x - 0.19, wm, 0.38, label="Watermarked", color="steelblue")
    b2 = ax.bar(x + 0.19, un, 0.38, label="Plain", color="tomato")
    ax.bar_label(b1, fmt="%.2f", fontsize=9)
    ax.bar_label(b2, fmt="%.2f", fontsize=9)
    ax.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1.2, label=f"Threshold = {THRESHOLD}")
    ax.set_xticks(x, names)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Mean detection z-score", fontsize=11)
axes[1].legend(fontsize=9, loc="upper right")
fig.suptitle("Mean detection z-score by model", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "zscore_by_model.png"), dpi=150)
plt.close(fig)

comparison_df = evaluator.compare_models(results_for_eval)
evaluator.save_comparison_csv(comparison_df, filename="comparison_table.csv")
comparison_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
print(comparison_df.to_string(index=False))
print("\nDone. Figures in report/figures/, data in report/results/.")