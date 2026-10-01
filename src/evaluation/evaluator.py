"""
Evaluation utilities for LLM watermarking experiments.

Provides:
  * Detection metrics  (TPR, FPR, accuracy, AUC-ROC)
  * Score distribution plots
  * ROC curve plots
  * Cross-model comparison table
  * JSON / CSV result persistence
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:  # seaborn is optional
    _HAS_SEABORN = False

logger = logging.getLogger(__name__)


class WatermarkEvaluator:
    """
    Computes and persists watermark detection evaluation metrics.

    Parameters
    ----------
    output_dir : str
        Directory where plots and result files are written.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------

    def compute_detection_metrics(
        self,
        watermarked_scores: List[float],
        unwatermarked_scores: List[float],
        threshold: float,
    ) -> Dict[str, Any]:
        """
        Compute binary-classification metrics at a fixed *threshold*.

        A sample is classified as watermarked when score ≥ threshold.

        Returns
        -------
        dict with keys: tpr, fpr, accuracy, precision, f1, tp, fp, tn, fn, auc_roc
        """
        # --- binary decisions ----------------------------------------
        tp = sum(s >= threshold for s in watermarked_scores)
        fn = len(watermarked_scores) - tp
        fp = sum(s >= threshold for s in unwatermarked_scores)
        tn = len(unwatermarked_scores) - fp

        n_pos = len(watermarked_scores)
        n_neg = len(unwatermarked_scores)
        n_total = n_pos + n_neg

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (
            2 * precision * tpr / (precision + tpr)
            if (precision + tpr) > 0
            else 0.0
        )

        # --- AUC-ROC -------------------------------------------------
        all_scores = list(watermarked_scores) + list(unwatermarked_scores)
        all_labels = [1] * n_pos + [0] * n_neg
        fpr_curve, tpr_curve, _ = roc_curve(all_labels, all_scores)
        auc_roc = float(auc(fpr_curve, tpr_curve))

        return {
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "auc_roc": round(auc_roc, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_watermarked": n_pos,
            "n_unwatermarked": n_neg,
            "threshold": threshold,
        }

    # ------------------------------------------------------------------
    def compute_roc(
        self,
        watermarked_scores: List[float],
        unwatermarked_scores: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Return (fpr_array, tpr_array, auc_score) for ROC plotting.
        """
        all_scores = list(watermarked_scores) + list(unwatermarked_scores)
        all_labels = [1] * len(watermarked_scores) + [0] * len(unwatermarked_scores)
        fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_scores)
        auc_score = float(auc(fpr_arr, tpr_arr))
        return fpr_arr, tpr_arr, auc_score

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def compare_models(
        self, results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Build a tidy DataFrame comparing models and algorithms.

        Parameters
        ----------
        results : dict
            ``results[model_display_name][algorithm_name]["metrics"]``

        Returns
        -------
        pd.DataFrame with columns:
            Model, Algorithm, TPR, FPR, Accuracy, F1, AUC-ROC
        """
        rows = []
        for model_name, algo_dict in results.items():
            for algo_name, algo_data in algo_dict.items():
                m = algo_data.get("metrics", {})
                rows.append(
                    {
                        "Model": model_name,
                        "Algorithm": algo_name,
                        "TPR": m.get("tpr", float("nan")),
                        "FPR": m.get("fpr", float("nan")),
                        "Accuracy": m.get("accuracy", float("nan")),
                        "F1": m.get("f1", float("nan")),
                        "AUC-ROC": m.get("auc_roc", float("nan")),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_score_distributions(
        self,
        watermarked_scores: List[float],
        unwatermarked_scores: List[float],
        algorithm_name: str,
        model_name: str,
        threshold: float,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot overlapping score histograms for watermarked vs. unwatermarked.

        Parameters
        ----------
        save : bool
            If True, write the figure to *output_dir*.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.hist(
            watermarked_scores,
            bins=25,
            alpha=0.65,
            color="steelblue",
            label="Watermarked",
            density=True,
        )
        ax.hist(
            unwatermarked_scores,
            bins=25,
            alpha=0.65,
            color="tomato",
            label="Not Watermarked",
            density=True,
        )
        ax.axvline(
            x=threshold,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold = {threshold:.2f}",
        )

        ax.set_xlabel("Detection Score (z-score)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(
            f"{algorithm_name} — Score Distribution\nModel: {model_name}", fontsize=13
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save:
            fname = (
                f"score_dist_{algorithm_name.replace(' ', '_').replace('-', '')}"
                f"_{model_name.replace(' ', '_')}.png"
            )
            fig.savefig(os.path.join(self.output_dir, fname), dpi=150)
            logger.info("Saved: %s", fname)

        return fig

    # ------------------------------------------------------------------
    def plot_roc_curves(
        self,
        results: Dict[str, Dict[str, Any]],
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot ROC curves for every (model, algorithm) combination in *results*.
        """
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        fig, ax = plt.subplots(figsize=(8, 6))

        ci = 0
        for model_name, algo_dict in results.items():
            for algo_name, algo_data in algo_dict.items():
                fpr_arr, tpr_arr, auc_val = self.compute_roc(
                    algo_data["watermarked_scores"],
                    algo_data["unwatermarked_scores"],
                )
                label = f"{model_name} / {algo_name}  (AUC={auc_val:.3f})"
                ax.plot(fpr_arr, tpr_arr, color=colors[ci % 10], lw=1.8, label=label)
                ci += 1

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — Watermark Detection", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save:
            fname = "roc_curves_comparison.png"
            fig.savefig(os.path.join(self.output_dir, fname), dpi=150)
            logger.info("Saved: %s", fname)

        return fig

    # ------------------------------------------------------------------
    def plot_comparison_heatmap(
        self,
        comparison_df: pd.DataFrame,
        metric: str = "AUC-ROC",
        save: bool = True,
    ) -> plt.Figure:
        """
        Heat-map of a chosen *metric* across models (rows) × algorithms (cols).
        """
        pivot = comparison_df.pivot(index="Model", columns="Algorithm", values=metric)

        fig, ax = plt.subplots(
            figsize=(max(5, len(pivot.columns) * 2.5), max(3, len(pivot) * 1.5))
        )
        if _HAS_SEABORN:
            import seaborn as sns  # noqa: F811
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".3f",
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                ax=ax,
                linewidths=0.5,
            )
        else:
            im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
            ax.set_yticklabels(pivot.index)
            for i in range(len(pivot)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"{metric} — Model × Algorithm", fontsize=13)
        fig.tight_layout()

        if save:
            fname = f"heatmap_{metric.replace('-', '').replace(' ', '_')}.png"
            fig.savefig(os.path.join(self.output_dir, fname), dpi=150)
            logger.info("Saved: %s", fname)

        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """
        Serialise *results* to JSON.  Returns the file path.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{ts}"

        # Strip non-serialisable objects (e.g. long text lists if huge)
        def _clean(obj: Any) -> Any:
            if isinstance(obj, (np.floating, float)):
                return round(float(obj), 6)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        path = os.path.join(self.output_dir, f"{filename}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_clean(results), fh, indent=2)
        logger.info("Results saved → %s", path)
        return path

    def save_comparison_csv(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "comparison_table.csv",
    ) -> str:
        """Write the comparison DataFrame to CSV."""
        path = os.path.join(self.output_dir, filename)
        comparison_df.to_csv(path, index=False, float_format="%.4f")
        logger.info("Comparison table saved → %s", path)
        return path
