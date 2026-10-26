"""Metrics, result files and plots. Everything is in original target units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def score(y_true: np.ndarray, y_pred: np.ndarray, target_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """R2 and RMSE per target."""
    out = {}
    for j, name in enumerate(target_cols):
        out[name] = {"R2": float(r2_score(y_true[:, j], y_pred[:, j])),
                     "RMSE": float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j])))}
    return out


def metrics_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups, target_cols: Sequence[str],
                     min_n: int = 3) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-group (e.g. per-Day) metrics; groups with fewer than min_n rows are skipped."""
    groups = np.asarray(groups).astype(str)
    out = {}
    for g in sorted(np.unique(groups)):
        m = groups == g
        if m.sum() >= min_n:
            out[g] = score(y_true[m], y_pred[m], target_cols)
            out[g]["n"] = int(m.sum())
    return out


def save_results(splits: Dict[str, Tuple[np.ndarray, np.ndarray]], target_cols: Sequence[str],
                 out_dir, meta: Optional[Dict[str, pd.DataFrame]] = None,
                 verbose: bool = True, plot: bool = True) -> dict:
    """splits = {"train": (y_true, y_pred), "val": ..., "test": ...}, original units.

    Writes metrics.json, predictions_<split>.csv and (if plot) parity_plot.png.
    """
    out_dir = Path(out_dir)
    metrics = {name: score(t, p, target_cols) for name, (t, p) in splits.items()}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for name, (t, p) in splits.items():
        cols = {}
        for j, tgt in enumerate(target_cols):
            cols[f"{tgt}_actual"] = t[:, j]
            cols[f"{tgt}_predicted"] = p[:, j]
        table = pd.DataFrame(cols)
        if meta and name in meta:
            table = pd.concat([meta[name].reset_index(drop=True), table], axis=1)
        table.to_csv(out_dir / f"predictions_{name}.csv", index=False)

    if plot:
        colors = {"train": "tab:blue", "val": "tab:green", "test": "tab:red"}
        fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 4.5), squeeze=False)
        for j, tgt in enumerate(target_cols):
            ax = axes[0, j]
            lo, hi = np.inf, -np.inf
            for name, (t, p) in splits.items():
                m = metrics[name][tgt]
                ax.scatter(t[:, j], p[:, j], s=10, alpha=0.6, color=colors.get(name),
                           label=f"{name} (RMSE={m['RMSE']:.2f}, R2={m['R2']:.3f})")
                lo, hi = min(lo, t[:, j].min()), max(hi, t[:, j].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(tgt)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "parity_plot.png", dpi=150)
        plt.close(fig)

    if verbose:
        for name, per_target in metrics.items():
            for tgt, m in per_target.items():
                print(f"{name:6s} {tgt:8s} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f}")
    return metrics


def save_loss_curve(history: Dict[str, Sequence[float]], out_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, values in history.items():
        ax.plot(values, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardised units)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "loss_curve.png", dpi=150)
    plt.close(fig)


def summarize_runs(runs_by_split: Dict[int, Sequence[dict]], seeds: Sequence[int]) -> dict:
    """summary.json content: mean/std over ALL runs, the seed lists and per-split means.

    runs_by_split maps split seed -> list of {"train","val","test"} metric dicts (one per
    training seed).
    """
    all_runs = [r for runs in runs_by_split.values() for r in runs]
    summary = mean_std(all_runs)
    summary["split_seeds"] = [int(s) for s in runs_by_split]
    summary["seeds"] = [int(s) for s in seeds]
    summary["per_split"] = {str(ss): _means(mean_std(runs)) for ss, runs in runs_by_split.items()}
    return summary


def _means(node: dict) -> dict:
    """Reduce a mean_std tree to its means."""
    if set(node) == {"mean", "std"}:
        return node["mean"]
    return {k: _means(v) for k, v in node.items()}


def mean_std(runs: Sequence[dict]) -> dict:
    """Recursively combine a list of identically-shaped nested metric dicts into mean/std."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: mean_std([r[k] for r in runs]) for k in first}
    vals = np.array(runs, dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std())}
