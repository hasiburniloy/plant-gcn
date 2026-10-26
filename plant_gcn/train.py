"""Train Plant-GCN (full-batch, transductive) on the shared pipeline.

    python -m plant_gcn.train --config plant_gcn/config.yaml [--data x.xlsx] [--out runs/x]
                              [--set training.epochs=600 ...] [--repeat-splits N]

Graph nodes are all train + val spectra; the loss is MSE on train nodes only and val
RMSE drives the LR schedule and early stopping. Test spectra are appended to the graph
after training (labels unused) and scored in original units (paper section 2.6).
With augmentation.mode=online the graph is built once from the clean features of the real
plants and every training epoch replaces the node features by a fresh offset/slope jitter;
validation, early stopping and all predictions use the clean features.
For each seed the output folder seed_<s>/ gets model.pt, metrics.json,
predictions_{train,val,test}.csv, metrics_test_by_Day.json, parity_plot.png and
loss_curve.png; summary.json holds mean/std across all runs plus per-split means.
data.split_seed may be a list (or use --repeat-splits N): every split seed x training
seed is run, in split_<ss>/seed_<s>/ folders when there is more than one split seed.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from common import SplitData, get_device, load_config, prepare_dataset, set_seed
from common.metrics import metrics_by_group, save_loss_curve, save_results, summarize_runs
from common.utils import apply_overrides, ensure_dir, repeat_splits, split_seeds

from .graph import append_nodes, build_graph
from .model import build_model

MODEL_KEYS = ("hidden1", "hidden2", "dropout")


def _val_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over targets of RMSE, in standardised units."""
    return float(((pred - target) ** 2).mean(dim=0).sqrt().mean())


def run_one(cfg: Dict[str, Any], seed: int, out_dir, data: Optional[SplitData] = None,
            plots: bool = True, verbose: bool = True,
            split_seed: Optional[int] = None) -> Dict[str, Any]:
    """Train one seed and write its outputs. Returns the metrics dict.

    Pass `data` to reuse a prepared dataset (e.g. during tuning); it must have been
    built with the same seed. `split_seed` overrides the first data.split_seed.
    """
    out_dir = ensure_dir(out_dir)
    set_seed(seed)
    device = get_device()
    tcfg, mcfg, k = cfg["training"], cfg["model"], int(cfg["graph"]["k"])
    if data is None:
        data = prepare_dataset(cfg, seed, split_seed)
    used_split_seed = int(split_seed if split_seed is not None else split_seeds(cfg)[0])

    n_tr, n_va = len(data.X_train), len(data.X_val)
    ref_X = np.vstack([data.X_train, data.X_val])
    ref_Y = np.vstack([data.Y_train, data.Y_val])
    graph = build_graph(ref_X, ref_Y, k).to(device)
    online = data.augmenter is not None and data.X_train_raw is not None
    ref_raw = np.vstack([data.X_train_raw, data.X_val_raw]) if online else None
    train_mask = torch.zeros(n_tr + n_va, dtype=torch.bool, device=device)
    train_mask[:n_tr] = True
    val_mask = ~train_mask

    model_kwargs = {key: mcfg[key] for key in MODEL_KEYS if key in mcfg}
    model = build_model(mcfg.get("name", "gcn"), ref_X.shape[1], ref_Y.shape[1], **model_kwargs).to(device)
    lr = float(tcfg["lr"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(tcfg["weight_decay"]))
    warmup = int(tcfg.get("warmup_epochs", 50))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=int(tcfg.get("plateau_patience", 100)))
    loss_fn = torch.nn.MSELoss()

    best_val, best_epoch, best_state, bad = float("inf"), 0, None, 0
    hist = {"train_loss": [], "val_loss": []}
    epochs, patience = int(tcfg["epochs"]), int(tcfg["early_stopping_patience"])
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if epoch <= warmup:  # linear warm-up from lr*0.01 to lr
            for g in opt.param_groups:
                g["lr"] = lr * (0.01 + 0.99 * (epoch - 1) / max(warmup - 1, 1))
        model.train()
        opt.zero_grad()
        if online:  # fresh jitter every epoch; edges stay those of the clean graph
            x_in = torch.as_tensor(data.augmenter.jitter(ref_raw), dtype=torch.float, device=device)
        else:
            x_in = graph.x
        out = model(x_in, graph.edge_index)
        loss = loss_fn(out[train_mask], graph.y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
            val_loss = float(loss_fn(out[val_mask], graph.y[val_mask]))
            val_rmse = _val_rmse(out[val_mask], graph.y[val_mask])
        if epoch > warmup:
            sched.step(val_rmse)
        hist["train_loss"].append(loss.item())
        hist["val_loss"].append(val_loss)

        if val_rmse < best_val:
            best_val, best_epoch, bad = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and (epoch % 100 == 0 or epoch == 1):
            print(f"[seed {seed}] epoch {epoch:4d} train_mse={loss.item():.4f} "
                  f"val_rmse={val_rmse:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if bad >= patience:
            if verbose:
                print(f"[seed {seed}] early stop at epoch {epoch} (best {best_epoch}, val_rmse={best_val:.4f})")
            break
    if verbose:
        print(f"[seed {seed}] done in {time.time() - t0:.0f}s, best epoch {best_epoch}, val_rmse={best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    inv = data.y_scaler.inverse_transform
    with torch.no_grad():
        pred_ref = model(graph.x, graph.edge_index).cpu().numpy()
        test_graph, new_idx = append_nodes(ref_X, ref_Y, data.X_test, k)
        test_graph = test_graph.to(device)
        pred_test = model(test_graph.x, test_graph.edge_index)[new_idx.start:new_idx.stop].cpu().numpy()

    splits = {
        "train": (inv(data.Y_train), inv(pred_ref[:n_tr])),
        "val": (inv(data.Y_val), inv(pred_ref[n_tr:])),
        "test": (inv(data.Y_test), inv(pred_test)),
    }
    metrics = save_results(splits, data.target_cols, out_dir, meta={"test": data.meta_test},
                           verbose=verbose, plot=plots)
    metrics["val_rmse_scaled"] = best_val
    metrics["best_epoch"] = best_epoch
    metrics["split_seed"] = used_split_seed

    group_col = (cfg.get("evaluation") or {}).get("group_by", "Day")
    if group_col in data.meta_test.columns:
        by_group = metrics_by_group(splits["test"][0], splits["test"][1],
                                    data.meta_test[group_col].to_numpy(), data.target_cols)
        with open(out_dir / f"metrics_test_by_{group_col}.json", "w") as f:
            json.dump(by_group, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if plots:
        save_loss_curve(hist, out_dir)

    torch.save({
        "state_dict": {key: v.cpu() for key, v in model.state_dict().items()},
        "model_name": mcfg.get("name", "gcn"),
        "model_kwargs": {"in_dim": ref_X.shape[1], "out_dim": ref_Y.shape[1], **model_kwargs},
        "reference_X": ref_X, "reference_Y": ref_Y,
        "preprocessor": data.preprocessor, "y_scaler": data.y_scaler,
        "k": k, "target_cols": data.target_cols, "spectral_cols": data.spectral_cols,
        "config": cfg, "seed": seed,
    }, out_dir / "model.pt")
    return metrics


def run_seeds(cfg: Dict[str, Any], out_dir, plots: bool = True) -> Dict[str, Any]:
    """Run every split seed x training seed; write summary.json (mean/std over all runs).

    With one split seed the folders are seed_<s>/; with several they are split_<ss>/seed_<s>/.
    summary.json also lists "split_seeds", "seeds" and a "per_split" block of per-split means.
    """
    out_dir = ensure_dir(out_dir)
    sseeds, seeds = split_seeds(cfg), [int(s) for s in cfg.get("seeds", [42])]
    runs_by_split: Dict[int, List[dict]] = {}
    for ss in sseeds:
        runs_by_split[ss] = []
        for seed in seeds:
            sub = out_dir / f"seed_{seed}" if len(sseeds) == 1 else out_dir / f"split_{ss}" / f"seed_{seed}"
            m = run_one(cfg, seed, sub, plots=plots, split_seed=ss)
            runs_by_split[ss].append({s: m[s] for s in ("train", "val", "test")})
    summary = summarize_runs(runs_by_split, seeds)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default="plant_gcn/config.yaml")
    ap.add_argument("--data", default=None, help="override data.path")
    ap.add_argument("--out", default=None, help="override output_dir")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a config value, e.g. --set training.epochs=600 (repeatable)")


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config, args.overrides)
    extra = []
    if args.data:
        extra.append(f"data.path={json.dumps(args.data)}")
    if args.out:
        extra.append(f"output_dir={json.dumps(args.out)}")
    cfg = apply_overrides(cfg, extra)
    if getattr(args, "repeat_splits", None):
        cfg = repeat_splits(cfg, args.repeat_splits)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    ap.add_argument("--repeat-splits", type=int, default=None, metavar="N",
                    help="evaluate on N test splits (first N of 42, 7, 123, 2024, 99, 11, 3, 77, 500, 2025)")
    args = ap.parse_args()
    cfg = config_from_args(args)
    out_dir = Path(cfg["output_dir"])
    summary = run_seeds(cfg, out_dir)
    print(f"[done] {out_dir / 'summary.json'}")
    for tgt, m in summary["test"].items():
        print(f"test {tgt:8s} R2={m['R2']['mean']:.4f}+-{m['R2']['std']:.4f} "
              f"RMSE={m['RMSE']['mean']:.4f}+-{m['RMSE']['std']:.4f}")


if __name__ == "__main__":
    main()
