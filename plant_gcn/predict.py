"""Predict with a trained Plant-GCN.

    python -m plant_gcn.predict --model runs/plant_gcn/seed_42/model.pt --data new.xlsx --out preds.csv

New spectra are preprocessed with the saved preprocessor, appended to the trained
train+val graph (k-NN rebuilt), and the new nodes' outputs are read off in original
units. If the target columns are present, R2/RMSE are printed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common.io import load_table
from common.metrics import score

from .graph import append_nodes
from .model import build_model


def predict_table(ckpt: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return df's non-spectral columns plus <target>_predicted columns."""
    spectral, targets = ckpt["spectral_cols"], ckpt["target_cols"]
    missing = [c for c in spectral if c not in df.columns]
    if missing:
        raise ValueError(f"{len(missing)} spectral columns used in training are missing from the data "
                         f"(e.g. {missing[:5]}). Expected {spectral[0]}..{spectral[-1]} ({len(spectral)} bands).")
    X = df[spectral].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("New data contains NaN in spectral columns.")
    X = ckpt["preprocessor"].transform(X)

    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    graph, new_idx = append_nodes(ckpt["reference_X"], ckpt["reference_Y"], X, ckpt["k"])
    with torch.no_grad():
        out = model(graph.x, graph.edge_index)[new_idx.start:new_idx.stop].numpy()
    pred = ckpt["y_scaler"].inverse_transform(out)

    result = df.drop(columns=spectral).copy()
    for j, tgt in enumerate(targets):
        result[f"{tgt}_predicted"] = pred[:, j]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="model.pt written by plant_gcn.train")
    ap.add_argument("--data", required=True, help="xlsx/csv with the same spectral columns")
    ap.add_argument("--out", required=True, help="output predictions csv")
    args = ap.parse_args()

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    df = load_table(args.data)
    result = predict_table(ckpt, df)
    result.to_csv(args.out, index=False)
    print(f"[predict] wrote {args.out}: {len(result)} rows")

    targets = ckpt["target_cols"]
    if all(t in df.columns for t in targets):
        ok = df[targets].notna().all(axis=1).to_numpy()
        y_true = df.loc[ok, targets].to_numpy(dtype=float)
        y_pred = result.loc[ok, [f"{t}_predicted" for t in targets]].to_numpy()
        for tgt, m in score(y_true, y_pred, targets).items():
            print(f"{tgt:8s} R2={m['R2']:.4f} RMSE={m['RMSE']:.4f}")


if __name__ == "__main__":
    main()
