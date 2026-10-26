import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_example_data import make_example  # noqa: E402

from plant_gcn import predict, train  # noqa: E402
from plant_gcn.common.preprocessing import SpectralPreprocessor  # noqa: E402
from plant_gcn.common.utils import apply_overrides, load_config  # noqa: E402

PKG_CONFIG = Path(train.__file__).resolve().parent / "config.yaml"


def _csv(tmp_path):
    path = tmp_path / "example.csv"
    make_example(150).to_csv(path, index=False)
    return path


def _train(tmp_path, csv):
    cfg = load_config(PKG_CONFIG)
    cfg = apply_overrides(cfg, [
        f"data.path={csv}", f"output_dir={tmp_path / 'run'}", "seeds=[0]", "training.epochs=3", "training.warmup_epochs=1",
    ])
    return train.run_seeds(cfg, tmp_path / "run", plots=False)


def test_train_writes_outputs(tmp_path):
    csv = _csv(tmp_path)
    _train(tmp_path, csv)
    run = tmp_path / "run"
    assert (run / "summary.json").exists()
    assert (run / "seed_0" / "model.pt").exists()
    metrics = json.loads((run / "seed_0" / "metrics.json").read_text())
    assert np.isfinite(metrics["test"]["RWC"]["RMSE"])


def test_predict_cli(tmp_path, monkeypatch):
    csv = _csv(tmp_path)
    _train(tmp_path, csv)
    out = tmp_path / "pred.csv"
    monkeypatch.setattr(sys, "argv", ["predict", "--model", str(tmp_path / "run" / "seed_0" / "model.pt"),
                                      "--data", str(csv), "--out", str(out)])
    predict.main()
    pred = pd.read_csv(out)
    assert len(pred) == 150
    assert "RWC_predicted" in pred.columns


def test_preprocessor_shape_and_finite():
    X = np.random.default_rng(0).uniform(0.01, 1.0, (20, 180))
    pre = SpectralPreprocessor().fit(X)
    Z = pre.transform(X)
    assert Z.shape == X.shape
    assert np.isfinite(Z).all()
