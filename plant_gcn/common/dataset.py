"""Data preparation shared by both models: `prepare_dataset(cfg) -> SplitData`.

Order of operations follows the Plant-GCN paper (Rahman et al. 2025, section 2.5):

 1. load, drop NaN rows, optional row filter
 2. pick spectral columns
 3. test split FIRST (stratified); test rows are never balanced or augmented
 4. optional SMOTE-ENN balancing (train-val only)
 5. offset/slope augmentation (train-val only), targets copied not jittered.
    augmentation.mode "static" (default here) makes the copies now; "online" makes none
    and returns raw arrays plus an OnlineAugmenter so the trainer jitters on the fly
 6. train/val split of the augmented set; group-aware is our fix for copy leakage
    (set split.group_aware=false to reproduce the paper's plain random split)
 7. preprocessing fitted on train only: log10(1/R) -> MSC -> mean centring
 8. targets standardised on train only; metrics are reported after inverse_transform

All Y arrays in SplitData are in standardised units; use `y_scaler.inverse_transform`
to get original units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .augment import OnlineAugmenter, augment_spectra
from .balance import smote_enn
from .io import load_table, select_spectral_columns
from .preprocessing import SpectralPreprocessor
from .split import group_split, make_strata, stratified_holdout
from .utils import split_seeds


@dataclass
class SplitData:
    X_train: np.ndarray
    Y_train: np.ndarray
    groups_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    groups_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    meta_test: pd.DataFrame
    preprocessor: SpectralPreprocessor
    y_scaler: StandardScaler
    target_cols: List[str]
    spectral_cols: List[str]
    stratified: bool = True
    X_train_raw: Optional[np.ndarray] = None   # raw reflectance, online mode only
    X_val_raw: Optional[np.ndarray] = None
    augmenter: Optional[OnlineAugmenter] = None

    @property
    def wavelengths(self) -> np.ndarray:
        return np.array([float(c) for c in self.spectral_cols])

    def summary(self) -> str:
        s = (f"[data] train={len(self.X_train)} val={len(self.X_val)} test={len(self.X_test)} "
             f"bands={self.X_train.shape[1]} targets={len(self.target_cols)} "
             f"test_stratified={self.stratified}")
        print(s)
        return s


def _load_clean(dcfg: Dict[str, Any], path: Optional[str] = None) -> pd.DataFrame:
    df = load_table(path or dcfg["path"])
    flt = dcfg.get("filter")
    if flt and flt.get("column"):
        if flt["column"] not in df.columns:
            raise ValueError(f"filter column {flt['column']!r} not in data")
        df = df[df[flt["column"]].astype(str).isin([str(v) for v in flt["values"]])]
    return df


def prepare_dataset(cfg: Dict[str, Any], seed: Optional[int] = None,
                    split_seed: Optional[int] = None) -> SplitData:
    """Build train/val/test arrays from a config dict (see the model config.yaml).

    `split_seed` overrides cfg["data"]["split_seed"] (an int or a list; the first entry
    is used when no override is given).
    """
    dcfg = cfg["data"]
    if seed is None:
        seed = int((cfg.get("seeds") or [42])[0])
    targets = list(dcfg["target_columns"])

    df = _load_clean(dcfg)
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise ValueError(f"Target columns not found: {missing}")
    spectral = select_spectral_columns(df.columns, dcfg.get("spectral_columns", "auto"),
                                       dcfg.get("wavelength_range"), exclude=targets)
    df = df.dropna(subset=spectral + targets).reset_index(drop=True)
    X_all = df[spectral].to_numpy(dtype=float)
    Y_all = df[targets].to_numpy(dtype=float)
    meta = df.drop(columns=spectral + targets)

    # test split before anything else (paper section 2.5); fixed seed so all runs share one test set
    strata = make_strata(df, dcfg.get("stratify_columns", []))
    tv_idx, te_idx, stratified = stratified_holdout(
        len(df), strata, float(dcfg.get("test_size", 0.15)), int(split_seed if split_seed is not None else split_seeds(cfg)[0]))
    X_tv, Y_tv = X_all[tv_idx], Y_all[tv_idx]
    X_test, Y_test = X_all[te_idx], Y_all[te_idx]
    meta_test = meta.iloc[te_idx].reset_index(drop=True)

    groups = np.arange(len(X_tv))
    bcfg = cfg.get("balance") or {}
    if bcfg.get("enabled", False):
        t_idx = targets.index(bcfg.get("target", targets[0]))
        X_tv, Y_tv, groups = smote_enn(X_tv, Y_tv, float(bcfg["threshold"]), t_idx, seed)

    acfg = cfg.get("augmentation") or {}
    mode = str(acfg.get("mode", "static")).lower()
    if mode not in ("static", "online"):
        raise ValueError(f"augmentation.mode must be 'static' or 'online', got {mode!r}")
    online = bool(acfg.get("enabled", True)) and mode == "online"
    raw_std = float(X_tv.std())  # same definition as augment_spectra uses
    if acfg.get("enabled", True) and not online:
        X_tv, Y_tv, groups = augment_spectra(
            X_tv, Y_tv, int(acfg.get("n_copies", 5)), float(acfg.get("offset_percent", 0.06)),
            float(acfg.get("slope_variation", 0.05)), bool(acfg.get("include_original", True)),
            groups, seed)

    scfg = cfg.get("split") or {}
    tr, va = group_split(groups, float(scfg.get("val_size", 0.3)), seed,
                         bool(scfg.get("group_aware", True)))

    pre = SpectralPreprocessor()
    X_train = pre.fit_transform(X_tv[tr])
    X_val, X_test = pre.transform(X_tv[va]), pre.transform(X_test)

    augmenter = X_train_raw = X_val_raw = None
    if online:
        X_train_raw, X_val_raw = X_tv[tr], X_tv[va]
        augmenter = OnlineAugmenter(raw_std, float(acfg.get("offset_percent", 0.06)),
                                    float(acfg.get("slope_variation", 0.05)), seed, pre)

    y_scaler = StandardScaler().fit(Y_tv[tr])
    data = SplitData(
        X_train=X_train, Y_train=y_scaler.transform(Y_tv[tr]), groups_train=groups[tr],
        X_val=X_val, Y_val=y_scaler.transform(Y_tv[va]), groups_val=groups[va],
        X_test=X_test, Y_test=y_scaler.transform(Y_test), meta_test=meta_test,
        preprocessor=pre, y_scaler=y_scaler, target_cols=targets, spectral_cols=spectral,
        stratified=stratified, X_train_raw=X_train_raw, X_val_raw=X_val_raw, augmenter=augmenter)
    data.summary()
    return data
