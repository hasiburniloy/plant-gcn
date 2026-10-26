"""Train/test and train/val splitting.

The test split is stratified and happens before any balancing or augmentation (paper
section 2.5). The train/val split is group-aware: our fix for augmented-copy leakage,
where copies of one spectrum could otherwise land in both train and val.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, train_test_split


def make_strata(df: pd.DataFrame, columns: Sequence[str]) -> Optional[np.ndarray]:
    """Join the given columns into one string key per row (None if no column exists)."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy()


def stratified_holdout(n: int, strata: Optional[np.ndarray], test_size: float,
                       seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (train_idx, test_idx, stratified). Falls back to unstratified if a stratum is too small."""
    idx = np.arange(n)
    if strata is not None:
        _, counts = np.unique(strata, return_counts=True)
        n_test = int(np.ceil(test_size * n))
        if counts.min() >= 2 and n_test >= len(counts) and n - n_test >= len(counts):
            tr, te = train_test_split(idx, test_size=test_size, stratify=strata, random_state=seed)
            return np.sort(tr), np.sort(te), True
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return np.sort(tr), np.sort(te), False


def group_split(groups: np.ndarray, val_size: float, seed: int,
                group_aware: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx). Group-aware: no group appears on both sides."""
    dummy = np.zeros(len(groups))
    if group_aware:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr, va = next(splitter.split(dummy, groups=groups))
    else:  # plain random split, as in the paper
        splitter = ShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr, va = next(splitter.split(dummy))
    return tr, va
