"""SMOTE-ENN balancing of low-value samples (used by the transformer model).

The class label is Y[:, target_index] < threshold. SMOTEENN runs on the joint matrix
[X | Y] so synthetic targets are interpolated consistently with their spectra.
Apply to train-val only, never to the test set.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE


def smote_enn(X: np.ndarray, Y: np.ndarray, threshold: float, target_index: int = 0,
              seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_res, Y_res, source_group).

    source_group is the original row index for retained real rows and a fresh unique id
    (>= len(X)) for every synthetic row.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    label = (Y[:, target_index] < threshold).astype(int)
    n_min = int(min(label.sum(), (1 - label).sum()))
    if n_min < 2:
        raise ValueError(f"Too few samples in one class (min={n_min}) for SMOTE-ENN; check threshold.")
    joint = np.hstack([X, Y])
    sampler = SMOTEENN(random_state=seed, smote=SMOTE(random_state=seed, k_neighbors=min(5, n_min - 1)))
    joint_res, _ = sampler.fit_resample(joint, label)
    X_res, Y_res = joint_res[:, :X.shape[1]], joint_res[:, X.shape[1]:]

    # real rows come back bit-identical, so match them by content to recover their index
    lookup = {row.tobytes(): i for i, row in enumerate(joint)}
    groups = np.empty(len(joint_res), dtype=int)
    next_id = len(X)
    for i, row in enumerate(joint_res):
        j = lookup.get(row.tobytes())
        if j is None:
            groups[i] = next_id
            next_id += 1
        else:
            groups[i] = j
    return X_res, Y_res, groups
