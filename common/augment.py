"""Spectral offset/slope augmentation (paper section 2.5).

x' = x * slope + offset, slope ~ U(1-s, 1+s), offset ~ U(-o*std, +o*std), where std is
the scalar standard deviation of the whole training reflectance matrix. Targets are
copied unchanged (the original notebook jittered them by accident; not reproduced).
Each copy keeps the group id of its source spectrum so a group-aware split can keep
copies of one plant on the same side.

Two ways to use the same jitter: `augment_spectra` makes static copies once (mode
"static"); `OnlineAugmenter` draws a fresh jitter on every call (mode "online").
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def jitter_reflectance(X: np.ndarray, rng: np.random.Generator, std: float,
                       offset_percent: float = 0.06, slope_variation: float = 0.05,
                       min_reflectance: float = 1e-4) -> np.ndarray:
    """One random slope and offset per spectrum: x' = x * slope + offset (floored at min_reflectance)."""
    X = np.asarray(X, dtype=float)
    slope = rng.uniform(1 - slope_variation, 1 + slope_variation, size=(len(X), 1))
    offset = rng.uniform(-offset_percent * std, offset_percent * std, size=(len(X), 1))
    # keep reflectance positive so log10(1/R) stays defined for dark pixels
    return np.maximum(X * slope + offset, min_reflectance)


class OnlineAugmenter:
    """Fresh offset/slope jitter on every call, followed by the fitted preprocessing.

    `std` is the scalar std of the raw train-val reflectance matrix (as in static mode).
    """

    def __init__(self, std: float, offset_percent: float, slope_variation: float, seed: int,
                 preprocessor, min_reflectance: float = 1e-4):
        self.std = float(std)
        self.offset_percent = float(offset_percent)
        self.slope_variation = float(slope_variation)
        self.min_reflectance = float(min_reflectance)
        self.preprocessor = preprocessor
        self.rng = np.random.default_rng(seed)

    def jitter(self, X_raw: np.ndarray) -> np.ndarray:
        """Jitter raw reflectance, then return the preprocessed array (same shape)."""
        Xj = jitter_reflectance(X_raw, self.rng, self.std, self.offset_percent,
                                self.slope_variation, self.min_reflectance)
        return self.preprocessor.transform(Xj)


def augment_spectra(X: np.ndarray, Y: np.ndarray, n_copies: int = 5,
                    offset_percent: float = 0.06, slope_variation: float = 0.05,
                    include_original: bool = True, groups: Optional[np.ndarray] = None,
                    seed: int = 0, min_reflectance: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_aug, Y_aug, groups). Group = id of the source spectrum."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    groups = np.arange(len(X)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    std = X.std()
    Xs, Ys, Gs = ([X], [Y], [groups]) if include_original else ([], [], [])
    for _ in range(n_copies):
        Xs.append(jitter_reflectance(X, rng, std, offset_percent, slope_variation, min_reflectance))
        Ys.append(Y.copy())
        Gs.append(groups.copy())
    if not Xs:
        raise ValueError("n_copies=0 with include_original=False leaves no data.")
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(Gs)
