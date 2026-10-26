"""Spectral preprocessing (paper section 2.4), fitted on the training split only.

    absorbance  = log10(1 / reflectance)
    MSC         : regress each spectrum on the training mean spectrum,
                  corrected = (x - intercept) / slope
    MeanCenter  : subtract the training mean spectrum

Adapted from the lab's preprocessing.py (Tanzeel, 2020); same math on numpy arrays.
All classes are plain Python and picklable.
"""

from __future__ import annotations

import numpy as np


def trans2absr(reflectance) -> np.ndarray:
    """Reflectance -> pseudo-absorbance, log10(1/R)."""
    return np.log10(1.0 / np.asarray(reflectance, dtype=float))


class MSC:
    """Multiplicative Scatter Correction. Fit on training data, apply to anything."""

    def __init__(self) -> None:
        self.reference = None

    def _correct(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.reference.shape[0]:
            raise ValueError(f"Expected {self.reference.shape[0]} bands, got {X.shape[1]}")
        r = self.reference - self.reference.mean()
        # closed-form least squares of each row on the reference: x = slope * ref + intercept
        slope = (X - X.mean(axis=1, keepdims=True)) @ r / (r @ r)
        intercept = X.mean(axis=1) - slope * self.reference.mean()
        slope = np.where(np.abs(slope) < 1e-12, 1e-12, slope)
        return (X - intercept[:, None]) / slope[:, None]

    def fit_transform(self, X) -> np.ndarray:
        self.reference = np.asarray(X, dtype=float).mean(axis=0)
        return self._correct(X)

    def transform(self, X) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("MSC is not fitted. Call fit_transform on training data first.")
        return self._correct(X)


class MeanCenter:
    """Subtract the training mean spectrum."""

    def __init__(self) -> None:
        self.reference = None

    def fit_transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.reference = X.mean(axis=0)
        return X - self.reference

    def transform(self, X) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("MeanCenter is not fitted. Call fit_transform on training data first.")
        return np.asarray(X, dtype=float) - self.reference


class SpectralPreprocessor:
    """log10(1/R) -> MSC -> mean centring, with fit/transform semantics."""

    def __init__(self, absorbance: bool = True, msc: bool = True, mean_center: bool = True) -> None:
        self.absorbance, self.use_msc, self.use_mean_center = absorbance, msc, mean_center
        self.msc = MSC()
        self.centre = MeanCenter()
        self.n_bands = None

    def fit(self, X) -> "SpectralPreprocessor":
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.n_bands = X.shape[1]
        if self.absorbance:
            X = trans2absr(X)
        if self.use_msc:
            X = self.msc.fit_transform(X)
        if self.use_mean_center:
            X = self.centre.fit_transform(X)
        return X

    def transform(self, X) -> np.ndarray:
        if self.n_bands is None:
            raise RuntimeError("SpectralPreprocessor is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} bands, got {X.shape[1]}")
        if self.absorbance:
            X = trans2absr(X)
        if self.use_msc:
            X = self.msc.transform(X)
        if self.use_mean_center:
            X = self.centre.transform(X)
        return X
