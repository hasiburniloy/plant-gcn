"""Table loading and spectral-column selection.

Headers are always normalised to str so Excel (int headers) and CSV (str headers)
behave identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd


def _norm_header(h) -> str:
    if isinstance(h, float) and h.is_integer():
        h = int(h)
    return str(h).strip()


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read .xlsx/.xls/.csv into a DataFrame with str column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [_norm_header(c) for c in df.columns]
    return df


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def select_spectral_columns(columns: Sequence[str],
                            spec: Union[str, Sequence, None] = "auto",
                            wavelength_range: Optional[Sequence[float]] = None,
                            exclude: Sequence[str] = ()) -> list:
    """Return the list of spectral column names.

    spec:
      "auto"            every column whose header parses as a float (minus `exclude`)
      [start, end]      two ints -> positional slice columns[start:end]
      [name, name, ...] explicit names
    wavelength_range [lo, hi] then keeps only columns with lo <= wavelength <= hi.
    """
    columns = [str(c) for c in columns]
    if spec is None or spec == "auto":
        cols = [c for c in columns if c not in exclude and _is_float(c)]
    elif isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(s, int) for s in spec):
        cols = columns[spec[0]:spec[1]]
    elif isinstance(spec, (list, tuple)):
        cols = [str(c) for c in spec]
        missing = [c for c in cols if c not in columns]
        if missing:
            raise ValueError(f"Spectral columns not found: {missing[:5]}")
    else:
        raise ValueError(f"Unsupported spectral_columns value: {spec!r}")
    if wavelength_range:
        lo, hi = wavelength_range
        cols = [c for c in cols if _is_float(c) and lo <= float(c) <= hi]
    if not cols:
        raise ValueError("No spectral columns selected; check spectral_columns / wavelength_range.")
    return cols
