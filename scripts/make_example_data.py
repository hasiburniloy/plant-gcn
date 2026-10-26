"""Write a small synthetic dataset in the expected layout (180 bands, 514..1000 nm).

For checking the pipeline end to end without real data:

    python scripts/make_example_data.py --out example_data.csv
"""

import argparse

import numpy as np
import pandas as pd


def make_example(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wl = np.linspace(514, 1000, 180).round().astype(int)
    treatments = rng.choice(["Well watered", "Mild stress", "Extreme stress"], size=n)
    day = rng.choice([f"Day{i}" for i in range(1, 6)], size=n)
    stress = pd.Series(treatments).map({"Well watered": 0, "Mild stress": 1, "Extreme stress": 2}).to_numpy()
    day_n = np.array([int(d[-1]) for d in day])

    rwc = 85 - 6 * stress * day_n / 5 + rng.normal(0, 2, n)
    traits = pd.DataFrame({
        "RWC": rwc,
        "ETR": 100 + 2 * (rwc - 70) + rng.normal(0, 5, n),
        "gsw": 0.2 + 0.006 * (rwc - 70) + rng.normal(0, 0.02, n),
        "PhiPS2": 0.3 + 0.004 * (rwc - 70) + rng.normal(0, 0.02, n),
        "Emm": 2 + 0.05 * (rwc - 70) + rng.normal(0, 0.2, n),
    })

    # vegetation-like curve: low visible, green bump, red edge, NIR plateau, water dip near 970 nm
    base = 0.05 + 0.05 * np.exp(-((wl - 550) / 30) ** 2) + 0.4 / (1 + np.exp(-(wl - 715) / 12))
    water = np.exp(-((wl - 970) / 25) ** 2)
    spectra = (base[None] * rng.uniform(0.9, 1.1, (n, 1))
               - 0.002 * (rwc[:, None] - 70) * water[None]
               + rng.normal(0, 0.003, (n, len(wl))))
    spectra = np.clip(spectra, 0.01, 1.0)

    meta = pd.DataFrame({"Day": day, "PlantID": np.arange(n),
                         "Genotype": rng.choice(["G1", "G2", "G3"], size=n), "Treatment": treatments})
    return pd.concat([meta, traits, pd.DataFrame(spectra, columns=[str(w) for w in wl])], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="example_data.csv")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    df = make_example(args.n)
    if args.out.endswith(".xlsx"):
        df.to_excel(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {df.shape[0]} rows, {df.shape[1] - 9} bands")


if __name__ == "__main__":
    main()
