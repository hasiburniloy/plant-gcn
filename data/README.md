# Data

Data files are not part of this repository. Place your own table here (or anywhere and pass `--data <file>`).

Expected layout, one row per plant, `.xlsx` or `.csv`:

| Day | PlantID | Genotype | Treatment | RWC | ... | 514 | 516 | ... | 1000 |
|---|---|---|---|---|---|---|---|---|---|
| Day1 | P001 | Star | WW | 71.2 | ... | 0.031 | 0.032 | ... | 0.612 |

- Reflectance columns: headers are wavelengths in nm (numeric), values in (0, 1]. They are detected automatically; `data.spectral_columns` / `data.wavelength_range` in the config can restrict them.
- Target columns: named in `data.target_columns` of the config.
- Metadata columns: anything else. `data.stratify_columns` (default `[Treatment, Day]`) must exist, or be set to `[]`.
- No missing values in the target or spectral columns (rows with NaN are dropped).
