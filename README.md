# Plant-GCN

Graph convolutional network that predicts blueberry physiological traits (relative water content RWC, transpiration Emm, stomatal conductance gsw, PhiPS2, ETR) from canopy hyperspectral reflectance (514–1000 nm).

Code for: Rahman, M. H., Busby, S., Hanif, S., Maruf, M. M., Ahmad, F., Ru, S., Sanz-Saez, A., Zheng, J., Rehman, T. U. (2025). *A graph convolutional network approach for hyperspectral image analysis of blueberries physiological traits under drought stress*. **Smart Agricultural Technology** 10:100743, doi:[10.1016/j.atech.2024.100743](https://doi.org/10.1016/j.atech.2024.100743).

No data is included; see [Data format](#data-format).

## How it works

Each preprocessed spectrum is a node; edges link it to its k nearest neighbours in spectral space (undirected, k = 3). A two-layer GCN (64, 32) with dropout 0.3 and a linear head regresses the five traits at once. Training is full-batch and transductive: all train + val spectra form one graph, the loss uses train nodes only. Test spectra are appended to that graph after training and read at their new positions; their labels are never used.

Data pipeline (`common/`):

1. Stratified 85:15 test split, made first and never augmented.
2. Offset/slope augmentation of the train-val spectra (offset 6 % of the band std, slope ±5 %).
3. Group-aware 70:30 train/val split (copies of one plant stay on one side).
4. log10(1/R) → multiplicative scatter correction → mean-centring, fitted on the training split only.
5. Targets standardised with the training mean and std; metrics are reported in original units.

`augmentation.mode: online` (default) builds the graph over the real plants and redraws the jitter on the node features every epoch, with edges fixed from the clean features. `static` draws `n_copies` fixed copies per plant and makes all of them nodes (the paper's protocol).

```
spectra (N x 180) -> log10(1/R), MSC, mean-centre -> k-NN graph -> GCN 64 -> GCN 32 -> dropout -> linear -> 5 traits
```

## Install

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric may need a platform-specific install; see [pytorch.org](https://pytorch.org) and [pyg.org](https://pyg.org).

## Data format

One row per plant, `.xlsx` or `.csv`: metadata columns (e.g. `Day`, `PlantID`, `Genotype`, `Treatment`), the target columns, then reflectance columns whose headers are wavelengths in nm (values in (0, 1]). Spectral columns are detected automatically as the numeric-named columns; `data.spectral_columns` and `data.wavelength_range` in `config.yaml` can restrict them. If your table has no `Treatment`/`Day` columns, set `data.stratify_columns: []`. Details in `data/README.md`.

## Train

Run from the repo root.

```bash
python -m plant_gcn.train --config plant_gcn/config.yaml --data path/to/data.xlsx --out runs/plant_gcn
```

Any config value can be overridden: `--set graph.k=5 --set training.epochs=2000 --set seeds=[1,2,3]`.

Test sets are small, so a single split is noisy. `--repeat-splits 5` runs the first 5 of the split seeds `[42, 7, 123, 2024, 99, 11, 3, 77, 500, 2025]` (or set `data.split_seed` to a list) and reports mean ± std over splits:

```bash
python -m plant_gcn.train --config plant_gcn/config.yaml --data path/to/data.xlsx --out runs/plant_gcn --repeat-splits 5
```

Outputs per run (`runs/plant_gcn/seed_<s>/`, or `split_<ss>/seed_<s>/` with several split seeds): `model.pt` (weights, fitted preprocessing, reference graph nodes), `metrics.json` (train / val / test R2 and RMSE per target, original units), `predictions_{train,val,test}.csv`, `metrics_test_by_Day.json`, `parity_plot.png`, `loss_curve.png`. `summary.json` has mean and std across all runs and a `per_split` block.

## Predict

```bash
python -m plant_gcn.predict --model runs/plant_gcn/seed_42/model.pt --data new_spectra.xlsx --out predictions.csv
```

The new file needs the same spectral columns as training. New spectra are preprocessed with the training statistics stored in `model.pt`, appended to the stored reference graph, and predicted. Output has the non-spectral columns plus `<target>_predicted`; R2/RMSE are printed if the targets are present.

## Configuration

All keys are documented in `plant_gcn/config.yaml`. Main groups: `data` (path, targets, stratification, test size, split seed), `augmentation` (mode, n_copies, offset_percent, slope_variation), `split` (val fraction, group_aware), `graph` (k, metric), `model` (hidden sizes, dropout), `training` (lr, weight_decay, warm-up, scheduler, epochs, early stopping), `seeds`.

## Protocol notes

Follows the paper's sections 2.5–2.6 with two deliberate differences: the train/val split is group-aware (`split.group_aware: false` restores the paper's random node split) and augmentation copies the targets unchanged. Augmented reflectance is floored at 1e-4 so log10(1/R) stays finite.

## Citation

```bibtex
@article{rahman2025plantgcn,
  title   = {A graph convolutional network approach for hyperspectral image analysis of blueberries physiological traits under drought stress},
  author  = {Rahman, Md. Hasibur and Busby, Savannah and Hanif, Sajid and Maruf, Md Mesbahul and Ahmad, Faraz and Ru, Sushan and Sanz-Saez, Alvaro and Zheng, Jingyi and Rehman, Tanzeel U.},
  journal = {Smart Agricultural Technology},
  volume  = {10},
  pages   = {100743},
  year    = {2025},
  doi     = {10.1016/j.atech.2024.100743}
}
```

## License

MIT, see `LICENSE`.
