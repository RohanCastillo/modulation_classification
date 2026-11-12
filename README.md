# Modulation Classifier

Given 1,024 raw radio samples, predict which of 26 modulation types produced
them, and measure how accuracy falls as the signal gets noisier. Two models
are compared: a cumulant-feature baseline with gradient boosting (planned)
and a small 1D CNN.

## Results

TODO: accuracy-vs-SNR chart and results table (26 modulations and 5 families).

## Data

[HisarMod2019.1](https://ieee-dataport.org/node/1785) (IEEE DataPort, free
account required): 780,000 simulated signals, each 1,024 complex I/Q samples,
at SNRs from -20 to +18 dB in 2 dB steps.

- 26 modulations in 5 families: PSK, QAM, FSK, PAM, and analog.
- Each signal passed through one of 5 unlabeled channels (ideal, static,
  Rayleigh, Rician, Nakagami-m).
- Authors' split: 520,000 train and 260,000 test signals (1,000 and 500 per
  modulation/SNR pair). Their train set is split 80/20 into train/validation
  here; their test set is held out for one final evaluation.
- Zip SHA-256:
  `9276c4bc445b6754ce770e997cf3c2f16a581db090ff0829286c5884c2ed79e0`

Citation: K. Tekbıyık, C. Keçeci, A. R. Ekti, A. Görçin, and G. Karabulut
Kurt, "HisarMod: A new challenging modulated signals dataset," IEEE Dataport,
2019. doi:10.21227/8k12-2g70

## How it works

1. **Convert** (`scripts/convert_hisarmod.py`, run once on Amarel): turns the
   MATLAB-style CSVs in `data/raw/HisarMod2019.1/` into float32 `.npy`
   arrays of shape `(n, 2, 1024)` plus label and SNR arrays in
   `data/processed/`.
2. **Prepare** ([modulation_classification.ipynb](modulation_classification.ipynb)):
   memory-map the arrays, normalize each signal to unit power, and make a
   fixed, stratified 80/20 train/validation split.
3. **Train**: a four-block 1D CNN (32 → 64 → 128 → 256 channels) with Adam,
   learning-rate decay, and early stopping on validation loss.
4. **Evaluate**: overall and per-SNR accuracy, plus confusion matrices, for
   both modulations and families (family is derived from the predicted
   modulation).

Outputs go to the gitignored `runs/` (settings, metrics, figures) and
`models/best.pt`; rerunning overwrites them.

## Running it

```bash
uv sync
mkdir -p runs models
```

On Amarel, after copying the CSVs to `data/raw/HisarMod2019.1/`:

```bash
sbatch scripts/convert_hisarmod.slurm   # one-time conversion (CPU)
sbatch scripts/run_notebook.slurm       # train and evaluate (GPU)
```

To run interactively, register the kernel with
`uv run python -m ipykernel install --user --name modulation-classifier`,
open the notebook, and run cells top to bottom. Settings live in
`TrainingParameters`. Running the training cell (or Run All) starts training.

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy scripts
```

Mypy cannot read notebooks, so extract the code cells first:

```bash
uv run python -c "import json; nb = json.load(open('modulation_classification.ipynb')); print('\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'))" > /tmp/modulation_notebook.py
uv run mypy /tmp/modulation_notebook.py
```
