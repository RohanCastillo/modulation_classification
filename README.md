# Modulation Classifier

Classify the modulation type of short raw radio snippets (HisarMod2019.1)
with a cumulant-feature baseline and a 1D CNN.

The first version focuses on a reproducible data pipeline, these two models,
a clear comparison, and a finished results write-up. Advanced experiments
and deployment are tracked under Future Improvements in [plan.md](plan.md).

## Problem

A radio receiver records a short burst of samples and has to work out how the
signal was encoded (for example QPSK, 16-QAM, or FM) without being told.
This project trains models to make that call from 1,024 raw I/Q samples,
across 26 modulation types in 5 families, and measures how accuracy falls as
the signal gets noisier.

## Results

TODO: accuracy-vs-SNR chart and results table (26 modulations and 5 families).

## Dataset

- Source: HisarMod2019.1 on IEEE DataPort
  (https://ieee-dataport.org/node/1785, DOI 10.21227/8k12-2g70). Downloading
  requires a free IEEE account.
- File: `HisarMod2019.1.zip` (5.5 GB) plus `readme.txt`, saved to `data/raw/`
- SHA-256 (zip):
  `9276c4bc445b6754ce770e997cf3c2f16a581db090ff0829286c5884c2ed79e0`
- License: none stated on the DataPort page; cite the dataset as below.
- Location: extract the ZIP to `data/raw/HisarMod2019.1/` (relative to the
  repo root); converted arrays go to `data/processed/`

Contents (CSV, one signal per row; rows sorted by SNR, then modulation):

| File | Rows | Contents |
|---|---|---|
| `Train/train_data.csv` (8.9 GB) | 520,000 | 1,024 complex samples per row, MATLAB format (`a+bi`) |
| `Train/train_labels.csv` | 520,000 | modulation code (see `readme.txt`) |
| `Train/train_snr.csv` | 520,000 | SNR in dB |
| `Test/test_data.csv` (4.5 GB) | 260,000 | as above |
| `Test/test_labels.csv` | 260,000 | as above |
| `Test/test_snr.csv` | 260,000 | as above |

Each (modulation, SNR) pair has 1,000 train and 500 test signals. The channel
type (ideal, static, Rayleigh, Rician, Nakagami-m) is not labeled.

Citation: K. Tekbıyık, C. Keçeci, A. R. Ekti, A. Görçin, and G. Karabulut
Kurt, "HisarMod: A new challenging modulated signals dataset," IEEE Dataport,
2019. doi:10.21227/8k12-2g70

## Setup

```
uv sync
```

PyTorch is pinned to 2.11.0 from the explicit CUDA 12.8 package index to
support Amarel's NVIDIA 570 driver. Other packages use PyPI. After pulling
dependency updates on Amarel, run `uv sync --frozen` before submitting a job;
the batch job uses the existing environment without syncing it.

## Usage

Convert the dataset on Amarel. Copy the extracted CSVs to
`data/raw/HisarMod2019.1/` in the repo there, then submit the Slurm job
from the repo root:

```bash
mkdir -p runs
sbatch scripts/convert_hisarmod.slurm
squeue -u $USER   # check whether it is queued or running
```

The job requests a CPU node with 16 GB of memory and writes its log to
`runs/convert_hisarmod_<job id>.out`. The `runs/` folder must exist before
submitting, because Slurm opens the log file before the job starts.

The script takes no arguments: it reads `data/raw/HisarMod2019.1/` and
writes `data/processed/`, which must not already exist. There is no row
limit, so every run converts the full dataset.

For each of `train` and `test`, it writes `{partition}_iq.npy` (float32,
`(n, 2, 1024)`, unnormalized, I then Q), `{partition}_labels.npy` (int64
raw codes), and `{partition}_snr_db.npy` (int64 dB). Row order is
preserved. Each partition is parsed fully in memory before saving, so the
train partition peaks at about 9 GB; request at least 12 GB on Amarel.
Normalization and splitting come later.

The row parser, `parse_iq_row`, assumes valid HisarMod rows with 1,024
finite samples; it does not validate the input.

## Learning notebook

Open [modulation_classification.ipynb](modulation_classification.ipynb) in
VS Code and select this project's `.venv/bin/python` as its kernel after
running `uv sync`. Restart the kernel and run cells from top to bottom.

The notebook contains the existing loading, power normalization, stratified
split, label mappings, accuracy-by-SNR helper, and CNN definition, with
explanations beside each step. Section 9 adds the CNN training loop with Adam, validation loss, early
stopping, and restoration of the best checkpoint. Running the execution cell starts training; Run All does too. The feature baseline, comparison plots, and final evaluation remain
planned in the notebook and [plan.md](plan.md).

After conversion on Amarel, configure `training_params = TrainingParameters(...)`
and run the notebook in order. Set `data_dir` to the converted arrays. Training uses only the authors' training set
with the fixed 80/20 split and selects CUDA when available. Initial settings
are 50 epochs with patience 5. For a short run, use a prepared real-data
subset and reduce `training_params.epochs`; synthetic smoke mode has been removed.

Section 9 follows three functions: `run_training_epoch` updates weights,
`evaluate` measures loss and accuracy, and `train` coordinates epochs, early
stopping, and best-weight restoration. A short execution cell starts training;
settings and JSON result saving are separate. The notebook writes settings and
losses under `runs/` and its best weights to `models/best.pt`.
The loop saves improvements based on validation loss and reloads the best
weights afterward. Epoch history is written after training returns; interrupted
runs retain their best checkpoint but may lose unsaved history. Checkpoints do not include optimizer state for resuming.
The test partition is never loaded by training. Validation accuracy is recorded each epoch; detailed SNR
and family reporting remain part of the planned evaluation work.

The former `src/` package has been removed. This is now a uv project for
managing the notebook environment, not an installable Python package.
No new dependencies are required to use the notebook in an existing editor.

### Amarel

Keep the conversion workflow above. On Amarel, run `uv sync` from the project
root, then register the environment for an existing Jupyter installation:

```bash
uv run python -m ipykernel install --user \
    --name modulation-classifier \
    --display-name "Modulation Classifier"
```

Open the notebook in an Amarel Jupyter compute session and choose
**Modulation Classifier**. The session must be able to access the project
and its environment. Use the repository root as the working directory for
relative data paths. Request a GPU allocation for CNN training;
do not train on a login node. See the
[Amarel Open OnDemand guide](https://user-guide-f4258a.gitlab.io/connect/open-ondemand/)
for cluster access. Available interactive apps and GPU choices depend on
Amarel's current configuration.

For full batch training, run these commands on the login node from the
repository root after conversion:

```bash
uv sync --frozen
mkdir -p runs models
sbatch scripts/run_notebook.slurm
```

The job requests one GPU, four CPU cores, 16 GB RAM, and a 12-hour time
limit (not an estimated runtime). It checks CUDA availability before running
the notebook's ordinary Python code cells in order, using the existing uv
environment without installing packages on the compute node. No notebook
execution dependency is needed. Notebook magics and shell commands are not
supported by this runner. It does not save an executed notebook; metrics,
weights, and figures are saved by the existing notebook cells.

Settings remain in `TrainingParameters`: by default, up to 50 epochs with
patience 5, using all converted training data and the fixed validation split.
For a short batch run, point `data_dir` at a prepared real-data subset and
reduce `epochs` before submission. Test data is not evaluated.

Follow progress with `tail -f runs/train_cnn_JOBID.out`, substituting the
submitted job ID. Successful execution ends with
`Training and validation reporting complete.` Check Slurm completion with
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed`.
Figures use a noninteractive backend. Existing output files are overwritten;
archive them before another run and do not run concurrent jobs in this checkout.
If the time limit is reached, the best weights remain saved, but history and
final reporting may be missing; the checkpoint cannot resume optimizer state.

### Development checks

Ruff checks the notebook directly:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy scripts
```

Mypy does not read `.ipynb` files directly. To type-check notebook definitions,
extract the code cells into a temporary Python file and check that file:

```bash
uv run python - <<'PYTHON'
import json
from pathlib import Path

notebook = json.loads(Path("modulation_classification.ipynb").read_text())
code = "\n\n".join(
    "".join(cell["source"])
    for cell in notebook["cells"]
    if cell["cell_type"] == "code"
)
Path("/tmp/modulation_notebook.py").write_text(code + "\n")
PYTHON
uv run mypy --config-file pyproject.toml /tmp/modulation_notebook.py
```

Definition cells can be run without data. The data-loading and training
cells require converted arrays; Run All loads data and starts training. Final test evaluation remains unimplemented
and must stay opt-in when added.

After training, the validation prediction cell collects `true_codes`,
`predicted_codes`, and `validation_snr_db` in matching validation-row order.
It uses the restored best weights and converts CNN class indices back to raw
codes. `calculate_metrics()` then returns a dictionary of overall modulation and
family accuracy plus a DataFrame of both accuracies per SNR in dB. A notebook
cell displays these validation results. Two separate figures then show modulation and family accuracy versus SNR,
with percentage axes. Four row-normalized confusion matrices compare families and all 26 modulations
at low SNR (< 0 dB) and high SNR (≥ 0 dB), with a shared 0–1 color scale.
Each figure has a separate pyplot cell, with color only and no plotting
helper. Both SNR groups must contain examples. Prediction arrays and reporting outputs are saved as described below.

The notebook also plots training and validation cross-entropy versus epoch
from the returned history, using a separate figure after training.

## Project layout

```text
modulation_classification.ipynb  # explanations and current implementations
scripts/
    convert_hisarmod.py          # one-time CSV-to-.npy conversion
    convert_hisarmod.slurm       # conversion job on Amarel
    run_notebook.slurm           # full CNN training and validation GPU job
pyproject.toml                  # uv dependencies and tool configuration
uv.lock                         # reproducible dependency versions
plan.md                         # remaining experiments and deliverables
```

Data, saved models, and run outputs remain in the gitignored `data/`,
`models/`, and `runs/` directories.

Before running the notebook on a fresh checkout or Amarel, create its output
folders from the project root:

```bash
mkdir -p runs models
```

The settings cell uses these existing folders. Settings (`runs/config.json`),
history (`runs/metrics.json`), and weights (`models/best.pt`) are overwritten
on subsequent runs; copy any results you want to retain before rerunning.
The folders remain gitignored and are not included in a clone.

Validation reporting saves the following files in the existing `runs/` folder:

- `validation_predictions.npz`: aligned true codes, predicted codes, and SNRs.
- `validation_metrics.json`: overall modulation and family accuracy.
- `validation_accuracy_by_snr.csv`: both accuracies indexed by SNR in dB.
- `figures/`: seven PNGs (loss, two accuracy curves, four confusion matrices).

The settings cell creates `runs/figures/`. Each plot saves at 200 dpi before
display. Rerunning replaces matching files; copy results you want to keep.
These outputs remain gitignored.
