#!/bin/bash
# Submit from the repository root after uv sync and mkdir -p runs models.
#SBATCH --job-name=train_cnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=runs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the repository root}"
mkdir -p runs models
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

nvidia-smi
uv run --frozen --no-sync python -u - <<'PYTHON'
import json
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError("This GPU job requires CUDA-enabled PyTorch.")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# The notebook uses plain Python; shared globals preserve cell execution order.
# Figures and metrics are saved by its cells, not embedded into the notebook.
notebook_path = Path("modulation_classification.ipynb")
notebook = json.loads(notebook_path.read_text())
for index, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] == "code":
        print(f"Running notebook cell {index}", flush=True)
        code = compile("".join(cell["source"]), f"{notebook_path}:cell_{index}", "exec")
        exec(code, globals())
print("Training and validation reporting complete.", flush=True)
PYTHON
