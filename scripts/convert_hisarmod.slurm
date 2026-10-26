#!/bin/bash
# Convert the extracted HisarMod CSVs into .npy arrays on an Amarel CPU node.
# Submit from the repo root (paths are relative to it):
#   mkdir -p runs && sbatch scripts/convert_hisarmod.slurm
#SBATCH --job-name=convert_hisarmod
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=runs/%x_%j.out

set -euo pipefail

uv run python scripts/convert_hisarmod.py
