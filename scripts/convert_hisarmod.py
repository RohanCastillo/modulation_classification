"""Convert extracted HisarMod CSV files into NumPy arrays."""

from pathlib import Path

import numpy as np


def parse_iq_row(row):
    sample_texts = row.strip().replace("i", "j").split(",")
    samples = np.array(sample_texts, dtype=np.complex64)
    return np.array([samples.real, samples.imag], dtype=np.float32)


def convert_partition(data_dir, output_dir, partition):
    folder = data_dir / partition.title()
    labels = np.loadtxt(
        folder / f"{partition}_labels.csv",
        dtype=np.int64,
        ndmin=1,
    )
    snr_db = np.loadtxt(
        folder / f"{partition}_snr.csv",
        dtype=np.int64,
        ndmin=1,
    )
    with (folder / f"{partition}_data.csv").open(encoding="utf-8") as rows:
        signals = np.array(
            [parse_iq_row(row) for row in rows], dtype=np.float32
        )
    np.save(output_dir / f"{partition}_iq.npy", signals)
    np.save(output_dir / f"{partition}_labels.npy", labels)
    np.save(output_dir / f"{partition}_snr_db.npy", snr_db)
    print(f"Converted {partition}: {len(labels):,} signals", flush=True)


def main():
    data_dir = Path("data/raw/HisarMod2019.1")
    output_dir = Path("data/processed")

    output_dir.mkdir(parents=True, exist_ok=False)
    for partition in ("train", "test"):
        convert_partition(data_dir, output_dir, partition)


if __name__ == "__main__":
    main()
