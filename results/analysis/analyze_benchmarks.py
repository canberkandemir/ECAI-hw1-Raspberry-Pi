import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
OUTPUT = ROOT / "results" / "analysis" / "results_comparison.csv"

RUNS = [1, 2]

results = {}
samples = {}

# Read benchmark results and individual timing samples.
for run in RUNS:
    csv_path = RAW / f"results_run{run}_recheck.csv"
    log_path = RAW / f"samples_run{run}_recheck.log"

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["status"] != "pass":
                continue

            kernel = row["kernel"]

            results.setdefault(kernel, {})[run] = (
                float(row["latency_median_us"]) / 1000
            )

    with log_path.open() as file:
        for line in file:
            if not line.startswith("SAMPLE,"):
                continue

            _, kernel, sample_number, latency_us = line.strip().split(",")

            samples.setdefault(kernel, []).append(
                float(latency_us) / 1000
            )

# Sparse scalar is the reference for speedup calculations.
baseline = statistics.median(samples["sparse_scalar"])

rows = []

for kernel, run_values in results.items():

    if len(run_values) != 2:
        raise ValueError(f"Missing benchmark run for {kernel}")

    kernel_samples = samples[kernel]

    if len(kernel_samples) != 10:
        raise ValueError(
            f"Expected 10 samples for {kernel}, got {len(kernel_samples)}"
        )

    run1 = run_values[1]
    run2 = run_values[2]

    median = statistics.median(kernel_samples)
    std = statistics.stdev(kernel_samples)

    run_variation = (
        abs(run2 - run1)
        / statistics.mean([run1, run2])
        * 100
    )

    speedup = baseline / median

    rows.append({
        "kernel": kernel,
        "run1_median_ms": round(run1, 3),
        "run2_median_ms": round(run2, 3),
        "combined_median_ms": round(median, 3),
        "combined_std_ms": round(std, 3),
        "run_variation_percent": round(run_variation, 3),
        "speedup_vs_sparse_scalar": round(speedup, 3),
        "total_samples": len(kernel_samples)
    })

# Save the analysis to CSV.
with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nCombined benchmark analysis (Run 1 + Run 2)")
print("-" * 75)

for row in rows:
    print(
        f"{row['kernel']:42s} "
        f"Median: {row['combined_median_ms']:8.3f} ms | "
        f"Std: {row['combined_std_ms']:7.3f} ms | "
        f"Speedup: {row['speedup_vs_sparse_scalar']:.2f}x"
    )

print(f"\nAnalysis saved to: {OUTPUT}")