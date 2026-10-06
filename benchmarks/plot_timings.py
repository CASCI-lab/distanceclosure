#!/usr/bin/env python3
"""Plot backbone benchmark speedups relative to the closure method."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMPARE = (
    "iterative",
    "flagged",
    "heuristic",
    "annotated",
    "annotated_twostep",
    "annotated_heuristic",
    "iterative_heuristic",
    "iterative_weightcut",
    "annotated_weightcut",
)
BASELINE = "closure"


def load_means(timings_path: Path) -> pd.DataFrame:
    df = pd.read_csv(timings_path)
    means = (
        df.groupby(["n", "algorithm"], as_index=False)["seconds"]
        .mean()
        .rename(columns={"seconds": "mean_seconds"})
    )
    return means


def speedup_table(means: pd.DataFrame) -> pd.DataFrame:
    pivot = means.pivot(index="n", columns="algorithm", values="mean_seconds")
    missing = [a for a in (BASELINE, *COMPARE) if a not in pivot.columns]
    if missing:
        raise ValueError(f"Missing algorithms in timings: {missing}")

    rows = []
    for n, row in pivot.iterrows():
        base = row[BASELINE]
        for algo in COMPARE:
            rows.append(
                {
                    "n": int(n),
                    "algorithm": algo,
                    "mean_seconds": row[algo],
                    "closure_seconds": base,
                    "speedup": base / row[algo],
                }
            )
    return pd.DataFrame(rows)


def make_plot(speedups: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)

    # Left: wall time (log) with closure included for context
    ax = axes[0]
    raw = speedups[["n", "algorithm", "mean_seconds"]].copy()
    closure = (
        speedups[["n", "closure_seconds"]]
        .drop_duplicates("n")
        .rename(columns={"closure_seconds": "mean_seconds"})
        .assign(algorithm=BASELINE)
    )
    raw = pd.concat([raw, closure], ignore_index=True)
    for algo in (*COMPARE, BASELINE):
        sub = raw[raw["algorithm"] == algo].sort_values("n")
        ax.plot(sub["n"], sub["mean_seconds"], marker="o", label=algo)
    ax.set_yscale("log")
    ax.set_xlabel("nodes (n)")
    ax.set_ylabel("mean runtime (s)")
    ax.set_title("Runtime vs size")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)

    # Right: speedup vs closure
    ax = axes[1]
    for algo in COMPARE:
        sub = speedups[speedups["algorithm"] == algo].sort_values("n")
        ax.plot(sub["n"], sub["speedup"], marker="o", label=algo)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1, label="closure (1×)")
    ax.set_xlabel("nodes (n)")
    ax.set_ylabel(f"speedup vs {BASELINE}  (T_closure / T)")
    ax.set_title("Speedup relative to closure")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Backbone algorithms vs closure", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timings",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "timings.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "speedup_vs_closure.png",
    )
    args = parser.parse_args()

    means = load_means(args.timings)
    speedups = speedup_table(means)
    print(speedups.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    make_plot(speedups, args.out)


if __name__ == "__main__":
    main()
