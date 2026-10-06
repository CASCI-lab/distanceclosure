#!/usr/bin/env python3
"""Benchmark the four exact distance-backbone algorithms on saved graphs."""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx

from distanceclosure import distance_backbone

ALGORITHMS = ("iterative", "flagged", "closure", "heuristic")
WEIGHT_ATTR = "weight"


def load_manifest(graphs_dir: Path) -> list[dict]:
    manifest_path = graphs_dir / "manifest.csv"
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f))


def edge_set(G: nx.Graph) -> set[tuple]:
    if G.is_directed():
        return set(G.edges())
    return {tuple(sorted(e)) for e in G.edges()}


def spot_check(graphs_dir: Path, rows: list[dict]) -> None:
    """Assert the four exact algorithms agree on the smallest graphs."""
    if not rows:
        return
    min_n = min(int(r["n"]) for r in rows)
    checked = 0
    for row in rows:
        if int(row["n"]) != min_n:
            continue
        G = nx.read_graphml(graphs_dir / row["path"])
        # GraphML may store node ids as strings; algorithms are fine with that.
        results = {
            algo: edge_set(
                distance_backbone(
                    G,
                    weight=WEIGHT_ATTR,
                    kind="metric",
                    algorithm=algo,
                    distortion=False,
                    self_loops=False,
                    verbose=False,
                )
            )
            for algo in ALGORITHMS
        }
        ref = results[ALGORITHMS[0]]
        for algo, edges in results.items():
            if edges != ref:
                raise AssertionError(
                    f"Edge-set mismatch on {row['path']}: {ALGORITHMS[0]} vs {algo} "
                    f"(|{ALGORITHMS[0]}|={len(ref)}, |{algo}|={len(edges)})"
                )
        checked += 1
        print(f"Spot-check OK: {row['path']} ({len(ref)} backbone edges)")
    print(f"Spot-check passed on {checked} graph(s) with n={min_n}")


def run_benchmarks(
    graphs_dir: Path,
    results_path: Path,
    repeats: int = 5,
    skip_spot_check: bool = False,
) -> Path:
    rows = load_manifest(graphs_dir)
    if not skip_spot_check:
        spot_check(graphs_dir, rows)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["n", "m", "seed", "algorithm", "trial", "seconds"]
    timing_rows: list[dict] = []

    for row in rows:
        path = graphs_dir / row["path"]
        n = int(row["n"])
        m = int(row["m"])
        seed = int(row["seed"])
        print(f"Benchmarking {path.name} (n={n}, m={m})")

        for algo in ALGORITHMS:
            for trial in range(repeats):
                G = nx.read_graphml(path)
                t0 = time.perf_counter()
                distance_backbone(
                    G,
                    weight=WEIGHT_ATTR,
                    kind="metric",
                    algorithm=algo,
                    distortion=False,
                    self_loops=False,
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0
                timing_rows.append(
                    {
                        "n": n,
                        "m": m,
                        "seed": seed,
                        "algorithm": algo,
                        "trial": trial,
                        "seconds": f"{elapsed:.6f}",
                    }
                )
                print(f"  {algo:10s} trial {trial}: {elapsed:.4f}s")

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_rows)

    print_summary(timing_rows)
    print(f"Wrote {results_path} ({len(timing_rows)} rows)")
    return results_path


def print_summary(timing_rows: list[dict]) -> None:
    buckets: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in timing_rows:
        buckets[(int(row["n"]), row["algorithm"])].append(float(row["seconds"]))

    print("\nMean seconds by (n, algorithm):")
    print(f"{'n':>6}  {'algorithm':12}  {'mean_s':>10}  {'trials':>6}")
    for (n, algo) in sorted(buckets.keys(), key=lambda x: (x[0], ALGORITHMS.index(x[1]))):
        vals = buckets[(n, algo)]
        mean = sum(vals) / len(vals)
        print(f"{n:6d}  {algo:12}  {mean:10.4f}  {len(vals):6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "graphs",
        help="Directory containing GraphML files and manifest.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "timings.csv",
        help="Output CSV path",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--skip-spot-check",
        action="store_true",
        help="Skip equality check of the four exact algorithms on smallest graphs",
    )
    args = parser.parse_args()

    run_benchmarks(
        graphs_dir=args.graphs_dir,
        results_path=args.results,
        repeats=args.repeats,
        skip_spot_check=args.skip_spot_check,
    )


if __name__ == "__main__":
    main()
