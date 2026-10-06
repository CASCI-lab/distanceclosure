#!/usr/bin/env python3
"""Generate a reproducible set of random weighted distance graphs for backbone benchmarks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import networkx as nx
import numpy as np

SIZES = (50, 100, 200, 400, 800)
MEAN_DEGREE = 8.0
REPLICATES = 3
MODEL = "fast_gnp"
WEIGHT_ATTR = "weight"
WEIGHT_LOW = 0.1
WEIGHT_HIGH = 1.0


def make_connected_er(n: int, mean_degree: float, seed: int) -> tuple[nx.Graph, int]:
    """Build a connected ER graph via ``fast_gnp_random_graph``, retrying seeds until connected."""
    p = mean_degree / (n - 1)
    attempt_seed = seed
    while True:
        G = nx.fast_gnp_random_graph(n, p, seed=attempt_seed)
        if nx.is_connected(G):
            return G, attempt_seed
        attempt_seed += 1


def assign_distance_weights(G: nx.Graph, seed: int) -> None:
    rng = np.random.default_rng(seed)
    weights = {(u, v): float(rng.uniform(WEIGHT_LOW, WEIGHT_HIGH)) for u, v in G.edges()}
    nx.set_edge_attributes(G, weights, WEIGHT_ATTR)


def generate_graphs(
    out_dir: Path,
    sizes: tuple[int, ...] = SIZES,
    mean_degree: float = MEAN_DEGREE,
    replicates: int = REPLICATES,
    base_seed: int = 0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    rows: list[dict] = []
    for n in sizes:
        for r in range(replicates):
            seed = base_seed + n * 1000 + r
            G, used_seed = make_connected_er(n, mean_degree, seed)
            assign_distance_weights(G, seed=used_seed + 10_000)

            filename = f"er_n{n}_r{r}.graphml"
            path = out_dir / filename
            nx.write_graphml(G, path)

            rows.append(
                {
                    "path": filename,
                    "n": n,
                    "m": G.number_of_edges(),
                    "seed": used_seed,
                    "model": MODEL,
                    "c": mean_degree,
                }
            )
            print(f"Wrote {path} (n={n}, m={G.number_of_edges()}, seed={used_seed})")

    fieldnames = ["path", "n", "m", "seed", "model", "c"]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest: {manifest_path} ({len(rows)} graphs)")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "graphs",
        help="Directory for GraphML files and manifest.csv",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SIZES),
        help="Node counts to generate",
    )
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--mean-degree", type=float, default=MEAN_DEGREE)
    args = parser.parse_args()

    generate_graphs(
        out_dir=args.out_dir,
        sizes=tuple(args.sizes),
        mean_degree=args.mean_degree,
        replicates=args.replicates,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
