import argparse

import numpy as np

from tsclust.clustering import cluster_time_series


def _generate_demo(n_per_cluster: int, length: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, length)
    data = []
    for _ in range(n_per_cluster):
        data.append(np.sin(t) + 0.1 * rng.standard_normal(length))
    for _ in range(n_per_cluster):
        data.append(np.cos(t) + 0.1 * rng.standard_normal(length))
    for _ in range(n_per_cluster):
        data.append(np.sin(2 * t) + 0.1 * rng.standard_normal(length))
    return np.vstack(data)


def _load_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolation Kernel + k-medoids demo")
    parser.add_argument("--input", type=str, default="", help="CSV path; rows are series")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--n-trees", type=int, default=200, help="Number of trees")
    parser.add_argument("--sample-size", type=int, default=256, help="Subsample size")
    parser.add_argument("--length", type=int, default=120, help="Demo series length")
    parser.add_argument("--per-cluster", type=int, default=30, help="Demo series per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.input:
        X = _load_csv(args.input)
    else:
        X = _generate_demo(args.per_cluster, args.length, args.seed)

    result = cluster_time_series(
        X,
        k=args.k,
        n_trees=args.n_trees,
        sample_size=args.sample_size,
        normalize=True,
        random_state=args.seed,
    )

    print("Medoids:", result.medoids)
    print("Labels (first 20):", result.labels[:20])


if __name__ == "__main__":
    main()
