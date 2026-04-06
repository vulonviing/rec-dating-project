from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rec_dating_project import ProjectPaths, RecDatingDataset, RoleBasedBipartiteNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sparse role-based bipartite sample.")
    parser.add_argument("--nrows", type=int, default=500_000, help="How many edges to use.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = ProjectPaths.default()
    paths.ensure_output_dirs()

    dataset = RecDatingDataset(paths.raw_edges_path)
    network = RoleBasedBipartiteNetwork(dataset)

    summary = dataset.compute_summary(nrows=args.nrows)
    snapshot = network.build_sparse_rating_matrix(nrows=args.nrows, summary=summary)

    report = {
        "nrows": args.nrows,
        "edge_count": snapshot.edge_count,
        "num_raters": snapshot.num_raters,
        "num_profiles": snapshot.num_profiles,
        "density": snapshot.density,
    }

    output_path = paths.output_data_dir / f"sample_snapshot_{args.nrows}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
